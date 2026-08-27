"""Jump Phase 1 and Phase 2, re-expressed on the unified multi-task observation.

Same two-stage structure as ``Go2-Jump-Phase1`` / ``Go2-Jump-Phase2`` -- Phase 1 learns quiet
standing, Phase 2 adds the assisted jump/backflip/sideflip and decays the assist away -- and
identical to them in every term that defines the task: commands, rewards, terminations, events,
episode length, the assist-force curriculum. The only difference is what the network sees: the
122/330-column superset shared with the locomotion side (see
:mod:`unitree_rl_lab.tasks.multitask.obs_spec`) instead of 47/56.

The point is to produce an acrobatics expert whose input layout already matches the multi-task
policy, so assembling the mixture of experts needs no weight surgery at all. Both phases have to
move together: Phase 2 resumes from Phase 1, so a Phase 1 left on the old observation could not
seed it.

Keeping the task itself untouched is deliberate. ``Go2-Jump-Phase2`` took ~6800 iterations through a
decaying assist curriculum to reach an unassisted flip, and changing the reward or the trigger
schedule in the same step as the observation would make any regression impossible to attribute.
Start from the existing checkpoints, widened by ``scripts/rsl_rl/widen_checkpoint.py``: the widened
network is mathematically identical, so it begins with the skill intact and only has to learn to use
the new columns.

The velocity command is present but pinned to zero. That fills its observation slot truthfully --
the robot really is being asked to hold still -- rather than feeding the network a command nothing
in the reward makes meaningful, which would teach it those three columns are noise. Training the
take-off from a *moving* start belongs in a follow-up phase, where the velocity command and the
reset velocity can be made to agree.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg import CommandsCfg, RobotEnvCfg
from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg_phase2 import CommandsCfgPhase2, RobotEnvCfgPhase2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import CommandsCfgPhase1
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    MultitaskEventCfg,
    MultitaskSceneCfg,
    UnifiedObservationsCfg,
    apply_multitask_post_init,
)

_ZERO_RANGES = mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
)

# Built by replacing fields on the locomotion command rather than constructing a fresh one, so every
# field this task does not care about keeps the value the locomotion policy was trained with -- the
# same command term has to serve the merged environment later.
_ZERO_VELOCITY_COMMAND = CommandsCfgPhase1().base_velocity.replace(
    ranges=_ZERO_RANGES,
    limit_ranges=_ZERO_RANGES,
    # Nothing to resample: the command is constant. Matches the jump command's own convention.
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
)


# =================================================================================================
# Phase 1 -- quiet standing on the unified observation
# =================================================================================================


@configclass
class MultitaskCommandsCfgPhase1(CommandsCfg):
    base_velocity = _ZERO_VELOCITY_COMMAND


@configclass
class RobotEnvCfgMultitaskPhase1(RobotEnvCfg):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    commands: MultitaskCommandsCfgPhase1 = MultitaskCommandsCfgPhase1()
    events: MultitaskEventCfg = MultitaskEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # The parent set up the acrobatics scene's timing; redo it for this scene, which also
        # carries the locomotion height scanner.
        apply_multitask_post_init(self)


@configclass
class RobotPlayEnvCfgMultitaskPhase1(RobotEnvCfgMultitaskPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.observations.policy.enable_corruption = False


# =================================================================================================
# Phase 2 -- assisted jump / backflip / sideflip on the unified observation
# =================================================================================================


@configclass
class MultitaskCommandsCfgPhase2(CommandsCfgPhase2):
    base_velocity = _ZERO_VELOCITY_COMMAND

    jump = CommandsCfgPhase2().jump.replace(
        # Held for the whole motion rather than its first third. The 0.5 s default drops the
        # command -- and with it the `enabled` flag the merged policy's gate reads -- a third of
        # the way through a ~1.2 s flip. Measured consequence in the merged task: 59% of the
        # action came from the locomotion expert while the robot was inverted, flip success
        # capped at 0.55, and the take-off speed curriculum stalled at 0.8 m/s. At 1.5 s the same
        # numbers are 0.7% locomotion, 0.79 success, and the curriculum reaches its 3.5 m/s
        # ceiling. Promoted from the Try-Long-Command sandbox.
        #
        # This value must equal the merged environment's ACRO_WINDOW_S and the deploy state's
        # command_duration_s: the expert, the reward window and the routing prior all key off the
        # same flag, and a disagreement is silent -- the robot still moves.
        command_duration_s=1.5,
        # Own curriculum state: assist decay must not be shared with the 0.5 s lineage.
        state_file="logs/rsl_rl/go2_multitask_jump_phase2/jump_curriculum_state.json",
    )


@configclass
class RobotEnvCfgMultitaskPhase2(RobotEnvCfgPhase2):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    commands: MultitaskCommandsCfgPhase2 = MultitaskCommandsCfgPhase2()
    events: MultitaskEventCfg = MultitaskEventCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_multitask_post_init(self)
        self.episode_length_s = 4.0


@configclass
class RobotPlayEnvCfgMultitaskPhase2(RobotEnvCfgMultitaskPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        # Play ignores the training curriculum's saved decay and runs with no external assist.
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = 0.0
        self.observations.policy.enable_corruption = False
