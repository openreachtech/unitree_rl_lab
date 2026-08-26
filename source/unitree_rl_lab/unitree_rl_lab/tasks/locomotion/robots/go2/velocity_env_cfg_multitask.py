"""Gallop Phase 1 and Phase 2, re-expressed on the unified multi-task observation.

The mirror image of ``dynamic/robots/go2/jump_env_cfg_multitask.py``. Same two-stage structure as
``Go2-Gallop-Phase1`` / ``Go2-Gallop-Phase2`` -- Phase 1 learns forward-only gallop-style running
behind a decaying tow assist, Phase 2 generalises to omnidirectional commands -- and identical to
them in every term that defines the task: commands, rewards, terminations, events, terrain, and both
curricula. Only the observation changes, to the 122/330-column superset shared with the acrobatics
side (see :mod:`unitree_rl_lab.tasks.multitask.obs_spec`).

The jump command is present but never triggered, which is what fills the five columns the locomotion
observation was missing. ``JumpCommand`` leaves ``enabled`` false for the whole episode when
``auto_trigger`` is off, so ``jump_command`` stays ``(0, 0, 0, 0)`` and ``jump_time`` stays 0 -- and
its assist force is gated on ``enabled & (trigger_step >= 0) & (assist_scale > 0)``, so with the
scale also pinned to zero there are two independent reasons it can never perturb the locomotion
physics. That matters: this has to be the same command term the merged environment uses, not a
stand-in that behaves differently once it starts firing.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg import CommandsCfg as JumpCommandsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_run import (
    CommandsCfgGo2GallopPhase1,
    CommandsCfgGo2GallopPhase2,
    RobotEnvCfgGo2GallopPhase1,
    RobotEnvCfgGo2GallopPhase2,
)
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    MultitaskEventCfg,
    MultitaskSceneCfg,
    UnifiedObservationsCfg,
    apply_multitask_post_init,
)

# The acrobatics task's own jump command, held inert. Reusing it rather than writing a zero stub
# keeps a single definition of what those five observation columns mean across both families.
_IDLE_JUMP_COMMAND = JumpCommandsCfg().jump.replace(
    auto_trigger=False,
    initial_assist_scale=0.0,
    state_file=None,
    debug_vis=False,
)


# =================================================================================================
# Phase 1 -- forward-only gallop on the unified observation
# =================================================================================================


@configclass
class MultitaskCommandsCfgGallopPhase1(CommandsCfgGo2GallopPhase1):
    jump = _IDLE_JUMP_COMMAND

    tow_assist = CommandsCfgGo2GallopPhase1().tow_assist.replace(
        # Own decay state: sharing it with the 117-column run would make it impossible to tell
        # which run's tow assist a given scale belongs to.
        state_file="logs/rsl_rl/go2_multitask_gallop_phase1/tow_assist_state.json"
    )


@configclass
class RobotEnvCfgMultitaskGallopPhase1(RobotEnvCfgGo2GallopPhase1):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    commands: MultitaskCommandsCfgGallopPhase1 = MultitaskCommandsCfgGallopPhase1()
    events: MultitaskEventCfg = MultitaskEventCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_multitask_post_init(self)


@configclass
class RobotPlayEnvCfgMultitaskGallopPhase1(RobotEnvCfgMultitaskGallopPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
        self.observations.policy.enable_corruption = False


# =================================================================================================
# Phase 2 -- omnidirectional commands on the unified observation
# =================================================================================================


@configclass
class MultitaskCommandsCfgGallopPhase2(CommandsCfgGo2GallopPhase2):
    jump = _IDLE_JUMP_COMMAND

    tow_assist = CommandsCfgGo2GallopPhase2().tow_assist.replace(
        state_file="logs/rsl_rl/go2_multitask_gallop_phase2/tow_assist_state.json"
    )


@configclass
class RobotEnvCfgMultitaskGallopPhase2(RobotEnvCfgGo2GallopPhase2):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    commands: MultitaskCommandsCfgGallopPhase2 = MultitaskCommandsCfgGallopPhase2()
    events: MultitaskEventCfg = MultitaskEventCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_multitask_post_init(self)


@configclass
class RobotPlayEnvCfgMultitaskGallopPhase2(RobotEnvCfgMultitaskGallopPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
        self.observations.policy.enable_corruption = False
