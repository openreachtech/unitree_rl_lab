"""Go2w-Jump-Phase2: jump + backflip + sideflip in one policy, one motion per episode.

Recombines the three single-motion tasks, each of which reached assist_scale 0 and
success 1.0 on its own from Phase 1 (2026-10-08):

- ``Go2w-Jump``     -- recipe base: measured stance 0.405 m, overshoot-tolerant success,
                       decay 0.005, Go2-Jump-60 reward fixes, xy-drift penalty.
- ``Go2w-Backflip`` -- 0.30 m launch + 200 N post-take-off pitch couple.
- ``Go2w-Sideflip`` -- 0.50 m launch + 65 N*m whole-body spin, no one-sided force.

Every per-motion setting is read from those tasks' command configs rather than copied, so
retuning one of them retunes Phase 2 with it. The rewards are Go2w-Jump's, which all three
already trained with.

One assist_scale is shared by the three motions and decays only while *each* of them is
at >= 0.60 success (``assist_force_decay``), so the slowest motion sets the pace.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_backflip import CommandsCfgBackflip
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_jump import (
    PLAY_ASSIST_SCALE,
    CommandsCfgJump,
    RobotEnvCfgJump,
)
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_sideflip import CommandsCfgSideflip

EXPERIMENT_DIR = "logs/rsl_rl/go2w_jump_phase2"

_BACKFLIP = CommandsCfgBackflip().jump
_SIDEFLIP = CommandsCfgSideflip().jump


@configclass
class CommandsCfgPhase2(CommandsCfgJump):
    jump = CommandsCfgJump().jump.replace(
        enable_jump=True,
        enable_backflip=True,
        enable_sideflip=True,
        # backflip
        target_pitch_turns_range=_BACKFLIP.target_pitch_turns_range,
        backflip_target_height=_BACKFLIP.backflip_target_height,
        backflip_assist_force=_BACKFLIP.backflip_assist_force,
        backflip_couple_force=_BACKFLIP.backflip_couple_force,
        backflip_couple_delay_s=_BACKFLIP.backflip_couple_delay_s,
        backflip_couple_duration_s=_BACKFLIP.backflip_couple_duration_s,
        # sideflip
        target_roll_turns_range=_SIDEFLIP.target_roll_turns_range,
        sideflip_target_height=_SIDEFLIP.sideflip_target_height,
        sideflip_assist_force=_SIDEFLIP.sideflip_assist_force,
        sideflip_spin_torque=_SIDEFLIP.sideflip_spin_torque,
        sideflip_spin_delay_s=_SIDEFLIP.sideflip_spin_delay_s,
        sideflip_spin_duration_s=_SIDEFLIP.sideflip_spin_duration_s,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgJump):
    """Assisted jump + backflip + sideflip (one motion sampled per env per episode)."""

    commands: CommandsCfgPhase2 = CommandsCfgPhase2()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = PLAY_ASSIST_SCALE
        self.commands.jump.debug_vis = True
