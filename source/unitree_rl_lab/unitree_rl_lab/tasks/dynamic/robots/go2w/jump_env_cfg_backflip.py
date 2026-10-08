"""Go2w-Backflip: one backward somersault, on its own, on top of Go2w-Jump's recipe.

Everything that made ``Go2w-Jump`` work carries over unchanged -- measured standing
height, overshoot-tolerant curriculum at 0.005/step, the Go2-Jump-60 reward fixes and
the horizontal-drift penalty. The motion is one turn of pitch (``target_pitch_turns =
-1``) with Phase 2's 350 N on the FRONT hips during the launch window, plus two assist
additions found by sweeping the Phase 1 policy under full assist (19.52 kg robot,
32 envs, 2026-10-08):

    assist                                   turns  max_h  success  fell  upright
    Phase 2 as-is (350 N front only)          --     --     0.00    1.00   0.00
    + lift 0.30 m + 200 N couple @0.25 s     1.00   0.48   1.00    0.00   1.00

- ``backflip_target_height = 0.30``: the jump's projectile-sized launch force, for air time.
  Without it the front force just tips the robot onto its back.
- ``backflip_couple_force = 200`` at 0.25-0.35 s after the trigger: up on the front
  hips, down on the rear, so it adds pitch with no lift, applied once airborne.

Note: ``backflip_assist_force`` is *assigned* to the front hips in
``JumpCommand._apply_assistance``, replacing the launch force there (175 N each instead of
launch + 175 N). The sweep and the trained policy both ran with that behaviour, so it is
kept; making it additive would need a new sweep.

The window is narrow -- 150 N or 250 N, or 0.20/0.30 s, each gave clearly worse
upright landings -- but it only has to hold at the start of the curriculum.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_jump import (
    PLAY_ASSIST_SCALE,
    CommandsCfgJump,
    RobotEnvCfgJump,
)

PITCH_TURNS = -1.0
EXPERIMENT_DIR = "logs/rsl_rl/go2w_backflip"


@configclass
class CommandsCfgBackflip(CommandsCfgJump):
    jump = CommandsCfgJump().jump.replace(
        enable_jump=False,
        enable_backflip=True,
        enable_sideflip=False,
        target_pitch_turns_range=(PITCH_TURNS, PITCH_TURNS),
        backflip_target_height=0.30,
        backflip_couple_force=200.0,
        backflip_couple_delay_s=0.25,
        backflip_couple_duration_s=0.10,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class RobotEnvCfgBackflip(RobotEnvCfgJump):
    """Assisted backflip-only task."""

    commands: CommandsCfgBackflip = CommandsCfgBackflip()


@configclass
class RobotPlayEnvCfgBackflip(RobotEnvCfgBackflip):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = PLAY_ASSIST_SCALE
        self.commands.jump.debug_vis = True
