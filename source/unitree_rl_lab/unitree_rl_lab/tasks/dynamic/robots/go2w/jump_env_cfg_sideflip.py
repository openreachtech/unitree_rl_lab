"""Go2w-Sideflip: one sideways somersault, on its own, on top of Go2w-Jump's recipe.

Go2w-Jump's settings unchanged; only the motion and its assist differ. Phase 2's sideflip
assist (350 N up on the right hips, no height) left the Phase 1 policy at ~0.3 turns and
on its side, and so did go2's Sideflip-Double recipe (launch + post-take-off hip couple,
up to 1800 N): max 0.71 turns, success 0.

Why: go2w's roll inertia is 4x go2's (whole-body Ixx 0.956 vs 0.236 kg*m^2), because the
calf (0.80 kg vs 0.15) and wheel (0.52 kg vs 0.04) sit at the far end of each leg. Hip
forces have only +/-0.047 m of roll lever and reach that inertia only through the soft
hip joints (kp 25), so pushing harder flexed the hips and the base sprang back (traced:
-11 rad/s on the base, -2.6 rad/s once the legs caught up).

The assist here is therefore:
- ``sideflip_target_height = 0.50``: the projectile launch force on all four hips, for air time.
- ``sideflip_assist_force = 0``: no one-sided lift (it tips the robot before take-off).
- ``sideflip_spin_torque = 65`` N*m from 0.26 s for 0.10 s after the trigger: a rigid
  whole-body spin (``JumpCommand._whole_body_spin``) -- every body gets its share of one
  angular acceleration, so nothing is transmitted through the joints. Measured
  -10 to -12 rad/s steady roll from 120 N*m, matching 120*0.1/0.956 = 12.5 rad/s.

Phase 1 policy under full assist (32 envs, 2026-10-08):

    lift  spin  delay | turns mean/min | max_h | success  upright_end
    0.50   65   0.26  |  1.01 / 0.92   | 0.64  |  0.44     0.41
    0.40   75   0.26  |  0.99 / 0.79   | 0.48  |  0.38     0.38
    0.50   70   0.26  |  1.09 / 0.89   | 0.65  |  0.16     0.06

The rotation is right; the failures are landings, which the standing Phase 1 policy was
never trained for. success sits under the 0.60 gate, so the assist stays at 1.0 until the
policy learns to land.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_jump import (
    PLAY_ASSIST_SCALE,
    CommandsCfgJump,
    RobotEnvCfgJump,
)

ROLL_TURNS = -1.0
EXPERIMENT_DIR = "logs/rsl_rl/go2w_sideflip"


@configclass
class CommandsCfgSideflip(CommandsCfgJump):
    jump = CommandsCfgJump().jump.replace(
        enable_jump=False,
        enable_backflip=False,
        enable_sideflip=True,
        target_roll_turns_range=(ROLL_TURNS, ROLL_TURNS),
        sideflip_target_height=0.50,
        sideflip_assist_force=0.0,
        sideflip_spin_torque=65.0,
        sideflip_spin_delay_s=0.26,
        sideflip_spin_duration_s=0.10,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class RobotEnvCfgSideflip(RobotEnvCfgJump):
    """Assisted sideflip-only task."""

    commands: CommandsCfgSideflip = CommandsCfgSideflip()


@configclass
class RobotPlayEnvCfgSideflip(RobotEnvCfgSideflip):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = PLAY_ASSIST_SCALE
        self.commands.jump.debug_vis = True
