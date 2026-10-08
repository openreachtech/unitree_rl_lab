"""Go2w-Jump: Phase 2's vertical jump on its own, without backflip/sideflip.

Phase 2 trains jump + backflip + sideflip together. On go2w it never got past full
assist (success 0.33 vs the 0.60 gate, assist_scale stuck at 1.0), so Play -- which runs
with zero assist -- shows a policy that has never jumped unaided, and the robot drifts
backwards instead. The backflip launch force is applied to the FRONT hips only, and on
free-rolling wheels that pitch-up turns straight into backward travel. This task drops
the flips to check the jump alone first.

Changes from Phase 2, each carried over from go2's ``Go2-Jump-60`` (feat/jump) unless
marked go2w:

- jump only, at a fixed ``TARGET_HEIGHT`` (a range collapsed learning on go2).
- (go2w) ``nominal_standing_height`` 0.45 -> 0.405, the measured stance (see below).
- (go2w) ``base_lin_vel_xy`` penalty: wheels roll, so fore-aft push turns into drift.
- ``pre_jump_standing_reward`` (plain gate): the windup variant pays the robot to hold
  still through exactly the window a crouch would use.
- ``action_rate`` -0.1 -> -0.01 and ``joint_torques`` removed: both oppose an explosive
  knee extension (L2 torque charges the 45 N*m calf 5x more than the 23.7 N*m thigh).
- ``non_target_rotation`` -0.05 -> -1.0: go2 left the ground with a systematic pitch/roll
  twist that the weak penalty never made worth fixing.
- landing upright gate 37 deg -> 26 deg.
- overshoot counts as success, and the assist decays at 0.005 per step instead of 0.01.
  Without them the first run sat at success 0 for ~700 its (0.43-0.47 m under full
  assist, above the 0.30 +/- 0.10 window), then lost the assist faster than it could
  follow (1.0 -> 0.70 in ~60 its, max_height 0.19 m). With both, assist_scale reached 0
  at ~it 1700 and the unaided jump kept improving (max_height 0.257 m, success 1.0 by
  it 2200; 2026-10-08).
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg_flip_base import (
    CommandsCfgFlipBase,
    CurriculumCfgFlipBase,
    FlipRewardsCfg,
    RobotEnvCfgFlipBase,
)

# Height gain over the measured stance. 0.30 m is a first check that the jump works at
# all -- above Phase 2's 0.20 m, below go2's 0.60 m. Keep it a single point, not a range.
TARGET_HEIGHT = 0.30
EXPERIMENT_DIR = "logs/rsl_rl/go2w_jump"
# Root height of the Phase 1 policy standing still: 0.4057 m mean over 64 envs, +/-0.001
# (2026-10-08, go2w_jump_phase1 model_299). The 0.45 in the base config is the spawn
# height; at 0.45 a correctly standing robot reads height_delta -0.044 and forfeits ~18%
# of the landing height reward. Fixed rather than captured at the trigger: capturing it
# was tried on go2 and exploited -- crouch just before the trigger, stand back up, and
# score a "jump" without leaving the ground (go2 sandbox/SUMMARY.md, feat/jump).
NOMINAL_STANDING_HEIGHT = 0.405
# Play runs without the training curriculum. 0.0 shows what the policy does alone; raise
# it (up to 1.0) to watch the assisted motion while assist_scale has not decayed yet.
# Overridable per run, e.g. to watch the bare assist force acting on a Phase 1 policy:
#   GO2W_JUMP_PLAY_ASSIST=1.0 rplay go2w_jump logs/rsl_rl/go2w_jump_phase1/<run>/model_299.pt
PLAY_ASSIST_SCALE = float(os.environ.get("GO2W_JUMP_PLAY_ASSIST", "0.0"))


@configclass
class CommandsCfgJump(CommandsCfgFlipBase):
    jump = CommandsCfgFlipBase().jump.replace(
        enable_jump=True,
        enable_backflip=False,
        enable_sideflip=False,
        target_height_range=(TARGET_HEIGHT, TARGET_HEIGHT),
        target_pitch_turns_range=(0.0, 0.0),
        target_roll_turns_range=(0.0, 0.0),
        nominal_standing_height=NOMINAL_STANDING_HEIGHT,
        landing_upright_threshold=-0.90,
        # Under full assist the jump overshoots (0.47 m against 0.30 +/- 0.10), and a
        # two-sided success check then holds assist_scale at 1.0 until the policy learns
        # to jump lower. The height reward stays two-sided.
        success_allow_overshoot=True,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class JumpRewardsCfg(FlipRewardsCfg):
    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = None
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-1.0,
        params={"command_name": "jump", "asset_cfg": SceneEntityCfg("robot")},
    )
    # A vertical jump should land where it took off. Applied over the whole episode, so
    # rolling away while idle is not a loophole either.
    base_lin_vel_xy = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-1.0)


@configclass
class CurriculumCfgJump(CurriculumCfgFlipBase):
    # 0.01 -> 0.005: at 0.01 the assist fell 1.0 -> 0.70 in ~60 iterations, faster than
    # the policy followed (max_height dropped to 0.19 m, success to 0.10).
    assist_force = CurrTerm(
        func=mdp.assist_force_decay,
        params={
            "command_name": "jump",
            "success_threshold": 0.60,
            "decay_step": 0.005,
            "minimum_episodes": 1024,
        },
    )


@configclass
class RobotEnvCfgJump(RobotEnvCfgFlipBase):
    """Assisted jump-only task (no backflip/sideflip)."""

    commands: CommandsCfgJump = CommandsCfgJump()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    curriculum: CurriculumCfgJump = CurriculumCfgJump()


@configclass
class RobotPlayEnvCfgJump(RobotEnvCfgJump):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = PLAY_ASSIST_SCALE
        # Draws the assist force on each hip: red = up (launch), blue = down (crouch pulse).
        self.commands.jump.debug_vis = True
