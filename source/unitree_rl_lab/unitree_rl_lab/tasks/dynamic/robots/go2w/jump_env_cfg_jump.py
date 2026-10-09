"""Go2w-Jump: the assisted vertical jump, and the recipe every go2w motion task builds on.

``Go2w-Backflip`` and ``Go2w-Sideflip`` inherit everything here and change only the
motion and its assist; ``Go2w-Jump-Phase2`` recombines the three.

The first go2w port trained jump + backflip + sideflip together with go2's settings and
never left full assist (success 0.33 vs the 0.60 gate), drifting backwards in Play: the
backflip force lifts the front hips only, and the rear wheels roll. The fixes, each
carried over from go2's ``Go2-Jump-60`` (feat/jump) unless marked go2w:

- jump only, at a fixed ``TARGET_HEIGHT`` (a range collapsed learning on go2).
- (go2w) ``nominal_standing_height`` 0.45 -> 0.405, the measured stance (see below).
- (go2w) ``base_lin_vel_xy`` penalty: wheels roll, so fore-aft push turns into drift.
- ``pre_jump_standing_reward`` gated on the command alone: a variant that kept paying for
  stillness until the assist landed paid the robot to hold still through exactly the
  window a crouch would use.
- ``action_rate`` -0.1 -> -0.01 and ``joint_torques`` removed: both oppose an explosive
  knee extension (L2 torque charges the 45 N*m calf 5x more than the 23.7 N*m thigh).
- ``non_target_rotation`` -0.05 -> -1.0: go2 left the ground with a systematic pitch/roll
  twist that the weak penalty never made worth fixing.
- landing upright gate 37 deg -> 26 deg.
- (go2w) overshoot counts as success, and the assist decays at 0.005 per step instead of
  0.01. Without them the first run sat at success 0 for ~700 its (0.43-0.47 m under full
  assist, above the 0.30 +/- 0.10 window), then lost the assist faster than it could
  follow (1.0 -> 0.70 in ~60 its, max_height 0.19 m). With both, assist_scale reached 0
  at ~it 1700 and the unaided jump kept improving (max_height 0.257 m, success 1.0 by
  it 2200; 2026-10-08).
- (go2w) ``wheel_vel`` penalty. Nothing else looks at the wheels (the joint penalties are
  leg-scoped), so a spinning wheel cost nothing: the first Phase 2 policy held the FL
  wheel command saturated at +8.5 and slipped it at ~23 rad/s while standing, creeping
  forward in MuJoCo (Isaac Lab: up to 2.0 m in 10 s idle). With it (fine-tuned 1000 its,
  2026-10-09) idle wheels sat at 0.07 rad/s with zero drift, and the robot also came to
  rest after a motion, with success unchanged (0.995-1.000).
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg import (
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    RobotEnvCfg,
    StandingRewardsCfg,
    TerminationsCfg,
)

# Height gain over the measured stance. 0.30 m is a first check that the jump works at
# all -- above the port's 0.20 m, below go2's 0.60 m. Keep it a single point, not a range.
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
class CommandsCfgJump(CommandsCfg):
    jump = CommandsCfg().jump.replace(
        auto_trigger=True,
        enable_jump=True,
        enable_backflip=False,
        enable_sideflip=False,
        # A less predictable trigger time makes an early, precommitted crouch a worse bet
        # on average, discouraging it alongside pre_jump_pose's explicit cost.
        trigger_time_range=(0.5, 2.0),
        target_height_range=(TARGET_HEIGHT, TARGET_HEIGHT),
        # Backflip lifts the front hips, sideflip the right hips; the crouch pulse and the
        # jump launch act on all four.
        backflip_assist_body_names=("FR_hip", "FL_hip"),
        sideflip_assist_body_names=("FR_hip", "RR_hip"),
        command_duration_s=0.50,
        assist_duration_s=0.10,
        # Crouch-assist: a brief downward 0 -> peak -> 0 pulse on all four hips right at
        # the trigger, so the robot physically experiences a crouch-load before push-off.
        crouch_assist_force=150.0,
        crouch_assist_duration_s=0.12,
        # The launch force ramps in exactly as the crouch pulse ends, so both sides of
        # the handoff are at ~0 force. A hard-step launch reliably broke training from a
        # cold Phase 1 resume on go2.
        assist_delay_s=0.12,
        assist_ramp_s=0.12,
        backflip_assist_force=350.0,
        sideflip_assist_force=350.0,
        initial_assist_scale=1.0,
        minimum_landing_time_s=0.80,
        nominal_standing_height=NOMINAL_STANDING_HEIGHT,
        landing_upright_threshold=-0.90,
        # Under full assist the jump overshoots (0.47 m against 0.30 +/- 0.10), and a
        # two-sided success check then holds assist_scale at 1.0 until the policy learns
        # to jump lower. The height reward stays two-sided.
        success_allow_overshoot=True,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class JumpRewardsCfg(StandingRewardsCfg):
    upright = None
    standing_pose = None
    stillness = None
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = None

    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress = RewTerm(
        func=mdp.motion_progress_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress_standing = RewTerm(
        func=mdp.motion_progress_standing_reward,
        weight=1.0,
        # Leg-scoped so the free-spinning wheels don't corrupt the landing pose term.
        params={
            "command_name": "jump",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-1.0,
        params={"command_name": "jump", "asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalizes hip (abduction/adduction) deviation from default so the pre-jump crouch
    # tucks legs by flexing thigh/calf instead of splaying hips outward.
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # Cost for holding a non-default joint pose while the jump command is idle, so an
    # early/anticipatory crouch right after spawn has an actual cost instead of being free.
    # Leg-scoped for the same wheel-drift reason as standing_pose.
    pre_jump_pose = RewTerm(
        func=mdp.pre_jump_pose_reward,
        weight=1.0,
        params={
            "command_name": "jump",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    # A vertical jump should land where it took off. Applied over the whole episode, so
    # rolling away while idle is not a loophole either.
    base_lin_vel_xy = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-1.0)
    # The other joint penalties skip the wheels, so this is the only thing that stops the
    # policy from spinning them. Always on: before, during and after the motion. A 23 rad/s
    # wheel costs 0.53 per step, comparable to the motion rewards.
    wheel_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )


@configclass
class JumpTerminationsCfg(TerminationsCfg):
    # Flips pass through every orientation; only a base contact ends the episode.
    bad_orientation = None


@configclass
class CurriculumCfgJump(CurriculumCfg):
    # decay_step 0.01 -> 0.005: at 0.01 the assist fell 1.0 -> 0.70 in ~60 iterations,
    # faster than the policy followed (max_height dropped to 0.19 m, success to 0.10).
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
class EventCfgJump(EventCfg):
    # Randomizes ground friction per-environment (sampled once at startup) so the policy
    # doesn't overfit to one grip level -- on go2 a backflip trained at a single fixed
    # friction transferred poorly to MuJoCo at every single friction value tried there.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class RobotEnvCfgJump(RobotEnvCfg):
    """Assisted jump-only task (no backflip/sideflip)."""

    commands: CommandsCfgJump = CommandsCfgJump()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    terminations: JumpTerminationsCfg = JumpTerminationsCfg()
    curriculum: CurriculumCfgJump = CurriculumCfgJump()
    events: EventCfgJump = EventCfgJump()


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
