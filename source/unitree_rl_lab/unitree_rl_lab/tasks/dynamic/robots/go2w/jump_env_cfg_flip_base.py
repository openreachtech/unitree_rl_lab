"""Shared base for the go2w assisted-motion tasks (Jump, Backflip, Sideflip, Phase 2).

This was ``Go2w-Jump-Phase2`` as first ported from go2: jump + backflip + sideflip with
go2's assist and reward settings. That task never left full assist on go2w, so it is no
longer registered; ``jump_env_cfg_jump.py`` fixes it up for the jump, the backflip and
sideflip files adapt the assist per motion, and ``jump_env_cfg_phase2.py`` recombines
the three.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg import (
    LEG_JOINT_NAMES,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    RobotEnvCfg,
    StandingRewardsCfg,
    TerminationsCfg,
)

@configclass
class CommandsCfgFlipBase(CommandsCfg):
    jump = CommandsCfg().jump.replace(
        auto_trigger=True,
        enable_jump=True,
        enable_backflip=True,
        enable_sideflip=True,
        # Widened from (0.8, 1.2): a less predictable trigger time makes an early,
        # precommitted crouch a worse bet on average, discouraging it alongside
        # pre_jump_pose's explicit cost below.
        trigger_time_range=(0.5, 2.0),
        # go2's value; every go2w task overrides it.
        target_height_range=(0.20, 0.20),
        target_pitch_turns_range=(-1.0, -1.0),
        target_roll_turns_range=(-1.0, -1.0),
        # RL_hip added so all 4 legs are resolved -- the crouch-assist pulse below
        # applies to every resolved body, making it symmetric across all four legs
        # like a real quadruped crouch. Backflip/sideflip launch forces are
        # unaffected: they use their own narrower index lists (FR_hip/FL_hip and
        # FR_hip/RR_hip respectively), so RL_hip never receives launch force.
        assist_body_names=["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
        backflip_assist_body_names=("FR_hip", "FL_hip"),
        sideflip_assist_body_names=("FR_hip", "RR_hip"),
        command_duration_s=0.50,
        assist_duration_s=0.10,
        # Crouch-assist: a brief downward pulse on all 4 legs right at trigger, before
        # the launch force, so the robot physically experiences a genuine crouch-load
        # instead of relying on reward shaping alone to elicit correct timing. Shaped
        # as a linear 0->peak->0 envelope (see JumpCommand._apply_assistance) so there's
        # no force discontinuity at either end.
        crouch_assist_force=150.0,
        crouch_assist_duration_s=0.12,
        # assist_delay_s == crouch_assist_duration_s: the launch force begins ramping
        # in exactly as the crouch pulse ends, so both sides of that handoff are at
        # ~0 force -- no discontinuity there either. assist_ramp_s smooths the launch
        # force's own onset the same way, instead of a hard step (which reliably broke
        # training from a cold Phase1 resume, regardless of magnitude, when tried
        # in isolation).
        assist_delay_s=0.12,
        assist_ramp_s=0.12,
        backflip_assist_force=350.0,
        # Was 600.0 -- nearly double backflip's force, which caused a chaotic, high
        # launch (max_height 0.9-1.1m vs backflip's usual 0.1-0.3m) and never
        # converged in isolated testing. 200.0 was too weak (never left the ground).
        # Matching backflip's proven 350.0 converged cleanly (success 0.99+ by
        # iteration 900).
        sideflip_assist_force=350.0,
        initial_assist_scale=1.0,
        minimum_landing_time_s=0.80,
        # Every task built on this base sets its own; a shared default would let two tasks
        # read and overwrite one curriculum state.
        state_file=None,
    )


@configclass
class FlipRewardsCfg(StandingRewardsCfg):
    upright = None
    standing_pose = None
    stillness = None

    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward_windup,
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
        weight=-0.05,
        params={
            "command_name": "jump",
            "asset_cfg": SceneEntityCfg("robot"),
        },
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


@configclass
class FlipTerminationsCfg(TerminationsCfg):
    bad_orientation = None


@configclass
class CurriculumCfgFlipBase(CurriculumCfg):
    assist_force = CurrTerm(
        func=mdp.assist_force_decay,
        params={
            "command_name": "jump",
            "success_threshold": 0.60,
            "decay_step": 0.01,
            "minimum_episodes": 1024,
        },
    )


@configclass
class EventCfgFlipBase(EventCfg):
    # Randomizes ground friction per-environment (sampled once at startup) so the policy
    # doesn't overfit to one assumed grip level -- sim2sim testing in MuJoCo showed a
    # backflip trained only at a single fixed friction value transferred poorly (mostly
    # under-rotating and landing on its back) regardless of which single MuJoCo friction
    # value was tried. Range brackets a smooth tile floor with margin on both sides.
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
class RobotEnvCfgFlipBase(RobotEnvCfg):
    """The original Phase 2 recipe: assisted jump + backflip + sideflip, Phase 2 rewards."""

    commands: CommandsCfgFlipBase = CommandsCfgFlipBase()
    rewards: FlipRewardsCfg = FlipRewardsCfg()
    terminations: FlipTerminationsCfg = FlipTerminationsCfg()
    curriculum: CurriculumCfgFlipBase = CurriculumCfgFlipBase()
    events: EventCfgFlipBase = EventCfgFlipBase()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4.0
