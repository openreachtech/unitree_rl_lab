"""Aliengo + Z1: whole-body end-effector twist tracking with a multi-critic policy.

"Multi-critic Learning for Whole-body End-effector Twist Tracking"
(doc/papers/Multi-critic_Learning_for_Whole-body_End-effector_Twist_Tracking.md), on top of the
Aliengo + Z1 locomotion task. One MLP drives the 12 leg and 6 arm joints, trained from scratch:
the actor is blind, the critics see privileged terms.

* Critics (3.3): one per reward group (``REWARD_GROUPS``): ``arm``, ``contact``, ``locomotion``.
* Commands (3.2): base velocity uniform in +-0.25; end-effector twist (``mdp.EeTwistCommand``);
  foot swing heights (``mdp.GaitScheduleCommand``). Task frame (7.2.1): the base frame for the
  first 3000 iterations, then the control frame.
* Terminations: contact of the base, thighs or calves.
* Terrain: Phase 1 flat; Phase 2 the locomotion Phase 2 terrain.

    python scripts/rsl_rl/train.py --task Aliengoz1-Twist-Tracking-Phase1 --headless
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.aliengo_z1 import locomotion_env_cfg as loco
from unitree_rl_lab.tasks.locomotion.robots.aliengo_z1.locomotion_env_cfg import (
    ARM_JOINTS,
    FEET,
    LEG_JOINTS,
)
from unitree_rl_lab.tasks.locomotion.robots.aliengo_z1.pose_tracking_env_cfg import EE_BODY

ARM_BODIES = ["link0[1-6]", "gripperStator", "gripperMover"]

FOOT_RADIUS = 0.0265
"""Aliengo foot sphere: foot origin height above the ground in contact [m]."""

STEPS_PER_ITERATION = 24
"""``num_steps_per_env`` of the runner, to express the frame curriculum in iterations."""


def _arm() -> dict:
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)}


# paper 3.2: the base velocity command is uniform in [-0.25, 0.25], with no curriculum
_PAPER_BASE_VEL = mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.25, 0.25), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.25, 0.25)
)


@configclass
class CommandsCfg(loco.CommandsCfg):
    arm = None
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        debug_vis=True,
        ranges=_PAPER_BASE_VEL,
        limit_ranges=_PAPER_BASE_VEL,
    )
    ee_twist = mdp.EeTwistCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        feet_body_names=FEET,
        resampling_time_range=(4.0, 8.0),
        # paper 3.2: a sphere around the shoulder, rejecting the torso and legs (radius: 85% of the reach)
        sphere_center=(0.253, 0.0, 0.159),
        sphere_radius=0.63,
        reject_box=((-0.45, 0.45), (-0.2, 0.2), (-1.0, 0.1)),
        min_height=-0.25,
        # 7.2.1: the base frame for the first 3000 iterations, then the control frame
        control_frame_start_step=3000 * STEPS_PER_ITERATION,
    )
    # paper eq. 5: a static walk, LF, RH, RF, LH a quarter cycle apart (FEET order FL, FR, RL, RR)
    gait = mdp.GaitScheduleCommandCfg(
        resampling_time_range=(1e9, 1e9), frequency=1.0, swing_height=0.08, phase_offsets=(0.0, 0.5, 0.75, 0.25)
    )


@configclass
class CommandsCfgPhase2(loco.CommandsCfgPhase2):
    arm = None
    base_velocity = CommandsCfg().base_velocity
    gait = CommandsCfg().gait
    # Phase 2: the control frame from the start
    ee_twist = CommandsCfg().ee_twist.replace(control_frame_start_step=0)


@configclass
class ActionsCfg:
    legs = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True)
    arm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg(loco.ObservationsCfg):
    @configclass
    class PolicyCfg(loco.ObservationsCfg.PolicyCfg):
        # twist = waypoint error / dt: clipped at 20 (0.4 m behind the waypoint), scaled to <= 2
        ee_twist = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_twist"}, clip=(-20.0, 20.0), scale=0.1
        )
        ee_goal = ObsTerm(func=mdp.ee_goal_pose, params={"command_name": "ee_twist"})
        gait = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait"})

    @configclass
    class CriticCfg(loco.ObservationsCfg.CriticCfg):
        arm_goal = None
        ee_twist = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_twist"}, clip=(-20.0, 20.0), scale=0.1
        )
        ee_goal = ObsTerm(func=mdp.ee_goal_pose, params={"command_name": "ee_twist"})
        gait = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait"})
        ee_pose = ObsTerm(func=mdp.ee_pose, params={"command_name": "ee_twist"})
        ee_waypoint_error = ObsTerm(func=mdp.ee_waypoint_error, params={"command_name": "ee_twist"})
        arm_contact_state = ObsTerm(
            func=mdp.contact_states, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_BODIES)}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(loco.RewardsCfg):
    """Locomotion rewards (critic ``locomotion``) + ``ee_*`` / ``arm_*`` (critic ``arm``)."""

    # -- locomotion. Leg action rate: L2, the paper's Gaussian leaves the legs jittery
    action_rate = RewTerm(func=mdp.action_rate_l2_term, weight=-0.1, params={"action_name": "legs"})
    # leg torques as in the paper: the locomotion task's L2 penalty makes walking not pay
    joint_torques = RewTerm(func=mdp.joint_torques_exp, weight=1e-5, params={"std_sq": 40.0, **loco._legs()})
    # velocity tracking narrower than the paper's (sigma^2 0.1 pays ~0.7 for standing still)
    track_lin_vel_xy = loco.RewardsCfg().track_lin_vel_xy.replace(
        weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.02)}
    )
    track_ang_vel_z = loco.RewardsCfg().track_ang_vel_z.replace(
        weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.05)}
    )
    # leg posture (not in the paper): without it the robot walks low with its legs skewed sideways
    joint_pos = loco.RewardsCfg().joint_pos.replace(weight=-0.2)
    # -- contact schedule (Table 6, CS), in place of the locomotion task's gait terms
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_sum,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)},
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact_schedule,
        weight=1.0,
        params={
            "command_name": "gait",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "scan_sensor_names": [f"foot_scan_{f}" for f in FEET],
            "foot_radius": FOOT_RADIUS,
        },
    )

    # -- arm (Table 6, Mani), wider than the paper's (sigma^2 0.005 / 0.01), which the arm never reaches
    ee_position_tracking = RewTerm(
        func=mdp.ee_position_tracking_exp, weight=5.0, params={"command_name": "ee_twist", "std_sq": 0.01}
    )
    ee_orientation_tracking = RewTerm(
        func=mdp.ee_orientation_tracking_exp, weight=4.0, params={"command_name": "ee_twist", "std_sq": 0.1}
    )
    arm_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_BODIES)},
    )
    arm_action_rate = RewTerm(func=mdp.action_rate_exp_term, weight=0.1, params={"action_name": "arm", "std_sq": 0.5})
    arm_joint_torques = RewTerm(func=mdp.joint_torques_exp, weight=1e-5, params={"std_sq": 40.0, **_arm()})
    arm_joint_vel = RewTerm(func=mdp.joint_vel_exp, weight=1e-4, params={"std_sq": 4.0, **_arm()})


@configclass
class RewardsCfgPhase2(RewardsCfg):
    """The locomotion Phase 2 relaxations for box terrain."""

    base_linear_velocity = loco.RewardsCfgPhase2().base_linear_velocity
    flat_orientation_l2 = loco.RewardsCfgPhase2().flat_orientation_l2


REWARD_GROUPS = {
    "arm": ["ee_.*", "arm_.*"],
    "contact": ["feet_contact", "feet_air_time", "air_time_variance"],
    "locomotion": [".*"],
}
"""Critic per reward group; a term goes to the first group it matches (``MultiCriticRLEnv``)."""


@configclass
class CurriculumCfg(loco.CurriculumCfg):
    lin_vel_cmd_levels = None  # fixed base velocity range (paper 3.2)


@configclass
class CurriculumCfgPhase2:
    terrain_levels = loco.CurriculumCfgPhase2().terrain_levels


@configclass
class TerminationsCfg(loco.TerminationsCfg):
    base_contact = loco.TerminationsCfg().base_contact.replace(
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh", ".*_calf"]), "threshold": 1.0}
    )
    bad_orientation = None


@configclass
class RobotEnvCfgPhase1(loco.RobotEnvCfgPhase1):
    """Phase 1: flat plane."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    reward_groups: dict = REWARD_GROUPS


@configclass
class RobotPlayEnvCfgPhase1(RobotEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # a trained policy: walking, in the control frame
        self.commands.ee_twist.control_frame_start_step = 0
        self.commands.ee_twist.debug_vis = True


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: random rough, boxes and pyramid slopes, terrain-level curriculum."""

    scene: loco.RobotSceneCfgPhase2 = loco.RobotSceneCfgPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()
    curriculum: CurriculumCfgPhase2 = CurriculumCfgPhase2()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_twist.debug_vis = True


# ---------------------------------------------------------------------------
# Arm only: fixed base, legs at default, to tune the arm part of the task in isolation (one critic).
# ---------------------------------------------------------------------------
ARM_ONLY_ROT_STD_SQ = [(0, 0.1), (1000 * STEPS_PER_ITERATION, 0.045)]
"""Orientation reward width (sigma^2) schedule: (env step, value); the paper's is 0.01."""


@configclass
class ArmOnlyActionsCfg:
    arm = ActionsCfg().arm


@configclass
class ArmOnlyRewardsCfg:
    ee_position_tracking = RewardsCfg().ee_position_tracking
    ee_orientation_tracking = RewardsCfg().ee_orientation_tracking.replace(
        params={"command_name": "ee_twist", "std_sq": ARM_ONLY_ROT_STD_SQ[0][1]}
    )
    arm_undesired_contacts = RewardsCfg().arm_undesired_contacts
    arm_action_rate = RewardsCfg().arm_action_rate
    arm_joint_torques = RewardsCfg().arm_joint_torques
    arm_joint_vel = RewardsCfg().arm_joint_vel


@configclass
class ArmOnlyCurriculumCfg:
    ee_orientation_width = CurrTerm(
        func=mdp.reward_param_schedule,
        params={"term_name": "ee_orientation_tracking", "param": "std_sq", "schedule": ARM_ONLY_ROT_STD_SQ},
    )


@configclass
class RobotEnvCfgArmOnly(RobotEnvCfgPhase1):
    actions: ArmOnlyActionsCfg = ArmOnlyActionsCfg()
    rewards: ArmOnlyRewardsCfg = ArmOnlyRewardsCfg()
    curriculum: ArmOnlyCurriculumCfg = ArmOnlyCurriculumCfg()
    reward_groups: dict = {"arm": [".*"]}

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.fix_base = True
        self.commands.gait = None
        self.observations.policy.gait = None
        self.observations.critic.gait = None
        for r in (self.commands.base_velocity.ranges, self.commands.base_velocity.limit_ranges):
            r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
        self.events.push_robot = None
        self.terminations.base_contact = None


@configclass
class RobotPlayEnvCfgArmOnly(RobotEnvCfgArmOnly):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.ee_twist.debug_vis = True
