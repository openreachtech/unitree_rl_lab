"""G1 (23dof) get-up from lying on the back, ported from HoST (OpenRobotLab/HoST, RSS 2025).

Source: ``legged_gym/envs/g1/g1_config_ground.py`` + ``legged_gym/envs/base/host_ground.py``.
Blind / proprioception only, no command.

Follows HoST's procedure, including its learner: train with the ``HoSTRunner`` agent cfg
(``unitree_rl_lab.utils.host_rl``: one critic per reward group, Tanh actor, smoothness
loss). Reward term names carry their group as prefix (task / regu / style / target).

A first port without HoST's learner (single critic, no smoothness loss, no Tanh) learned a
~0.25 s kip-up and kept shuffling once standing; HoST's per-group critics are what keep the
small regu / style terms from being drowned by the task reward.

Added on top of HoST (a HoST-only run stood with both hip rolls at their ~170 deg limit, the
twist undone by the other joints, feet wide apart, and kept waving its arms once up): hip
roll / yaw limits narrowed at startup, continuous hip yaw / roll deviation penalties and a
hip-width foot distance reward (style), and a joint-speed penalty once standing (target).

Remaining differences from HoST:

* Effort limits stay at unitree_rl_lab's values (hip_roll 139 / ankle 35 vs HoST 88 / 50);
  only the PD gains follow HoST.
* Isaac Lab's observation noise is added after the unactuated-phase mask (HoST masks after
  noise), and the history is ordered per term rather than per frame.
* Motor strength / actuation offset randomization is not ported (implicit PD).
"""

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG
from unitree_rl_lab.assets.robots.g1_host import UNITREE_G1_23DOF_HOST_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from . import mdp as getup_mdp

SIM_DT = 0.005
DECIMATION = 4
STEP_DT = SIM_DT * DECIMATION


# pelvis-height phases (HoST ``target_base_height_phase1/3``) and standing target
PHASE1_HEIGHT = 0.45
PHASE3_HEIGHT = 0.50  # HoST: 0.65; lowered so the post-task terms (stand still, leg pose) also act on a crouched stance
STANDING_HEIGHT = 0.75

# HoST ``default_joint_angles``
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_elbow_joint": 0.8,
    ".*_hip_(roll|yaw)_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    ".*_shoulder_.*": 0.0,
    ".*_wrist_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
}
# HoST ``target_joint_angles`` for the upper body (used by ``target_upper_dof_pos``)
# HoST target_joint_angles had straight arms held out (shoulder roll +-0.3, elbow 0), which
# looked unnatural; use unitree_rl_lab's relaxed G1 arm pose instead
UPPER_TARGET_POS = {
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}
LEG_JOINTS = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
UPPER_JOINTS = [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint", "waist_yaw_joint"]


def _supine(angle: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation of ``angle`` about y. -90 deg: lying on the back."""
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def _make_robot_cfg() -> ArticulationCfg:
    robot = copy.deepcopy(UNITREE_G1_23DOF_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.pos = (0.0, 0.0, 0.5)
    robot.init_state.rot = _supine(-math.pi / 2)
    robot.init_state.joint_pos = DEFAULT_JOINT_POS
    # HoST PD gains (kp / kd): hip 150/4, knee 200/6, ankle 40/2, arms & waist 100/4
    a = robot.actuators
    a["N7520-14.3"].stiffness = {".*_hip_.*": 150.0, "waist_yaw_joint": 100.0}
    a["N7520-14.3"].damping = {".*_hip_.*": 4.0, "waist_yaw_joint": 4.0}
    a["N7520-22.5"].stiffness = {".*_hip_roll_.*": 150.0, ".*_knee_.*": 200.0}
    a["N7520-22.5"].damping = {".*_hip_roll_.*": 4.0, ".*_knee_.*": 6.0}
    a["N5020-16"].stiffness = 100.0
    a["N5020-16"].damping = 4.0
    a["N5020-16-parallel"].stiffness = 40.0
    a["N5020-16-parallel"].damping = 2.0
    return robot


ROBOT = SceneEntityCfg("robot")
TORSO = SceneEntityCfg("robot", body_names="torso_link")
FEET = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
KNEES = SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"], preserve_order=True)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground (HoST ``g1_ground``)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.3,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = _make_robot_cfg()

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """HoST ``domain_rand``. Motor strength / actuation offset are not ported (implicit PD)."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    add_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            # HoST: U(-0.03, 0.03) scaled x4 for x/y, x2 for z
            "com_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "z": (-0.06, 0.06)},
        },
    )

    # not in HoST: keep the hips in a realistic range (the G1 hip roll goes to ~170 deg, and a
    # HoST-only run settled into standing with both hip rolls at that limit)
    hip_limits = EventTerm(
        func=getup_mdp.restrict_joint_limits,
        mode="startup",
        params={
            "limits": {
                "left_hip_roll_joint": (-0.52, 1.0),
                "right_hip_roll_joint": (-1.0, 0.52),
                ".*_hip_yaw_joint": (-1.0, 1.0),
            }
        },
    )

    # reset
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_pose = EventTerm(
        func=getup_mdp.reset_lying_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ActionsCfg:
    JointPositionDeltaAction = getup_mdp.DeltaJointPositionActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class ObservationsCfg:
    """HoST: 6-frame history of (ang_vel, gravity, absolute joint pos, joint vel, last action,
    action scale). The critic gets the same plus base linear velocity, without noise."""

    @configclass
    class PolicyCfg(ObsGroup):
        # every term is zeroed while unactuated (HoST)
        base_ang_vel = ObsTerm(
            func=getup_mdp.unactuated_masked,
            params={"term_func": mdp.base_ang_vel},
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=getup_mdp.unactuated_masked,
            params={"term_func": mdp.projected_gravity},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=getup_mdp.unactuated_masked, params={"term_func": mdp.joint_pos}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=getup_mdp.unactuated_masked,
            params={"term_func": mdp.joint_vel_rel},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=getup_mdp.unactuated_masked, params={"term_func": mdp.last_action})
        action_scale = ObsTerm(
            func=getup_mdp.unactuated_masked,
            params={"term_func": getup_mdp.action_scale_obs},
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )

        def __post_init__(self):
            self.history_length = 6
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        action_scale = ObsTerm(func=getup_mdp.action_scale_obs)

        def __post_init__(self):
            self.history_length = 6

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """HoST g1_ground rewards with HoST's scales. The name prefix is the reward group (one
    critic each, see ``unitree_rl_lab.utils.host_rl``); the group weights live in the agent cfg."""

    # -- task (HoST multiplies orientation and head height; not scaled by dt)
    task_orientation_head_height = RewTerm(
        func=getup_mdp.task_orientation_head_height,
        weight=1.0,
        # HoST: 1.0 m, which a deep crouch already meets on the G1 (standing straight is ~1.25 m)
        # HoST: target 1.0 m, margin 1.0 m. On the G1 a crouch or kneeling upright already gets
        # ~0.9 of that (standing straight is ~1.25 m), so the target is raised and the margin narrowed;
        # 1.15 m still allowed standing with bent knees.
        params={"torso_cfg": TORSO, "feet_cfg": FEET, "target_head_height": 1.22, "target_head_margin": 0.3},
    )

    # -- regu
    regu_dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    regu_action_rate = RewTerm(func=getup_mdp.delta_action_rate_l2, weight=-0.01)
    regu_smoothness = RewTerm(func=getup_mdp.delta_action_smoothness_l2, weight=-0.01)
    regu_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6)
    regu_joint_power = RewTerm(func=mdp.energy, weight=-2.5e-5)
    regu_dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    regu_joint_tracking_error = RewTerm(func=getup_mdp.joint_tracking_error_l2, weight=-2.5e-4)
    regu_dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-100.0)
    regu_dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})

    # -- style
    style_waist_deviation = RewTerm(
        func=getup_mdp.joint_abs_exceeds,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint"), "max_limit": 1.4},
    )
    style_hip_yaw_deviation = RewTerm(
        func=getup_mdp.joint_abs_exceeds,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw_joint"), "max_limit": 1.4, "min_limit": 0.9},
    )
    style_hip_roll_deviation = RewTerm(
        func=getup_mdp.joint_abs_exceeds,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint"),
            "max_limit": 1.4,
            "min_limit": 0.9,
        },
    )
    style_shoulder_roll_deviation = RewTerm(
        func=getup_mdp.shoulder_roll_inward,
        weight=-2.5,
        params={
            "left_cfg": SceneEntityCfg("robot", joint_names="left_shoulder_roll_joint"),
            "right_cfg": SceneEntityCfg("robot", joint_names="right_shoulder_roll_joint"),
        },
    )
    style_left_foot_displacement = RewTerm(
        func=getup_mdp.foot_displacement,
        weight=2.5,
        params={"foot_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"), "standup_height": PHASE3_HEIGHT},
    )
    style_right_foot_displacement = RewTerm(
        func=getup_mdp.foot_displacement,
        weight=2.5,
        params={"foot_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"), "standup_height": PHASE3_HEIGHT},
    )
    style_knee_deviation = RewTerm(
        func=getup_mdp.knee_out_of_range,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint")},
    )
    style_shank_orientation = RewTerm(
        func=getup_mdp.shank_orientation,
        weight=10.0,
        params={"knee_cfg": KNEES, "foot_cfg": FEET, "phase1_height": PHASE1_HEIGHT},
    )
    style_ground_parallel = RewTerm(func=getup_mdp.feet_ground_parallel, weight=20.0, params={"feet_cfg": FEET})
    style_feet_distance = RewTerm(func=getup_mdp.feet_too_far, weight=-10.0, params={"feet_cfg": FEET})
    style_ang_vel_xy = RewTerm(
        func=getup_mdp.ang_vel_xy_exp_gated, weight=1.0, params={"min_base_height": PHASE1_HEIGHT}
    )
    # not in HoST: no kneeling with the torso upright
    style_kneeling = RewTerm(func=getup_mdp.kneeling_upright, weight=-10.0, params={"knee_cfg": KNEES})
    # leg alignment (not in HoST): hips not twisted / splayed out, feet hip-width apart
    style_hip_yaw_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw_joint")},
    )
    style_hip_roll_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint")},
    )
    style_feet_width = RewTerm(
        func=getup_mdp.feet_lateral_distance_tol,
        weight=5.0,
        params={"feet_cfg": FEET, "min_base_height": PHASE1_HEIGHT},
    )

    # -- target (post-task: only once the pelvis is above PHASE3_HEIGHT)
    target_ang_vel_xy = RewTerm(
        func=getup_mdp.ang_vel_xy_exp_gated, weight=10.0, params={"min_base_height": PHASE3_HEIGHT}
    )
    target_lin_vel_xy = RewTerm(
        func=getup_mdp.lin_vel_xy_exp_gated, weight=10.0, params={"min_base_height": PHASE3_HEIGHT}
    )
    target_feet_height_var = RewTerm(
        func=getup_mdp.feet_height_diff_exp, weight=2.5, params={"feet_cfg": FEET, "min_base_height": PHASE3_HEIGHT}
    )
    target_upper_dof_pos = RewTerm(
        func=getup_mdp.upper_joint_pos_exp,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS),
            "target": UPPER_TARGET_POS,
            "min_base_height": PHASE3_HEIGHT,
            "sigma": -1.0,  # HoST: -0.1, too flat to pull the arms anywhere
        },
    )
    target_orientation = RewTerm(
        func=getup_mdp.flat_orientation_exp_gated, weight=10.0, params={"min_base_height": PHASE3_HEIGHT}
    )
    target_base_height = RewTerm(
        func=getup_mdp.base_height_exp_gated,
        weight=10.0,
        # HoST: sharpness 20, which gives ~0 a few cm below the target and so no push to straighten
        params={"target_height": STANDING_HEIGHT, "min_base_height": PHASE3_HEIGHT, "sharpness": 4.0},
    )
    # not in HoST: legs back to the nominal standing pose (knees forward and slightly bent)
    target_lower_body_pose = RewTerm(
        func=getup_mdp.stand_pose_exp,
        weight=10.0,
        params={
            "std": 1.5,
            "min_base_height": PHASE3_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )
    # not in HoST: continuous squared errors (the exp kernels give no gradient far from the pose)
    target_leg_pose_l2 = RewTerm(
        func=getup_mdp.joint_target_l2_gated,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS), "min_base_height": PHASE3_HEIGHT},
    )
    target_arm_pose_l2 = RewTerm(
        func=getup_mdp.joint_target_l2_gated,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"]),
            "target": UPPER_TARGET_POS,
            "min_base_height": PHASE3_HEIGHT,
        },
    )
    # not in HoST: no drifting / stepping once up
    target_stand_base_vel = RewTerm(
        func=getup_mdp.stand_base_lin_vel_l2, weight=-5.0, params={"min_base_height": PHASE3_HEIGHT}
    )
    target_stand_feet_vel = RewTerm(
        func=getup_mdp.stand_feet_vel_l2, weight=-20.0, params={"feet_cfg": FEET, "min_base_height": PHASE3_HEIGHT}
    )
    # not in HoST: hold still once up (the HoST-only run kept waving its arms)
    target_stand_joint_vel = RewTerm(
        func=getup_mdp.stand_joint_vel_l2, weight=-0.2, params={"min_base_height": PHASE3_HEIGHT}
    )


@configclass
class TerminationsCfg:
    """Timeout, plus HoST's blow-up guards. No contact / orientation termination:
    every episode starts lying down."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_out = DoneTerm(func=getup_mdp.joint_vel_out_of_bound, params={"limit": 300.0})
    base_vel_out = DoneTerm(func=getup_mdp.base_vel_out_of_bound, params={"limit": 20.0})


@configclass
class CurriculumCfg:
    getup = CurrTerm(func=getup_mdp.update_getup_curriculum, params={"torso_cfg": TORSO, "feet_cfg": FEET})


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """G1 23dof get-up from supine (HoST ``g1_ground``)."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 10.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # play with the end-of-curriculum setting: no pulling force, deploy action scale
        self.actions.JointPositionDeltaAction.scale_init = self.actions.JointPositionDeltaAction.scale
        self.actions.JointPositionDeltaAction.assist_frac_init = 0.0
        self.curriculum.getup = None


def _make_host_robot_cfg() -> ArticulationCfg:
    """HoST's own G1 URDF, explicit PD and URDF effort limits (see ``assets/robots/g1_host.py``)."""
    robot = copy.deepcopy(UNITREE_G1_23DOF_HOST_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.pos = (0.0, 0.0, 0.5)
    robot.init_state.rot = _supine(-math.pi / 2)
    robot.init_state.joint_pos = DEFAULT_JOINT_POS
    return robot


@configclass
class RobotHoSTModelEnvCfg(RobotEnvCfg):
    """Same task on HoST's robot model (URDF, collision, explicit PD) instead of unitree_rl_lab's USD."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _make_host_robot_cfg()


@configclass
class RobotHoSTModelPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _make_host_robot_cfg()


def _make_host_urdf_only_robot_cfg() -> ArticulationCfg:
    """HoST's URDF (geometry, mass, collision) with everything else as in ``_make_robot_cfg``:
    unitree_rl_lab's implicit actuators and effort limits with HoST's gains, and the same solver
    iterations as the unitree USD (8 / 4). Isolates the effect of the robot model itself."""
    robot = _make_robot_cfg()
    spawn = copy.deepcopy(UNITREE_G1_23DOF_HOST_CFG.spawn)
    spawn.articulation_props = copy.deepcopy(UNITREE_G1_23DOF_CFG.spawn.articulation_props)
    robot.spawn = spawn
    return robot


@configclass
class RobotHoSTURDFEnvCfg(RobotEnvCfg):
    """Same task with only the robot model swapped for HoST's URDF."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _make_host_urdf_only_robot_cfg()


@configclass
class RobotHoSTURDFPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _make_host_urdf_only_robot_cfg()


# HoST's values of the params the posture fine-tunes changed (run g1_getup/*_host)
G1_HOST_PARAMS = {
    "task_orientation_head_height/target_head_height": 1.0,
    "task_orientation_head_height/target_head_margin": 1.0,
    "style_left_foot_displacement/standup_height": 0.65,
    "style_right_foot_displacement/standup_height": 0.65,
    "target_ang_vel_xy/min_base_height": 0.65,
    "target_lin_vel_xy/min_base_height": 0.65,
    "target_feet_height_var/min_base_height": 0.65,
    "target_upper_dof_pos/min_base_height": 0.65,
    "target_orientation/min_base_height": 0.65,
    "target_base_height/min_base_height": 0.65,
    "target_base_height/sharpness": 20.0,
    "target_upper_dof_pos/target": {"left_shoulder_roll_joint": 0.3, "right_shoulder_roll_joint": -0.3},
    "target_upper_dof_pos/sigma": -0.1,
}


@configclass
class RobotStagedEnvCfg(RobotEnvCfg):
    """From scratch in one run: HoST's rewards until the robot gets up without help, then the
    posture terms of RobotEnvCfg are faded in (``staged.make_staged``)."""

    def __post_init__(self):
        super().__post_init__()
        from .staged import make_staged

        make_staged(self, G1_HOST_PARAMS)
