import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2W_CFG
from unitree_rl_lab.tasks.dynamic import mdp

# go2w has four extra wheel joints on top of the 12 leg joints. The legs are
# position-controlled and the wheels are velocity-controlled continuous joints, exactly
# as in the go2w velocity task (``locomotion/robots/go2w/velocity_env_cfg.py``). Every
# term that would otherwise treat a wheel like a leg joint -- position targets, pose
# penalties, joint limits -- is scoped to one of these two lists. A wheel's absolute
# position drifts without bound as it spins, so it must never enter a pose/limit term.
# fmt: off
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
# fmt: on

# Actuator model: go2's, as matched to unitree_mujoco on feat/jump (Go2-Jump-60's
# GO2_CORRECTED_ACTUATOR_CFG <-> simulate/config.yaml `torque_speed_curves`). Go2W's legs are
# the same GO-M8010-6 motors. UNITREE_GO2W_CFG's IdealPDActuator, which the locomotion tasks
# keep, has neither the torque-speed curve nor joint friction, and its only speed limit is
# PhysX's hard clamp from the USD (calf 20.07, others 30.1 rad/s); MuJoCo has neither clamp.
# Trained with it, the Phase 2 policy pushed off far harder in MuJoCo (calf 31-34 vs 20 rad/s,
# sideflip 0.27 vs 0.17 m) and over-rotated the sideflip onto its side. With this model on
# both sides the policy lands jump, backflip and sideflip in MuJoCo (2026-10-13; MuJoCo must
# be built from a source that has the curve -- check for the `[torque-speed]` lines at start).
#
#   legs    hip/thigh Y1 20.2 / Y2 23.4, calf Y1 39.22 / Y2 45.43, X1 13.5 / X2 30 rad/s;
#           Fs 0.2 / Fd 0.1 and armature 0.01 as go2w.xml's joint defaults.
#   wheels  no curve (no spec for the wheel motor), 23.7 N*m (go2w.xml ctrlrange), same joint
#           friction and armature; MuJoCo's curve config has a matching no-derate "wheel" entry.
#
# PhysX's velocity clamp is lifted to 100 rad/s: the curve takes the legs to zero torque at
# 30 rad/s, and MuJoCo has no clamp to match.
JOINT_FS = 0.2
JOINT_FD = 0.1
JOINT_ARMATURE = 0.01
VELOCITY_LIMIT_SIM = 100.0
ACTUATORS = {
    "legs": unitree_actuators.UnitreeActuatorCfg_Go2HV(
        joint_names_expr=LEG_JOINT_NAMES,
        stiffness=25.0,
        damping=0.5,
        friction=0.0,
        Fs=JOINT_FS,
        Fd=JOINT_FD,
        Y1={".*_hip_joint": 20.2, ".*_thigh_joint": 20.2, ".*_calf_joint": 45.43 * (20.2 / 23.4)},
        Y2={".*_hip_joint": 23.4, ".*_thigh_joint": 23.4, ".*_calf_joint": 45.43},
        X1=13.5,
        X2=30.0,
        armature=JOINT_ARMATURE,
        velocity_limit_sim=VELOCITY_LIMIT_SIM,
    ),
    "wheels": unitree_actuators.UnitreeActuatorCfg(
        joint_names_expr=WHEEL_JOINT_NAMES,
        stiffness=0.0,
        damping=0.5,
        friction=0.0,
        Fs=JOINT_FS,
        Fd=JOINT_FD,
        Y1=23.7,
        armature=JOINT_ARMATURE,
        velocity_limit_sim=VELOCITY_LIMIT_SIM,
    ),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", actuators=ACTUATORS)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    jump = mdp.JumpCommandCfg(
        asset_name="robot",
        assist_body_names=["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
        # go2w stands taller than go2 (init pos z=0.45 vs 0.40); the jump/landing
        # bookkeeping measures height deltas against this nominal.
        nominal_standing_height=0.45,
        resampling_time_range=(1.0e9, 1.0e9),
        auto_trigger=False,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    # The term names are load-bearing: ``export_deploy_cfg`` recognises
    # ``JointPositionAction``/``JointVelocityAction`` when writing ``deploy.yaml``.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    # Wheels stay in the action space as velocity-controlled joints (same convention as
    # the go2w velocity task). During a flip the policy is free to brake or spin them;
    # it is not forced to lock them. Wheel rate is ``action * scale`` rad/s.
    JointVelocityAction = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=8.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, params={"command_name": "jump"})
        # Wheel joints spin freely, so their absolute position carries no usable
        # information and drifts without bound -- only the leg joints are observed.
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        # Joint velocity is bounded and informative for every joint (including the
        # wheels), so it stays over the full set.
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, params={"command_name": "jump"})
        root_height = ObsTerm(func=mdp.root_height)
        root_roll_angle = ObsTerm(func=mdp.root_roll_angle)
        root_pitch_angle = ObsTerm(func=mdp.root_pitch_angle)
        maximum_jump_height = ObsTerm(func=mdp.maximum_jump_height, params={"command_name": "jump"})
        accumulated_root_pitch = ObsTerm(func=mdp.accumulated_root_pitch, params={"command_name": "jump"})
        accumulated_root_roll = ObsTerm(func=mdp.accumulated_root_roll, params={"command_name": "jump"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class StandingRewardsCfg:
    upright = RewTerm(func=mdp.upright_reward, weight=2.0)
    # Scoped to the legs: the free-spinning wheels drift without bound, so a full-joint
    # pose error would collapse this reward to ~0 regardless of the real standing pose.
    standing_pose = RewTerm(
        func=mdp.standing_pose_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    stillness = RewTerm(func=mdp.stillness_reward, weight=1.0)
    # Leg-scoped: rolling wheels legitimately run at high velocity/acceleration/torque,
    # so penalising those on the wheels would just fight the wheel-velocity action.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"],
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    assist_force = None


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: StandingRewardsCfg = StandingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
