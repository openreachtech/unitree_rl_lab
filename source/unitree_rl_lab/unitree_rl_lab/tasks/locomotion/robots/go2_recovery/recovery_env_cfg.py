"""Go2 walk / fall / get-up environment: one 20 s episode that walks, gets knocked down, and gets up.

Built from the two tasks the experts come from:

- ``Unitree-Go2-Velocity-v1`` (go2-blind): the observation layout (history 3) and the walking
  reward set. The locomotion expert loads with no change at all.
- the original 42-dim get-up policy (``Go2-GetUp`` ``model_2999``): the tumbled-pose reset and the get-up reward set.

Both use the stock ``UNITREE_GO2_CFG`` actuator model, so this does too -- not the corrected one the
multi-task experts run on. Switching it means retraining both experts under it first.

The episode never ends on a fall. That is the point of the task: a fall is a state the policy has to
leave, not a failure that resets it. The walking reward set fades out while the robot is down (see
``mdp.standing_weight``) and the get-up set fades in, so neither punishes the other's motion.
"""

import math

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2_recovery import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2_recovery.obs_spec import ACTION_SCALE, HISTORY

STANDING_HEIGHT = 0.30
"""Get-up target, as in the original get-up task."""

FEET = SceneEntityCfg("contact_forces", body_names=".*_foot")


def _standing(term, weight: float, **term_params) -> RewTerm:
    return RewTerm(
        func=mdp.state_gated,
        weight=weight,
        params={"state": mdp.STATE_STANDING, "term": term, "term_params": term_params},
    )


def _down(term, weight: float, **term_params) -> RewTerm:
    return RewTerm(
        func=mdp.state_gated, weight=weight, params={"state": mdp.STATE_DOWN, "term": term, "term_params": term_params}
    )


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground: both experts were trained or verified there, and the critic's base height
    observation assumes it."""

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
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    # startup
    resolve_gated_term_params = EventTerm(func=mdp.resolve_gated_term_params, mode="startup")

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset: mostly standing, the rest in a random tumbled pose, so get-up is practised from the
    # first iteration and not only after a knockdown.
    reset_pose = EventTerm(
        func=mdp.reset_fallen_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pos_range": (-3.14159, 3.14159),
            "standing_prob": 0.6,
        },
    )

    # interval: ordinary pushes the walking expert should absorb, and knockdowns it mostly cannot.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    knockdown = EventTerm(
        func=mdp.knockdown,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"angular_speed_range": (8.0, 16.0), "linear_speed_range": (1.0, 3.0)},
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.0, 1.0)),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=ACTION_SCALE, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Order and widths must match ``obs_spec.POLICY_LAYOUT`` / ``CRITIC_LAYOUT``: the experts are
    placed by column."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01), history_length=HISTORY
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=HISTORY,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100), history_length=HISTORY)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_height = ObsTerm(func=mdp.base_height)

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- walking, while on its feet (Go2 velocity task weights)
    track_lin_vel_xy = _standing(mdp.track_lin_vel_xy_exp, 1.5, command_name="base_velocity", std=math.sqrt(0.25))
    track_ang_vel_z = _standing(mdp.track_ang_vel_z_exp, 0.75, command_name="base_velocity", std=math.sqrt(0.25))
    base_linear_velocity = _standing(mdp.lin_vel_z_l2, -2.0)
    base_angular_velocity = _standing(mdp.ang_vel_xy_l2, -0.05)
    flat_orientation_l2 = _standing(mdp.flat_orientation_l2, -2.5)
    dof_pos_limits = _standing(mdp.joint_pos_limits, -10.0)
    action_rate = _standing(mdp.action_rate_l2, -0.1)
    joint_pos = _standing(
        mdp.joint_position_penalty,
        -0.7,
        asset_cfg=SceneEntityCfg("robot", joint_names=".*"),
        stand_still_scale=5.0,
        velocity_threshold=0.3,
    )
    feet_air_time = _standing(
        mdp.feet_air_time, 0.1, sensor_cfg=FEET, command_name="base_velocity", threshold=0.5
    )
    air_time_variance = _standing(mdp.air_time_variance_penalty, -1.0, sensor_cfg=FEET)
    feet_slide = _standing(
        mdp.feet_slide, -0.1, asset_cfg=SceneEntityCfg("robot", body_names=".*_foot"), sensor_cfg=FEET
    )
    undesired_contacts = _standing(
        mdp.undesired_contacts,
        -1.0,
        threshold=1,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
    )

    # -- getting up, everywhere: being upright at standing height is what the walking set needs too
    upright = RewTerm(func=mdp.orientation_l2, weight=3.0, params={"desired_gravity": [0.0, 0.0, -1.0]})
    base_height = RewTerm(func=mdp.base_height_exp, weight=3.0, params={"target_height": STANDING_HEIGHT, "std": 0.15})

    # -- getting up, while down (Go2 get-up task weights). The get-up expert's raw actions are twice
    # what they were standalone (action scale 0.5 -> 0.25), so its action-rate charge is quartered.
    joint_pose = _down(mdp.joint_pos_target_l2, -0.3)
    action_rate_down = _down(mdp.action_rate_l2, -0.05 / 4)
    is_down = RewTerm(func=mdp.is_down, weight=-1.0)

    # -- falling: soften the landing
    body_impact = RewTerm(
        func=mdp.body_impact,
        weight=-2e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*"]), "threshold": 300.0},
    )

    # -- effort, everywhere
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)


@configclass
class TerminationsCfg:
    """Timeout only: a fall is something to get up from, not the end of the episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.reset_pose.params["standing_prob"] = 1.0
        self.events.knockdown.interval_range_s = (4.0, 6.0)
        # follow robot 0 so a recorded video keeps it in frame while it walks off
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)


@configclass
class RobotFallenPlayEnvCfg(RobotPlayEnvCfg):
    """Every episode starts already down, in a random tumbled pose, and resets to a new one every 6 s.

    The knockdown play config mostly shows the robot catching itself before it lands; this one shows
    only the get-up from a robot that is lying on the ground.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0
        self.events.reset_pose.params["standing_prob"] = 0.0
        self.events.knockdown = None
        self.events.push_robot = None


@configclass
class RobotHardKnockdownPlayEnvCfg(RobotPlayEnvCfg):
    """Walking forward, knocked down every 5 s harder than anything it can catch.

    At this strength 99% of knockdowns end on the ground (``measure_recovery.py --angular 32 40
    --linear 4 6``), so every one shows a real fall and a get-up rather than a stumble.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.events.push_robot = None
        self.events.knockdown.interval_range_s = (5.0, 5.0)
        self.events.knockdown.params = {"angular_speed_range": (32.0, 40.0), "linear_speed_range": (4.0, 6.0)}


@configclass
class GetUpEnvCfg(RobotEnvCfg):
    """Get-up alone, with exactly the observation, action and deploy interface of the walk / fall /
    get-up mixture (and of the blind walking policy it was built from).

    Same 117-column observation, joint order, default pose and 0.25 action scale, so one
    ``State_RLBase`` runs either policy -- only ``policy_dir`` differs. The velocity command is held
    at zero: its columns carry no information, so a policy trained or converted here keeps zero
    weight on them and ignores whatever the deploy side feeds in. The get-up policy itself is the
    original 42-dim one, converted exactly (``build_recovery_moe_checkpoint.py
    --getup-only``); this environment is for playing, exporting and, if wanted, fine-tuning it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.events.reset_pose.params["standing_prob"] = 0.1
        self.events.knockdown = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class GetUpPlayEnvCfg(GetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 6.0
        self.events.reset_pose.params["standing_prob"] = 0.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)
