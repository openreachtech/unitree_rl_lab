"""Blind, proprioception-only get-up task for the Go2: reset into a random
tumbled pose and recover to standing. No velocity command, no cameras/LiDAR.

Reused as-is from ``tasks.locomotion.robots.go2.velocity_env_cfg``: the robot
articulation/actuator config (``ROBOT_CFG``), and the joint-position action
term. Everything else here (scene terrain, events, observations without a
command, rewards, terminations) is specific to getting up rather than
walking, following the reset/reward strategy in iit-DLSLab/get-up-isaaclab
(``getup_env.py`` / ``go2_env_cfg.py``) adapted to this repo's manager-based
``ObsTerm``/``RewTerm``/``EventTerm`` style instead of a hand-rolled
``DirectRLEnv``.
"""

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
from unitree_rl_lab.tasks.locomotion import mdp

from . import mdp as getup_mdp

# Approximate Go2 standing base height (m). Reused from iit-DLSLab/get-up-isaaclab's
# ``desired_base_height`` for the Go2 (go2_env_cfg.py); adjust if the trained
# policy settles at a visibly different height than intended.
STANDING_HEIGHT = 0.30


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene for get-up training (no terrain curriculum needed here)."""

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
    """Configuration for events."""

    # startup
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

    # reset: drop into a random tumbled pose (or, with small probability, stay standing)
    reset_fallen_pose = EventTerm(
        func=getup_mdp.reset_fallen_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pos_range": (-3.14159, 3.14159),
            "standing_prob": 0.1,
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP. Blind / proprioception only, no command."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged, noise-free)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP. Deliberately small: upright + height + joint
    pose + a few smoothness/effort penalties. Extend once this baseline is
    behaving (see task notes for reward-hacking checks to run first).
    """

    # -- task: get upright, then hold the standing height, then match the standing joint pose
    upright = RewTerm(func=mdp.orientation_l2, weight=3.0, params={"desired_gravity": [0.0, 0.0, -1.0]})
    base_height = RewTerm(
        func=getup_mdp.base_height_exp,
        weight=3.0,
        params={"target_height": STANDING_HEIGHT, "std": 0.15},
    )
    joint_pose = RewTerm(func=getup_mdp.joint_pos_target_l2, weight=-0.3)

    # -- smoothness / effort
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP. Timeout only -- a fallen start pose is
    expected, so base contact / bad orientation must NOT end the episode
    early or every episode would terminate at step 0.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 blind get-up environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 8.0
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
        # keep every play env's fall genuinely random too (no debugging bias toward standing)
        self.events.reset_fallen_pose.params["standing_prob"] = 0.0
