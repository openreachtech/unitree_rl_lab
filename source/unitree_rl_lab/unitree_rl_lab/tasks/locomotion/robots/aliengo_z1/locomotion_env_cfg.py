"""Aliengo + Z1: locomotion pre-training with a moving arm.

Pre-trains the locomotion policy that "Whole-Body End-Effector Pose Tracking"
(doc/papers/Whole-Body_End-Effector_Pose_Tracking.md, III-G) relies on: an MLP that drives the
12 leg joints only and walks with the arm in any configuration. The arm is moved by a scripted
command (``mdp.ArmJointCommand``): its PD targets travel toward goal configurations drawn from
the collision-free set made by ``scripts/tools/sample_z1_workspace.py`` (re-drawn every 3-6 s).
The policy observes the arm's joint positions and velocities, not its goal.

Terrain, rewards, commands and the privileged critic follow the Go2-Blind-GRU line
(``robots/go2/velocity_env_cfg_blind*.py``), with an MLP instead of the GRU:
  Phase 1  flat plane
  Phase 2  random rough / boxes / pyramid slope up / down, 25% each, terrain-level curriculum
Written standalone (only the foot-ring sensor helper is imported) so Go2 experiments do not
leak in, and so that every joint-space term can be restricted to the legs.

    python scripts/tools/sample_z1_workspace.py --headless   # once, makes the arm goal set
    python scripts/rsl_rl/train.py --task AliengoZ1-Locomotion-Phase1 --headless
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_ALIENGO_Z1_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import foot_ring_sensor

REPO_ROOT = Path(__file__).resolve().parents[7]
ARM_GOAL_DATASET = str(REPO_ROOT / "datasets" / "aliengo_z1" / "z1_workspace.pt")
"""Collision-free arm configurations, made by scripts/tools/sample_z1_workspace.py (git-ignored)."""

LEG_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
"""Joints driven by the policy."""

ARM_JOINTS = ["joint[1-6]"]
"""Z1 joints moved by the scripted arm command. The gripper is held closed and not observed."""

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Steps of proprioceptive history in the (memoryless, blind) policy observation.
POLICY_HISTORY_LENGTH = 5

_FOOT_SCAN_CFGS = [SceneEntityCfg(f"foot_scan_{name}") for name in FEET]


def _legs() -> dict:
    """``params`` restricting a joint-space term to the leg joints."""
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}


def _legs_and_arm() -> dict:
    """``params`` restricting a joint-state observation to legs + arm (no gripper): 18 joints."""
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS + ARM_JOINTS)}


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    # Go2-Blind-GRU-Phase2's rough and boxes, plus pyramid slopes up and down (ranges as in Isaac
    # Lab's ROUGH_TERRAINS_CFG). One column per sub-terrain: the generator assigns sub-terrains to
    # columns by proportion and difficulty to rows, so 4 x 25% gives rough | boxes | slope |
    # inverted slope, each over 10 difficulty rows.
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=4,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=2.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Phase 1: flat plane + Aliengo-Z1."""

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
    robot: ArticulationCfg = UNITREE_ALIENGO_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # Per-foot terrain rings for the privileged critic (Lee et al. 2020), as in Go2-Blind-GRU.
    foot_scan_FL_foot: RayCasterCfg = foot_ring_sensor("FL_foot")
    foot_scan_FR_foot: RayCasterCfg = foot_ring_sensor("FR_foot")
    foot_scan_RL_foot: RayCasterCfg = foot_ring_sensor("RL_foot")
    foot_scan_RR_foot: RayCasterCfg = foot_ring_sensor("RR_foot")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class RobotSceneCfgPhase2(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_type="generator", terrain_generator=PHASE2_TERRAIN_CFG, max_init_terrain_level=2
    )


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------
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

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # The arm starts every episode folded; the arm command moves it from there.
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]", "jointGripper"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Base velocity as in Go2-Blind-GRU-Phase1, plus the scripted arm motion."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )

    arm = mdp.ArmJointCommandCfg(
        asset_name="robot",
        dataset_path=ARM_GOAL_DATASET,
        resampling_time_range=(3.0, 6.0),
        max_joint_vel=0.7,
        default_pose_prob=0.2,
    )


@configclass
class CommandsCfgPhase2(CommandsCfg):
    base_velocity = CommandsCfg().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class ActionsCfg:
    """Legs only. The arm follows the scripted arm command through its PD actuators."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Joint states cover legs + arm (18 joints, no gripper); last_action is the 12 leg actions.

    The MLP policy is blind and has no memory, so its proprioceptive terms carry the last
    ``POLICY_HISTORY_LENGTH`` steps: the terrain under the feet can only be inferred from how the
    body and joints responded over the past few steps (Go2-Blind-GRU gets this from its GRU)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=POLICY_HISTORY_LENGTH,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=POLICY_HISTORY_LENGTH,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params=_legs_and_arm(),
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=POLICY_HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params=_legs_and_arm(),
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=POLICY_HISTORY_LENGTH,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100), history_length=POLICY_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic: Go2-Blind-GRU's terms, plus the arm goal."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        arm_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "arm"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params=_legs_and_arm(), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params=_legs_and_arm(), scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, params=_legs_and_arm(), scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # Terrain profile at the feet and foot-ground interaction (Lee et al. 2020).
        foot_height_scan = ObsTerm(
            func=mdp.foot_height_scan, params={"sensor_cfgs": _FOOT_SCAN_CFGS}, clip=(-1.0, 1.0)
        )
        foot_terrain_normal = ObsTerm(func=mdp.foot_terrain_normal, params={"sensor_cfgs": _FOOT_SCAN_CFGS})
        foot_contact_force = ObsTerm(
            func=mdp.contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
            scale=0.01,
        )
        foot_contact_state = ObsTerm(
            func=mdp.contact_states, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)}
        )
        thigh_contact_state = ObsTerm(
            func=mdp.contact_states, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh")}
        )
        calf_contact_state = ObsTerm(
            func=mdp.contact_states, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf")}
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Go2-Blind-GRU rewards (upstream Go2 + track_ang_vel_z at 1.0; its extra terms are all at
    weight 0 and left out). Joint-space regularizers act on the legs only, and arm contacts are
    not penalized: the policy does not control the arm, so it could not act on either."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params=_legs())
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params=_legs())
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4, params=_legs())
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params=_legs())
    energy = RewTerm(func=mdp.energy, weight=-2e-5, params=_legs())

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- other: legs (other than feet) touching anything
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class RewardsCfgPhase2(RewardsCfg):
    """Posture penalties relaxed for box terrain, as in Go2-Blind-GRU-Phase2: stepping onto a box
    needs vertical base velocity, body pitch and a non-default stance."""

    base_linear_velocity = RewardsCfg().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfg().flat_orientation_l2.replace(weight=-1.0)
    joint_pos = RewardsCfg().joint_pos.replace(weight=-0.4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Phase 1: flat plane, so no terrain levels."""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class CurriculumCfgPhase2(CurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
@configclass
class RobotEnvCfgPhase1(ManagerBasedRLEnvCfg):
    """Phase 1: flat plane."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        # Lay the generated terrain out by difficulty only when the terrain-level curriculum runs.
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.curriculum = getattr(self.curriculum, "terrain_levels", None) is not None


@configclass
class RobotPlayEnvCfgPhase1(RobotEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: random rough, boxes and pyramid slopes, terrain-level curriculum."""

    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()
    curriculum: CurriculumCfgPhase2 = CurriculumCfgPhase2()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 4 columns (rough | boxes | slope | inverted slope) x 5 rows, spawning over every row so all
        # difficulties are visible.
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
