"""Aliengo + Z1: whole-body end-effector pose tracking.

Reproduces "Whole-Body End-Effector Pose Tracking" (doc/papers/Whole-Body_End-Effector_Pose_Tracking.md)
on Aliengo + Z1. One MLP drives all 18 joints (12 legs + 6 arm; the gripper is held closed) to bring
the end-effector (``gripperStator``) to a commanded pose, without walking: the legs only tilt, crouch
and stretch the base to extend the arm's reach.

* Command (III-B/C): ``mdp.EePoseKeypointCommand`` -- pre-sampled arm-only poses
  (``scripts/tools/sample_z1_workspace.py``) under a random body transform, binned for uniform
  coverage, re-drawn when under the terrain, represented by 3 keypoints of a 0.2 m cube. A new pose
  every 4 s, 3 per 12 s episode.
* Observation (III-D): gravity, base velocities, joint positions, last actions, keypoint differences.
* Rewards (III-E): delayed keypoint tracking, progress, four-feet contact force, initial leg pose,
  and torque / acceleration / action-rate / joint-limit penalties, with the paper's weights except
  the feet contact force (normalized by body weight). Leg contacts are penalized as in Go2-Blind-GRU.
* Terminations: base contact only (see ``TerminationsCfg``).
* Terrain (III-F): the locomotion Phase 2 terrain (rough / boxes / slope up / down) with a fixed seed,
  and a curriculum on the episode's mean pose error.
* Reset (III-G): states recorded from the locomotion policy on the same sub-terrain
  (``scripts/tools/collect_init_states.py``).

Scaled from ANYmal + Dynaarm to Aliengo + Z1 (about 0.7x): body transform x/y +-0.15 m, z -0.2..+0.07 m
(paper +-0.2 m, -0.3..+0.1 m), cube side 0.2 m (paper 0.3 m). Action scale 0.25 on the legs (as the
locomotion policy), 0.5 on the arm (paper and Isaac Lab's arm tasks).

    python scripts/tools/collect_init_states.py --headless   # once, after Aliengoz1-Locomotion-Phase2
    python scripts/rsl_rl/train.py --task Aliengoz1-Pose-Tracking-Phase1 --headless
"""

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

from unitree_rl_lab.assets.robots.unitree import UNITREE_ALIENGO_Z1_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.aliengo_z1.locomotion_env_cfg import (
    ARM_GOAL_DATASET,
    ARM_JOINTS,
    FEET,
    LEG_JOINTS,
    PHASE2_TERRAIN_CFG,
    REPO_ROOT,
)

INIT_STATE_DATASET = str(REPO_ROOT / "datasets" / "aliengo_z1" / "init_states.pt")
"""Locomotion states on POSE_TRACKING_TERRAIN_CFG, made by scripts/tools/collect_init_states.py (git-ignored)."""

POSE_TRACKING_TERRAIN_CFG = PHASE2_TERRAIN_CFG.replace(seed=42)
"""The locomotion Phase 2 terrain, seeded: the recorded initial states only fit the exact same terrain."""

EE_BODY = "gripperStator"


def _joints(names: list[str]) -> dict:
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=names)}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=POSE_TRACKING_TERRAIN_CFG,
        max_init_terrain_level=2,
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    ee_pose = mdp.EePoseKeypointCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        dataset_path=ARM_GOAL_DATASET,
        resampling_time_range=(4.0, 4.0),
        tracking_window=2.0,
        cube_side=0.2,
        base_offset_ranges=mdp.EePoseKeypointCommandCfg.Ranges(
            x=(-0.15, 0.15),
            y=(-0.15, 0.15),
            z=(-0.2, 0.07),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(-math.pi / 6, math.pi / 6),
            yaw=(-math.pi / 6, math.pi / 6),
        ),
        terrain_margin=0.08,
    )


@configclass
class ActionsCfg:
    legs = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True)
    arm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Policy: o_t = [g_b, v_b, q, a_{t-1}] + keypoint differences (III-D), 3 + 6 + 18 + 18 + 9 = 54."""

    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, params=_joints(LEG_JOINTS + ARM_JOINTS), noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        last_action = ObsTerm(func=mdp.last_action)
        ee_keypoints = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """The policy's terms without noise, plus joint velocities and efforts and foot contacts."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params=_joints(LEG_JOINTS + ARM_JOINTS))
        last_action = ObsTerm(func=mdp.last_action)
        ee_keypoints = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params=_joints(LEG_JOINTS + ARM_JOINTS), scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, params=_joints(LEG_JOINTS + ARM_JOINTS), scale=0.01)
        foot_contact_force = ObsTerm(
            func=mdp.contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
            scale=0.01,
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """III-E, paper weights. Isaac Lab multiplies every term by the step dt, as Orbit did."""

    # -- task: R_T = 13 R_t + 80 R_p + w_f R_f + 0.4 R_q (R_f normalized by body weight, see feet_contact_force)
    ee_tracking = RewTerm(func=mdp.ee_keypoint_tracking, weight=13.0, params={"command_name": "ee_pose", "std": 0.05})
    ee_progress = RewTerm(func=mdp.ee_keypoint_progress, weight=80.0, params={"command_name": "ee_pose"})
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET), "threshold": 1.0},
    )
    leg_initial_pose = RewTerm(
        func=mdp.joint_pos_initial_exp,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS), "std": 0.05},
    )

    # -- penalties R_P (legs + arm)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-3e-5, params=_joints(LEG_JOINTS + ARM_JOINTS))
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-3e-6, params=_joints(LEG_JOINTS + ARM_JOINTS))
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-2)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.3, params=_joints(LEG_JOINTS + ARM_JOINTS))
    # Go2-Blind-GRU's leg contact penalty, in place of the paper's knee-contact termination
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """Terminate on base contact only.

    The paper (III-E) also terminates on knee contacts. With Aliengo's thigh and calf as the knee,
    98% of those were calves touching box edges, which ended a third of the episodes on the box
    terrain; leg contacts are penalized instead (``RewardsCfg.undesired_contacts``).
    Go2-Blind-GRU's tilt termination (0.8 rad) is left out as in the paper: reaching low or far goals
    tilts the base past it, and it ended 99% of the episodes once the policy learned to lean."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
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

    # reset: a recorded locomotion state on the robot's sub-terrain (III-G)
    reset_from_init_states = EventTerm(
        func=mdp.reset_from_init_states, mode="reset", params={"dataset_path": INIT_STATE_DATASET}
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_pose, params={"command_name": "ee_pose"})


@configclass
class RobotEnvCfgPhase1(ManagerBasedRLEnvCfg):
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
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        # sub-terrains laid out by difficulty, as when the initial states were recorded
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class RobotPlayEnvCfgPhase1(RobotEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        # goal (large) and current (small) end-effector frames
        self.commands.ee_pose.debug_vis = True
