from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.models.teacher_actor import TeacherActorCritic
from unitree_rl_lab.assets.models.modules.student_teacher import StudentTeacher
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distillation_cfg import BeliefDistillationRunnerCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.observations import height_scan_excluding_body
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

POLICY_HISTORY_LENGTH = 3
CRITIC_HISTORY_LENGTH = 3

# Must match RobotSceneCfg.height_scanner.pattern_cfg (velocity_env_cfg.py):
# patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=HEIGHT_SCAN_SIZE).
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_SIZE = (1.6, 1.0)


def _grid_pattern_num_points(resolution: float, size: tuple[float, float]) -> int:
    """Ray count produced by isaaclab.sensors.ray_caster.patterns.GridPatternCfg.

    Mirrors isaaclab's grid_pattern(): arange(-size/2, size/2 + eps, resolution) includes both
    endpoints, so each axis has round(size / resolution) + 1 points, not size / resolution.
    """
    num_x = round(size[0] / resolution) + 1
    num_y = round(size[1] / resolution) + 1
    return num_x * num_y


# LiDAR mount in base frame (LiDAR -> base translation). Matches deploy height_scan_pipeline.
GO2_LIDAR_OFFSET_X = 0.28945  # m, forward from base
GO2_LIDAR_OFFSET_Y = 0.0
GO2_LIDAR_OFFSET_Z = -0.046825  # m

# Nominal standing base height (world z above ground). Used to zero height_scan on flat terrain.
GO2_NOMINAL_BASE_Z = 0.32  # m

# Isaac mdp.height_scan offset: ground-to-sensor height at nominal stance (flat terrain -> ~0).
GO2_HEIGHT_SCAN_OFFSET = GO2_NOMINAL_BASE_Z + GO2_LIDAR_OFFSET_Z  # 0.273175 m

# RayCaster grid xy origin at LiDAR mount (matches unitree_mujoco utlidar site on base_link).
# Wired into RobotSceneCfg.height_scanner.offset in RobotEnvCfgGo2.__post_init__ below (z there
# is a fixed ray-start height for raycasting, unrelated to GO2_LIDAR_OFFSET_Z).
GO2_HEIGHT_SCANNER_OFFSET = (
    GO2_LIDAR_OFFSET_X,
    GO2_LIDAR_OFFSET_Y,
    GO2_LIDAR_OFFSET_Z,
)

# 17×11 grid, resolution 0.1m, size [1.6, 1.0]m
POLICY_HEIGHT_SCAN_CFG = ObsTerm(
    func=height_scan_excluding_body,
    params={
        "sensor_cfg": SceneEntityCfg("height_scanner"),
        "asset_cfg": SceneEntityCfg("robot"),
        "offset": GO2_HEIGHT_SCAN_OFFSET,
        # Mask points under body footprint (in base xy, meters).
        "exclude_half_extent_x": 0.22,
        "exclude_half_extent_y": 0.12,
        "fill_value": 0.0,
    },
    clip=(-1.0, 5.0),
    # noise=Unoise(n_min=-0.05, n_max=0.05),
    history_length=0,
)

# Height scan grid matches RobotSceneCfgGo2V2.height_scanner (LiDAR origin, 17×11 @ 0.1 m).
CRITIC_HEIGHT_SCAN_CFG = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": GO2_HEIGHT_SCAN_OFFSET},
    clip=(-1.0, 5.0),
    history_length=0,
)


@configclass
class PolicyCfgGo2(ObservationsCfg.PolicyCfg):
    """Go2 policy: per-term observation history for temporal context."""

    joint_pos_rel = ObservationsCfg.PolicyCfg().joint_pos_rel.replace(history_length=POLICY_HISTORY_LENGTH)
    joint_vel_rel = ObservationsCfg.PolicyCfg().joint_vel_rel.replace(history_length=POLICY_HISTORY_LENGTH)
    last_action = ObservationsCfg.PolicyCfg().last_action.replace(history_length=POLICY_HISTORY_LENGTH)
    height_scan = POLICY_HEIGHT_SCAN_CFG

    def __post_init__(self):
        super().__post_init__()


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic with explicit order: proprio -> extero(height_scan) -> privileged."""

    base_ang_vel = ObservationsCfg.CriticCfg().base_ang_vel
    projected_gravity = ObservationsCfg.CriticCfg().projected_gravity
    velocity_commands = ObservationsCfg.CriticCfg().velocity_commands
    joint_pos_rel = ObservationsCfg.CriticCfg().joint_pos_rel.replace(history_length=CRITIC_HISTORY_LENGTH)
    joint_vel_rel = ObservationsCfg.CriticCfg().joint_vel_rel.replace(history_length=CRITIC_HISTORY_LENGTH)
    last_action = ObservationsCfg.CriticCfg().last_action.replace(history_length=CRITIC_HISTORY_LENGTH)
    height_scan = CRITIC_HEIGHT_SCAN_CFG
    # critic-only privileged terms
    base_lin_vel = ObservationsCfg.CriticCfg().base_lin_vel
    joint_effort = ObservationsCfg.CriticCfg().joint_effort


@configclass
class ObservationsCfgGo2(ObservationsCfg):
    """Go2 observations: policy history; extended critic for privileged training."""

    policy: PolicyCfgGo2 = PolicyCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class CommandsCfgGo2(CommandsCfg):
    """Go2 v1: reduce standing-only env fraction."""

    base_velocity = CommandsCfg().base_velocity.replace(rel_standing_envs=0.01)


@configclass
class RewardsCfgGo2(RewardsCfg):
    """Go2-specific reward tuning."""

    track_ang_vel_z = RewardsCfg().track_ang_vel_z.replace(weight=1.0)

    wild_foot_clearance = RewTerm(
        func=mdp.wild_foot_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],  # trot: FR+RL vs FL+RR
            "radius": 0.1,
            "target_clearance": 0.05,
        },
    )

    foot_clearance_terrain_adaptive = RewTerm(
        func=mdp.foot_clearance_terrain_adaptive,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.05,
            "command_name": "base_velocity",
        },
    )

    forward_command_progress = RewTerm(
        func=mdp.forward_command_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RobotEnvCfgGo2(RobotEnvCfg):
    """Shared Go2 v1 MDP settings."""

    observations: ObservationsCfgGo2 = ObservationsCfgGo2()
    commands: CommandsCfgGo2 = CommandsCfgGo2()
    rewards: RewardsCfgGo2 = RewardsCfgGo2()

    def __post_init__(self):
        super().__post_init__()
        # Show height-scan rays/hits in Isaac Sim GUI for Go2 tasks.
        self.scene.height_scanner.debug_vis = True
        # Shift the grid to the LiDAR mount (z is just a fixed ray-start height for raycasting,
        # unrelated to GO2_LIDAR_OFFSET_Z); matches unitree_mujoco utlidar site / deploy
        # HeightScanUpdater.
        _, _, z = self.scene.height_scanner.offset.pos
        self.scene.height_scanner.offset.pos = (GO2_LIDAR_OFFSET_X, GO2_LIDAR_OFFSET_Y, z)
        # ordering="yx": inner loop over y, outer loop over x (idx = ix * Ny + iy), matching the
        # flatten order used by unitree_mujoco height_map_simulator and deploy HeightScanUpdater.
        self.scene.height_scanner.pattern_cfg.ordering = "yx"


def _go2_obs_block_dims() -> tuple[int, int, int]:
    """Go2 policy/critic observation block widths (proprio, extero, priv).

    Must match ``ObservationsCfgGo2`` / ``ObservationsCfgStudent``:
      proprio: ang_vel(3)+gravity(3)+cmd(3)+joint_pos(12*H)+joint_vel(12*H)+action(12*H)
      extero:  height-scan grid
      priv:    critic-only base_lin_vel(3)+joint_effort(12)
    """
    proprio = 3 + 3 + 3 + 12 * POLICY_HISTORY_LENGTH * 3
    extero = _grid_pattern_num_points(HEIGHT_SCAN_RESOLUTION, HEIGHT_SCAN_SIZE)
    priv = 3 + 12
    return proprio, extero, priv


@configclass
class TeacherPPORunnerCfg(BasePPORunnerCfg):
    """Use custom teacher actor-critic with rsl-rl PPO runner."""

    def __post_init__(self):
        super().__post_init__()
        # Ensure class symbol is imported in this module for config serialization/debug.
        _ = TeacherActorCritic
        self.policy.class_name = "TeacherActorCritic"
        proprio, extero, priv = _go2_obs_block_dims()
        self.policy.proprio_obs_dim = proprio
        self.policy.extero_obs_dim = extero
        self.policy.priv_obs_dim = priv


@configclass
class StudentDistillationRunnerCfg(BeliefDistillationRunnerCfg):
    """Belief-encoder student distillation (BC + height-map reconstruction)."""

    def __post_init__(self):
        # Ensure class symbol is imported for config serialization/debug.
        _ = StudentTeacher
        proprio, extero, priv = _go2_obs_block_dims()
        self.policy.proprio_obs_dim = proprio
        self.policy.extero_obs_dim = extero
        # TeacherポリシーはPreveledgedInfoを使っていないが、ゼロ埋めしている。
        # そのため、Studentポリシーも同じゼロ埋めを行う必要がある。
        self.policy.priv_obs_dim = priv

        # TeacherポリシーのExteroceptiveEncoderをStudentポリシーに転送しない。
        # self.policy.transfer_extero_encoder_from_teacher = False
