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

import os
import math
import torch
import scipy.spatial.transform as transform
from .lidar_cfg import get_go2_lidar_cfg
from .heightmap_visualizer import visualize_heightmap
import sys
# リポジトリルート (sourceとdeployが同居するフォルダ) を走査して自動取得
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = None
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "deploy")) and os.path.exists(os.path.join(current_dir, "source")):
        root_dir = current_dir
        break
    current_dir = os.path.dirname(current_dir)

if root_dir is None:
    # フォールバック (7個上に遡る)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../"))

ROOT_DIR = root_dir
HEIGHTMAP_DIR = os.path.join(ROOT_DIR, "deploy/robots/go2/unitree_go2_locomotion_heightmap")
sys.path.append(HEIGHTMAP_DIR)

from lidar_processor import HeightmapProcessor
import lidar_processor as lp

YAML_PATH = os.path.join(HEIGHTMAP_DIR, "heightmap_spec.yaml")
lidar_processor = HeightmapProcessor(config_yaml_path=YAML_PATH, device="cuda:0")

class LidarRotaryFilter:
    def __init__(self, spin_freq: float = 10.0, num_layers: int = 28, phase_shift_per_cycle: float = 3.214):
        self.spin_freq = spin_freq                # 回転速度 (Hz)
        self.num_layers = num_layers              # 1回転あたりのレイヤ数
        self.angle_step_rad = 2.0 * math.pi / num_layers # レイヤ間の角度間隔 (rad)
        self.phase_shift_rad = math.radians(phase_shift_per_cycle) # 周回ごとの位相シフト量 (rad)
        self.epsilon_rad = math.radians(1.5)      # 検出漏れを防ぐためのサンプリング角度許容誤差 (rad)

    def filter_points(self, pos_w: torch.Tensor, sensor_pos_w: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        N, R, _ = pos_w.shape
        device = pos_w.device
        pos_rel = pos_w - sensor_pos_w.unsqueeze(1)
        yaw_points = torch.atan2(pos_rel[..., 1], pos_rel[..., 0]) # [N, R]
        total_rotations = current_time * self.spin_freq # [N]
        base_angle = (total_rotations * 2.0 * math.pi) % (2.0 * math.pi) # [N]
        cycle_idx = total_rotations.long() # [N]
        phase_offset = (cycle_idx.float() * self.phase_shift_rad) % self.angle_step_rad # [N]
        
        layer_indices = torch.arange(self.num_layers, device=device).float() # [num_layers]
        valid_angles = layer_indices.view(1, -1) * self.angle_step_rad + base_angle.unsqueeze(1) + phase_offset.unsqueeze(1) # [N, num_layers]
        valid_angles = torch.atan2(torch.sin(valid_angles), torch.cos(valid_angles)) # [N, num_layers]
        
        diff = torch.abs(yaw_points.unsqueeze(-1) - valid_angles.unsqueeze(1)) # [N, R, num_layers]
        diff = torch.minimum(diff, 2.0 * math.pi - diff)
        
        mask = torch.any(diff < self.epsilon_rad, dim=-1) # [N, R]
        filtered_pos_w = pos_w.clone()
        filtered_pos_w[..., 2] = torch.where(mask, filtered_pos_w[..., 2], torch.tensor(-1e9, device=device))
        return filtered_pos_w

lidar_filter = LidarRotaryFilter()

def go2_lidar_heightmap(env, randomize: bool = False):
    """毎ステップ呼び出されるLiDAR観測関数"""
    current_time = env.episode_length_buf.float() * (env.cfg.sim.dt * env.cfg.decimation)
    filtered_pos_w = lidar_filter.filter_points(
        pos_w=env.scene["lidar"].data.ray_hits_w,
        sensor_pos_w=env.scene["lidar"].data.pos_w,
        current_time=current_time
    )
    heightmap = lidar_processor.process(
        pos_w=filtered_pos_w,
        root_pos_w=env.scene["robot"].data.root_pos_w,
        root_quat_w=env.scene["robot"].data.root_quat_w,
        randomize=randomize
    )
    if env.num_envs <= 8:
        visualize_heightmap(env, heightmap, lidar_processor)
    return heightmap

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

POLICY_HEIGHT_SCAN_CFG = ObsTerm(
    func=go2_lidar_heightmap,
    params={"randomize": True},
    clip=(-1.0, 5.0),
)

# Height scan grid matches RobotSceneCfgGo2V2.height_scanner (LiDAR origin, 17×11 @ 0.1 m).
CRITIC_HEIGHT_SCAN_CFG = ObsTerm(
    func=go2_lidar_heightmap,
    params={"randomize": False},
    clip=(-1.0, 5.0),
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

    # Heightmap usage flag
    use_heightmap: bool = True

    observations: ObservationsCfgGo2 = ObservationsCfgGo2()
    commands: CommandsCfgGo2 = CommandsCfgGo2()
    rewards: RewardsCfgGo2 = RewardsCfgGo2()

    def __post_init__(self):
        super().__post_init__()
        
        # use_heightmap が False の場合はハイトマップ観測とLiDARセンサーを無効化
        if not self.use_heightmap:
            if hasattr(self.observations.policy, "height_scan"):
                delattr(self.observations.policy, "height_scan")
            if hasattr(self.observations.critic, "height_scan"):
                delattr(self.observations.critic, "height_scan")
            if hasattr(self.scene, "lidar"):
                delattr(self.scene, "lidar")
        else:
            # LiDAR センサーを有効化し、不要になった height_scanner を削除
            self.scene.lidar = get_go2_lidar_cfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/lidar",
                config_yaml_path=YAML_PATH
            )
            self.scene.lidar.update_period = self.decimation * self.sim.dt
            if hasattr(self.scene, "height_scanner"):
                delattr(self.scene, "height_scanner")


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
