from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def height_scan_excluding_body(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exclude_half_extent_x: float = 0.22,
    exclude_half_extent_y: float = 0.12,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Height scan with points under the robot body masked out.

    This computes the same height feature as the default height_scan
    (base height minus ray-hit z minus offset), then replaces rays whose
    hit points fall inside a center rectangle around the robot base.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2] - offset

    # Use hit-point XY in world frame relative to robot root XY.
    rel_xy = sensor.data.ray_hits_w[:, :, :2] - asset.data.root_pos_w[:, :2].unsqueeze(1)
    under_body_mask = (rel_xy[..., 0].abs() <= exclude_half_extent_x) & (rel_xy[..., 1].abs() <= exclude_half_extent_y)

    return torch.where(under_body_mask, torch.full_like(heights, fill_value), heights)
