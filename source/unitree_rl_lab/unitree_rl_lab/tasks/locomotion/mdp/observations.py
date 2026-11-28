from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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


def height_scan_root(env: ManagerBasedRLEnv, sensor_cfg, offset: float, asset_cfg=None) -> torch.Tensor:
    """``root z - hit z - offset`` per ray: the same values as ``OdomElevationMap.sample`` (0 on flat
    ground at nominal stance), from a top-down ray caster. Missed rays read as a deep hole (1.0)."""
    from isaaclab.managers import SceneEntityCfg

    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    hits_z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w[..., 2]
    h = asset.data.root_pos_w[:, 2:3] - hits_z - offset
    # a miss comes back as +inf z, i.e. -inf here: treat as far below (a hole), not as a wall
    return torch.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)
