"""Reset events that depend on the terrain column."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["reset_root_facing_axes"]


def reset_root_facing_axes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    terrain_types: tuple[int, ...],
    xy_range: float = 0.5,
    yaw_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Re-place the envs of the given terrain columns with yaw 0 / 90 / 180 / 270 deg
    (+- ``yaw_noise``), so walking forward meets the pyramid stair edges square on.

    Ported from tsubame_isaac_env. Put it after ``reset_root_state_uniform`` (full yaw);
    it overwrites the pose of those envs only: spawn point + uniform xy (+- ``xy_range``),
    zero velocity.
    """
    terrain = env.scene.terrain
    if getattr(terrain, "terrain_types", None) is None or not terrain_types:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids[torch.isin(terrain.terrain_types[env_ids], torch.tensor(terrain_types, device=env.device))]
    if len(ids) == 0:
        return

    n = len(ids)
    root = asset.data.default_root_state[ids].clone()
    pos = root[:, :3] + env.scene.env_origins[ids]
    pos[:, :2] += torch.empty(n, 2, device=env.device).uniform_(-xy_range, xy_range)
    yaw = torch.randint(0, 4, (n,), device=env.device).float() * (math.pi / 2)
    yaw += torch.empty(n, device=env.device).uniform_(-yaw_noise, yaw_noise)
    zeros = torch.zeros(n, device=env.device)
    asset.write_root_pose_to_sim(torch.cat([pos, quat_from_euler_xyz(zeros, zeros, yaw)], dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
