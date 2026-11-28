from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["fell_into_hole"]


def fell_into_hole(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[int, ...],
    nominal_height: float = 0.78,
    max_drop: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base more than ``max_drop`` below standing height over the spawn level, on the terrain
    columns in ``terrain_types`` (gaps, stepping stones), whose only ground below the spawn
    level is the floor of a hole. A height-above-terrain check cannot see this: in the hole the
    floor *is* the terrain, and the robot stands on it at a normal height."""
    terrain = env.scene.terrain
    types = torch.as_tensor(terrain_types, device=env.device)
    on_hole_terrain = torch.isin(terrain.terrain_types, types)
    z = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    return on_hole_terrain & (z < env.scene.env_origins[:, 2] + nominal_height - max_drop)
