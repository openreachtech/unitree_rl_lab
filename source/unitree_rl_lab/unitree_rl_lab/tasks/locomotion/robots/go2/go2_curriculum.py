from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainGeneratorCfg


def _column_indices_by_sub_terrain(terrain_generator_cfg: TerrainGeneratorCfg, num_cols: int) -> dict[str, list[int]]:
    """Map each sub-terrain name to terrain-grid column indices (Isaac Lab proportion layout)."""
    names = list(terrain_generator_cfg.sub_terrains.keys())
    cumsum = np.cumsum([terrain_generator_cfg.sub_terrains[n].proportion for n in names], dtype=np.float64)
    cumsum /= cumsum[-1]
    # Same rule as TerrainGenerator._generate_curriculum_terrains: col / num_cols + 0.001
    sub_indices = np.searchsorted(cumsum, np.arange(num_cols) / num_cols + 0.001)
    col_map = {name: [] for name in names}
    for col, idx in enumerate(sub_indices):
        col_map[names[int(idx)]].append(col)
    return col_map


def _sample_column_indices(
    env_ids: Sequence[int] | slice,
    num_envs: int,
    allowed_cols: list[int],
    device: torch.device,
) -> torch.Tensor:
    columns = torch.tensor(allowed_cols, dtype=torch.long, device=device)
    count = num_envs if isinstance(env_ids, slice) else len(env_ids)
    pick = torch.randint(0, columns.shape[0], (count,), device=device)
    return columns[pick]


def terrain_manual_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    level = int(getattr(env.cfg, "curriculum_level", 1))
    terrain_gen_cfg = terrain.cfg.terrain_generator
    col_map = _column_indices_by_sub_terrain(terrain_gen_cfg, terrain_gen_cfg.num_cols)

    if level <= 1:
        allowed_cols = col_map["flat"]
        terrain.terrain_types[env_ids] = _sample_column_indices(env_ids, env.num_envs, allowed_cols, env.device)
        terrain.terrain_levels[env_ids] = 0
        terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
        return torch.mean(terrain.terrain_levels.float())

    if level == 2:
        allowed_cols: list[int] = []
        for name in ("random_rough", "boxes"):
            allowed_cols.extend(col_map.get(name, []))
    else:
        allowed_cols = []
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            allowed_cols.extend(col_map.get(name, []))

    terrain.terrain_types[env_ids] = _sample_column_indices(env_ids, env.num_envs, allowed_cols, env.device)
    # Row difficulty uses Isaac Lab default (updates levels + env_origins via update_env_origins).
    return mdp.terrain_levels_vel(env, env_ids, asset_cfg)


@configclass
class CurriculumCfgGo2(CurriculumCfg):
    def __post_init__(self):
        self.terrain_levels = CurrTerm(func=terrain_manual_levels)
