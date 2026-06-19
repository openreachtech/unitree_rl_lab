from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands import UniformLevelVelocityCommandCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainGeneratorCfg

Ranges = UniformLevelVelocityCommandCfg.Ranges

# Privileged critic (policy does not use height_scan).
CRITIC_HEIGHT_SCAN_CFG = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    clip=(-1.0, 5.0),
)

# Per manual curriculum_level: starting command ranges and in-phase expansion caps.
PHASE_VEL_START: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)),
    2: Ranges(lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)),
    3: Ranges(lin_vel_x=(-0.05, 0.1), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.5, 0.5)),
}

PHASE_VEL_LIMIT: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)),
    2: Ranges(lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.1, 1.1)),
    3: Ranges(lin_vel_x=(-0.8, 1.2), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.0, 1.0)),
}

# In-place spin focus command ranges (keyed by curriculum level bucket).
SPIN_FOCUS_VEL_START: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-1.0, 1.0)),
    2: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-1.0, 1.0)),
    3: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.8, 0.8)),
}
SPIN_FOCUS_VEL_LIMIT: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-1.5, 1.5)),
    2: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-1.5, 1.5)),
    3: Ranges(lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-1.0, 1.0)),
}

def curriculum_level_key(curriculum_level: int) -> int:
    if curriculum_level <= 1:
        return 1
    if curriculum_level == 2:
        return 2
    return 3


def apply_phase_velocity_ranges(env_cfg) -> None:
    """Apply level-based velocity ranges (or spin-focus ranges when enabled)."""
    key = curriculum_level_key(env_cfg.curriculum_level)
    if getattr(env_cfg, "focus_spin_in_place", False):
        env_cfg.commands.base_velocity.ranges = SPIN_FOCUS_VEL_START[key]
        env_cfg.commands.base_velocity.limit_ranges = SPIN_FOCUS_VEL_LIMIT[key]
        return
    env_cfg.commands.base_velocity.ranges = PHASE_VEL_START[key]
    env_cfg.commands.base_velocity.limit_ranges = PHASE_VEL_LIMIT[key]


def apply_play_velocity_ranges(env_cfg) -> None:
    """Play mode: use level-based velocity limits for keyboard/resampling."""
    key = curriculum_level_key(env_cfg.curriculum_level)
    if getattr(env_cfg, "focus_spin_in_place", False):
        play_ranges = SPIN_FOCUS_VEL_LIMIT[key]
    else:
        play_ranges = PHASE_VEL_LIMIT[key]
    env_cfg.commands.base_velocity.ranges = play_ranges
    env_cfg.commands.base_velocity.limit_ranges = play_ranges


def apply_phase_terrain_settings(env_cfg) -> None:
    """Set ``max_init_terrain_level`` for the current manual curriculum level."""
    level = env_cfg.curriculum_level
    if level <= 1:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    else:
        num_rows = env_cfg.scene.terrain.terrain_generator.num_rows
        env_cfg.scene.terrain.max_init_terrain_level = min(2, num_rows - 1)

def _set_reward_weight(rewards, name: str, weight: float) -> None:
    if hasattr(rewards, name):
        getattr(rewards, name).weight = weight

def apply_phase_reward_settings(env_cfg) -> None:
    """Apply level-specific reward weights (stairs: prioritize forward motion over staying put)."""
    rewards = env_cfg.rewards
    if curriculum_level_key(env_cfg.curriculum_level) >= 3:
        _set_reward_weight(rewards, "feet_height_body_stairs", -0.15)
        _set_reward_weight(rewards, "wild_foot_clearance", 0.4)
    else:
        _set_reward_weight(rewards, "feet_height_body_stairs", 0.0)
        _set_reward_weight(rewards, "wild_foot_clearance", 0.0)



def apply_manual_curriculum_level(env_cfg) -> None:
    """Apply static env settings for the current ``curriculum_level`` (terrain + velocity)."""
    apply_phase_terrain_settings(env_cfg)
    apply_phase_velocity_ranges(env_cfg)
    apply_phase_reward_settings(env_cfg)


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
        self.lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
