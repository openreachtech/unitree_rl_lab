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
    3: Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)),
}

PHASE_VEL_LIMIT: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)),
    2: Ranges(lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)),
    3: Ranges(lin_vel_x=(-1.0, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)),
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

# Train: per-level terrain spawn mix (independent of terrain mesh column proportions).
TERRAIN_SPAWN_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "flat": 1.0,
    },
    2: {
        "flat": 0.10,
        "random_rough": 0.40,
        "boxes": 0.50,
    },
    3: {
        "flat": 0.04,
        "random_rough": 0.08,
        "boxes": 0.08,
        "pyramid_stairs": 0.30,
        "pyramid_stairs_inv": 0.50,
    },
}

# Play: one terrain type per phase (no flat/rough mix on stairs levels).
PLAY_TERRAIN_SPAWN_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"flat": 1.0},
    2: {"random_rough": 0.2, "boxes": 0.8},
    3: {"pyramid_stairs": 0.2, "pyramid_stairs_inv": 0.8},
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
        env_cfg.scene.terrain.max_init_terrain_level = min(3, num_rows - 1)

def _set_reward_weight(rewards, name: str, weight: float) -> None:
    if hasattr(rewards, name):
        getattr(rewards, name).weight = weight


def _set_reward_param(rewards, name: str, param: str, value) -> None:
    if hasattr(rewards, name):
        getattr(rewards, name).params[param] = value


def apply_phase_reward_settings(env_cfg) -> None:
    """Apply level-specific reward weights for stair vs flat/rough phases."""
    rewards = env_cfg.rewards
    if curriculum_level_key(env_cfg.curriculum_level) >= 3:
        _set_reward_weight(rewards, "flat_orientation_l2", -1.0)
        _set_reward_weight(rewards, "base_linear_velocity", -0.5)
        _set_reward_weight(rewards, "joint_torques", -1e-4)
        _set_reward_weight(rewards, "action_rate", -0.05)
        # _set_reward_weight(rewards, "feet_air_time", 0.07)
        # _set_reward_param(rewards, "feet_air_time", "threshold", 0.35)
        # _set_reward_weight(rewards, "feet_height_body_stairs", -0.15)
        _set_reward_weight(rewards, "wild_foot_clearance", 0.4)



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


def terrain_manual_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    phase = curriculum_level_key(int(getattr(env.cfg, "curriculum_level", 1)))
    spawn_table = PLAY_TERRAIN_SPAWN_WEIGHTS if getattr(env.cfg, "play_mode", False) else TERRAIN_SPAWN_WEIGHTS
    spawn = spawn_table[phase]
    terrain_gen = terrain.cfg.terrain_generator
    col_map = _column_indices_by_sub_terrain(terrain_gen, terrain_gen.num_cols)

    names = [n for n in spawn if spawn[n] > 0 and col_map.get(n)]
    count = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
    device = env.device

    # 1) pick sub-terrain by TERRAIN_SPAWN_WEIGHTS, 2) pick a random column within it
    type_pick = torch.multinomial(torch.tensor([spawn[n] for n in names], device=device), count, replacement=True)
    sampled_cols = torch.empty(count, dtype=torch.long, device=device)
    for t, name in enumerate(names):
        mask = type_pick == t
        if not mask.any():
            continue
        cols = torch.tensor(col_map[name], device=device)
        sampled_cols[mask] = cols[torch.randint(len(cols), (int(mask.sum()),), device=device)]

    terrain.terrain_types[env_ids] = sampled_cols
    return mdp.terrain_levels_vel(env, env_ids, asset_cfg)


@configclass
class CurriculumCfgGo2(CurriculumCfg):
    def __post_init__(self):
        self.terrain_levels = CurrTerm(func=terrain_manual_levels)
        self.lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
