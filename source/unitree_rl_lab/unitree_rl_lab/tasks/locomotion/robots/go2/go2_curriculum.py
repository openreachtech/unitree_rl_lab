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
CRITIC_HISTORY_LENGTH_FLAT = 5
CRITIC_HISTORY_LENGTH_ROUGH = 5


def _new_height_scan_term() -> ObsTerm:
    """Fresh ObsTerm instance per call; avoids sharing mutable state (history_length) across obs groups."""
    return ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 5.0),
    )

# Per manual curriculum_level: starting command ranges and in-phase expansion caps.
#
# Level 3 (stairs) is FORWARD-BIASED on purpose: lin_vel_x stays positive and the
# yaw range is shrunk. The terrain curriculum advances a robot by net displacement
# from its spawn, but a zero-mean (symmetric +/-) command makes the robot wander in
# place with ~0 net displacement, so it could never satisfy the move-up threshold on
# stairs. A forward-biased command turns "track the command well" into "actually make
# forward progress up the stairs", which is what we need.
PHASE_VEL_START: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.0, 1.0)),
    2: Ranges(lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-1.0, 1.0)),
    3: Ranges(lin_vel_x=(0.1, 0.4), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)),
}

PHASE_VEL_LIMIT: dict[int, Ranges] = {
    1: Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)),
    2: Ranges(lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-1.0, 1.0)),
    3: Ranges(lin_vel_x=(0.1, 0.6), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)),
}

# Play / deploy: fixed command ranges (keyboard W/S uses lin_vel_x min/max).
PLAY_VEL_RANGES = Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))


def apply_phase_velocity_ranges(env_cfg) -> None:
    """Reset ``base_velocity.ranges`` to the start of the current manual curriculum level."""
    level = env_cfg.curriculum_level
    key = 1 if level <= 1 else (2 if level == 2 else 3)
    env_cfg.commands.base_velocity.ranges = PHASE_VEL_START[key]


def apply_phase_terrain_settings(env_cfg) -> None:
    """Set ``max_init_terrain_level`` for the current manual curriculum level."""
    level = env_cfg.curriculum_level
    if level <= 1:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    else:
        num_rows = env_cfg.scene.terrain.terrain_generator.num_rows
        env_cfg.scene.terrain.max_init_terrain_level = min(2, num_rows - 1)


def apply_phase_height_scanner(env_cfg) -> None:
    """Enable RayCaster + height_scan (policy: single-frame, critic: history) for all levels."""
    if env_cfg.curriculum_level <= 1:
        env_cfg.observations.policy.height_scan = _new_height_scan_term()
        env_cfg.observations.critic.height_scan = _new_height_scan_term()
        env_cfg.observations.critic.history_length = CRITIC_HISTORY_LENGTH_FLAT
    else:
        env_cfg.observations.policy.height_scan = _new_height_scan_term()
        env_cfg.observations.critic.height_scan = _new_height_scan_term()
        env_cfg.observations.critic.history_length = CRITIC_HISTORY_LENGTH_ROUGH


def apply_manual_curriculum_level(env_cfg) -> None:
    """Apply static env settings for the current ``curriculum_level`` (terrain + velocity + critic obs)."""
    apply_phase_terrain_settings(env_cfg)
    apply_phase_velocity_ranges(env_cfg)
    apply_phase_height_scanner(env_cfg)


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


def terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain-difficulty ratchet tuned for slow, hard terrain (stairs).

    Replacement for ``mdp.terrain_levels_vel``. The stock function:
      * move_up  when distance-from-spawn > terrain_size/2  (= 4.0 m)
      * move_down when distance < commanded_speed * episode_time * 0.5

    On stairs at a throttled command speed, a 4.0 m net displacement inside one
    episode is essentially unreachable, while the velocity-scaled move_down floor
    fires almost every reset -> levels collapse to 0 (observed: 0.057).

    This version:
      * move_up  at a reachable fraction of the tile (35 % = ~2.8 m), so a robot
        that genuinely climbs a few steps forward is promoted.
      * move_down only when the robot barely moved (< 0.5 m), i.e. it actually
        failed. A robot making partial progress stays on its level and keeps
        practising instead of being demoted. This turns the curriculum into a
        one-way ratchet that tracks real skill instead of net wandering.
    """
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    asset = env.scene[asset_cfg.name]
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    tile_size = terrain.cfg.terrain_generator.size[0]
    move_up = distance > tile_size * 0.35
    move_down = (distance < 0.5) & (~move_up)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


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
    # Row difficulty via the climb-tuned ratchet (updates levels + env_origins).
    return terrain_levels_climb(env, env_ids, asset_cfg)


def lin_vel_cmd_levels_go2(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Expand velocity ranges via ``mdp.lin_vel_cmd_levels``, capped per ``curriculum_level``."""
    level = int(getattr(env.cfg, "curriculum_level", 1))
    cmd_cfg = env.command_manager.get_term("base_velocity").cfg
    saved_limit = cmd_cfg.limit_ranges
    limit_key = 1 if level <= 1 else (2 if level == 2 else 3)
    cmd_cfg.limit_ranges = PHASE_VEL_LIMIT[limit_key]
    try:
        return mdp.lin_vel_cmd_levels(env, env_ids, reward_term_name)
    finally:
        cmd_cfg.limit_ranges = saved_limit


@configclass
class CurriculumCfgGo2(CurriculumCfg):
    def __post_init__(self):
        self.terrain_levels = CurrTerm(func=terrain_manual_levels)
        self.lin_vel_cmd_levels = CurrTerm(func=lin_vel_cmd_levels_go2)
