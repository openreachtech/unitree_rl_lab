from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def height_scan_noise_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log the height-scan noise curriculum factor set by ``HeightScanExcludingBodyNoisy``."""
    return torch.tensor(getattr(env, "height_scan_noise_level", 0.0), device=env.device)


def custom_terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
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
