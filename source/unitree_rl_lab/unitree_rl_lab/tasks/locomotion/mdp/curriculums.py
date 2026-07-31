from __future__ import annotations

import json
import os
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


def _save_tow_assist_state(command) -> None:
    """Persist the tow-assist EFGCL curriculum state so it survives a training restart."""
    if command.cfg.state_file is None:
        return
    state = {
        "assist_scale": command.assist_scale,
        "curriculum_success_rate": command.curriculum_success_rate,
        "curriculum_episode_count": command.curriculum_episode_count,
        "curriculum_success_count": command.curriculum_success_count,
    }
    os.makedirs(os.path.dirname(command.cfg.state_file), exist_ok=True)
    tmp_path = command.cfg.state_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, command.cfg.state_file)


def tow_assist_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tow_command_name: str = "tow_assist",
    velocity_command_name: str = "base_velocity",
    error_threshold: float = 0.4,
    success_threshold: float = 0.6,
    decay_step: float = 0.01,
    minimum_episodes: int = 1024,
    min_episode_duration_s: float = 2.0,
) -> torch.Tensor:
    """EFGCL-style success-rate-gated decay of :class:`TowAssistCommand`'s assist force,
    following the same ``assist_scale = max(0, assist_scale - decay_step)`` schedule used for
    the jump/backflip/sideflip assist (see ``feat/jump``'s ``assist_force_decay``), but
    triggered by continuous velocity-tracking success instead of a one-shot landing check.

    Reuses ``velocity_command_name``'s own per-env running tracking-error metric
    (``UniformVelocityCommand._update_metrics`` accumulates this every step and it is only
    reset -- by ``CommandManager.reset()`` -- *after* the curriculum manager's ``compute()``
    runs for the same ``env_ids``, so it is still valid here) rather than duplicating that
    bookkeeping inside the tow-assist command itself.

    That metric is a running sum pre-divided by a *fixed* window
    (``resampling_time_range[1] / step_dt``), meant to approximate a mean error over a full
    command window -- but an episode that ends after only a few steps (e.g. the robot falls
    almost immediately) has barely accumulated anything, so it reads as a tiny, "good" error
    regardless of how badly tracking actually went. Left uncorrected, early failures under a
    random initial policy get miscounted as tracking successes, decaying the assist force
    before it's earned (confirmed via smoke test: assist_scale dropped from 1.0 to ~0.95
    within 3 iterations). Episodes shorter than ``min_episode_duration_s`` are still counted
    toward ``curriculum_episode_count`` (a fall is a real failure, not a no-op) but are never
    counted as a success, regardless of what the under-accumulated metric reads.
    """
    tow_command = env.command_manager.get_term(tow_command_name)
    if len(env_ids) == 0:
        return torch.tensor(tow_command.assist_scale, device=env.device)

    tracking_error = env.command_manager.get_term(velocity_command_name).metrics["error_vel_xy"][env_ids]
    elapsed_s = env.episode_length_buf[env_ids].float() * env.step_dt
    ran_long_enough = elapsed_s >= min_episode_duration_s
    successes = int(((tracking_error < error_threshold) & ran_long_enough).sum().item())

    tow_command.curriculum_episode_count += len(env_ids)
    tow_command.curriculum_success_count += successes

    if tow_command.curriculum_episode_count >= minimum_episodes:
        success_rate = tow_command.curriculum_success_count / tow_command.curriculum_episode_count
        tow_command.curriculum_success_rate = success_rate
        if success_rate >= success_threshold:
            tow_command.assist_scale = max(0.0, tow_command.assist_scale - decay_step)
        tow_command.curriculum_episode_count = 0
        tow_command.curriculum_success_count = 0
        _save_tow_assist_state(tow_command)

    return torch.tensor(tow_command.assist_scale, device=env.device)
