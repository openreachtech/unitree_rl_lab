from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    threshold: float = 0.8,
) -> torch.Tensor:
    """Widen the range by 0.1 when the mean episode tracking reward exceeds ``threshold`` x weight."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * threshold:
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
    threshold: float = 0.8,
) -> torch.Tensor:
    """Widen the range by 0.1 when the mean episode tracking reward exceeds ``threshold`` x weight."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * threshold:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_vel_walls(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    distance_types: tuple[int, ...] = (),
    promote_distance: float = 2.3,
    demote_distance: float = 2.0,
    fail_termination_names: tuple[str, ...] = ("bad_orientation", "base_height"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, float]:
    """Terrain row ratchet: Isaac Lab's ``terrain_levels_vel``, except on wall columns.

    Columns not in ``distance_types``: up when the robot got half a tile from its spawn,
    down when it covered less than half of what its command asked for.

    Columns in ``distance_types`` (walls, forward-only commands): up when the radial
    distance from spawn exceeds ``promote_distance``, down when it stays under
    ``demote_distance`` or the episode ended in a fall (from tsubame_isaac_env's
    ``terrain_levels_tracking``). Clearing a wall needs a pause, so a robot that gets
    over falls behind its command and the stock rule would not promote it; with the first
    ring at 1.65 m, 2.3 / 2.0 m means "got past the first wall" / "did not".

    Logged per column (``level_<col>_<terrain>``, ``fall_<col>_<terrain>``) as well as the
    mean: each env stays in its column, and the mean alone does not show which terrain
    is stuck.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (distance < torch.norm(command[ids, :2], dim=1) * env.max_episode_length_s * 0.5) & ~move_up

    fell = torch.zeros_like(move_up)
    for name in fail_termination_names:
        fell |= env.termination_manager.get_term(name)[ids]
    if distance_types:
        on_wall = torch.isin(terrain.terrain_types[ids], torch.tensor(distance_types, device=env.device))
        up_d = distance > promote_distance
        down_d = ((distance < demote_distance) | fell) & ~up_d
        move_up = torch.where(on_wall, up_d, move_up)
        move_down = torch.where(on_wall, down_d, move_down)
    terrain.update_env_origins(ids, move_up, move_down)

    generator = terrain.cfg.terrain_generator
    weights = torch.tensor([sub.proportion for sub in generator.sub_terrains.values()])
    cumulative = torch.cumsum(weights / weights.sum(), dim=0)
    names = list(generator.sub_terrains.keys())
    out = {"mean": float(terrain.terrain_levels.float().mean())}
    for c in range(terrain.terrain_origins.shape[1]):
        # same rule as the generator: the first sub-terrain whose cumulative share exceeds c / num_cols
        name = names[int((cumulative > c / generator.num_cols + 0.001).nonzero()[0])]
        out[f"level_{c}_{name}"] = float(terrain.terrain_levels[terrain.terrain_types == c].float().mean())
        done_in_col = terrain.terrain_types[ids] == c
        if bool(done_in_col.any()):
            out[f"fall_{c}_{name}"] = float(fell[done_in_col].float().mean())
    return out
