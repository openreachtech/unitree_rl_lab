"""Curriculum for the hind-stance wall crossing task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["wall_terrain_levels"]


def wall_terrain_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "wallcross") -> torch.Tensor:
    """Up a row after a successful crossing, down a row after a jump that did not get over.

    Runs before the command terms reset, so the episode's outcome is still there to read. Episodes
    that carried no jump leave the level where it is. Both directions, unlike Go2W's one-way
    ratchet, whose robots piled up on rows they could not do (its SUMMARY, Lesson 3)."""
    wall = env.command_manager.get_term(command_name)
    ids = torch.as_tensor(env_ids, device=env.device)
    up = wall.success[ids]
    down = wall.jump_episode[ids] & wall.triggered[ids] & ~wall.success[ids]
    env.scene.terrain.update_env_origins(ids, up, down)
    return torch.mean(env.scene.terrain.terrain_levels.float())
