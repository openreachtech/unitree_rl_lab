"""Gait reward for Anaguma's ordinary (trotting) walking expert."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def diagonal_trot_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward a trot from contact states alone: diagonal pairs together, the two diagonals apart.

    ``sensor_cfg`` must list the feet as ``[FL, FR, RL, RR]`` (``preserve_order=True``). Three
    terms, each 0 or 1, averaged to 0..1:

    * FL and RR share the same contact state (one diagonal moves as a unit),
    * FR and RL share the same contact state (the other diagonal),
    * FL and FR differ (the two diagonals alternate rather than all four landing together).

    Self-referential on purpose, like ``paired_gait_reward``: ``feet_gait`` grades against a clock
    the policy does not observe, which a non-recurrent policy on the unified observation cannot
    track. Exact same-instant contact is what a trot is, so instantaneous states are enough here
    (the reason ``paired_gait`` needed stride phases was that a gallop's pairs are *offset*).

    Zero when the commanded velocity is below ``moving_threshold``, so standing still is free to
    keep all four feet down.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    stance = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # (N, 4) FL FR RL RR
    diag_a = (stance[:, 0] == stance[:, 3]).float()
    diag_b = (stance[:, 1] == stance[:, 2]).float()
    alternate = (stance[:, 0] != stance[:, 1]).float()
    reward = (diag_a + diag_b + alternate) / 3.0
    cmd = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs() > moving_threshold
    return reward * moving.float()
