from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def jump_time_encoding(env, command_name: str = "jump", time_scale: float = 1.0) -> torch.Tensor:
    """Bounded time encoding measured from the jump-command rising edge."""
    command = env.command_manager.get_term(command_name)
    scaled_time = command.time_since_trigger / time_scale
    encoded = scaled_time**3 / (1.0 + scaled_time**3)
    return encoded.unsqueeze(-1)


def root_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """World-frame root height for the privileged critic."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def root_roll_angle(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root roll angle in radians for the privileged critic."""
    asset = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return roll.unsqueeze(-1)


def root_pitch_angle(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root pitch angle in radians for the privileged critic."""
    asset = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return pitch.unsqueeze(-1)


def maximum_jump_height(env, command_name: str = "jump") -> torch.Tensor:
    """Maximum root-height increase since the current jump command."""
    command = env.command_manager.get_term(command_name)
    return command.max_height.unsqueeze(-1)


def accumulated_root_pitch(env, command_name: str = "jump") -> torch.Tensor:
    """Accumulated pitch rotation in radians since the command trigger."""
    command = env.command_manager.get_term(command_name)
    return command.accumulated_pitch.unsqueeze(-1)


def accumulated_root_roll(env, command_name: str = "jump") -> torch.Tensor:
    """Accumulated roll rotation in radians since the command trigger."""
    command = env.command_manager.get_term(command_name)
    return command.accumulated_roll.unsqueeze(-1)
