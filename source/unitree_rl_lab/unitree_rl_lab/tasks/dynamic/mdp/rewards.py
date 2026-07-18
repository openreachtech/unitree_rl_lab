from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def upright_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt / std**2)


def standing_pose_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.exp(-error / std**2)


def stillness_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    linear_std: float = 0.25,
    angular_std: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    linear_error = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    angular_error = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return torch.exp(-linear_error / linear_std**2 - angular_error / angular_std**2)


def pre_jump_standing_reward(
    env,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return upright_reward(env, asset_cfg) * stillness_reward(env, asset_cfg) * (~command.enabled).float()


def jump_progress_reward(
    env,
    command_name: str = "jump",
    scale: float = 0.01,
) -> torch.Tensor:
    """EFGCL task progress based on maximum height reached."""
    command = env.command_manager.get_term(command_name)
    error = command.max_height - command.target_height
    attempted = command.trigger_step >= 0
    return torch.exp(-torch.square(error) / scale) * attempted.float()


def jump_progress_standing_reward(
    env,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_scale: float = 0.01,
    joint_scale: float = 0.25,
) -> torch.Tensor:
    """Product of jump progress and post-jump standing quality."""
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    task_progress = jump_progress_reward(env, command_name)
    height_term = torch.exp(-torch.square(command.height_delta) / height_scale)
    joint_error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    pose_term = torch.exp(-joint_error / joint_scale)
    return task_progress * (height_term + pose_term)


def motion_progress_reward(
    env,
    command_name: str = "jump",
    height_scale: float = 0.01,
    rotation_scale: float = math.pi**2,
) -> torch.Tensor:
    """Shared EFGCL progress reward for jump, backflip, and sideflip."""
    command = env.command_manager.get_term(command_name)
    jump_error = command.max_height - command.target_height
    pitch_error = command.accumulated_pitch - command.target_pitch_turns * (2.0 * math.pi)
    roll_error = command.accumulated_roll - command.target_roll_turns * (2.0 * math.pi)

    progress = torch.zeros(env.num_envs, device=env.device)
    progress = torch.where(
        command.motion_code == command.MOTION_JUMP,
        torch.exp(-torch.square(jump_error) / height_scale),
        progress,
    )
    progress = torch.where(
        command.motion_code == command.MOTION_BACKFLIP,
        torch.exp(-torch.square(pitch_error) / rotation_scale),
        progress,
    )
    progress = torch.where(
        command.motion_code == command.MOTION_SIDEFLIP,
        torch.exp(-torch.square(roll_error) / rotation_scale),
        progress,
    )
    return progress * (command.trigger_step >= 0).float()


def motion_progress_standing_reward(
    env,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_scale: float = 0.01,
    joint_scale: float = 0.25,
) -> torch.Tensor:
    """Motion progress multiplied by stable standing after landing."""
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    task_progress = motion_progress_reward(env, command_name)
    height_term = torch.exp(-torch.square(command.height_delta) / height_scale)
    joint_error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    pose_term = torch.exp(-joint_error / joint_scale)
    return task_progress * (height_term + pose_term)


def non_target_angular_velocity_penalty(
    env,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Suppress rotation outside the axis targeted by the commanded motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    angular_velocity = asset.data.root_ang_vel_b
    all_axes = torch.sum(torch.square(angular_velocity), dim=1)
    if command_name is None:
        return all_axes

    command = env.command_manager.get_term(command_name)
    attempted = command.trigger_step >= 0
    backflip_penalty = torch.square(angular_velocity[:, 0]) + torch.square(angular_velocity[:, 2])
    sideflip_penalty = torch.square(angular_velocity[:, 1]) + torch.square(angular_velocity[:, 2])
    penalty = torch.where(
        attempted & (command.motion_code == command.MOTION_BACKFLIP),
        backflip_penalty,
        all_axes,
    )
    return torch.where(
        attempted & (command.motion_code == command.MOTION_SIDEFLIP),
        sideflip_penalty,
        penalty,
    )
