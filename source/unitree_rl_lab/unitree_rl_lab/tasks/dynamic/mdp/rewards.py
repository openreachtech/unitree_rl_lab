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
    # Scope to ``asset_cfg.joint_ids`` so a wheeled robot (go2w) can exclude its
    # continuous wheel joints: their absolute position drifts without bound, so
    # including them in ``joint_pos - default_joint_pos`` makes this reward collapse
    # to ~0 regardless of the actual standing pose. ``joint_ids`` defaults to
    # ``slice(None)`` (all joints), so the legged go2 behaviour is unchanged.
    joint_ids = asset_cfg.joint_ids
    error = torch.sum(
        torch.square(
            asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
        ),
        dim=1,
    )
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


def pre_jump_pose_reward(
    env,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """Cost for holding a non-default joint pose whenever the jump command is idle.

    Without this, an anticipatory crouch is free to hold indefinitely before the
    command fires (and after landing), since ``pre_jump_standing_reward`` only checks
    upright/stillness, not joint angles. Gated the same way so it only applies while
    the command is not enabled.
    """
    command = env.command_manager.get_term(command_name)
    return standing_pose_reward(env, asset_cfg, std) * (~command.enabled).float()


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
    # See ``standing_pose_reward``: scope the pose error to ``asset_cfg.joint_ids`` so a
    # wheeled robot can drop its free-spinning wheel joints. Defaults to all joints.
    joint_ids = asset_cfg.joint_ids
    joint_error = torch.sum(
        torch.square(
            asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
        ),
        dim=1,
    )
    pose_term = torch.exp(-joint_error / joint_scale)
    return task_progress * (height_term + pose_term)


def base_lin_vel_xy_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base velocity (world frame).

    A wheeled robot (go2w) can roll on its free wheels, so any fore-aft component of the
    push-off turns into drift instead of being held by ground friction as a legged robot's
    planted feet would. Measured in the world frame so pitching mid-air does not mix the
    vertical take-off velocity into it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


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
