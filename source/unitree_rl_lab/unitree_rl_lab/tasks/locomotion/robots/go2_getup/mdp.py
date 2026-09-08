"""Get-up-specific MDP terms for the Go2.

These are the pieces that don't exist in ``tasks.locomotion.mdp`` because the
locomotion task always resets standing and upright: a reset that drops the
robot into a random tumbled pose, and a couple of reward terms (target base
height, target joint pose) that don't depend on a velocity command.

Reset strategy follows iit-DLSLab/get-up-isaaclab (``getup_env.py:_reset_idx``):
sample a uniformly random full-SO(3) base orientation plus a wide uniform
offset on every joint (clamped to the robot's own limits), so the robot
starts each episode collapsed in some arbitrary tangle. A small fraction of
resets (``standing_prob``) is left at the nominal standing pose instead, so
the policy always keeps some "how do I stay standing" signal alongside
"how do I get up" -- without that, a freshly-initialized policy sees nothing
but falling poses and can take much longer to discover the target basin.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_fallen_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos_range: tuple[float, float] = (-3.14159, 3.14159),
    standing_prob: float = 0.1,
):
    """Reset to a random fallen pose (or, with ``standing_prob``, to standing)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # -- joints: default + wide uniform offset, clamped to the robot's own limits.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*joint_pos_range)
    joint_limits = asset.data.default_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)

    # -- root: random full-SO(3) orientation, lifted slightly so the tumbled
    # pose doesn't spawn in self/ground collision.
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 2] += 0.15
    root_state[:, 7:] = 0.0

    keep_standing = torch.rand(len(env_ids), device=env.device) < standing_prob
    rand_quat = math_utils.random_orientation(len(env_ids), device=env.device)
    root_state[:, 3:7] = torch.where(keep_standing.unsqueeze(-1), root_state[:, 3:7], rand_quat)
    joint_pos[keep_standing] = asset.data.default_joint_pos[env_ids][keep_standing]

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-((z - target) / std)^2)``, gated to only pay out once the base
    is roughly upright -- otherwise the reward can be farmed by pushing a
    sideways- or upside-down torso up to the target height without actually
    standing up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    upright = asset.data.projected_gravity_b[:, 2] < -0.5
    return torch.exp(-height_error / std**2) * upright


def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint-position distance from the nominal standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
