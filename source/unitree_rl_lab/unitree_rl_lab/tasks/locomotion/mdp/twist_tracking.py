"""MDP terms for whole-body end-effector twist tracking.

Observations and rewards of "Multi-critic Learning for Whole-body End-effector Twist Tracking"
(doc/papers/Multi-critic_Learning_for_Whole-body_End-effector_Twist_Tracking.md, 7.1). They work
with :class:`EeTwistCommand`, which holds the trajectory, the task frame and the current waypoint.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .commands.ee_twist_command import rot6d_from_quat

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands.ee_twist_command import EeTwistCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> EeTwistCommand:
    return env.command_manager.get_term(command_name)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def ee_goal_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Final pose of the trajectory in the task frame: position and 6D rotation. Shape (N, 9)."""
    command = _command(env, command_name)
    return torch.cat([command.goal_pos, rot6d_from_quat(command.goal_quat)], dim=-1)


def ee_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Current end-effector pose in the task frame: position and 6D rotation. Shape (N, 9)."""
    pos, quat = _command(env, command_name).ee_pose()
    return torch.cat([pos, rot6d_from_quat(quat)], dim=-1)


def ee_waypoint_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Current waypoint minus end-effector pose: position and rotation vector. Shape (N, 6)."""
    return torch.cat(_command(env, command_name).waypoint_error(), dim=-1)


# ---------------------------------------------------------------------------
# Rewards (Table 6, Mani)
# ---------------------------------------------------------------------------
def ee_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std_sq: float) -> torch.Tensor:
    """exp(-|r_EE,t - (r_EE,t-1 + v_EE dt)|^2 / std_sq): the waypoint the last twist pointed at."""
    command = _command(env, command_name)
    pos_err, _ = command.waypoint_error()
    return _phi(pos_err, std_sq) * command.valid.float()


def ee_orientation_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std_sq: float) -> torch.Tensor:
    """exp(-|R_EE,t [-] (R_EE,t-1 [+] w_EE dt)|^2 / std_sq)."""
    command = _command(env, command_name)
    _, rot_err = command.waypoint_error()
    return _phi(rot_err, std_sq) * command.valid.float()


def _phi(v: torch.Tensor, std_sq: float) -> torch.Tensor:
    """Phi(v, sigma^2) = exp(-|v|^2 / sigma^2) over the last dim (Table 6)."""
    return torch.exp(-v.square().sum(dim=-1) / std_sq)


def action_rate_exp_term(env: ManagerBasedRLEnv, action_name: str, std_sq: float) -> torch.Tensor:
    """Phi(a_t - a_t-1, sigma^2) restricted to one action term."""
    sl = _action_slice(env, action_name)
    manager = env.action_manager
    return _phi(manager.action[:, sl] - manager.prev_action[:, sl], std_sq)


def joint_torques_exp(env: ManagerBasedRLEnv, std_sq: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phi(tau, sigma^2) over the selected joints."""
    asset = env.scene[asset_cfg.name]
    return _phi(asset.data.applied_torque[:, asset_cfg.joint_ids], std_sq)


def joint_vel_exp(env: ManagerBasedRLEnv, std_sq: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phi(qdot, sigma^2) over the selected joints."""
    asset = env.scene[asset_cfg.name]
    return _phi(asset.data.joint_vel[:, asset_cfg.joint_ids], std_sq)


def _action_slice(env: ManagerBasedRLEnv, action_name: str) -> slice:
    manager = env.action_manager
    start = 0
    for name, term in manager._terms.items():
        if name == action_name:
            return slice(start, start + term.action_dim)
        start += term.action_dim
    raise ValueError(f"Unknown action term '{action_name}'. Available: {list(manager._terms)}")


def action_rate_l2_term(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """``action_rate_l2`` restricted to one action term (e.g. the legs or the arm)."""
    sl = _action_slice(env, action_name)
    manager = env.action_manager
    return (manager.action[:, sl] - manager.prev_action[:, sl]).square().sum(dim=1)


# ---------------------------------------------------------------------------
# Contact schedule rewards (Table 6, CS), with :class:`GaitScheduleCommand`
# ---------------------------------------------------------------------------
def feet_contact_schedule(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    scan_sensor_names: list[str],
    foot_radius: float,
) -> torch.Tensor:
    """sum_f (1 - C_f) Phi(F_f, 1.0) Phi(h_f^ - h_f, 0.05) + C_f n(F_fz > 1) Phi(v_f,xy, 0.01).

    A foot scheduled in swing earns for no contact force and for its height above the ground
    matching the schedule; a foot scheduled in stance for touching the ground without slipping.
    ``sensor_cfg`` / ``asset_cfg`` name the feet in the schedule's order (``preserve_order``); the
    ground under each foot is the mean of its ring scan (``scan_sensor_names``, same order).
    """
    command = env.command_manager.get_term(command_name)
    stance = command.contact_schedule
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    ground = torch.stack(
        [env.scene.sensors[name].data.ray_hits_w[..., 2].nan_to_num(posinf=0.0, neginf=0.0).mean(-1) for name in scan_sensor_names],
        dim=-1,
    )
    height = foot_pos[..., 2] - ground - foot_radius
    swing = (1.0 - stance) * _phi(forces, 1.0) * torch.exp(-torch.square(command.command - height) / 0.05)
    planted = stance * (forces[..., 2] > 1.0).float() * _phi(foot_vel, 0.01)
    return (swing + planted).sum(-1)


def feet_air_time_sum(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum_f n_contact,f t_air,f: the air time of each foot, paid when it touches down."""
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    return (first_contact * sensor.data.last_air_time[:, sensor_cfg.body_ids]).sum(-1)

