from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import MODE_FRONT_BIPED, MODE_HIND_BIPED, MODE_QUAD
from .observations import com_cop_vector_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Bipedal posture.

``base_height_l2`` and ``joint_deviation_l1`` (used to keep the swing/tucked legs
near their default pose, mirroring the reference's per-leg ``*_hip/thigh/calf_motion``
penalties) are generic Isaac Lab velocity-mdp terms, reused as-is via
``unitree_rl_lab.tasks.locomotion.mdp`` -- see ``robots/go2/biped_env_cfg.py``.
``flat_orientation_l2`` (``sum(g_xy^2)``) is likewise reused, but with a *positive*
weight: the goal here is exactly the tilted-away-from-flat posture that term
normally penalizes.
"""


def gravity_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the gravity z-component in the body frame.

    ``g_z^2`` is 1 for a normal flat/quadruped stance (gravity along the body
    z-axis) and shrinks toward 0 as the body pitches toward vertical. Used with a
    *negative* weight so, combined with the positive-weighted ``flat_orientation_l2``
    reuse above, both terms push the same direction: away from flat, toward biped
    stance (mirrors the reference ``bipedal_dog``'s ``orientation``/``orientation_3``
    reward pair).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 2])


"""
CoM-CoP balance (TumblerNet, Xiao et al. 2025).
"""


def pendulum_angle_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tilt of the CoM-CoP vector from vertical (TumblerNet Eq. 10).

    Treats the CoM-CoP vector as an inverted pendulum: ``theta`` is the angle
    between the vector and *world* vertical (the paper: "the angle between the
    vector CoM-CoP and the vertical z-axis"). Uses the **world-frame**
    ``com_cop_vector_world`` -- not the body-frame ``com_cop_vector`` used for
    observations -- since the body pitches ~70-90 degrees in biped stance, so
    the body's own z-axis is not a valid stand-in for gravity-vertical here
    (matches the reference's ``_reward_inv_pendulum``, which independently
    recomputes CoM/CoP from unrotated world positions rather than reusing its
    body-frame ``self.com_cop``). Large ``theta`` means the CoM is falling away
    from directly above the support point.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    norm = torch.linalg.norm(c, dim=-1).clamp(min=1.0e-6)
    cos_theta = (c[:, 2] / norm).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    return torch.square(theta)


def pendulum_instability_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Proxy for the angular acceleration of the pendulum tilt angle (TumblerNet Eq. 11).

    ``sin^2(theta) / ||c||^2`` grows both as the CoM-CoP vector tilts away from
    vertical *and* as it shortens -- a short, tilted "pendulum" is closer to
    tipping over than a long one at the same angle. World frame; see
    ``pendulum_angle_penalty`` for why.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    norm_sq = torch.sum(torch.square(c), dim=-1).clamp(min=1.0e-4)
    cos_theta = (c[:, 2] / norm_sq.sqrt()).clamp(-1.0, 1.0)
    sin_sq = 1.0 - torch.square(cos_theta)
    return sin_sq / norm_sq


def handle_length_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the horizontal CoM-CoP offset (TumblerNet Eq. 12 "handle length").

    The larger the horizontal distance between the CoM's ground projection and
    the CoP, the longer the lever arm gravity has to tip the robot over. World
    frame (world-horizontal, i.e. the actual ground plane); see
    ``pendulum_angle_penalty`` for why body frame would be wrong here.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    return torch.linalg.norm(c[:, :2], dim=-1)


"""
Multimode (quad / hind-biped / front-biped) reward shaping.

These wrap the same underlying quantities as generic locomotion rewards
(orientation, base height, joint-default deviation) but read the current per-env
stance from the ``GaitModeCommand`` (see ``commands.py``) and blend between a
quadruped-appropriate and a biped-appropriate target/sign, since the two regimes
want opposite things from the same signal (e.g. flat vs. tilted orientation).

The CoM-CoP rewards above (``pendulum_angle_penalty`` etc.) need no such blending:
called with all four feet as the candidate stance set, the force-weighted CoP
naturally collapses onto whichever 1-2 feet are actually bearing weight, so the
same reward call is already mode-agnostic.
"""


def _mode_ids(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).mode_ids


def mode_orientation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    quad_weight: float,
    biped_weight: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Flat-orientation reward for quad mode, tilt-encouraging reward for biped modes.

    Both modes score the same underlying quantity (``sum(g_xy^2)``: 0 when
    upright/flat, growing as the body tilts) but want opposite signs: quad wants it
    near 0 (flat), biped wants it large (pitched up onto two legs). Leave this
    term's own ``RewardTermCfg.weight`` at 1.0 -- the per-mode magnitude is already
    applied here since a single post-hoc weight can't flip sign per env.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    is_quad = _mode_ids(env, command_name) == MODE_QUAD
    return torch.where(is_quad, -quad_weight * g_xy_sq, biped_weight * g_xy_sq)


def mode_base_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    biped_target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height error from ``biped_target_height``, in biped modes only.

    Quad mode has no explicit height target here, matching the reference
    quadruped task (``Unitree-Go2-Velocity-v0``), which relies on natural gait +
    joint limits to hold a sensible standing height rather than an explicit height
    reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    is_quad = _mode_ids(env, command_name) == MODE_QUAD
    err2 = torch.square(asset.data.root_pos_w[:, 2] - biped_target_height)
    return torch.where(is_quad, torch.zeros_like(err2), err2)


def _cached_joint_ids(env: ManagerBasedRLEnv, asset: Articulation, front_joint_names, hind_joint_names):
    cache_attr = "_multimode_leg_joint_id_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache = getattr(env, cache_attr)
    key = (tuple(front_joint_names), tuple(hind_joint_names))
    if key not in cache:
        front_ids, _ = asset.find_joints(list(front_joint_names))
        hind_ids, _ = asset.find_joints(list(hind_joint_names))
        cache[key] = (front_ids, hind_ids)
    return cache[key]


def lifted_leg_joint_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    front_joint_names: list[str],
    hind_joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default pose of whichever leg pair is currently
    "lifted" (front pair in hind-biped mode, hind pair in front-biped mode); zero
    in quad mode, where every leg is load-bearing and free to move naturally.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    front_ids, hind_ids = _cached_joint_ids(env, asset, front_joint_names, hind_joint_names)

    dev = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos)
    dev_front = torch.sum(dev[:, front_ids], dim=1)
    dev_hind = torch.sum(dev[:, hind_ids], dim=1)

    mode_ids = _mode_ids(env, command_name)
    zeros = torch.zeros_like(dev_front)
    return torch.where(mode_ids == MODE_HIND_BIPED, dev_front, torch.where(mode_ids == MODE_FRONT_BIPED, dev_hind, zeros))


def _cached_body_ids(env: ManagerBasedRLEnv, asset: Articulation, front_body_names, hind_body_names):
    cache_attr = "_multimode_leg_body_id_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache = getattr(env, cache_attr)
    key = (tuple(front_body_names), tuple(hind_body_names))
    if key not in cache:
        front_ids, _ = asset.find_bodies(list(front_body_names))
        hind_ids, _ = asset.find_bodies(list(hind_body_names))
        cache[key] = (front_ids, hind_ids)
    return cache[key]


def lifted_leg_contact_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    front_foot_names: list[str],
    hind_foot_names: list[str],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize ground-contact force on whichever foot pair is currently "lifted".

    Fills the TumblerNet paper's r4 front/hind-foot contact-force term (Xiao et al.
    2025): the paper penalizes the swing pair's contact force directly to push it
    off the ground during the quad->biped transition. This generalizes that to
    whichever pair the current mode designates as swing, and is zero in quad mode
    (all four feet are legitimately load-bearing there).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    front_ids, hind_ids = _cached_body_ids(env, asset, front_foot_names, hind_foot_names)

    force_z = contact_sensor.data.net_forces_w[:, :, 2].clamp(min=0.0)
    force_front = torch.sum(force_z[:, front_ids], dim=1)
    force_hind = torch.sum(force_z[:, hind_ids], dim=1)

    mode_ids = _mode_ids(env, command_name)
    zeros = torch.zeros_like(force_front)
    return torch.where(mode_ids == MODE_HIND_BIPED, force_front, torch.where(mode_ids == MODE_FRONT_BIPED, force_hind, zeros))


def mode_gated_joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_command_name: str,
    stand_still_scale: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``joint_position_penalty`` (pull toward default pose when not commanded to
    move), active in quad mode only.

    In biped modes the "default pose" *is* the quadruped stance, so pulling toward
    it at zero velocity command would fight the entire point of holding a biped
    stance -- the CoM-CoP/tilt rewards already shape what "quiet standing" should
    look like there.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(vel_command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )

    is_quad = _mode_ids(env, command_name) == MODE_QUAD
    return torch.where(is_quad, reward, torch.zeros_like(reward))
