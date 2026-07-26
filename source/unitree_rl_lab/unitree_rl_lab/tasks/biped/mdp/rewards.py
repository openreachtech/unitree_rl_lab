from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
