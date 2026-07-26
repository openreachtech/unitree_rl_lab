from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # pragma: no cover - depends on installed isaaclab version
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def com_cop_vector(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body-frame vector from the center-of-mass (CoM) to the center-of-pressure (CoP).

    Ground truth used both as the biped estimator's supervised training target
    (see ``BipedPPO``) and as the input to the CoM-CoP balance rewards. Matches
    "TumblerNet" (Xiao et al. 2025) / the reference ``bipedal_dog.py`` construction:

    - CoP: force-weighted mean of the *stance* feet positions (``asset_cfg`` /
      ``sensor_cfg`` must list only the stance-leg feet, e.g. the hind feet for a
      hind-leg biped stance), weighted by each foot's vertical contact force.
    - CoM: mass-weighted mean of every rigid body's position.

    Both are expressed in the base body frame before differencing, so the result is
    independent of the robot's world pose/heading.

    :param asset_cfg: robot articulation; ``body_ids`` must select the stance feet.
    :param sensor_cfg: contact sensor; ``body_ids`` must select the same stance feet,
        in the same order as ``asset_cfg``.
    :return: [(N, 3)] com - cop, in the base body frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Per-body mass is fixed after startup domain randomization (see e.g.
    # RewardsCfg/EventCfg add_base_mass); cache once instead of querying physx
    # every call.
    if not hasattr(env, "_biped_body_masses"):
        env._biped_body_masses = asset.root_physx_view.get_masses().to(env.device)  # (N, B)
    masses = env._biped_body_masses

    total_mass = torch.sum(masses, dim=1, keepdim=True)  # (N, 1)
    com_w = torch.sum(asset.data.body_pos_w * masses.unsqueeze(-1), dim=1) / total_mass  # (N, 3)
    com_b = quat_apply_inverse(asset.data.root_quat_w, com_w - asset.data.root_pos_w)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, F, 3)
    foot_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1).reshape(-1, 4),
        (foot_pos_w - asset.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(foot_pos_w.shape)

    # Vertical contact force per stance foot; a small epsilon avoids a 0/0 CoP when
    # every stance foot is briefly airborne (e.g. right after a stumble).
    force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0) + 1.0e-6  # (N, F)
    cop_b = torch.sum(foot_pos_b * force_z.unsqueeze(-1), dim=1) / torch.sum(force_z, dim=1, keepdim=True)

    return com_b - cop_b


def com_cop_vector_world(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame vector from the center-of-mass (CoM) to the center-of-pressure (CoP).

    Used by the CoM-CoP balance *rewards* (``pendulum_angle_penalty`` etc.), as
    opposed to ``com_cop_vector`` (body frame), which is used for the
    observation / estimator-training-target. This split matches the reference
    ``bipedal_dog.py`` exactly: its ``self.com_cop`` (fed into
    ``privileged_obs_buf``, i.e. our observation/estimator target) is computed
    in the body frame, but its actual active pendulum-angle reward,
    ``_reward_inv_pendulum`` in ``legged_robot.py``, independently recomputes
    CoM/CoP from raw (unrotated) ``rigid_body_state`` -- i.e. in the world
    frame -- rather than reusing ``self.com_cop``.

    This distinction is not cosmetic: the paper defines the pendulum angle
    theta as the angle between the CoM-CoP vector and *the vertical z-axis*,
    i.e. the world gravity axis. Once the robot pitches ~70-90 degrees into
    biped stance, the body's own z-axis is nearly horizontal in world space,
    so using the body-frame vector there would measure alignment with the
    body's spine axis instead of true balance against gravity -- exactly
    backwards for a stability reward.

    Root translation cancels out of the CoM - CoP difference, so unlike
    ``com_cop_vector`` this needs no translation or rotation relative to the
    root at all -- it's the difference of two raw world positions.

    :param asset_cfg: robot articulation; ``body_ids`` must select the stance feet.
    :param sensor_cfg: contact sensor; ``body_ids`` must select the same stance feet,
        in the same order as ``asset_cfg``.
    :return: [(N, 3)] com - cop, in the world frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if not hasattr(env, "_biped_body_masses"):
        env._biped_body_masses = asset.root_physx_view.get_masses().to(env.device)  # (N, B)
    masses = env._biped_body_masses

    total_mass = torch.sum(masses, dim=1, keepdim=True)  # (N, 1)
    com_w = torch.sum(asset.data.body_pos_w * masses.unsqueeze(-1), dim=1) / total_mass  # (N, 3)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, F, 3)
    force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0) + 1.0e-6  # (N, F)
    cop_w = torch.sum(foot_pos_w * force_z.unsqueeze(-1), dim=1) / torch.sum(force_z, dim=1, keepdim=True)

    return com_w - cop_w
