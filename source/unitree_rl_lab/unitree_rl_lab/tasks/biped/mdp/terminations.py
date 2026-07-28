"""Mode-aware terminations for the multimode (quad / hind-biped / front-biped) task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import MODE_QUAD

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hip_contact_biped_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """``illegal_contact`` on the hips, active only in the two biped modes.

    In quad mode the hips sit naturally close to the ground (normal quadruped
    geometry, ~0.32m stance height) and can graze it during ordinary exploration
    without indicating a real fall -- the reference quadruped task
    (``Unitree-Go2-Velocity-v0``) only hard-terminates on base/trunk contact and
    treats hip contact as a soft reward penalty instead (see the
    ``undesired_contacts`` reward, which includes ``.*_hip``). In biped mode the
    body is deliberately much higher (0.55m target), so hip contact there is an
    unambiguous collapse signal and should still hard-terminate, same as the
    validated single-stance biped tasks.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    hip_down = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    mode_ids = env.command_manager.get_term(command_name).mode_ids
    return hip_down & (mode_ids != MODE_QUAD)


def bad_orientation_quad_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``bad_orientation`` (hard tilt-limit termination), active only in quad mode.

    Biped modes need to pitch ~70-90 degrees by design, so the plain tilt-limit
    termination used by the reference quadruped task would immediately end any
    biped-mode episode; the biped-specific ``collapsed``/``base_contact``
    terminations already catch genuine biped falls.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    angle = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs()
    mode_ids = env.command_manager.get_term(command_name).mode_ids
    return (angle > limit_angle) & (mode_ids == MODE_QUAD)
