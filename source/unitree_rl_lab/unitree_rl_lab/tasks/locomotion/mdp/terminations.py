from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_excluding_top(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    vertical_margin: float = 20.0,
) -> torch.Tensor:
    """Like ``isaaclab.envs.mdp.illegal_contact``, but exempts contacts that read as
    resting/pushing down onto a flat top surface (upward-dominant reaction force) from
    termination, using a horizontal-vs-vertical directional split to tell "bumping a
    vertical face" apart from "rolling/resting contact".

    For a climbing task, resting body weight on the wall's top edge -- or a limb
    catching the ledge mid-climb -- is legitimate technique, not a crash, and shouldn't
    end the episode the same way slamming into the wall's front face should. Direction
    is read from the *latest* step in the contact history (the magnitude threshold still
    uses the historical max, same as ``illegal_contact``, to catch brief impact spikes);
    a contact whose current reaction force is upward-dominant is treated as "resting,"
    even if an earlier spike in the same history window looked more like an impact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, hist, B, 3)

    latest_forces = net_contact_forces[:, -1]  # (N, B, 3)
    horizontal = torch.norm(latest_forces[..., :2], dim=-1)
    vertical_up = latest_forces[..., 2].clamp(min=0.0)
    resting_on_top = vertical_up > (horizontal + vertical_margin)

    magnitude = torch.norm(net_contact_forces, dim=-1).max(dim=1)[0]  # (N, B)
    illegal = (magnitude > threshold) & (~resting_on_top)
    return torch.any(illegal, dim=1)
