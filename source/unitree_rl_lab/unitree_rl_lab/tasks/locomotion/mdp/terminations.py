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
    resting on a surface -- an upward-dominant reaction force -- from termination, using
    a horizontal-vs-vertical split to tell "bumping a vertical face" apart from "putting
    weight on something".

    Ported from the Go2W Phase 5 campaign. Measured need on this lineage: crossing a
    30 cm wall, 86 % of robots put the trunk on the wall's top edge, and the force at
    that moment has a *median horizontal component of 0.0 N* -- it is pure vertical
    support, not a collision. The stock rule fires at 1 N on magnitude alone, so it was
    ending those episodes; disabling it raised the measured crossing rate from 87.9 % to
    99.6 % on a solid wall and 60.9 % to 78.5 % on a floating one. Exempting the vertical
    case recovers that without also excusing a head-on hit, whose horizontal component
    stays large (the 90th percentile of horizontal force at first touch is 60-160 N).

    Direction is read from the *latest* step in the contact history; the magnitude test
    still uses the historical maximum, as ``illegal_contact`` does, so a brief impact
    spike is not missed. A contact whose current reaction force is upward-dominant counts
    as resting even if an earlier spike in the same window looked like an impact.

    The Go2W campaign also has a looser variant that additionally exempts *pressing* into
    a face (its climb technique leans the chest on the wall at 60-80 deg of pitch). That
    is not ported: this robot steps over a wall rather than climbing up it, so a
    horizontal-dominant trunk force here is still a failure.
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
