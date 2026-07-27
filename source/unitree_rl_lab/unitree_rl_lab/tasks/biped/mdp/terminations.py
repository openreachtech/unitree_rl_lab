"""Mode-aware terminations for the multimode (quad / hind-biped / front-biped) task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MODE_QUAD

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
