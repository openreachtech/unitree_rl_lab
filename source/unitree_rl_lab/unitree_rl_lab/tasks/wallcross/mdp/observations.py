"""Observations for the hind-stance wall crossing task: a height map of the wall ahead."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["wall_heightmap", "HEIGHTMAP_X", "HEIGHTMAP_Y", "HEIGHTMAP_DIM", "constant_columns"]

HEIGHTMAP_X = torch.arange(-0.20, 1.30 + 1e-6, 0.05)   # 31 cells along the heading
HEIGHTMAP_Y = torch.arange(-0.20, 0.20 + 1e-6, 0.10)   # 5 cells across
HEIGHTMAP_DIM = len(HEIGHTMAP_X) * len(HEIGHTMAP_Y)    # 155
CELL_HALF = 0.025


def heading_xy(quat_w: torch.Tensor) -> torch.Tensor:
    """Horizontal heading that stays defined in the hind stance.

    ``yaw_quat`` projects the body x axis, which points up in the stance (pitch ~85 deg) and flips at
    90 deg. ``x_b - z_b`` projects to the forward direction both on four legs (x forward, z up) and
    nose up (x up, z backward)."""
    w, x, y, z = quat_w.unbind(-1)
    bx = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), dim=-1)
    bz = torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x)), dim=-1)
    f = bx - bz
    return f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def wall_heightmap(env: ManagerBasedRLEnv, command_name: str = "wallcross", asset_name: str = "robot") -> torch.Tensor:
    """Height of the ground under a 31 x 5 grid (5 cm x 10 cm cells) ahead of the robot, in metres.

    Computed from the wall table rather than ray cast: the 0.1 m height scanner of the critic can fall
    between a 3 cm wall's faces, and the hind stance breaks its ``yaw`` alignment. A cell reads the
    wall's height when the wall overlaps the cell along the heading (max over the cell, as an
    elevation map does), the floor (0) otherwise. Grid from 0.2 m behind to 1.3 m ahead of the base,
    +-0.2 m across, aligned with :func:`heading_xy`. Flattened x-major (31 rows of 5)."""
    wall = env.command_manager.get_term(command_name)
    robot = env.scene[asset_name]
    dev = robot.device
    f = heading_xy(robot.data.root_quat_w)                        # (N, 2)
    left = torch.stack((-f[:, 1], f[:, 0]), dim=-1)
    gx, gy = torch.meshgrid(HEIGHTMAP_X.to(dev), HEIGHTMAP_Y.to(dev), indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)                       # (M,)
    base = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    pts = base[:, None, :] + gx[None, :, None] * f[:, None, :] + gy[None, :, None] * left[:, None, :]
    cheb = pts.abs().amax(dim=-1)                                 # (N, M) tile-centred
    half, height, thick = wall.wall()
    hit = (cheb + CELL_HALF >= half[:, None]) & (cheb - CELL_HALF <= (half + thick)[:, None])
    return hit.float() * height[:, None]


def constant_columns(env, value: float, dim: int) -> torch.Tensor:
    """``dim`` columns of ``value`` (evaluation ablations: a blanked sensor)."""
    return torch.full((env.num_envs, dim), value, device=env.device)
