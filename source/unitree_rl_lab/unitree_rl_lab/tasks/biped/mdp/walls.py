"""Shared terms for the circular walls of :mod:`unitree_rl_lab.tasks.biped.terrains`.

Per-environment wall geometry (:class:`WallGeometry`), the terrain-column velocity command, the
on-the-wall exemption from ``illegal_contact`` and the Play spread over rows, used by the wall-jump
tasks (``walljump.py``) and ``scripts/rsl_rl/measure_wall_crossing.py``.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import terrains

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "TerrainTypeVelocityCommand",
    "TerrainTypeVelocityCommandCfg",
    "illegal_contact_off_wall",
    "spread_over_rows",
]


class WallGeometry:
    """Per-environment wall geometry, read from :data:`terrains.WALL_TABLE` and the tile origin."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        walls = next(c for c in generator.sub_terrains.values() if isinstance(c, terrains.MeshCircularWallTerrainCfg))
        self.radii = torch.tensor(terrains.ring_radii(walls, generator.size), device=env.device)
        self.table = torch.as_tensor(terrains.WALL_TABLE, dtype=torch.float, device=env.device)

    @property
    def height_thickness(self) -> tuple[torch.Tensor, torch.Tensor]:
        terrain = self.env.scene.terrain
        wall = self.table[terrain.terrain_levels, terrain.terrain_types]
        return wall[:, 0], wall[:, 1]

    def radial(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Horizontal distance from the tile centre, for ``(num_envs, ..., 3)`` world positions."""
        origin = self.env.scene.env_origins[:, :2]
        offset = pos_w[..., :2] - origin.view(-1, *([1] * (pos_w.dim() - 2)), 2)
        return torch.linalg.norm(offset, dim=-1)

    def in_band(self, radial: torch.Tensor, margin: float) -> torch.Tensor:
        """Whether ``radial`` lies within ``margin`` of any ring, inner face to outer face."""
        _, thickness = self.height_thickness
        half = 0.5 * thickness.view(-1, *([1] * (radial.dim() - 1)), 1)
        r = radial.unsqueeze(-1)
        return ((r > self.radii - half - margin) & (r < self.radii + half + margin)).any(dim=-1)


def _geometry(env: ManagerBasedRLEnv) -> WallGeometry:
    geometry = getattr(env, "_wall_geometry", None)
    if geometry is None:
        geometry = env._wall_geometry = WallGeometry(env)
    return geometry


class TerrainTypeVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` with its own ranges on given terrain columns.

    Anaguma Phase 5's: forward only at the walls, where a lateral or yaw command could be satisfied
    without going near one, and the full range on the rough column, so that driving sideways,
    backwards and turning are not forgotten. A robot's column is fixed for its lifetime (the
    curriculum moves rows only).
    """

    cfg: TerrainTypeVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        types = self._env.scene.terrain.terrain_types[env_ids]
        for column, ranges in self.cfg.column_ranges.items():
            ids = env_ids[types == int(column)]
            if len(ids) == 0:
                continue
            r = torch.empty(len(ids), device=self.device)
            self.vel_command_b[ids, 0] = r.uniform_(*ranges.lin_vel_x)
            self.vel_command_b[ids, 1] = r.uniform_(*ranges.lin_vel_y)
            self.vel_command_b[ids, 2] = r.uniform_(*ranges.ang_vel_z)


@configclass
class TerrainTypeVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TerrainTypeVelocityCommand

    column_ranges: dict[str, UniformVelocityCommandCfg.Ranges] = {}
    """Ranges per terrain column, keyed by the column index as a string (``to_dict`` drops int keys)."""


# -- terminations ------------------------------------------------------------------------------


class illegal_contact_off_wall(ManagerTermBase):
    """``illegal_contact``, except for contact on the wall.

    Contact counts as a fall unless the body is radially within ``margin`` of a ring (inner face to
    outer face) *and* higher than ``min_height_fraction`` of the wall: resting the trunk on the
    wall top is how the robot gets over, and the height condition keeps the exemption from
    covering a robot lying on the floor at the foot of the wall. The test is geometric because the
    force direction does not separate the two -- Anaguma's Phase 5 found the horizontal component
    of a trunk scraping forward over the edge put most such contacts on the wrong side of a
    direction test. Taken from ``feat/wallcross``.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        sensor = env.scene.sensors[sensor_cfg.name]
        names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        # Resolved once: the sensor's and the articulation's body orders differ.
        self.robot_ids, _ = env.scene["robot"].find_bodies(names, preserve_order=True)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        margin: float = 0.1,
        min_height_fraction: float = 0.5,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        touching = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold
        geometry = _geometry(env)
        height, _ = geometry.height_thickness
        pos = env.scene["robot"].data.body_pos_w[:, self.robot_ids]
        high = (pos[..., 2] - env.scene.env_origins[:, 2:3]) > (min_height_fraction * height).unsqueeze(-1)
        on_wall = geometry.in_band(geometry.radial(pos), margin) & high
        return (touching & ~on_wall).any(dim=-1)


def spread_over_rows(env: ManagerBasedRLEnv, env_ids) -> None:
    """Startup event for Play: the same number of robots on every row (wall height), instead of
    the importer's random draw."""
    terrain = env.scene.terrain
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    index = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[:] = index % num_rows
    terrain.terrain_types[:] = (index // num_rows) % num_cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
