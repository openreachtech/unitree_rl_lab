"""Elevation map from a LiDAR fan, kept the way elevation_mapping_cupy keeps it.

``OdomElevationMap`` bins each step's returns into an odom-aligned grid that scrolls with the
robot by whole cells and holds absolute terrain heights: the highest return per cell replaces
the held value, and a cell keeps its value until a new return lands on it. On the G1 the head
MID-360 never sees the ground within ~0.95 m, so everything under the feet comes from memory.

Noise (from feat/mid360, fitted to a MID-360 rosbag), drawn as one of three conditions per
episode (``LidarNoiseCfg``):

* per ray, before binning: distance error along the ray, outliers, and a tilt of the returns
  (per step: the IMU's gravity error; per episode: the mount off axis);
* odometry: the robot's pose in odom is integrated from the true per-step motion plus an error
  proportional to the distance travelled (per step and a per-episode bias). Returns are placed
  with that estimated pose and the map is read back with it, so drift shows up as the map
  sliding relative to the ground, as on hardware. Heights stay absolute (no vertical drift).
"""

from __future__ import annotations

__all__ = ["LidarNoiseCfg", "LidarNoiseConditionCfg", "OdomElevationMap"]

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_angle_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class LidarNoiseConditionCfg:
    """One noise condition (metres and degrees, standard deviations unless noted)."""

    probability: float = 0.0
    """Relative chance of an episode drawing this condition."""
    range_std: float = 0.0
    """Per-ray distance error along the ray, resampled every step."""
    tilt_step_std: float = 0.0
    """Per-step tilt of the returns about the sensor origin (deg)."""
    tilt_episode_std: float = 0.0
    """Per-episode tilt (deg): the mount slightly off axis."""
    outlier_prob: float = 0.0
    """Per-ray, per-step chance of a grossly wrong distance."""
    outlier_range: float = 0.0
    """Magnitude of an outlier's distance error (m), uniform in +-this."""
    odom_xy_step_std: float = 0.0
    """Per-step odometry translation error, as a fraction of the distance moved that step."""
    odom_xy_bias_std: float = 0.0
    """Per-episode odometry translation bias, same units (does not average out: drift)."""
    odom_yaw_step_std: float = 0.0
    """Per-step odometry heading error (deg per metre travelled)."""
    odom_yaw_bias_std: float = 0.0
    """Per-episode odometry heading bias (deg per metre travelled)."""


@configclass
class LidarNoiseCfg:
    """Per-episode noise conditions, plus one knob to scale all of them."""

    weak: LidarNoiseConditionCfg = LidarNoiseConditionCfg()
    nominal: LidarNoiseConditionCfg = LidarNoiseConditionCfg()
    strong: LidarNoiseConditionCfg = LidarNoiseConditionCfg()
    scale: float = 1.0
    """Multiplies every magnitude; 0.0 disables the noise."""


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw angle of a (w, x, y, z) quaternion. Shape (..., 4) -> (...,)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class OdomElevationMap:
    """Odom-aligned scrolling height map from a LiDAR's ``ray_hits_w``.

    Call ``update(env)`` once per step, read with ``sample(points_b)``: ``root_z - terrain_z -
    offset`` at points in the yaw-aligned base frame (nearest cell). Never-measured cells read as
    flat ground at the lower foot's sole (``feet_cfg``).
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        feet_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        offset: float = 0.0,
        resolution: float = 0.05,
        size: float = 2.8,
        sole_height: float = 0.0,
        noise: LidarNoiseCfg | None = None,
        min_range: float = 0.0,
        debug_vis: bool = False,
    ):
        self.num_envs, self.device = env.num_envs, env.device
        self._sensor_name, self._asset_name = sensor_cfg.name, asset_cfg.name
        self._feet_ids = feet_cfg.body_ids
        self._offset, self._res, self._sole = offset, resolution, sole_height
        self._min_range, self._debug_vis = min_range, debug_vis
        self._m = round(size / resolution)  # square, cells per side
        n, m = self.num_envs, self._m
        self._z = torch.zeros(n, m, m, device=self.device)
        self._has = torch.zeros(n, m, m, dtype=torch.bool, device=self.device)
        self._new = torch.zeros_like(self._has)
        self._origin = torch.zeros(n, 2, dtype=torch.long, device=self.device)  # odom cell of grid [0, 0]
        self._est_xy = torch.zeros(n, 2, device=self.device)
        self._est_yaw = torch.zeros(n, device=self.device)
        self._true_xy = torch.zeros(n, 2, device=self.device)
        self._true_yaw = torch.zeros(n, device=self.device)
        self._root_z = torch.zeros(n, device=self.device)
        self._fill = torch.zeros(n, 1, device=self.device)
        self._pose_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ar = torch.arange(m, device=self.device)

        self._noise = noise
        if noise is not None:
            conds = [noise.weak, noise.nominal, noise.strong]

            def col(attr: str) -> torch.Tensor:
                return torch.tensor([getattr(c, attr) for c in conds], device=self.device)

            w = col("probability")
            self._cond_weights = w / w.sum()
            self._range_std = col("range_std")
            self._tilt_step_std = torch.deg2rad(col("tilt_step_std"))
            self._tilt_episode_std = torch.deg2rad(col("tilt_episode_std"))
            self._outlier_prob = col("outlier_prob")
            self._outlier_range = col("outlier_range")
            self._odom_xy_step_std = col("odom_xy_step_std")
            self._odom_xy_bias_std = col("odom_xy_bias_std")
            self._odom_yaw_step_std = torch.deg2rad(col("odom_yaw_step_std"))
            self._odom_yaw_bias_std = torch.deg2rad(col("odom_yaw_bias_std"))
            self._condition = torch.zeros(n, dtype=torch.long, device=self.device)
            self._episode_tilt = torch.zeros(n, 2, device=self.device)
            self._odom_bias_xy = torch.zeros(n, 2, device=self.device)
            self._odom_bias_yaw = torch.zeros(n, device=self.device)
            self._draw_conditions(torch.arange(n, device=self.device))

    # ---------------------------------------------------------------- episode
    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._has[ids] = False
        self._pose_valid[ids] = False  # the reset teleports the robot: restart odom at the truth
        if self._noise is not None:
            self._draw_conditions(ids)

    def _draw_conditions(self, ids: torch.Tensor) -> None:
        """Noise condition, mount tilt and odometry bias for each new episode."""
        if ids.numel() == 0:
            return
        k = ids.numel()
        drawn = torch.multinomial(self._cond_weights, k, replacement=True)
        self._condition[ids] = drawn
        self._episode_tilt[ids] = torch.randn(k, 2, device=self.device) * self._tilt_episode_std[drawn].unsqueeze(-1)
        self._odom_bias_xy[ids] = torch.randn(k, 2, device=self.device) * self._odom_xy_bias_std[drawn].unsqueeze(-1)
        self._odom_bias_yaw[ids] = torch.randn(k, device=self.device) * self._odom_yaw_bias_std[drawn]

    # ---------------------------------------------------------------- noise
    def _perturb(self, rel: torch.Tensor, finite: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance error / outliers along each ray, then a tilt about the sensor origin.
        ``rel``: hits relative to the sensor origin. A return pushed under ``min_range`` is a
        dropout, not a point at the sensor."""
        cond = self._condition
        distance = rel.norm(dim=-1, keepdim=True)
        distance = torch.where(finite.unsqueeze(-1), distance, torch.zeros_like(distance))
        direction = rel / distance.clamp(min=1.0e-6)
        error = torch.randn_like(distance) * (scale * self._range_std[cond]).view(-1, 1, 1)
        is_outlier = torch.rand_like(distance) < self._outlier_prob[cond].view(-1, 1, 1)
        outlier = (torch.rand_like(distance) * 2.0 - 1.0) * (scale * self._outlier_range[cond]).view(-1, 1, 1)
        new_distance = distance + torch.where(is_outlier, outlier, error)
        if self._min_range > 0.0:
            finite = finite & (new_distance.squeeze(-1) >= self._min_range)
        rel = direction * new_distance.clamp(min=0.0)

        step_tilt = torch.randn(self.num_envs, 2, device=self.device) * self._tilt_step_std[cond].unsqueeze(-1)
        tilt = scale * (step_tilt + self._episode_tilt)
        angle = torch.linalg.vector_norm(tilt, dim=-1)
        axis = torch.nn.functional.normalize(
            torch.stack([tilt[:, 0], tilt[:, 1], torch.zeros_like(angle)], dim=-1), dim=-1, eps=1.0e-9
        )
        rot = matrix_from_quat(quat_from_angle_axis(angle, axis))
        return torch.einsum("nij,nrj->nri", rot, rel), finite

    def _corrupt_odometry(
        self, step_xy: torch.Tensor, step_yaw: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's ego-motion as an estimator would report it: error proportional to the
        distance travelled (a standing robot does not drift), in the robot's own frame."""
        travelled = torch.linalg.vector_norm(step_xy, dim=-1, keepdim=True)
        cond = self._condition
        xy_err = self._odom_bias_xy + torch.randn_like(step_xy) * (scale * self._odom_xy_step_std[cond]).unsqueeze(-1)
        yaw_err = self._odom_bias_yaw + torch.randn_like(step_yaw) * (scale * self._odom_yaw_step_std[cond])
        return step_xy + travelled * xy_err * scale, step_yaw + travelled.squeeze(-1) * yaw_err * scale

    # ---------------------------------------------------------------- pose
    def _update_pose(self, root_xy: torch.Tensor, yaw: torch.Tensor, scale: float) -> None:
        """Integrate the estimated odom pose (a fresh episode starts with odom = truth), then
        scroll the grid by whole cells to keep the estimated pose at its centre."""
        fresh = ~self._pose_valid
        c, s = torch.cos(self._true_yaw), torch.sin(self._true_yaw)
        d = root_xy - self._true_xy
        step_xy = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)
        step_yaw = torch.atan2(torch.sin(yaw - self._true_yaw), torch.cos(yaw - self._true_yaw))
        if self._noise is not None and scale > 0.0:
            step_xy, step_yaw = self._corrupt_odometry(step_xy, step_yaw, scale)
        ce, se = torch.cos(self._est_yaw), torch.sin(self._est_yaw)
        est_xy = self._est_xy + torch.stack(
            [ce * step_xy[:, 0] - se * step_xy[:, 1], se * step_xy[:, 0] + ce * step_xy[:, 1]], dim=-1
        )
        self._est_xy = torch.where(fresh.unsqueeze(-1), root_xy, est_xy)
        self._est_yaw = torch.where(fresh, yaw, self._est_yaw + step_yaw)
        self._true_xy, self._true_yaw = root_xy.clone(), yaw.clone()
        self._pose_valid[:] = True

        origin = torch.round(self._est_xy / self._res).long() - self._m // 2
        shift = torch.where(fresh.unsqueeze(-1), torch.zeros_like(origin), origin - self._origin)
        self._has &= ~fresh.view(-1, 1, 1)
        if bool((shift != 0).any()):
            ix = self._ar.view(1, -1) + shift[:, :1]  # (N, M): old row feeding new row
            iy = self._ar.view(1, -1) + shift[:, 1:]
            okx, oky = (ix >= 0) & (ix < self._m), (iy >= 0) & (iy < self._m)
            flat = ix.clamp(0, self._m - 1).unsqueeze(-1) * self._m + iy.clamp(0, self._m - 1).unsqueeze(1)
            self._z = torch.gather(self._z.flatten(1), 1, flat.flatten(1)).view_as(self._z)
            has = torch.gather(self._has.flatten(1), 1, flat.flatten(1)).view_as(self._has)
            self._has = has & okx.unsqueeze(-1) & oky.unsqueeze(1)
        self._origin = origin

    def _body_to_odom(self, xy_b: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(self._est_yaw).unsqueeze(-1), torch.sin(self._est_yaw).unsqueeze(-1)
        return self._est_xy.unsqueeze(1) + torch.stack(
            [c * xy_b[..., 0] - s * xy_b[..., 1], s * xy_b[..., 0] + c * xy_b[..., 1]], dim=-1
        )

    def _world_to_odom(self, xy_w: torch.Tensor) -> torch.Tensor:
        """World xy (N, P, 2) -> odom xy, through the true body pose and the estimated odom pose."""
        rel = xy_w - self._true_xy.unsqueeze(1)
        c, s = torch.cos(self._true_yaw).unsqueeze(-1), torch.sin(self._true_yaw).unsqueeze(-1)
        return self._body_to_odom(torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1))

    # ---------------------------------------------------------------- io
    def update(self, env: ManagerBasedRLEnv) -> None:
        sensor = env.scene.sensors[self._sensor_name]
        asset = env.scene[self._asset_name]
        hits_w = sensor.data.ray_hits_w
        root_pos = asset.data.root_pos_w
        finite = torch.isfinite(hits_w).all(dim=-1)
        self._root_z = root_pos[:, 2].clone()

        scale = self._noise.scale if self._noise is not None else 0.0
        if scale > 0.0:
            origin = sensor.data.pos_w + quat_apply(sensor.data.quat_w, sensor.ray_starts[:, 0])
            perturbed, finite = self._perturb(hits_w - origin.unsqueeze(1), finite, scale)
            hits_w = origin.unsqueeze(1) + perturbed

        sole_z = asset.data.body_pos_w[:, self._feet_ids, 2].min(dim=1).values - self._sole
        self._fill = (root_pos[:, 2] - sole_z - self._offset).unsqueeze(-1)

        self._update_pose(root_pos[:, :2], _yaw_from_quat(asset.data.root_quat_w), scale)

        # highest return per cell replaces the held value
        hits = torch.where(finite.unsqueeze(-1), hits_w, torch.zeros_like(hits_w))
        cell = torch.round(self._world_to_odom(hits[..., :2]) / self._res).long() - self._origin.unsqueeze(1)
        valid = finite & ((cell >= 0) & (cell < self._m)).all(dim=-1)
        flat = torch.where(valid, cell[..., 0] * self._m + cell[..., 1], torch.zeros_like(cell[..., 0]))
        zmax = torch.full((self.num_envs, self._m * self._m), -torch.inf, device=self.device)
        zmax.scatter_reduce_(1, flat, torch.where(valid, hits[..., 2], -torch.inf), reduce="amax", include_self=True)
        self._new = torch.isfinite(zmax).view_as(self._z)
        self._z = torch.where(self._new, zmax.view_as(self._z), self._z)
        self._has |= self._new

        if self._debug_vis:
            self._visualize()

    def sample(self, points_b: torch.Tensor) -> torch.Tensor:
        """``root_z - terrain_z - offset`` at yaw-aligned base-frame points (N, P, 2), nearest cell;
        never-measured or out-of-window cells read the fill."""
        cell = torch.round(self._body_to_odom(points_b) / self._res).long() - self._origin.unsqueeze(1)
        inside = ((cell >= 0) & (cell < self._m)).all(dim=-1)
        flat = cell.clamp(0, self._m - 1)
        flat = flat[..., 0] * self._m + flat[..., 1]
        z = torch.gather(self._z.flatten(1), 1, flat)
        has = torch.gather(self._has.flatten(1), 1, flat) & inside
        h = self._root_z.unsqueeze(-1) - z - self._offset
        return torch.where(has, h, self._fill.expand_as(h))

    def _visualize(self, env_index: int = 0) -> None:
        """Green = returned this step, red = held; drawn where the map puts them (estimated pose)."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if not hasattr(self, "_visualizer"):
            sphere = lambda rgb: sim_utils.SphereCfg(  # noqa: E731
                radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
            )
            self._visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/OdomElevationMap",
                    markers={"measured": sphere((0.0, 1.0, 0.0)), "held": sphere((1.0, 0.0, 0.0))},
                )
            )
        e = env_index
        has = self._has[e].flatten()
        if not bool(has.any()):
            return
        g = torch.stack(torch.meshgrid(self._ar, self._ar, indexing="ij"), dim=-1).view(-1, 2)
        odom = (g[has] + self._origin[e]).float() * self._res
        # odom -> world through the estimated -> true pose (where the map is wrong, so is this)
        rel = odom - self._est_xy[e]
        d = self._true_yaw[e] - self._est_yaw[e]
        c, s = torch.cos(d), torch.sin(d)
        w = self._true_xy[e] + torch.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1)
        pts = torch.cat([w, self._z[e].flatten()[has].unsqueeze(-1)], -1)
        self._visualizer.visualize(translations=pts, marker_indices=(~self._new[e].flatten()[has]).long())
