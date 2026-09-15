"""Height grid built from a LiDAR fan instead of a top-down raycast.

``height_scan_excluding_body`` fires 609 rays straight down from 20 m above the
robot, so it returns the true terrain everywhere -- including the far side of a
wall and the ground behind the robot, neither of which any real sensor can
measure. Feeding that to the policy trains it on information the hardware cannot
supply, and no amount of additive noise fixes it: the missing structure is
*which cells are measurable at all*, and that is decided by geometry.

This term builds the same grid the way the robot will: fire a static fan from the
LiDAR mount, bin the returns into the 5 cm cells, and take the highest return per
cell. Occlusion, the limited field of view and the density falloff with range then
fall out of the raycast rather than being modelled. Cells that receive no return
hold their previous value, which is the cheapest stand-in for the temporal
accumulation a real elevation map performs.

The output has the same layout, order and units as ``height_scan_excluding_body`` --
one value per kept grid cell -- so it drops into the policy observation group in place
of that term while the critic keeps the clean top-down scan as privileged input. The
cell count follows whatever exclusion rectangle the caller passes, and the LiDAR tasks
widen theirs, so the two are not interchangeable at a fixed width; see
``velocity_env_cfg_lidar.py``.

Measurement noise is applied per ray, before the returns are binned -- see
``LidarNoiseCfg``. That placement is what makes it faithful: perturbing distance along
the ray produces the lateral error a real sensor makes at shallow incidence, which
cannot be expressed by adding noise to a finished height grid.

``scripts/tools/check_lidar_map_coverage.py`` predicts the noise-free coverage
analytically; this implementation is checked against it.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_angle_axis

from .observations import _height_scan_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_UNOBSERVED = 1.0e4
"""Sentinel for a cell no beam reached. Large, so it loses every ``amin``."""


@configclass
class LidarNoiseConditionCfg:
    """One noise condition: how badly the sensor behaves for this episode.

    Standard deviations, in metres and degrees. Everything scales with
    ``LidarNoiseCfg.scale``, so tuning overall severity does not mean editing these.
    """

    probability: float = 0.0
    """Relative chance of an episode drawing this condition. Normalised across conditions."""
    range_std: float = 0.0
    """Per-ray distance error (m), resampled every step. The paper's *Position*, applied
    along the ray so the lateral component follows from the incidence angle."""
    tilt_step_std: float = 0.0
    """Per-step tilt of the returns about the sensor origin (deg). The paper's *Tilt*.
    Physically the IMU's gravity-direction error, which is what fixes "up" for the height
    values -- so it fluctuates with the estimate rather than being fixed per run."""
    tilt_episode_std: float = 0.0
    """Per-episode tilt held for the run (deg): the sensor bolted on slightly off axis.
    Not from the paper -- there is no mounting error in a simulated point cloud -- but it
    is the other half of a real tilt and costs nothing on top of the per-step term."""
    outlier_prob: float = 0.0
    """Per-ray, per-step chance of a grossly wrong distance. The paper's *Outliers*."""
    outlier_range: float = 0.0
    """Magnitude of an outlier's distance error (m), uniform in +-this."""


@configclass
class LidarNoiseCfg:
    """Per-episode noise conditions, plus one knob to scale all of them.

    The paper varies noise strength to control how hard the reconstruction task is; the
    same effect here comes from ``scale`` (dial everything at once) and the condition
    mix (how often an episode is easy or hard). Defaults are 60% weak / 30% nominal /
    10% strong.

    Two of the paper's augmentations are absent on purpose, because the fan already
    produces them from geometry rather than from a noise model:

    *Pruning*  -- occlusion behind obstacles, the field of view, and the fan's angular
                  sparsity already leave 35.8% of cells with no return on flat ground.
                  Adding stochastic dropout on top would only decorrelate the gaps from
                  the terrain, and would further bury the shadow it is meant to sit
                  alongside.
    *Height* / *Robot Pose* -- unobserved cells hold their previous value with no motion
                  compensation, so stale readings drift out of register as the robot
                  advances. That is the same class of error, arising from the temporal
                  fill rather than being injected.
    """

    weak: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.60, range_std=0.01, tilt_step_std=0.5, tilt_episode_std=0.25,
        outlier_prob=0.005, outlier_range=0.15,
    )
    nominal: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.30, range_std=0.02, tilt_step_std=1.0, tilt_episode_std=0.5,
        outlier_prob=0.01, outlier_range=0.30,
    )
    strong: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.10, range_std=0.04, tilt_step_std=2.0, tilt_episode_std=1.0,
        outlier_prob=0.03, outlier_range=0.60,
    )

    scale: float = 1.0
    """Multiplies every magnitude above. 0.0 disables noise entirely without touching
    the conditions, which is how to get a clean reference run."""

    num_steps_per_env: int = 24
    """Rollout length of one iteration, to turn env steps into iterations for the ramp."""
    start_iteration: int = 0
    """Iterations of noise-free returns before the ramp begins."""
    full_iteration: int = 0
    """Iteration at which ``scale`` is reached. Equal to ``start_iteration`` means no ramp."""


def _curriculum_level(common_step_counter: int, cfg: LidarNoiseCfg) -> float:
    """Fraction of the configured magnitude to apply at the current iteration."""
    if cfg.full_iteration <= cfg.start_iteration:
        return 1.0
    iteration = common_step_counter // max(cfg.num_steps_per_env, 1)
    if iteration >= cfg.full_iteration:
        return 1.0
    if iteration <= cfg.start_iteration:
        return 0.0
    return (iteration - cfg.start_iteration) / (cfg.full_iteration - cfg.start_iteration)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw angle of a (w, x, y, z) quaternion. Shape (..., 4) -> (...,)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LidarElevationMap(ManagerTermBase):
    """Bin a LiDAR fan's returns into the body-centered height grid.

    The heavy state is one buffer of held cell values; everything else is a
    stateless reduction over the fan's hits.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._resolution: float = p["resolution"]
        self._size: tuple[float, float] = p["size"]
        self._offset: float = p["offset"]
        self._flat_fill: float = p["flat_fill"]
        self._lidar_offset: tuple[float, float, float] = p["lidar_offset"]

        self._num_x = round(self._size[0] / self._resolution) + 1
        self._num_y = round(self._size[1] / self._resolution) + 1
        self._num_cells = self._num_x * self._num_y
        self._x0 = -self._size[0] / 2 + p["scanner_offset_xy"][0]
        self._y0 = -self._size[1] / 2 + p["scanner_offset_xy"][1]

        # ordering="yx" (idx = ix * num_y + iy), matching _height_scan_indices.
        self._keep_indices, _ = _height_scan_indices(
            self._resolution,
            self._size[0],
            self._size[1],
            p["scanner_offset_xy"][0],
            p["scanner_offset_xy"][1],
            p["exclude_half_extent_x"],
            p["exclude_half_extent_y"],
            self.device,
        )

        # Cell centers in the yaw-aligned base frame, for the diagnostics and markers.
        cx = torch.linspace(self._x0, self._x0 + self._size[0], self._num_x, device=self.device)
        cy = torch.linspace(self._y0, self._y0 + self._size[1], self._num_y, device=self.device)
        gx, gy = torch.meshgrid(cx, cy, indexing="ij")
        self._cell_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (num_cells, 2)

        kept_xy = self._cell_xy.index_select(0, self._keep_indices)
        # Cells outside the fan's azimuth wedge can never receive a beam, so counting
        # them as unobserved would report a field-of-view choice as a density problem.
        # With a full turn every cell qualifies and the mask is all-true.
        h_fov = p["horizontal_fov"]
        if h_fov[1] - h_fov[0] >= 359.9:
            self._in_fov = torch.ones(kept_xy.shape[0], dtype=torch.bool, device=self.device)
        else:
            azimuth = torch.rad2deg(
                torch.atan2(kept_xy[:, 1] - self._lidar_offset[1], kept_xy[:, 0] - self._lidar_offset[0])
            )
            self._in_fov = (azimuth >= h_fov[0]) & (azimuth <= h_fov[1])
        radius = torch.linalg.vector_norm(kept_xy - kept_xy.new_tensor(self._lidar_offset[:2]), dim=-1)
        self._band_near = self._in_fov & (radius < 0.30)
        self._band_mid = self._in_fov & (radius >= 0.30) & (radius < 0.50)
        self._band_far = self._in_fov & (radius >= 0.50)

        self._hold = torch.full((self.num_envs, self._num_cells), self._flat_fill, device=self.device)

        # Noise conditions, drawn per episode. Absent config means a noise-free sensor.
        self._noise: LidarNoiseCfg | None = p.get("noise")
        if self._noise is not None:
            conds = [self._noise.weak, self._noise.nominal, self._noise.strong]

            def _column(attr: str) -> torch.Tensor:
                return torch.tensor([getattr(c, attr) for c in conds], device=self.device)

            weights = _column("probability")
            if torch.any(weights < 0.0) or weights.sum() <= 0.0:
                raise ValueError("LiDAR noise condition probabilities must be >= 0 and sum to > 0.")
            self._cond_weights = weights / weights.sum()
            self._range_std = _column("range_std")
            self._tilt_step_std = torch.deg2rad(_column("tilt_step_std"))
            self._tilt_episode_std = torch.deg2rad(_column("tilt_episode_std"))
            self._outlier_prob = _column("outlier_prob")
            self._outlier_range = _column("outlier_range")
            self._condition = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_tilt = torch.zeros(self.num_envs, 2, device=self.device)
            self._draw_conditions(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._hold[ids] = self._flat_fill
        if self._noise is not None:
            self._draw_conditions(ids)

    def _draw_conditions(self, env_ids: torch.Tensor) -> None:
        """Pick a noise condition per episode, and the tilt this run is stuck with."""
        if env_ids.numel() == 0:
            return
        drawn = torch.multinomial(self._cond_weights, env_ids.numel(), replacement=True)
        self._condition[env_ids] = drawn
        # Drawn at full magnitude; the ramp scales it when the returns are perturbed.
        self._episode_tilt[env_ids] = (
            torch.randn(env_ids.numel(), 2, device=self.device) * self._tilt_episode_std[drawn].unsqueeze(-1)
        )

    def _perturb(self, rel: torch.Tensor, finite: torch.Tensor, level: float) -> torch.Tensor:
        """Apply the paper's Position, Outliers and Tilt to the returns, per ray.

        ``rel`` is each hit relative to the sensor origin, so distance and direction are
        separable here -- which is the whole point of doing this before binning. Noise on
        the distance moves the point along its own ray, so a shallow beam's 2 cm error
        lands 6 cm away across the ground, exactly as the sensor would. Adding the same
        error to a finished height grid could only ever move it vertically.
        """
        cond = self._condition
        distance = rel.norm(dim=-1, keepdim=True)
        # Misses come back as inf; park them at the origin so nothing propagates NaN, and
        # rely on ``finite`` to drop them afterwards.
        distance = torch.where(finite.unsqueeze(-1), distance, torch.zeros_like(distance))
        direction = rel / distance.clamp(min=1.0e-6)

        # Position: Gaussian along the ray.
        error = torch.randn_like(distance) * (level * self._range_std[cond]).view(-1, 1, 1)
        # Outliers: a few rays return a grossly wrong distance instead.
        outlier_mag = (level * self._outlier_range[cond]).view(-1, 1, 1)
        is_outlier = torch.rand_like(distance) < self._outlier_prob[cond].view(-1, 1, 1)
        outlier = (torch.rand_like(distance) * 2.0 - 1.0) * outlier_mag
        error = torch.where(is_outlier, outlier, error)
        rel = direction * (distance + error).clamp(min=0.0)

        # Tilt: rotate the returns about the sensor origin. Per-step stands in for the
        # IMU's gravity-direction error, which is what fixes "up" for the height values;
        # the per-episode part is the mount being slightly off axis.
        step_tilt = torch.randn(self.num_envs, 2, device=self.device) * self._tilt_step_std[cond].unsqueeze(-1)
        tilt = level * (step_tilt + self._episode_tilt)
        angle = torch.linalg.vector_norm(tilt, dim=-1)
        axis = torch.nn.functional.normalize(
            torch.stack([tilt[:, 0], tilt[:, 1], torch.zeros_like(angle)], dim=-1), dim=-1, eps=1.0e-9
        )
        rot = matrix_from_quat(quat_from_angle_axis(angle, axis))
        return torch.einsum("nij,nrj->nri", rot, rel)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        offset: float = 0.0,
        resolution: float = 0.05,
        size: tuple[float, float] = (1.4, 1.0),
        scanner_offset_xy: tuple[float, float] = (0.0, 0.0),
        exclude_half_extent_x: float = 0.30,
        exclude_half_extent_y: float = 0.20,
        lidar_offset: tuple[float, float, float] = (0.19, 0.0, 0.10),
        horizontal_fov: tuple[float, float] = (-180.0, 180.0),
        flat_fill: float = 0.0,
        noise: LidarNoiseCfg | None = None,
        debug_vis: bool = False,
        debug_vis_env_index: int | None = 0,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        asset = env.scene[asset_cfg.name]
        hits_w = sensor.data.ray_hits_w  # (N, num_rays, 3)
        root_pos = asset.data.root_pos_w  # (N, 3)
        finite = torch.isfinite(hits_w).all(dim=-1)

        if self._noise is not None:
            level = self._noise.scale * _curriculum_level(env.common_step_counter, self._noise)
            env.lidar_noise_level = level
            if level > 0.0:
                # Every ray leaves one point in the fan, so perturb them relative to that
                # origin. ray_starts carries the mount offset in the sensor frame.
                origin = sensor.data.pos_w + quat_apply(sensor.data.quat_w, sensor.ray_starts[:, 0])
                perturbed = self._perturb(hits_w - origin.unsqueeze(1), finite, level)
                hits_w = origin.unsqueeze(1) + perturbed

        # Hits into the yaw-aligned base frame. Only the xy rotation is needed to pick
        # the cell, and a 2D rotation avoids expanding the quaternion to every ray.
        rel = hits_w - root_pos.unsqueeze(1)
        yaw = _yaw_from_quat(asset.data.root_quat_w).unsqueeze(1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        bx = cos_y * rel[..., 0] + sin_y * rel[..., 1]
        by = -sin_y * rel[..., 0] + cos_y * rel[..., 1]

        ix = torch.round((bx - self._x0) / self._resolution).long()
        iy = torch.round((by - self._y0) / self._resolution).long()
        valid = (
            (ix >= 0)
            & (ix < self._num_x)
            & (iy >= 0)
            & (iy < self._num_y)
            & finite  # from the raw returns: noise must not resurrect a miss
        )

        # Same feature as height_scan_excluding_body: sensor height - terrain - offset.
        # Higher terrain means a smaller value, so the elevation map's "highest return
        # in the cell" is an amin here, not an amax.
        heights = root_pos[:, 2:3] - hits_w[..., 2] - offset
        heights = torch.where(valid, heights, heights.new_full((), _UNOBSERVED))
        flat_idx = torch.where(valid, ix * self._num_y + iy, torch.zeros_like(ix))

        grid = torch.full_like(self._hold, _UNOBSERVED)
        grid.scatter_reduce_(1, flat_idx, heights, reduce="amin", include_self=True)

        unobserved = grid >= _UNOBSERVED * 0.5
        grid = torch.where(unobserved, self._hold, grid)
        self._hold = grid

        kept = grid.index_select(1, self._keep_indices)
        # Per-cell, over the whole grid, for whoever wants to split a metric by
        # whether a beam actually landed. Published rather than returned: the policy
        # must never read it -- a real spinning LiDAR cannot say which cells it
        # covered inside one 20 ms control step -- but an evaluator computing
        # reconstruction error per region is measuring, not observing.
        env.lidar_map_unobserved_cells = unobserved
        self._record_diagnostics(env, unobserved.index_select(1, self._keep_indices))

        if debug_vis:
            self._visualize(env, asset, kept, unobserved, offset, debug_vis_env_index)
        return kept

    def _record_diagnostics(self, env: ManagerBasedRLEnv, unobserved_kept: torch.Tensor) -> None:
        """Publish per-band unobserved rates for the curriculum logger.

        Split by band because the two causes are not separable in the aggregate: on
        flat terrain every unobserved in-FOV cell is a density shortfall, while the
        rise over that baseline on obstacle terrain is the occlusion signal. With a
        sparse fan the baseline is nowhere near zero -- see ``LIDAR_H_RES`` for the
        measured figures -- so these are only meaningful as a difference against a
        flat-terrain run of the same fan.
        """

        def rate(mask: torch.Tensor) -> float:
            if not bool(mask.any()):
                return 0.0
            return float(unobserved_kept[:, mask].float().mean())

        env.lidar_map_unobserved_rate = rate(self._in_fov)
        env.lidar_map_unobserved_near = rate(self._band_near)
        env.lidar_map_unobserved_mid = rate(self._band_mid)
        env.lidar_map_unobserved_far = rate(self._band_far)

    def _visualize(
        self,
        env: ManagerBasedRLEnv,
        asset,
        kept: torch.Tensor,
        unobserved: torch.Tensor,
        offset: float,
        env_index: int | None,
    ) -> None:
        """Green = measured this step, red = held from an earlier step.

        The split is the whole point of the visualization: red marks exactly the cells
        the real robot would have no data for right now, so a wall's shadow should show
        up as a solid red region and flat ground as almost none.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if not hasattr(self, "_visualizer"):
            self._visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Go2LidarMap",
                    markers={
                        "measured": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                        "held": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    },
                )
            )

        kept_xy = self._cell_xy.index_select(0, self._keep_indices)  # (K, 2)
        held = unobserved.index_select(1, self._keep_indices)
        if env_index is None:
            env_ids = torch.arange(kept.shape[0], device=self.device)
        else:
            env_ids = torch.tensor([min(max(env_index, 0), kept.shape[0] - 1)], device=self.device)

        root_pos = asset.data.root_pos_w[env_ids]
        yaw = _yaw_from_quat(asset.data.root_quat_w[env_ids]).unsqueeze(-1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cell_x, cell_y = kept_xy[:, 0].unsqueeze(0), kept_xy[:, 1].unsqueeze(0)

        shown = kept[env_ids]
        if self.cfg.clip is not None:
            shown = shown.clamp(min=self.cfg.clip[0], max=self.cfg.clip[1])
        positions = torch.stack(
            [
                root_pos[:, 0:1] + cos_y * cell_x - sin_y * cell_y,
                root_pos[:, 1:2] + sin_y * cell_x + cos_y * cell_y,
                root_pos[:, 2:3] - shown - offset,
            ],
            dim=-1,
        ).reshape(-1, 3)
        marker_indices = held[env_ids].reshape(-1).long()

        finite = torch.isfinite(positions).all(dim=-1)
        if not bool(finite.any()):
            return
        self._visualizer.visualize(
            translations=positions[finite], marker_indices=marker_indices[finite]
        )


def lidar_map_unobserved_rate(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Fraction of in-FOV cells with no return this step (all bands)."""
    return torch.tensor(getattr(env, "lidar_map_unobserved_rate", 0.0), device=env.device)


def lidar_map_unobserved_near(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Unobserved rate for in-FOV cells within 0.30 m of the LiDAR mount."""
    return torch.tensor(getattr(env, "lidar_map_unobserved_near", 0.0), device=env.device)


def lidar_map_unobserved_mid(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Unobserved rate for in-FOV cells 0.30-0.50 m from the LiDAR mount."""
    return torch.tensor(getattr(env, "lidar_map_unobserved_mid", 0.0), device=env.device)


def lidar_map_unobserved_far(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Unobserved rate for in-FOV cells beyond 0.50 m -- where shadows land."""
    return torch.tensor(getattr(env, "lidar_map_unobserved_far", 0.0), device=env.device)


def lidar_noise_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Effective noise magnitude in force this step (``scale`` times the ramp)."""
    return torch.tensor(getattr(env, "lidar_noise_level", 0.0), device=env.device)
