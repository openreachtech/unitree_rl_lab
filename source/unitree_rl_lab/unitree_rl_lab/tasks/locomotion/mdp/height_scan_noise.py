"""Height-scan noise with per-episode mapping conditions and a training curriculum.

Follows the height-sample randomization of "Learning robust perceptive locomotion
for quadrupedal robots in the wild" (Miki et al., 2022), section S8 / Figure 6B:
each episode draws one of three *mapping conditions* (nominal / large offset /
large noise), and the noise magnitude ramps up over training instead of being
active from the first iteration (section S3: "We start the student training
without height sample noise and gradually increase the noise level through a
student curriculum factor which linearly increases over training epochs").

Deviations from the paper, and why:

* The paper samples heights in rings around each foot, so its noise is organised
  per foot (``eps_f*`` per step, ``w_*`` per episode). This project uses one
  body-centered grid, so those become one scan-wide offset per step
  (``per_step_offset_std``) and one scan-wide offset per episode
  (``per_episode_offset_std``).
* Lateral scan-point displacement (the paper's ``eps_px/py``, ``eps_fx/fy``,
  ``w_x/w_y``) is not implemented: the grid is a fixed raycast pattern, so
  shifting sample positions means re-gathering cells rather than perturbing a
  value. Only the height-domain terms are applied here.
* The paper states its ``z`` vector holds variances, but its published values
  are hard to reconcile with that reading for every component (e.g. the nominal
  per-foot height term would become a 0.20 m offset resampled every control
  step), and the vector as printed lists 7 numbers for the 8 named parameters
  ``z0..z7``. The tables in ``velocity_env_cfg_student.py`` therefore give
  standard deviations directly, derived from the paper where the value is
  unambiguous and chosen for this grid where it is not.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .observations import _height_scan_indices, height_scan_excluding_body

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class HeightScanNoiseConditionCfg:
    """One mapping condition (paper Figure 6B), in metres of standard deviation."""

    probability: float = 0.0
    """Relative chance of this condition being drawn. Normalised across conditions."""
    per_point_std: float = 0.0
    """Independent noise on every scan cell, resampled every control step."""
    per_step_offset_std: float = 0.0
    """One offset shared by the whole scan, resampled every control step."""
    per_episode_offset_std: float = 0.0
    """One offset shared by the whole scan, held for the episode (map/pose drift)."""
    outlier_std: float = 0.0
    """Magnitude of the intermittent outliers added on top of the other terms."""
    outlier_prob: float = 0.0
    """Per-cell, per-step chance of an outlier."""


@configclass
class HeightScanNoiseCfg:
    """The three mapping conditions plus the iteration schedule that scales them."""

    nominal: HeightScanNoiseConditionCfg = HeightScanNoiseConditionCfg()
    """Normal mapping conditions."""
    large_offset: HeightScanNoiseConditionCfg = HeightScanNoiseConditionCfg()
    """Coherent but displaced map (pose-estimation drift, deformable ground)."""
    large_noise: HeightScanNoiseConditionCfg = HeightScanNoiseConditionCfg()
    """Map carrying essentially no terrain information (occlusion, sensor failure)."""
    num_steps_per_env: int = 200
    """Rollout length of one training iteration, used to turn env steps into iterations."""
    start_iteration: int = 0
    """Iterations of completely clean height scans before the ramp begins."""
    full_iteration: int = 0
    """Iteration at which the conditions reach their configured magnitude."""
    resample_mid_episode: bool = True
    """Redraw the condition halfway through an episode, as the paper does."""


class HeightScanExcludingBodyNoisy(ManagerTermBase):
    """``height_scan_excluding_body`` with the noise model described above.

    The noise lives inside the observation term rather than in an Isaac Lab
    ``NoiseModelCfg`` because it needs the environment: the curriculum factor is
    derived from ``common_step_counter``, and the mid-episode condition redraw
    needs ``episode_length_buf``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._noise_cfg: HeightScanNoiseCfg = cfg.params["scan_noise"]
        conditions = [self._noise_cfg.nominal, self._noise_cfg.large_offset, self._noise_cfg.large_noise]

        def _column(attr: str) -> torch.Tensor:
            return torch.tensor([getattr(c, attr) for c in conditions], device=self.device)

        weights = _column("probability")
        if torch.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("Height-scan noise condition probabilities must be non-negative and sum to > 0.")
        self._condition_weights = weights / weights.sum()
        self._per_point_std = _column("per_point_std")
        self._per_step_offset_std = _column("per_step_offset_std")
        self._per_episode_offset_std = _column("per_episode_offset_std")
        self._outlier_std = _column("outlier_std")
        self._outlier_prob = _column("outlier_prob")

        self._condition = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_offset = torch.zeros(self.num_envs, device=self.device)
        self._redrawn_this_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._draw_conditions(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = self._as_env_ids(env_ids)
        self._draw_conditions(env_ids)
        self._redrawn_this_episode[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        scan_noise: HeightScanNoiseCfg,
        offset: float = 0.0,
        resolution: float = 0.05,
        size: tuple[float, float] = (1.0, 1.0),
        scanner_offset_xy: tuple[float, float] = (0.0, 0.0),
        exclude_half_extent_x: float = 0.25,
        exclude_half_extent_y: float = 0.15,
        debug_vis_excluded_body: bool = False,
        debug_vis_env_index: int = 0,
        debug_vis_noisy_scan: bool = False,
    ) -> torch.Tensor:
        heights = height_scan_excluding_body(
            env,
            sensor_cfg,
            offset=offset,
            resolution=resolution,
            size=size,
            scanner_offset_xy=scanner_offset_xy,
            exclude_half_extent_x=exclude_half_extent_x,
            exclude_half_extent_y=exclude_half_extent_y,
            debug_vis_excluded_body=debug_vis_excluded_body,
            debug_vis_env_index=debug_vis_env_index,
        )

        level = _curriculum_level(env.common_step_counter, scan_noise)
        # Published under Curriculum/height_scan_noise by mdp.height_scan_noise_level.
        env.height_scan_noise_level = level
        if level > 0.0:
            if scan_noise.resample_mid_episode:
                self._redraw_mid_episode()
            heights = self._corrupt(heights, level)

        if debug_vis_noisy_scan:
            self._visualize(
                env,
                sensor_cfg,
                heights,
                offset,
                resolution,
                size,
                scanner_offset_xy,
                exclude_half_extent_x,
                exclude_half_extent_y,
                debug_vis_env_index,
            )
        return heights

    def _corrupt(self, heights: torch.Tensor, level: float) -> torch.Tensor:
        condition = self._condition
        heights = heights + torch.randn_like(heights) * (level * self._per_point_std[condition]).unsqueeze(1)
        per_step_offset = torch.randn(self.num_envs, device=self.device) * self._per_step_offset_std[condition]
        heights = heights + (level * (per_step_offset + self._episode_offset)).unsqueeze(1)

        outlier_prob = (level * self._outlier_prob[condition]).unsqueeze(1)
        is_outlier = torch.rand_like(heights) < outlier_prob
        outliers = torch.randn_like(heights) * (level * self._outlier_std[condition]).unsqueeze(1)
        return torch.where(is_outlier, heights + outliers, heights)

    def _visualize(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        heights: torch.Tensor,
        offset: float,
        resolution: float,
        size: tuple[float, float],
        scanner_offset_xy: tuple[float, float],
        exclude_half_extent_x: float,
        exclude_half_extent_y: float,
        env_index: int,
    ) -> None:
        """Draw the corrupted scan as cyan spheres, over the raw ray hits.

        The raw RayCaster markers show the terrain as it actually is; these show the
        same cells as the policy receives them, so the gap between the two is the
        noise. ``ObservationTermCfg.clip`` is applied here as well, because the
        observation manager clips after this term returns.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        sensor = env.scene.sensors[sensor_cfg.name]
        if not hasattr(sensor, "_noisy_scan_visualizer"):
            sensor._noisy_scan_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Go2HeightScan/NoisyScan",
                    markers={
                        "noisy_scan": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                        )
                    },
                )
            )

        keep_indices, _ = _height_scan_indices(
            resolution,
            size[0],
            size[1],
            scanner_offset_xy[0],
            scanner_offset_xy[1],
            exclude_half_extent_x,
            exclude_half_extent_y,
            heights.device,
        )
        env_index = min(max(env_index, 0), heights.shape[0] - 1)
        shown = heights[env_index]
        if self.cfg.clip is not None:
            shown = shown.clamp(min=self.cfg.clip[0], max=self.cfg.clip[1])

        # height = sensor_z - hit_z - offset, so the corrupted hit sits at sensor_z - height - offset.
        positions = sensor.data.ray_hits_w[env_index].index_select(0, keep_indices).clone()
        positions[:, 2] = sensor.data.pos_w[env_index, 2] - shown - offset
        positions = positions[torch.isfinite(positions).all(dim=-1)]
        if positions.shape[0] == 0:
            return
        sensor._noisy_scan_visualizer.visualize(translations=positions)

    def _as_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _draw_conditions(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        drawn = torch.multinomial(self._condition_weights, env_ids.numel(), replacement=True)
        self._condition[env_ids] = drawn
        # Drawn at full magnitude; the curriculum scales it when the scan is built.
        self._episode_offset[env_ids] = (
            torch.randn(env_ids.numel(), device=self.device) * self._per_episode_offset_std[drawn]
        )

    def _redraw_mid_episode(self) -> None:
        due = (~self._redrawn_this_episode) & (self._env.episode_length_buf >= self._env.max_episode_length // 2)
        env_ids = due.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self._draw_conditions(env_ids)
        self._redrawn_this_episode[env_ids] = True


def _curriculum_level(common_step_counter: int, cfg: HeightScanNoiseCfg) -> float:
    """Fraction of the configured noise magnitude to apply at the current iteration."""
    iteration = common_step_counter // max(cfg.num_steps_per_env, 1)
    if iteration >= cfg.full_iteration:
        return 1.0
    if iteration <= cfg.start_iteration:
        return 0.0
    return (iteration - cfg.start_iteration) / max(cfg.full_iteration - cfg.start_iteration, 1)
