from __future__ import annotations

import torch
from functools import lru_cache
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


@lru_cache(maxsize=None)
def _height_scan_indices(
    resolution: float,
    size_x: float,
    size_y: float,
    scanner_offset_x: float,
    scanner_offset_y: float,
    exclude_half_extent_x: float,
    exclude_half_extent_y: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return kept and excluded ``ordering="yx"`` flattened grid indices."""
    num_x = round(size_x / resolution) + 1
    num_y = round(size_y / resolution) + 1
    x = torch.linspace(-size_x / 2, size_x / 2, num_x, device=device) + scanner_offset_x
    y = torch.linspace(-size_y / 2, size_y / 2, num_y, device=device) + scanner_offset_y
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    # Include cells on the footprint boundary despite floating-point roundoff.
    eps = resolution * 1.0e-4
    under_body = (x_grid.abs() <= exclude_half_extent_x + eps) & (
        y_grid.abs() <= exclude_half_extent_y + eps
    )
    under_body = under_body.flatten()
    keep_indices = (~under_body).nonzero(as_tuple=False).squeeze(-1)
    excluded_indices = under_body.nonzero(as_tuple=False).squeeze(-1)
    return keep_indices, excluded_indices


def _visualize_height_scan_points(
    sensor,
    indices: torch.Tensor,
    env_index: int | None,
    visualizer_attr: str,
    prim_path: str,
    color: tuple[float, float, float],
    radius: float = 0.018,
) -> None:
    """Overlay a subset of the grid's ray-hit points with spheres in Isaac Sim.

    ``env_index=None`` draws every environment (only cheap enough for play-sized
    envs; training leaves this at a single index to avoid the marker overhead).
    The visualizer is cached on the sensor under ``visualizer_attr`` so several
    subsets (kept cells, excluded cells) can be drawn in different colors at once.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    if not hasattr(sensor, visualizer_attr):
        marker_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "points": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            },
        )
        setattr(sensor, visualizer_attr, VisualizationMarkers(marker_cfg))

    if env_index is None:
        positions = sensor.data.ray_hits_w.index_select(1, indices).reshape(-1, 3)
    else:
        env_index = min(max(env_index, 0), sensor.data.ray_hits_w.shape[0] - 1)
        positions = sensor.data.ray_hits_w[env_index].index_select(0, indices)
    positions = positions[torch.isfinite(positions).all(dim=-1)]
    if positions.shape[0] == 0:
        return
    getattr(sensor, visualizer_attr).visualize(translations=positions)


def height_scan_excluding_body(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.0,
    resolution: float = 0.05,
    size: tuple[float, float] = (1.0, 1.0),
    scanner_offset_xy: tuple[float, float] = (0.0, 0.0),
    exclude_half_extent_x: float = 0.25,
    exclude_half_extent_y: float = 0.15,
    debug_vis_excluded_body: bool = False,
    debug_vis_scan: bool = False,
    debug_vis_env_index: int | None = 0,
) -> torch.Tensor:
    """Height scan with cells under the robot body removed from the observation.

    This computes the same height feature as the default height_scan
    (sensor height minus ray-hit z minus offset), then returns only fixed grid
    cells outside a rectangle centered on the robot base. Grid cells are
    ordered with x as the outer axis and y as the inner axis.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2] - offset

    keep_indices, excluded_indices = _height_scan_indices(
        resolution,
        size[0],
        size[1],
        scanner_offset_xy[0],
        scanner_offset_xy[1],
        exclude_half_extent_x,
        exclude_half_extent_y,
        heights.device,
    )
    # Not gated on sensor.cfg.debug_vis: that flag only controls the raw RayCaster's own
    # marker drawing, which we may want off (see RobotPlayEnvCfgPhase1/2) so these
    # overlays are visible on their own instead of buried among the full raw grid's markers.
    if debug_vis_excluded_body:
        _visualize_height_scan_points(
            sensor,
            excluded_indices,
            debug_vis_env_index,
            visualizer_attr="_excluded_body_visualizer",
            prim_path="/Visuals/Go2HeightScan/ExcludedBody",
            color=(1.0, 0.0, 1.0),  # magenta
        )
    # The cells the policy actually receives, at the terrain points they were read
    # from -- red so they read against the magenta exclusion overlay.
    if debug_vis_scan:
        _visualize_height_scan_points(
            sensor,
            keep_indices,
            debug_vis_env_index,
            visualizer_attr="_scan_visualizer",
            prim_path="/Visuals/Go2HeightScan/Scan",
            color=(1.0, 0.0, 0.0),  # red
        )
    return heights.index_select(1, keep_indices)
