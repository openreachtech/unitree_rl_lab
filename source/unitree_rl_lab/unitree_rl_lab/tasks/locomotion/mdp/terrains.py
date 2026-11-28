"""Custom sub-terrains.

G1-GRU-Phase3, after tsubame_isaac_env (feat/pos-target):

* pyramid stairs whose tread narrows as the step height rises with difficulty
* concentric circular walls standing on flat ground

G1-FPM (Watch Your Step terrain suite, trimmed):

* ridge stairs / ridge slope: up, then back down to the spawn level, in every heading,
  so one column covers both directions
* a gap around the spawn platform, with a floor to fall onto
* stepping stones over a floor

The robot spawns on the center platform and walks outward, so the pyramid (platform on
top) is stairs *down* and the inverted one (platform at the bottom) is stairs *up*.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.terrains import (
    HfTerrainBaseCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    SubTerrainBaseCfg,
)
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

__all__ = [
    "MeshPyramidStairsVariableWidthCfg",
    "MeshInvertedPyramidStairsVariableWidthCfg",
    "MeshCircularWallTerrainCfg",
    "MeshThinWallTerrainCfg",
    "MeshRidgeStairsTerrainCfg",
    "HfRidgeSlopeTerrainCfg",
    "MeshGapWithFloorTerrainCfg",
    "MeshSteppingStonesTerrainCfg",
]


def _lerp(value_range: tuple[float, float], difficulty: float) -> float:
    return value_range[0] + difficulty * (value_range[1] - value_range[0])


def pyramid_stairs_variable_width_terrain(difficulty: float, cfg: MeshPyramidStairsVariableWidthCfg):
    """Isaac Lab's pyramid stairs with the tread taken from ``step_width_range``."""
    return mesh_terrains.pyramid_stairs_terrain(
        difficulty, cfg.replace(step_width=_lerp(cfg.step_width_range, difficulty))
    )


def inverted_pyramid_stairs_variable_width_terrain(difficulty: float, cfg: MeshInvertedPyramidStairsVariableWidthCfg):
    """Isaac Lab's inverted pyramid stairs with the tread taken from ``step_width_range``."""
    return mesh_terrains.inverted_pyramid_stairs_terrain(
        difficulty, cfg.replace(step_width=_lerp(cfg.step_width_range, difficulty))
    )


@configclass
class MeshPyramidStairsVariableWidthCfg(MeshPyramidStairsTerrainCfg):
    """Pyramid stairs (platform on top) whose tread narrows with difficulty."""

    function = pyramid_stairs_variable_width_terrain

    step_width: float = 0.0
    """Unused: set from ``step_width_range``."""
    step_width_range: tuple[float, float] = MISSING
    """Tread (m) at difficulty 0 and 1: ``(wide_easy, narrow_hard)``."""


@configclass
class MeshInvertedPyramidStairsVariableWidthCfg(MeshInvertedPyramidStairsTerrainCfg):
    """Inverted pyramid stairs (platform at the bottom) whose tread narrows with difficulty."""

    function = inverted_pyramid_stairs_variable_width_terrain

    step_width: float = 0.0
    """Unused: set from ``step_width_range``."""
    step_width_range: tuple[float, float] = MISSING
    """Tread (m) at difficulty 0 and 1: ``(wide_easy, narrow_hard)``."""


def circular_wall_terrain(difficulty: float, cfg: MeshCircularWallTerrainCfg):
    """Flat ground with concentric circular walls around the center spawn platform.

    Ring radii, outermost first: ``R_k = (size/2 - border) - (k + 0.5) * spacing`` for the
    ``(size - 2 border - platform) // (2 spacing) + 1`` rings that fit. A circle puts the
    wall at the same distance in every heading, so a distance-from-spawn curriculum means
    the same thing whichever way the robot walks.
    """
    height = _lerp(cfg.wall_height_range, difficulty)
    thickness = _lerp(cfg.wall_thickness_range, difficulty)
    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    floor = cfg.floor_thickness
    meshes = [
        trimesh.creation.box(
            (size[0], size[1], floor), trimesh.transformations.translation_matrix((center[0], center[1], -floor / 2))
        )
    ]
    if cfg.border_width > 0.0:
        meshes += make_border(cfg.size, size, floor, [center[0], center[1], -floor / 2])
    lift = trimesh.transformations.translation_matrix((center[0], center[1], height / 2))
    for k in range(int((size[0] - cfg.platform_width) // (2 * cfg.wall_spacing) + 1)):
        r = size[0] / 2 - (k + 0.5) * cfg.wall_spacing
        if r > thickness:
            meshes.append(
                trimesh.creation.annulus(
                    r_min=r - thickness / 2, r_max=r + thickness / 2, height=height, sections=96, transform=lift
                )
            )
    return meshes, np.array([center[0], center[1], 0.0])


@configclass
class MeshCircularWallTerrainCfg(SubTerrainBaseCfg):
    """Concentric circular walls (hurdles) on flat ground, to step over."""

    function = circular_wall_terrain

    border_width: float = 0.0
    wall_height_range: tuple[float, float] = (0.05, 0.25)
    """Wall height (m) at difficulty 0 and 1."""
    wall_thickness_range: tuple[float, float] = (0.01, 0.01)
    """Wall thickness (m) at difficulty 0 and 1."""
    wall_spacing: float = 1.5667
    """Distance between consecutive rings (m)."""
    platform_width: float = 3.0
    """Only decides how many rings fit, not where they are."""
    floor_thickness: float = 0.05


def thin_wall_terrain(
    difficulty: float, cfg: "MeshThinWallTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Concentric free-standing thin walls (hurdles) around a center spawn platform (from
    feat/wall).

    Same ring layout as :func:`isaaclab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain`
    (concentric square rings around a flat center platform) but each ring is a
    free-standing wall instead of a raised/lowered stair tread -- ground stays flat
    at z=0 everywhere except the walls themselves, which the robot must step over.

    Both wall height and wall thickness scale with difficulty, the same way
    ``step_height_range``/``step_width_range`` already do for the pyramid-stairs
    family: low and thick (easy to clear, easy to see) at difficulty 0, tall and
    thin (a real height challenge, harder to detect via height-scan) at difficulty
    1. Ring-to-ring spacing (``cfg.wall_spacing``) is a fixed walking gap, decoupled
    from thickness, so consecutive hurdles don't get closer together as they get
    thinner.
    """
    wall_height = _lerp(cfg.wall_height_range, difficulty)
    wall_thickness = _lerp(cfg.wall_thickness_range, difficulty)
    wall_spacing = cfg.wall_spacing

    num_walls_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * wall_spacing) + 1
    num_walls_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * wall_spacing) + 1
    num_walls = int(min(num_walls_x, num_walls_y))

    meshes_list = list()

    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.floor_thickness]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, cfg.floor_thickness, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)

    # Flat ground everywhere; walls stand on top of it.
    meshes_list.append(
        _box(
            (terrain_size[0], terrain_size[1], cfg.floor_thickness),
            (terrain_center[0], terrain_center[1], -cfg.floor_thickness / 2),
        )
    )
    wall_z = wall_height / 2.0
    for k in range(num_walls):
        box_size = (terrain_size[0] - 2 * k * wall_spacing, terrain_size[1] - 2 * k * wall_spacing)
        box_offset = (k + 0.5) * wall_spacing

        # top / bottom wall segments
        wall_dims_tb = (box_size[0], wall_thickness, wall_height)
        y_top = terrain_center[1] + terrain_size[1] / 2.0 - box_offset
        y_bottom = terrain_center[1] - terrain_size[1] / 2.0 + box_offset
        meshes_list += [
            _box(wall_dims_tb, (terrain_center[0], y_top, wall_z)),
            _box(wall_dims_tb, (terrain_center[0], y_bottom, wall_z)),
        ]

        # left / right wall segments, notched so corners don't double up with top/bottom
        wall_dims_lr = (wall_thickness, box_size[1] - 2 * wall_thickness, wall_height)
        x_right = terrain_center[0] + terrain_size[0] / 2.0 - box_offset
        x_left = terrain_center[0] - terrain_size[0] / 2.0 + box_offset
        meshes_list += [
            _box(wall_dims_lr, (x_right, terrain_center[1], wall_z)),
            _box(wall_dims_lr, (x_left, terrain_center[1], wall_z)),
        ]

    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    return meshes_list, origin


@configclass
class MeshThinWallTerrainCfg(SubTerrainBaseCfg):
    """Concentric thin free-standing walls the robot must step over, centered on a
    flat spawn platform -- same ring layout as the pyramid-stairs family, but each
    ring is a hurdle instead of a stair tread."""

    function = thin_wall_terrain

    border_width: float = 0.0
    wall_height_range: tuple[float, float] = (0.05, 0.25)
    """Wall height (in m) at difficulty 0 and difficulty 1: ``(low_easy, tall_hard)``."""
    wall_thickness_range: tuple[float, float] = (0.15, 0.03)
    """Wall thickness (in m) at difficulty 0 and difficulty 1: ``(thick_easy, thin_hard)``."""
    wall_spacing: float = 0.60
    """Fixed center-to-center spacing between consecutive concentric wall rings (in m),
    independent of wall_thickness_range."""
    platform_width: float = 1.0
    floor_thickness: float = 0.05


# ---------------------------------------------------------------------------
# G1-FPM terrains. All keep the spawn platform at z = 0 and the tile edge at z = 0,
# so neighbouring tiles line up and "below the spawn level" only happens in a hole.
# ---------------------------------------------------------------------------


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))


def _ridge_profile(num_steps: int) -> list[int]:
    """Step levels outward from the platform: up to the peak, then down (1, 2, .., p, .., 2, 1); the
    last ring steps down onto the ground at the tile border."""
    return [min(k + 1, num_steps - k) for k in range(num_steps)]


def ridge_stairs_terrain(difficulty: float, cfg: MeshRidgeStairsTerrainCfg):
    """Square stair rings around the spawn platform: up ``n/2`` steps, then down to the ground.

    Ring k spans Chebyshev distance ``platform/2 + k * tread`` to ``+ (k + 1) * tread``; the
    tread narrows and the riser grows with difficulty (as the Phase 3 stairs)."""
    step_h = _lerp(cfg.step_height_range, difficulty)
    tread = _lerp(cfg.step_width_range, difficulty)
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    run = 0.5 * (min(cfg.size) - cfg.platform_width) - cfg.border_width
    levels = _ridge_profile(int(run // tread))
    base = cfg.floor_thickness
    meshes = [_box((cfg.size[0], cfg.size[1], base), (cx, cy, -base / 2))]  # ground, top at 0
    for k, level in enumerate(levels):
        if level == 0:
            continue
        h = level * step_h
        inner = cfg.platform_width + 2 * k * tread
        outer = inner + 2 * tread
        meshes += make_border((outer, outer), (inner, inner), h + base, [cx, cy, (h - base) / 2])
    return meshes, np.array([cx, cy, 0.0])


@configclass
class MeshRidgeStairsTerrainCfg(SubTerrainBaseCfg):
    """Stairs up then down (a square ridge) around the spawn platform."""

    function = ridge_stairs_terrain

    step_height_range: tuple[float, float] = (0.05, 0.25)
    """Riser (m) at difficulty 0 and 1."""
    step_width_range: tuple[float, float] = (0.35, 0.275)
    """Tread (m) at difficulty 0 and 1."""
    platform_width: float = 3.0
    border_width: float = 0.25
    """Flat ground between the last step and the tile edge (m)."""
    floor_thickness: float = 0.05


@height_field_to_mesh
def ridge_slope_terrain(difficulty: float, cfg: HfRidgeSlopeTerrainCfg) -> np.ndarray:
    """A square ridge of slope ``s``: flat platform, up to the peak half way out, down to the edge."""
    slope = _lerp(cfg.slope_range, difficulty)
    nx, ny = int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale)
    x = (np.arange(nx) + 0.5) * cfg.horizontal_scale - 0.5 * cfg.size[0]
    y = (np.arange(ny) + 0.5) * cfg.horizontal_scale - 0.5 * cfg.size[1]
    d = np.maximum(np.abs(x)[:, None], np.abs(y)[None, :]) - 0.5 * cfg.platform_width  # out from the platform
    half = 0.5 * (0.5 * min(cfg.size) - 0.5 * cfg.platform_width)
    height = slope * np.clip(half - np.abs(d - half), 0.0, None)
    height[d < 0.0] = 0.0
    return np.rint(height / cfg.vertical_scale).astype(np.int16)


@configclass
class HfRidgeSlopeTerrainCfg(HfTerrainBaseCfg):
    """Slope up then down (a square ridge) around the spawn platform."""

    function = ridge_slope_terrain

    slope_range: tuple[float, float] = (0.0, 0.5)
    """Gradient (rise / run) at difficulty 0 and 1."""
    platform_width: float = 3.0


def gap_with_floor_terrain(difficulty: float, cfg: MeshGapWithFloorTerrainCfg):
    """Concentric square gaps (moats) around the spawn platform, ``ring_width`` of solid ground
    between them, a floor ``gap_depth`` down (a robot that steps in lands on something and the
    height map sees a bottom). The last ring of ground runs out to the tile edge, so tiles join
    on solid ground and a robot crosses a gap every ``gap + ring_width`` metres."""
    gap = _lerp(cfg.gap_width_range, difficulty)
    depth = cfg.gap_depth
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    half_tile = 0.5 * min(cfg.size)
    r = 0.5 * cfg.platform_width
    meshes = [
        _box((cfg.size[0], cfg.size[1], 0.05), (cx, cy, -depth - 0.025)),  # floor
        _box((2 * r, 2 * r, depth), (cx, cy, -depth / 2)),  # spawn platform
    ]
    while True:
        inner = r + gap
        outer = inner + cfg.ring_width
        if inner >= half_tile:
            break
        if outer + gap >= half_tile:  # no room for another gap: solid out to the tile edge
            meshes += make_border(cfg.size, (2 * inner, 2 * inner), depth, [cx, cy, -depth / 2])
            break
        meshes += make_border((2 * outer, 2 * outer), (2 * inner, 2 * inner), depth, [cx, cy, -depth / 2])
        r = outer
    return meshes, np.array([cx, cy, 0.0])


@configclass
class MeshGapWithFloorTerrainCfg(SubTerrainBaseCfg):
    """Concentric gaps around the spawn platform to step over, with a floor at ``gap_depth``."""

    function = gap_with_floor_terrain

    gap_width_range: tuple[float, float] = (0.1, 0.5)
    """Gap width (m) at difficulty 0 and 1."""
    ring_width: float = 1.0
    """Solid ground between two gaps (m)."""
    platform_width: float = 2.5
    gap_depth: float = 0.5


def stepping_stones_terrain(difficulty: float, cfg: MeshSteppingStonesTerrainCfg):
    """Square stones on a uniform lattice centred on the tile, tops at z = 0, over a floor
    ``hole_depth`` down. The spawn platform covers the inner ``m x m`` lattice block and is
    sized so its edge is exactly one gap from the first stones (at least ``platform_width``)."""
    stone = _lerp(cfg.stone_width_range, difficulty)
    gap = _lerp(cfg.gap_width_range, difficulty)
    pitch = stone + gap
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    depth = cfg.hole_depth
    m = int(np.ceil((0.5 * cfg.platform_width + gap + 0.5 * stone) / pitch))  # first lattice ring outside
    half_p = m * pitch - 0.5 * stone - gap
    meshes = [
        _box((cfg.size[0], cfg.size[1], 0.05), (cx, cy, -depth - 0.025)),  # floor
        _box((2 * half_p, 2 * half_p, depth), (cx, cy, -depth / 2)),  # spawn platform
    ]
    n = int((0.5 * min(cfg.size) - 0.5 * stone) // pitch)
    # solid ground from one gap past the last stones to the tile edge, so tiles join on solid ground
    inner = 2 * (n * pitch + 0.5 * stone + gap)
    if inner < min(cfg.size):
        meshes += make_border(cfg.size, (inner, inner), depth, [cx, cy, -depth / 2])
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if max(abs(i), abs(j)) < m:
                continue  # under the platform
            meshes.append(_box((stone, stone, depth), (cx + i * pitch, cy + j * pitch, -depth / 2)))
    return meshes, np.array([cx, cy, 0.0])


@configclass
class MeshSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """Stepping stones around the spawn platform, over a floor (Watch Your Step: stones 0.3-0.5 m,
    gaps 0.1-0.2 m)."""

    function = stepping_stones_terrain

    stone_width_range: tuple[float, float] = (0.5, 0.3)
    """Stone side (m) at difficulty 0 and 1."""
    gap_width_range: tuple[float, float] = (0.05, 0.2)
    """Gap between stones (m) at difficulty 0 and 1."""
    platform_width: float = 2.5
    hole_depth: float = 0.5
