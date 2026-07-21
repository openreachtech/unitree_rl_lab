"""Stair terrains whose tread width narrows with difficulty, not just step height.

Isaac Lab's stock ``MeshPyramidStairsTerrainCfg``/``MeshInvertedPyramidStairsTerrainCfg``
interpolate ``step_height`` across ``step_height_range`` based on the row's difficulty
(0 at the easiest row, 1 at the hardest), but treat ``step_width`` (tread depth) as a
single fixed value for every row. That means a curriculum that promotes robots to
harder rows only ever makes the risers taller -- the tread is exactly as short on the
very first row as on the last.

These variants add a ``step_width_range`` that interpolates the same way
``step_height_range`` already does, so a robot climbing the built-in
``terrain_levels`` curriculum sees a *wide, easy* tread at low difficulty and only
reaches the *narrow, hard* tread once it has already been promoted through the
easier rows -- a native curriculum on tread width, not just riser height, with no
runtime mesh regeneration required (every row is still baked once at scene creation,
exactly like the stock terrain).
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def _lerp(value_range: tuple[float, float], difficulty: float) -> float:
    return value_range[0] + difficulty * (value_range[1] - value_range[0])


def pyramid_stairs_variable_width_terrain(
    difficulty: float, cfg: "MeshPyramidStairsVariableWidthCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pyramid stairs where both step height and tread width scale with difficulty.

    Identical to :func:`isaaclab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain`
    except ``step_width`` is interpolated from ``cfg.step_width_range`` the same way
    ``step_height`` is interpolated from ``cfg.step_height_range``, instead of being a
    single fixed value.
    """
    step_height = _lerp(cfg.step_height_range, difficulty)
    step_width = _lerp(cfg.step_width_range, difficulty)

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = (k + 2) * step_height

        box_dims = (box_size[0], step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))

        if cfg.holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))

        meshes_list += [box_top, box_bottom, box_right, box_left]

    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
    return meshes_list, origin


def inverted_pyramid_stairs_variable_width_terrain(
    difficulty: float, cfg: "MeshInvertedPyramidStairsVariableWidthCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Inverted pyramid stairs where both step height and tread width scale with difficulty.

    Identical to
    :func:`isaaclab.terrains.trimesh.mesh_terrains.inverted_pyramid_stairs_terrain`
    except ``step_width`` is interpolated from ``cfg.step_width_range`` the same way
    ``step_height`` is interpolated from ``cfg.step_height_range``, instead of being a
    single fixed value.
    """
    step_height = _lerp(cfg.step_height_range, difficulty)
    step_width = _lerp(cfg.step_width_range, difficulty)

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = total_height - (k + 1) * step_height

        box_dims = (box_size[0], step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))

        if cfg.holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))

        meshes_list += [box_top, box_bottom, box_right, box_left]

    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
    return meshes_list, origin


@configclass
class MeshPyramidStairsVariableWidthCfg(SubTerrainBaseCfg):
    """Pyramid stairs whose tread width narrows with difficulty, like step height already does."""

    function = pyramid_stairs_variable_width_terrain

    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width_range: tuple[float, float] = MISSING
    """The tread width (in m) at difficulty 0 and difficulty 1. Unlike the stock
    ``step_width`` this is a range: ``(wide_easy, narrow_hard)``."""
    platform_width: float = 1.0
    holes: bool = False


@configclass
class MeshInvertedPyramidStairsVariableWidthCfg(MeshPyramidStairsVariableWidthCfg):
    """Inverted pyramid stairs whose tread width narrows with difficulty."""

    function = inverted_pyramid_stairs_variable_width_terrain


def _box(dims: tuple[float, float, float], pos: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))


def thin_wall_terrain(
    difficulty: float, cfg: "MeshThinWallTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Concentric free-standing thin walls (hurdles) around a center spawn platform.

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


def floating_thin_wall_terrain(
    difficulty: float, cfg: "MeshFloatingThinWallTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Concentric floating thin walls: same ring layout as thin_wall_terrain,
    but each ring has no solid wall body -- only a thin horizontal tread
    hovering at ``wall_height``, with nothing connecting it to the ground.

    Mirrors how :func:`floating_pyramid_stairs_terrain` hollows out
    :func:`pyramid_stairs_terrain`'s solid risers, keeping only a thin tread
    at the top of each step. Here there's just one "step" per ring (the wall),
    so the whole solid box (z=0 to wall_height) is replaced by a thin slab
    whose top surface still sits at wall_height -- the "upper flat area" the
    solid wall used to have at its top, with the solid body beneath it removed.
    """
    wall_height = _lerp(cfg.wall_height_range, difficulty)
    wall_thickness = _lerp(cfg.wall_thickness_range, difficulty)
    wall_spacing = cfg.wall_spacing
    tread_thickness = cfg.tread_thickness
    stringer_width = cfg.stringer_width

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

    # Flat ground everywhere; the floating tread hovers above it, untouched.
    meshes_list.append(
        _box(
            (terrain_size[0], terrain_size[1], cfg.floor_thickness),
            (terrain_center[0], terrain_center[1], -cfg.floor_thickness / 2),
        )
    )

    tread_z = wall_height - tread_thickness / 2.0
    for k in range(num_walls):
        box_size = (terrain_size[0] - 2 * k * wall_spacing, terrain_size[1] - 2 * k * wall_spacing)
        box_offset = (k + 0.5) * wall_spacing

        # top / bottom treads
        tread_dims_tb = (box_size[0], wall_thickness, tread_thickness)
        y_top = terrain_center[1] + terrain_size[1] / 2.0 - box_offset
        y_bottom = terrain_center[1] - terrain_size[1] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_tb, (terrain_center[0], y_top, tread_z)),
            _box(tread_dims_tb, (terrain_center[0], y_bottom, tread_z)),
        ]

        # left / right treads, notched so corners don't double up with top/bottom
        tread_dims_lr = (wall_thickness, box_size[1] - 2 * wall_thickness, tread_thickness)
        x_right = terrain_center[0] + terrain_size[0] / 2.0 - box_offset
        x_left = terrain_center[0] - terrain_size[0] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_lr, (x_right, terrain_center[1], tread_z)),
            _box(tread_dims_lr, (x_left, terrain_center[1], tread_z)),
        ]

        if cfg.add_stringers:
            post_height = max(wall_height - tread_thickness, tread_thickness)
            post_z = post_height / 2.0
            post_dims = (stringer_width, stringer_width, post_height)
            half_x = box_size[0] / 2.0 - stringer_width / 2.0
            half_y_lr = tread_dims_lr[1] / 2.0 - stringer_width / 2.0
            for y in (y_top, y_bottom):
                meshes_list += [
                    _box(post_dims, (terrain_center[0] + half_x, y, post_z)),
                    _box(post_dims, (terrain_center[0] - half_x, y, post_z)),
                ]
            for x in (x_right, x_left):
                meshes_list += [
                    _box(post_dims, (x, terrain_center[1] + half_y_lr, post_z)),
                    _box(post_dims, (x, terrain_center[1] - half_y_lr, post_z)),
                ]

    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    return meshes_list, origin


@configclass
class MeshFloatingThinWallTerrainCfg(MeshThinWallTerrainCfg):
    """Floating thin walls: no solid wall body, just a thin tread hovering at
    wall_height with nothing connecting it to the ground -- only the "upper
    flat area" of MeshThinWallTerrainCfg survives. Same ring layout and
    wall_height_range/wall_thickness_range/wall_spacing as the solid version."""

    function = floating_thin_wall_terrain

    tread_thickness: float = 0.04
    """Thickness of the floating tread (in m)."""
    stringer_width: float = 0.03
    """Cross-section width of optional vertical support posts (in m)."""
    add_stringers: bool = False
    """If True, add thin vertical posts connecting the tread down to the
    ground as a simple frame. Default False: purely a floating bar with
    nothing below it."""


def floating_pyramid_stairs_terrain(
    difficulty: float, cfg: "MeshFloatingPyramidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pyramid stairs made of thin treads and vertical stringers (no risers).

    Same ring layout as :func:`isaaclab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain`,
    but each step is a thin horizontal tread instead of a solid riser-filled box, with
    optional posts under the tread ends as a simple frame (骨組み).
    """
    step_height = _lerp(cfg.step_height_range, difficulty)
    step_width = _lerp(cfg.step_width_range, difficulty)
    tread_thickness = cfg.tread_thickness
    stringer_width = cfg.stringer_width

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)

    # Floor under the open risers so feet that slip through land on ground.
    meshes_list.append(
        _box((terrain_size[0], terrain_size[1], tread_thickness), (terrain_center[0], terrain_center[1], -tread_thickness / 2))
    )

    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_offset = (k + 0.5) * step_width
        tread_top_z = (k + 1) * step_height
        tread_z = tread_top_z - tread_thickness / 2.0

        # top / bottom treads
        tread_dims_tb = (box_size[0], step_width, tread_thickness)
        y_top = terrain_center[1] + terrain_size[1] / 2.0 - box_offset
        y_bottom = terrain_center[1] - terrain_size[1] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_tb, (terrain_center[0], y_top, tread_z)),
            _box(tread_dims_tb, (terrain_center[0], y_bottom, tread_z)),
        ]

        # left / right treads
        if cfg.holes:
            tread_dims_lr = (step_width, box_size[1], tread_thickness)
        else:
            tread_dims_lr = (step_width, box_size[1] - 2 * step_width, tread_thickness)
        x_right = terrain_center[0] + terrain_size[0] / 2.0 - box_offset
        x_left = terrain_center[0] - terrain_size[0] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_lr, (x_right, terrain_center[1], tread_z)),
            _box(tread_dims_lr, (x_left, terrain_center[1], tread_z)),
        ]

        if cfg.add_stringers:
            post_height = max(tread_top_z - tread_thickness, tread_thickness)
            post_z = post_height / 2.0
            post_dims = (stringer_width, stringer_width, post_height)
            half_x = box_size[0] / 2.0 - stringer_width / 2.0
            half_y_lr = tread_dims_lr[1] / 2.0 - stringer_width / 2.0
            # posts under ends of top/bottom treads
            for y in (y_top, y_bottom):
                meshes_list += [
                    _box(post_dims, (terrain_center[0] + half_x, y, post_z)),
                    _box(post_dims, (terrain_center[0] - half_x, y, post_z)),
                ]
            # posts under ends of left/right treads
            for x in (x_right, x_left):
                meshes_list += [
                    _box(post_dims, (x, terrain_center[1] + half_y_lr, post_z)),
                    _box(post_dims, (x, terrain_center[1] - half_y_lr, post_z)),
                ]

    platform_top_z = (num_steps + 1) * step_height
    platform_size = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
    )
    meshes_list.append(
        _box(
            (platform_size[0], platform_size[1], tread_thickness),
            (terrain_center[0], terrain_center[1], platform_top_z - tread_thickness / 2.0),
        )
    )
    if cfg.add_stringers:
        post_height = max(platform_top_z - tread_thickness, tread_thickness)
        post_z = post_height / 2.0
        post_dims = (stringer_width, stringer_width, post_height)
        half_x = platform_size[0] / 2.0 - stringer_width / 2.0
        half_y = platform_size[1] / 2.0 - stringer_width / 2.0
        for dx in (half_x, -half_x):
            for dy in (half_y, -half_y):
                meshes_list.append(_box(post_dims, (terrain_center[0] + dx, terrain_center[1] + dy, post_z)))

    origin = np.array([terrain_center[0], terrain_center[1], platform_top_z])
    return meshes_list, origin


def floating_inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: "MeshFloatingInvertedPyramidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Inverted pyramid stairs made of thin treads and vertical stringers (no risers).

    Same ring layout as
    :func:`isaaclab.terrains.trimesh.mesh_terrains.inverted_pyramid_stairs_terrain`,
    but each step is a thin horizontal tread instead of a solid riser-filled box.
    """
    step_height = _lerp(cfg.step_height_range, difficulty)
    step_width = _lerp(cfg.step_width_range, difficulty)
    tread_thickness = cfg.tread_thickness
    stringer_width = cfg.stringer_width

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # No full floor here: a slab at z=0 would seal the pit. The border is the
    # top walking surface; open space under treads is intentional.

    bottom_z = -total_height

    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        box_offset = (k + 0.5) * step_width
        tread_top_z = -(k + 1) * step_height
        tread_z = tread_top_z - tread_thickness / 2.0

        tread_dims_tb = (box_size[0], step_width, tread_thickness)
        y_top = terrain_center[1] + terrain_size[1] / 2.0 - box_offset
        y_bottom = terrain_center[1] - terrain_size[1] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_tb, (terrain_center[0], y_top, tread_z)),
            _box(tread_dims_tb, (terrain_center[0], y_bottom, tread_z)),
        ]

        if cfg.holes:
            tread_dims_lr = (step_width, box_size[1], tread_thickness)
        else:
            tread_dims_lr = (step_width, box_size[1] - 2 * step_width, tread_thickness)
        x_right = terrain_center[0] + terrain_size[0] / 2.0 - box_offset
        x_left = terrain_center[0] - terrain_size[0] / 2.0 + box_offset
        meshes_list += [
            _box(tread_dims_lr, (x_right, terrain_center[1], tread_z)),
            _box(tread_dims_lr, (x_left, terrain_center[1], tread_z)),
        ]

        if cfg.add_stringers:
            # Posts hang from the tread down toward the bottom platform.
            post_height = max(tread_top_z - tread_thickness - bottom_z, tread_thickness)
            post_z = tread_top_z - tread_thickness - post_height / 2.0
            post_dims = (stringer_width, stringer_width, post_height)
            half_x = box_size[0] / 2.0 - stringer_width / 2.0
            half_y_lr = tread_dims_lr[1] / 2.0 - stringer_width / 2.0
            for y in (y_top, y_bottom):
                meshes_list += [
                    _box(post_dims, (terrain_center[0] + half_x, y, post_z)),
                    _box(post_dims, (terrain_center[0] - half_x, y, post_z)),
                ]
            for x in (x_right, x_left):
                meshes_list += [
                    _box(post_dims, (x, terrain_center[1] + half_y_lr, post_z)),
                    _box(post_dims, (x, terrain_center[1] - half_y_lr, post_z)),
                ]

    platform_size = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
    )
    meshes_list.append(
        _box(
            (platform_size[0], platform_size[1], tread_thickness),
            (terrain_center[0], terrain_center[1], bottom_z - tread_thickness / 2.0),
        )
    )

    origin = np.array([terrain_center[0], terrain_center[1], bottom_z])
    return meshes_list, origin


@configclass
class MeshFloatingPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Floating pyramid stairs: thin treads + optional posts, no riser walls.

    Both step height and tread width scale with difficulty via ranges.
    """

    function = floating_pyramid_stairs_terrain

    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    """The tread width (in m) at difficulty 0 and difficulty 1:
    ``(wide_easy, narrow_hard)``."""
    platform_width: float = 1.0
    holes: bool = False
    tread_thickness: float = 0.04
    """Thickness of each floating tread (in m)."""
    stringer_width: float = 0.03
    """Cross-section width of vertical support posts (in m)."""
    add_stringers: bool = True
    """If True, add vertical posts under tread ends as a simple frame."""


@configclass
class MeshFloatingInvertedPyramidStairsTerrainCfg(MeshFloatingPyramidStairsTerrainCfg):
    """Floating inverted pyramid stairs: thin treads + optional posts, no riser walls."""

    function = floating_inverted_pyramid_stairs_terrain
