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
