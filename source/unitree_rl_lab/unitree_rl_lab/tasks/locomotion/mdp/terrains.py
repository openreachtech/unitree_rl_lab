"""Sub-terrains for G1-GRU-Phase3, after tsubame_isaac_env (feat/pos-target):

* pyramid stairs whose tread narrows as the step height rises with difficulty
* concentric circular walls standing on flat ground

The robot spawns on the center platform and walks outward, so the pyramid (platform on
top) is stairs *down* and the inverted one (platform at the bottom) is stairs *up*.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg, SubTerrainBaseCfg
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

__all__ = [
    "MeshPyramidStairsVariableWidthCfg",
    "MeshInvertedPyramidStairsVariableWidthCfg",
    "MeshCircularWallTerrainCfg",
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
