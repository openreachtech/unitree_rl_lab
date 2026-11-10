"""Circular wall rings for the wall-jump tasks.

The layout is ``Anaguma-Perceptive-Mid360-Phase5``'s (tsubame_isaac_env, ``mdp/terrains.py``):
concentric rings on flat ground around the spawn point, circular rather than square so that the
distance to the wall is the same in every heading.

Ring centre-lines sit at ``R_k = (size/2 - border) - (k + 0.5) * spacing``, the thin-wall
terrain's formula; with ``size`` 8.0, ``border`` 0.0, ``platform`` 3.0 and ``spacing`` 1.5667 there
are two, at 1.65 and 3.22 m (``num_rings`` keeps the inner ones only).

Each tile's wall is recorded in :data:`WALL_TABLE` as ``(height, thickness)``, indexed like
``TerrainImporter.terrain_origins``: the commands, rewards and terminations need to know where the
wall top is, and the generator's curriculum draws a tile's difficulty with jitter inside its row
(``(row + U(0, 1)) / num_rows``), so the height cannot be recomputed from the row.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import HfRandomUniformTerrainCfg, SubTerrainBaseCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass

CIRCULAR_WALL_SECTIONS = 96
"""Facets of each ring. At 3.2 m the chord sag is R * (1 - cos(pi / 96)) = 1.7 mm."""

WALL_TABLE: np.ndarray | None = None
"""``(rows, cols, 2)``: each tile's wall ``(height, thickness)``; NaN for tiles without one."""

_last_wall: tuple[float, float] | None = None


def ring_radii(cfg: MeshCircularWallTerrainCfg, size: tuple[float, float] | None = None) -> list[float]:
    """Centre-line radii of the rings, innermost first. ``size`` defaults to the sub-terrain's own
    (which the generator fills in from its tile size)."""
    size = size or cfg.size
    half = size[0] / 2.0 - cfg.border_width
    n = int((size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.wall_spacing) + 1)
    radii = sorted(half - (k + 0.5) * cfg.wall_spacing for k in range(max(n, 0)))
    return radii[: cfg.num_rings] if cfg.num_rings is not None else radii


def _annulus(r_mid: float, thickness: float, height: float, center) -> trimesh.Trimesh:
    return trimesh.creation.annulus(
        r_min=max(r_mid - thickness / 2.0, 1e-4),
        r_max=r_mid + thickness / 2.0,
        height=height,
        sections=CIRCULAR_WALL_SECTIONS,
        transform=trimesh.transformations.translation_matrix((center[0], center[1], height / 2.0)),
    )


def circular_wall_terrain(difficulty: float, cfg: MeshCircularWallTerrainCfg):
    global _last_wall
    if cfg.row_heights:
        # One height per row: the generator's curriculum difficulty is (row + U(0, 1)) / num_rows.
        height = cfg.row_heights[min(int(difficulty * len(cfg.row_heights)), len(cfg.row_heights) - 1)]
    else:
        low, high = cfg.wall_height_range
        height = low + (high - low) * difficulty
    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
    floor = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], cfg.floor_thickness),
        trimesh.transformations.translation_matrix((center[0], center[1], -cfg.floor_thickness / 2.0)),
    )
    meshes = [floor] + [_annulus(r, cfg.wall_thickness, height, center) for r in ring_radii(cfg)]
    _last_wall = (height, cfg.wall_thickness)
    return meshes, np.array(center)


@configclass
class MeshCircularWallTerrainCfg(SubTerrainBaseCfg):
    """Concentric circular walls standing on flat ground around the spawn point."""

    function = circular_wall_terrain

    wall_height_range: tuple[float, float] = (0.05, 0.50)
    """Wall height at difficulty 0 and 1."""
    row_heights: tuple[float, ...] | None = None
    """Exact wall heights, one per row, in place of ``wall_height_range`` -- for looking at given
    heights in Play. ``num_rows`` must match."""
    wall_thickness: float = 0.05
    """Fixed, as in Anaguma's Phase 5: with height and thickness moving together it is not clear
    which one a change in the crossing rate came from."""
    wall_spacing: float = 1.5667
    platform_width: float = 3.0
    """Decides only how many rings are generated, not where: 3.0 gives two."""
    num_rings: int | None = None
    """Keep only the innermost this many rings (``None``: all)."""
    border_width: float = 0.0
    floor_thickness: float = 0.05


class WallTerrainGenerator(TerrainGenerator):
    """Terrain generator that also records each tile's wall in :data:`WALL_TABLE`."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        global WALL_TABLE
        WALL_TABLE = np.full((cfg.num_rows, cfg.num_cols, 2), np.nan)
        super().__init__(cfg, device)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        global _last_wall
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if isinstance(sub_terrain_cfg, MeshCircularWallTerrainCfg) and _last_wall is not None:
            WALL_TABLE[row, col] = _last_wall
        _last_wall = None


CIRCULAR_WALLS = MeshCircularWallTerrainCfg(proportion=1.0)

ROUGH = HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25)
"""Anaguma Phase 5's rough column, for driving. ``random_uniform_terrain`` ignores the difficulty,
so every row is the same ground."""

WALL_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=WallTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    # A cached tile skips the generating function, and WALL_TABLE would not be filled.
    use_cache=False,
    curriculum=True,
    sub_terrains={"walls": CIRCULAR_WALLS},
)
"""The generator settings; tasks replace the rows, columns and sub-terrains."""
