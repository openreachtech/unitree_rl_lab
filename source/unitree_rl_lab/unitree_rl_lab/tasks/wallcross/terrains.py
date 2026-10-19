"""Concentric square-wall terrain for the hind-stance wall crossing task.

Each tile is flat ground with a square ring of wall centred on the tile origin, where the robot
spawns. The robot rises into the hind-leg stance, walks out to the ring and crosses it once, so a
tile carries a single ring: the generator is written for ``num_rings`` of them, for when crossing
several in a row becomes part of the task, but the rings past the first are never reached by a
one-crossing episode and would only cost mesh.

Curriculum
----------
One difficulty axis in ``[0, 1]`` walks four stages, each moving one quantity:

======  ====================  =====================
stage   height (m)            thickness (m)
======  ====================  =====================
A0      0.01 -> 0.20          0.50
A       0.20 -> 0.40          0.50
B       0.40 -> 0.80          0.50
C       0.60 .. 0.80          0.50 -> 0.03
======  ====================  =====================

With ``order="thin_first"`` the thickness stage moves in front of the last height stage instead:
the wall is thinned at 0.30-0.40 m, then raised to 0.80 m while it stays a few centimetres thick.

======  ====================  =====================
stage   height (m)            thickness (m)
======  ====================  =====================
A0      0.01 -> 0.20          0.50
A       0.20 -> 0.40          0.50
C'      0.30 .. 0.40          0.50 -> 0.03
B'      0.40 -> 0.80          0.03 .. 0.07
======  ====================  =====================

The stage boundaries are set from the number of rows each stage gets, so one row moves height by at most
~3 cm or thickness by ~2 cm. Go2W's Phase 5 stalled at a coarser step and recovered when the rows
were doubled (its SUMMARY, Lesson 3).

The value above is the *frontier*: the hardest wall of that level. Most tiles are drawn from a
narrow band just below it, so a level is promoted on walls that are actually at that level; the
rest (``easy_fraction``) are drawn from anywhere between the easiest wall and the frontier, so the
walls the robot has already learned keep appearing and are not forgotten.

Wall geometry is recorded
-------------------------
Rewards and terminations need to know where the wall is (contact on the wall's top edge is
technique, contact with the floor is a fall), and the terrain importer discards its generator once
the mesh is built. :class:`WallTerrainGenerator` therefore keeps a ``(rows, cols, 3)`` table of
``(inner half-size, height, thickness)`` in :data:`WALL_TABLE`, indexed the same way as
``TerrainImporter.terrain_origins``. Tiles of any other sub-terrain type hold NaN.
"""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass

WALL_TABLE: np.ndarray | None = None
"""``(num_rows, num_cols, 3)``: inner half-size, height, thickness of each tile's first ring, in
metres. Filled by the most recent :class:`WallTerrainGenerator`; one terrain per process."""

TERRAIN_MESH: trimesh.Trimesh | None = None
"""The combined terrain mesh of the most recent :class:`WallTerrainGenerator`, kept for inspection
(the importer does not keep it)."""

_last_wall: tuple[float, float, float] | None = None
"""Handoff from :func:`concentric_walls_terrain` to :meth:`WallTerrainGenerator._add_sub_terrain`,
which is called right after it with the tile's row and column."""


def stage_bounds(cfg: MeshConcentricWallsTerrainCfg) -> list[float]:
    """Difficulty at which each stage after the first begins."""
    total = sum(cfg.stage_rows)
    return list(np.cumsum(cfg.stage_rows)[:-1] / total)


def _stage(difficulty: float, cfg: MeshConcentricWallsTerrainCfg) -> tuple[int, float]:
    """Index of the stage this difficulty falls in, and how far through it (0..1)."""
    edges = [0.0] + stage_bounds(cfg) + [1.0]
    d = float(np.clip(difficulty, 0.0, 1.0))
    for i in range(len(edges) - 1):
        if d < edges[i + 1] or i == len(edges) - 2:
            return i, (d - edges[i]) / (edges[i + 1] - edges[i])
    raise AssertionError


def frontier(difficulty: float, cfg: MeshConcentricWallsTerrainCfg) -> tuple[float, float]:
    """The hardest ``(height, thickness)`` at this difficulty."""
    i, u = _stage(difficulty, cfg)
    if cfg.order == "thin_first":
        return _frontier_thin_first(i, u, cfg)
    if i < len(cfg.height_path) - 1:
        lo, hi = cfg.height_path[i], cfg.height_path[i + 1]
        return lo + u * (hi - lo), cfg.thickness_thick
    return cfg.height_path[-1], cfg.thickness_thick + u * (cfg.thickness_thin - cfg.thickness_thick)


def _frontier_thin_first(i: int, u: float, cfg: MeshConcentricWallsTerrainCfg) -> tuple[float, float]:
    """Stages A0, A (height, thick), C' (thickness at ``height_path[2]``), B' (height, thin)."""
    hp = cfg.height_path
    if i < 2:
        return hp[i] + u * (hp[i + 1] - hp[i]), cfg.thickness_thick
    if i == 2:
        return hp[2], cfg.thickness_thick + u * (cfg.thickness_thin - cfg.thickness_thick)
    return hp[2] + u * (hp[3] - hp[2]), cfg.thickness_thin


def sample_wall(difficulty: float, rng: np.random.Generator, cfg: MeshConcentricWallsTerrainCfg):
    """Draw one tile's ``(inner half-size, height, thickness)``."""
    if cfg.order == "thin_first":
        return _sample_wall_thin_first(difficulty, rng, cfg)
    i, _ = _stage(difficulty, cfg)
    h_hi, t_lo = frontier(difficulty, cfg)
    easy = rng.uniform() < cfg.easy_fraction
    lowest = cfg.height_path[0]
    if i < len(cfg.height_path) - 1:
        # Height stages: height is the moving axis, thickness stays at the thick end.
        h_floor = lowest if easy else max(lowest, h_hi - cfg.frontier_band_height)
        height = rng.uniform(h_floor, h_hi)
        thickness = cfg.thickness_thick
    else:
        # Thickness stage: height stays high and varies within the target band.
        height = rng.uniform(cfg.height_stage_c_min, cfg.height_path[-1])
        t_ceil = cfg.thickness_thick if easy else min(cfg.thickness_thick, t_lo + cfg.frontier_band_thickness)
        thickness = rng.uniform(t_lo, t_ceil)
    half = rng.uniform(*cfg.inner_half_size_range)
    return float(half), float(height), float(thickness)


def _sample_wall_thin_first(difficulty: float, rng: np.random.Generator, cfg: MeshConcentricWallsTerrainCfg):
    """``sample_wall`` for ``order="thin_first"``.

    An easy tile is a frontier tile of a lower difficulty drawn uniformly from ``[0, difficulty]``,
    so it can come from any stage already passed: low thick walls, 0.4 m walls of any thickness, and
    thin walls lower than the frontier."""
    if rng.uniform() < cfg.easy_fraction:
        difficulty = rng.uniform(0.0, difficulty)
    i, u = _stage(difficulty, cfg)
    h_hi, t_lo = _frontier_thin_first(i, u, cfg)
    lowest = cfg.height_path[0]
    if i < 2:
        height = rng.uniform(max(lowest, h_hi - cfg.frontier_band_height), h_hi)
        thickness = cfg.thickness_thick
    elif i == 2:
        height = rng.uniform(cfg.height_stage_thin_min, h_hi)
        thickness = rng.uniform(t_lo, min(cfg.thickness_thick, t_lo + cfg.frontier_band_thickness))
    else:
        height = rng.uniform(max(cfg.height_path[2], h_hi - cfg.frontier_band_height), h_hi)
        thickness = rng.uniform(cfg.thickness_thin, cfg.thickness_thin + cfg.frontier_band_thickness)
    half = rng.uniform(*cfg.inner_half_size_range)
    return float(half), float(height), float(thickness)


def ring_boxes(center: tuple[float, float], half: float, height: float, thickness: float) -> list[trimesh.Trimesh]:
    """Four boxes forming a square ring whose *inner* faces are ``half`` from ``center``.

    The two walls along x span the full outer width so the corners are closed; the two along y fit
    between them. Boxes overlap nothing, so no face is buried inside another."""
    cx, cy = center
    outer = half + thickness
    mid = half + 0.5 * thickness
    z = 0.5 * height
    specs = [
        ((2.0 * outer, thickness, height), (cx, cy + mid, z)),
        ((2.0 * outer, thickness, height), (cx, cy - mid, z)),
        ((thickness, 2.0 * half, height), (cx + mid, cy, z)),
        ((thickness, 2.0 * half, height), (cx - mid, cy, z)),
    ]
    return [trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)) for dim, pos in specs]


def concentric_walls_terrain(
    difficulty: float, cfg: MeshConcentricWallsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat ground with ``num_rings`` square rings of wall around the tile centre."""
    global _last_wall
    # Seeded from the difficulty (which carries the generator's per-tile random offset) and the
    # generator seed, so the same tile comes out identically with or without the terrain cache.
    key = f"{difficulty!r}:{getattr(cfg, 'seed', None)!r}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.md5(key).digest()[:8], "little"))

    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    ground_depth = 1.0
    meshes = [
        trimesh.creation.box(
            (cfg.size[0], cfg.size[1], ground_depth),
            trimesh.transformations.translation_matrix((center[0], center[1], -0.5 * ground_depth)),
        )
    ]
    half, height, thickness = sample_wall(difficulty, rng, cfg)
    _last_wall = (half, height, thickness)
    ring_half, ring_height, ring_thickness = half, height, thickness
    for ring in range(cfg.num_rings):
        if ring > 0:
            ring_half = ring_half + ring_thickness + rng.uniform(*cfg.ring_gap_range)
            _, ring_height, ring_thickness = sample_wall(difficulty, rng, cfg)
        if ring_half + ring_thickness > 0.5 * min(cfg.size) - cfg.edge_margin:
            break
        meshes += ring_boxes(center, ring_half, ring_height, ring_thickness)
    origin = np.array([center[0], center[1], 0.0])
    return meshes, origin


@configclass
class MeshConcentricWallsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`concentric_walls_terrain`."""

    function = concentric_walls_terrain

    stage_rows: tuple[int, ...] = (10, 10, 14, 24)
    """Rows given to each stage: A0, A, B (height) and C (thickness). With the defaults each row moves
    height by 2 cm (A0, A), 2.9 cm (B) or thickness by 2 cm (C). ``num_rows`` of the generator must
    equal the sum."""

    height_path: tuple[float, ...] = (0.01, 0.20, 0.40, 0.80)
    """Frontier height at the start of each height stage, and at the end of the last one.

    A0 (0.01 -> 0.20 m) was added after the first full run: started from the hind-stance expert on
    0.20 m walls, 3000 iterations produced no crossing at all (0 of 1002 jumps on evaluation) -- the
    expert has never come down off its hind legs, so there was nothing for the wall rows to build on.
    A 1 cm wall is a line on the floor: crossing it is coming down onto four legs and walking on."""

    order: str = "height_first"
    """``"height_first"``: the stages in the table above (A0, A, B, C). ``"thin_first"``: A0, A,
    then the wall is thinned at 0.30-0.40 m (C') before it is raised with thin walls (B'); with this
    order ``stage_rows`` lists A0, A, C', B'. run4 (height first) learned to climb onto the top of
    the 0.5 m walls and walk over it: on 0.71 m walls 7 cm thick its base reached the top edge
    (apex minus wall p90 +0.08 m) but only 8% of its jumps got over, where the thick walls of the
    same height gave 14-24%. A thin wall has no top to climb onto, so it is given before the climb
    on high thick walls is learned."""

    height_stage_c_min: float = 0.60
    height_stage_thin_min: float = 0.30
    """``thin_first`` only: lowest wall of stage C', which is thinned with heights up to
    ``height_path[2]``."""
    thickness_thick: float = 0.50
    thickness_thin: float = 0.03

    easy_fraction: float = 0.5
    """Share of tiles drawn from the whole range already covered instead of the frontier band. Was
    0.3 in run3, which by its end had lost the lowest walls: on the 1 cm row the share of jumps that
    crossed fell from 96% (model_3000) to 40% (model_4498), the rest coming down badly."""
    frontier_band_height: float = 0.04
    frontier_band_thickness: float = 0.04

    inner_half_size_range: tuple[float, float] = (1.0, 2.0)
    """Distance from the spawn point to the ring's inner face. The robot rises in 0.3-0.4 s and walks
    out at about 60% of its command in the hind stance. At 2.5 m two thirds of the first run's jumps
    were forced by the time limit, still ~0.9 m short of the wall."""

    num_rings: int = 1
    ring_gap_range: tuple[float, float] = (1.5, 3.0)
    edge_margin: float = 1.0
    """Clear floor kept between the outermost wall and the tile edge, so a robot landing past the
    wall comes down on flat ground. Neighbouring tiles add their own margin on the other side."""


class WallTerrainGenerator(TerrainGenerator):
    """Terrain generator that also records each tile's wall in :data:`WALL_TABLE`."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        global WALL_TABLE
        global TERRAIN_MESH
        WALL_TABLE = np.full((cfg.num_rows, cfg.num_cols, 3), np.nan)
        super().__init__(cfg, device)
        TERRAIN_MESH = self.terrain_mesh

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        global _last_wall
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if isinstance(sub_terrain_cfg, MeshConcentricWallsTerrainCfg) and _last_wall is not None:
            WALL_TABLE[row, col] = _last_wall
        _last_wall = None


THIN_FIRST_WALLS_CFG = MeshConcentricWallsTerrainCfg(proportion=1.0, order="thin_first", stage_rows=(10, 10, 24, 14))
"""Stages A0, A, C', B' at 2 cm (A0, A, C') and 2.9 cm (B') per row, the same 58 rows as the default."""


WALL_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=WallTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=sum(MeshConcentricWallsTerrainCfg().stage_rows),
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    # Off: a cached tile skips the generating function, so WALL_TABLE would not be filled.
    use_cache=False,
    curriculum=True,
    sub_terrains={"walls": MeshConcentricWallsTerrainCfg(proportion=1.0)},
)


WALL_TERRAIN_THIN_FIRST_CFG = WALL_TERRAIN_CFG.replace(
    num_rows=sum(THIN_FIRST_WALLS_CFG.stage_rows), sub_terrains={"walls": THIN_FIRST_WALLS_CFG}
)
