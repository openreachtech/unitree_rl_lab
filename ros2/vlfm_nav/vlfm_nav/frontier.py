"""Frontier detection on an occupancy grid. Pure numpy/scipy -- no ROS.

Grid convention (nav_msgs/OccupancyGrid.data reshaped to (H, W)):
  -1 unknown, 0..100 occupancy probability. Free <= FREE_MAX, occupied >= OCC_MIN.

A *frontier* cell is a free cell with an unknown 8-neighbor -- the boundary the
robot has to cross to see new space (Yamauchi 1997; VLFM section IV-A). Clusters
of frontier cells are the exploration candidates; each gets a *navigation goal*
placed in nearby well-cleared free space, because the frontier itself hugs the
unknown (and often the inflation layer), where Nav2 goals go to die -- the M3
lesson, kept here as logic instead of tribal knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FREE_MAX = 10
OCC_MIN = 65

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Cluster:
    """One frontier cluster, in grid cells (row, col)."""

    cells: np.ndarray  # (N, 2) [row, col]
    centroid: np.ndarray  # (2,) [row, col], float

    @property
    def size(self) -> int:
        return len(self.cells)


def exclusion_mask(shape: tuple[int, int], cells: list[tuple[int, int]], radius_cells: int) -> np.ndarray:
    """Boolean mask with a filled disc of ``radius_cells`` around each (row, col).

    Cells inside are removed from the frontier before clustering: the area around
    a visited or written-off goal never yields frontier cells or goals again.
    """
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    rr = radius_cells
    yy, xx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
    disc = (yy * yy + xx * xx) <= rr * rr
    for row, col in cells:
        r0, r1 = max(0, row - rr), min(h, row + rr + 1)
        c0, c1 = max(0, col - rr), min(w, col + rr + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        mask[r0:r1, c0:c1] |= disc[r0 - (row - rr) : r1 - (row - rr), c0 - (col - rr) : c1 - (col - rr)]
    return mask


def find_clusters(
    grid: np.ndarray, min_cells: int = 8, exclusion: np.ndarray | None = None
) -> list[Cluster]:
    """Frontier cells, 8-connected into clusters of at least ``min_cells``.
    Cells under ``exclusion`` (see :func:`exclusion_mask`) are not frontier."""
    free = (grid >= 0) & (grid <= FREE_MAX)
    unknown = grid == -1
    frontier = free & ndimage.binary_dilation(unknown, structure=_EIGHT)
    if exclusion is not None:
        frontier &= ~exclusion
    labels, n = ndimage.label(frontier, structure=_EIGHT.astype(int))
    clusters = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) < min_cells:
            continue
        cells = np.stack([ys, xs], axis=1)
        clusters.append(Cluster(cells=cells, centroid=cells.mean(axis=0)))
    return clusters


def has_frontier_near(grid: np.ndarray, row: int, col: int, radius_cells: int) -> bool:
    """Is there any frontier cell within ``radius_cells`` (square window) of (row, col)?

    Used while NAVIGATING: a goal exists to look past a frontier, so once no
    frontier is left near it -- consumed by the scans made en route -- the goal
    has served its purpose without being reached. The window is padded by one
    cell so the free/unknown adjacency at the window edge is judged correctly.
    """
    h, w = grid.shape
    r0, r1 = max(0, row - radius_cells - 1), min(h, row + radius_cells + 2)
    c0, c1 = max(0, col - radius_cells - 1), min(w, col + radius_cells + 2)
    sub = grid[r0:r1, c0:c1]
    free = (sub >= 0) & (sub <= FREE_MAX)
    unknown = sub == -1
    return bool((free & ndimage.binary_dilation(unknown, structure=_EIGHT)).any())


def clearance_cells(grid: np.ndarray) -> np.ndarray:
    """Per-cell distance (in cells) to the nearest occupied cell."""
    occ = grid >= OCC_MIN
    return ndimage.distance_transform_edt(~occ)


def goal_for_cluster(cluster: Cluster, near: np.ndarray | None = None) -> np.ndarray:
    """The frontier cell to aim the robot at: the cluster cell closest to ``near``
    (the robot), or -- with no ``near`` -- the one closest to the centroid.

    The goal IS a frontier cell, not a cleared stand-in offset from it. The robot
    never has to stand on it: it either consumes the frontier en route (the scans
    reveal what was behind it) or arrives within goal_reached_m, both well before
    the goal cell itself. Aiming straight at the frontier keeps goal and frontier
    at the SAME point, so the consumed check and the visited-retire disc are both
    centred on the frontier and need only a small radius -- no goal<->frontier gap
    to span, which is what made those two radii so fragile when the goal was offset
    (2026-09-29). A frontier that hugs a wall lands the goal in inflation and Nav2
    aborts it -> blacklist, which is the right call for an unreachable pocket.

    Anchoring on the nearest cell (not the centroid) matters right after the spin:
    the frontier is then a RING around the robot whose centroid is the robot
    itself, and a goal there deadlocks before the first move (2026-09-28)."""
    ref = near if near is not None else cluster.centroid
    k = int(np.argmin(((cluster.cells - ref) ** 2).sum(axis=1)))
    return cluster.cells[k]
