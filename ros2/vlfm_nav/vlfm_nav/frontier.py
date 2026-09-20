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


def find_clusters(grid: np.ndarray, min_cells: int = 8) -> list[Cluster]:
    """Frontier cells, 8-connected into clusters of at least ``min_cells``."""
    free = (grid >= 0) & (grid <= FREE_MAX)
    unknown = grid == -1
    frontier = free & ndimage.binary_dilation(unknown, structure=_EIGHT)
    labels, n = ndimage.label(frontier, structure=_EIGHT.astype(int))
    clusters = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) < min_cells:
            continue
        cells = np.stack([ys, xs], axis=1)
        clusters.append(Cluster(cells=cells, centroid=cells.mean(axis=0)))
    return clusters


def clearance_cells(grid: np.ndarray) -> np.ndarray:
    """Per-cell distance (in cells) to the nearest occupied cell."""
    occ = grid >= OCC_MIN
    return ndimage.distance_transform_edt(~occ)


def goal_for_cluster(
    cluster: Cluster,
    grid: np.ndarray,
    clearance: np.ndarray,
    min_clearance_cells: float,
    search_radius_cells: int,
) -> np.ndarray | None:
    """A navigable stand-in for the cluster: the free cell with enough clearance
    nearest to the cluster centroid, searched within ``search_radius_cells``.

    Returns (row, col) or None when the whole neighborhood is cramped -- such a
    cluster (e.g. a sliver of frontier in a wall gap) is not worth a goal.
    """
    h, w = grid.shape
    cy, cx = cluster.centroid
    y0, y1 = max(0, int(cy) - search_radius_cells), min(h, int(cy) + search_radius_cells + 1)
    x0, x1 = max(0, int(cx) - search_radius_cells), min(w, int(cx) + search_radius_cells + 1)
    sub = grid[y0:y1, x0:x1]
    ok = (sub >= 0) & (sub <= FREE_MAX) & (clearance[y0:y1, x0:x1] >= min_clearance_cells)
    if not ok.any():
        return None
    ys, xs = np.nonzero(ok)
    d2 = (ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2
    k = int(np.argmin(d2))
    return np.array([ys[k] + y0, xs[k] + x0])
