"""Frontier selection with two kinds of memory. Pure logic -- no ROS.

BLACKLIST -- failures. A goal Nav2 aborted (with a feasible start) is written
off: a permanent small disc (blacklist_radius_m) that goal SELECTION skips.

VISITED -- successes. A goal the robot actually arrived at is done: its
neighborhood is excluded from FRONTIER DETECTION itself (see
frontier.exclusion_mask, radius = the goal's consumption radius), so the area
never yields frontier cells or goals again. Stronger than a blacklist entry on
purpose: re-placement jitter cannot resurrect a goal next to a visited spot.

The one amnesty is ``clear_recent`` and applies to the BLACKLIST only:
failures accrued while the robot's own position was temporarily bad (sealed
start) say nothing about the goals. Visits are definitive and never amnestied.
"""

from __future__ import annotations

import numpy as np


class FrontierSelector:
    def __init__(self, blacklist_radius_m: float = 0.15):
        self._blacklist: list[tuple[float, float, float]] = []  # failures (x, y, created)
        self._visited: list[tuple[float, float, float]] = []  # arrivals (x, y, created)
        self._radius = blacklist_radius_m

    # ---------------------------------------------------------------- recording
    def blacklist(self, x: float, y: float, now: float) -> None:
        self._blacklist.append((x, y, now))

    def visit(self, x: float, y: float, now: float) -> None:
        self._visited.append((x, y, now))

    def clear_recent(self, now: float, window_s: float) -> int:
        """Amnesty: forget the blacklist entries of the last ``window_s`` seconds.
        Returns how many were dropped. Visits are never amnestied."""
        before = len(self._blacklist)
        self._blacklist = [e for e in self._blacklist if e[2] < now - window_s]
        return before - len(self._blacklist)

    # ------------------------------------------------------------------ queries
    def blacklist_entries(self) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in self._blacklist]

    def visited_entries(self) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in self._visited]

    @property
    def radius(self) -> float:
        return self._radius

    def _blocked(self, xy: np.ndarray) -> np.ndarray:
        if not self._blacklist or len(xy) == 0:
            return np.zeros(len(xy), dtype=bool)
        bl = np.array([(bx, by) for bx, by, _ in self._blacklist])
        d = np.hypot(xy[:, 0, None] - bl[None, :, 0], xy[:, 1, None] - bl[None, :, 1])
        return (d < self._radius).any(axis=1)

    # ---------------------------------------------------------------- selection
    def choose(
        self,
        goals_xy: np.ndarray,
        values: np.ndarray,
        robot_xy: np.ndarray,
        reached_radius_m: float,
    ) -> int | None:
        """Index of the frontier goal to pursue, or None if none is eligible.

        Goals closer than ``reached_radius_m`` are skipped -- not as a preference,
        but by definition: that is the arrival radius, so the robot already stands
        there and re-targeting it would complete instantly and loop.

        Score = value - distance-cost. With uniform (zero) values this is
        nearest-first; a VLM value in [0, 1] outweighs up to ~10 m of detour
        (cost = 0.1/m), the same order of trade-off VLFM's value map induces.
        """
        if len(goals_xy) == 0:
            return None
        d = np.hypot(goals_xy[:, 0] - robot_xy[0], goals_xy[:, 1] - robot_xy[1])
        eligible = (~self._blocked(goals_xy)) & (d >= reached_radius_m)
        if not eligible.any():
            return None
        score = values - 0.1 * d
        score[~eligible] = -np.inf
        return int(np.argmax(score))
