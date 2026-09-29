"""Frontier selection with failure memory. Pure logic -- no ROS.

Combines the value score (M5: VLM; M4: uniform) with travel cost, and remembers
where navigation failed so those spots are never targeted again. Entries come
from exactly one event -- Nav2 aborting a goal -- and are PERMANENT: a failed
goal is written off for the run. This is what lets "no frontiers left (outside
the blacklist)" be the exploration's single termination concept.

The one amnesty is ``clear_since``/``clear_recent``: goals that failed while the
robot's own position was sealed (every plan insta-failing because of the START,
not the goal) say nothing about the goals -- the escape that frees the robot
drops those entries again.
"""

from __future__ import annotations

import numpy as np


class FrontierSelector:
    def __init__(self, blacklist_radius_m: float = 0.15):
        self._blacklist: list[tuple[float, float, float]] = []  # (x, y, created)
        self._radius = blacklist_radius_m

    def blacklist(self, x: float, y: float, now: float) -> None:
        self._blacklist.append((x, y, now))

    def clear_since(self, t: float) -> None:
        """Amnesty: forget the failures recorded at/after ``t``."""
        self._blacklist = [b for b in self._blacklist if b[2] < t]

    def clear_recent(self, now: float, window_s: float) -> None:
        """Amnesty: forget the failures of the last ``window_s`` seconds."""
        self.clear_since(now - window_s)

    def entries(self) -> list[tuple[float, float]]:
        """Blacklist centers, for display."""
        return [(x, y) for x, y, _ in self._blacklist]

    @property
    def radius(self) -> float:
        return self._radius

    def _blocked(self, xy: np.ndarray) -> np.ndarray:
        if not self._blacklist or len(xy) == 0:
            return np.zeros(len(xy), dtype=bool)
        bl = np.array([(bx, by) for bx, by, _ in self._blacklist])
        d = np.hypot(xy[:, 0, None] - bl[None, :, 0], xy[:, 1, None] - bl[None, :, 1])
        return (d < self._radius).any(axis=1)

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
        there and re-targeting it would complete instantly and loop. Anything
        beyond it is eligible however close.

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
