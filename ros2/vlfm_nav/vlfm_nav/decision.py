"""Frontier selection with failure memory. Pure logic -- no ROS.

Combines the value score (M5: VLM; M4: uniform) with travel cost, and remembers
where navigation recently failed so one unreachable frontier cannot deadlock the
exploration loop.
"""

from __future__ import annotations

import numpy as np


class FrontierSelector:
    def __init__(self, blacklist_radius_m: float = 0.8, blacklist_ttl_s: float = 180.0):
        self._blacklist: list[tuple[float, float, float]] = []  # (x, y, expiry_time)
        self._radius = blacklist_radius_m
        self._ttl = blacklist_ttl_s

    def blacklist(self, x: float, y: float, now: float) -> None:
        self._blacklist.append((x, y, now + self._ttl))

    def _blocked(self, xy: np.ndarray, now: float) -> np.ndarray:
        self._blacklist = [b for b in self._blacklist if b[2] > now]
        if not self._blacklist:
            return np.zeros(len(xy), dtype=bool)
        bl = np.array([(bx, by) for bx, by, _ in self._blacklist])
        d = np.hypot(xy[:, 0, None] - bl[None, :, 0], xy[:, 1, None] - bl[None, :, 1])
        return (d < self._radius).any(axis=1)

    def choose(
        self,
        goals_xy: np.ndarray,
        values: np.ndarray,
        robot_xy: np.ndarray,
        now: float,
        min_goal_distance_m: float = 0.6,
    ) -> int | None:
        """Index of the frontier goal to pursue, or None if none is eligible.

        Score = value - distance-cost. With uniform (zero) values this is
        nearest-first; a VLM value in [0, 1] outweighs up to ~10 m of detour
        (cost = 0.1/m), the same order of trade-off VLFM's value map induces.
        """
        if len(goals_xy) == 0:
            return None
        d = np.hypot(goals_xy[:, 0] - robot_xy[0], goals_xy[:, 1] - robot_xy[1])
        eligible = (~self._blocked(goals_xy, now)) & (d >= min_goal_distance_m)
        if not eligible.any():
            return None
        score = values - 0.1 * d
        score[~eligible] = -np.inf
        return int(np.argmax(score))
