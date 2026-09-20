"""Frontier scoring. M4 stub; M5 replaces the scorer with the VLM value map.

The contract the VLM version will honor: ``score(points_xy) -> np.ndarray`` of
values in [0, 1], higher = more promising. The decision layer combines this
with travel cost; a scorer returning all-zeros therefore degrades exactly to
nearest-frontier exploration, which is the M4 milestone.

M5 plan (paper: VLFM section IV-B): a BLIP-2 ITM scorer runs in a separate
process behind a ROS service, projecting per-image cosine scores into a
top-down value map by camera frustum, confidence-averaged over observations.
"""

from __future__ import annotations

import numpy as np


class UniformValueMap:
    """All frontiers equally valuable -> selection reduces to nearest-first."""

    def score(self, points_xy: np.ndarray) -> np.ndarray:
        return np.zeros(len(points_xy))
