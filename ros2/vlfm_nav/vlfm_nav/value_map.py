"""Language-grounded value map. Pure logic -- no ROS, no model.

The contract the decision layer sees is unchanged: ``score(points_xy) -> np.ndarray`` of
values in [0, 1], higher = more promising. Scoring all-zeros degrades exactly to
nearest-frontier exploration, which is the M4 milestone and what runs when no VLM is up
-- ``GridValueMap`` does that by itself while it is waiting for its first grid, so there
is no separate placeholder class and no switch to throw.

``ValueMap`` is the M5 implementation. See doc/design/vlfm_nav.md 7.5 for the design and
the measurements behind it; the short version:

**The whole trick is that the camera pose is known, so "what is visible right now" is a
fan on the map.** A score computed from the image is painted over that fan. VLFM paints
one scalar over the whole cone; here the 90 deg view is cut into four 22.5 deg
sub-fans, each carrying the score of its own image strip -- four times the angular
resolution for the same inference.

Two details that are not optional:

* **The fan is cut where the view is blocked**, by raycasting the occupancy grid (not a
  depth image -- the papers used one because a Habitat agent has neither a LiDAR nor a
  trustworthy global map, and we have both). Without the cut, "the next room looks like
  a bathroom" bleeds through walls and frontier selection breaks.
* **Cells are blended by confidence, not overwritten.** Confidence falls off from the
  optical axis as cos^2, so something glimpsed at the edge of the view does not
  overwrite what was seen head-on (VLFM IV-B).

Values go in raw. SigLIP's sigmoid is already 0-1 but heavily skewed low -- measured
2026-10-01, a frame-filling bed scores 0.46 on its strip and the median frame scores
0.000 -- so ``score()`` normalises against the run's own observed spread at read time.
Normalising at write time instead would make old paint and new paint disagree whenever
the spread moved.
"""

from __future__ import annotations

import math

import numpy as np


class GridValueMap:
    """The consumer side: score frontiers from the grids the VLM node publishes.

    The scorer lives in the VLM process because that is where the camera, the model and
    the raycast are. The exploration node only needs to read the result, and it reads it
    off the wire as two ``nav_msgs/OccupancyGrid`` -- the same two topics Foxglove draws,
    so what steers the robot and what a human checks are by construction the same thing.

    Until the first grid arrives (no VLM running, or it is still loading a model) every
    point scores ``neutral``, which is 0, so selection falls through to distance alone.
    Running without the VLM therefore needs no switch: the stack degrades to
    nearest-first, which is the M4 behaviour.
    """

    def __init__(self, radius_m: float = 0.5, neutral: float = 0.0):
        self.radius = radius_m
        self.neutral = neutral
        self._val = None  # (array 0..100 with -1 unobserved, origin, resolution)
        self._cnf = None

    def set_value(self, grid, origin, resolution) -> None:
        self._val = (np.asarray(grid), np.asarray(origin, dtype=float), float(resolution))

    def set_confidence(self, grid, origin, resolution) -> None:
        self._cnf = (np.asarray(grid), np.asarray(origin, dtype=float), float(resolution))

    def score(self, points_xy: np.ndarray) -> np.ndarray:
        """Confidence-weighted mean of the published value around each point.

        Mirrors ``ValueMap.score``: a radius because a frontier cell jitters as the map
        grows, a weighted mean because a max chases single-cell noise. Weighting needs
        the confidence layer, so without it this stays neutral rather than guessing --
        an unweighted mean would let one glancing look outvote a long hard stare.
        """
        pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
        out = np.full(len(pts), self.neutral)
        if self._val is None or self._cnf is None or len(pts) == 0:
            return out
        vg, vo, vr = self._val
        cg, co, cr = self._cnf
        if vg.shape != cg.shape:
            return out
        k = int(math.ceil(self.radius / vr))
        h, w = vg.shape
        cols = np.floor((pts[:, 0] - vo[0]) / vr).astype(int)
        rows = np.floor((pts[:, 1] - vo[1]) / vr).astype(int)
        for n, (r, c) in enumerate(zip(rows, cols)):
            rs, re = max(0, r - k), min(h, r + k + 1)
            cs, ce = max(0, c - k), min(w, c + k + 1)
            if rs >= re or cs >= ce:
                continue
            v = vg[rs:re, cs:ce].astype(float)
            q = cg[rs:re, cs:ce].astype(float)
            m = (v >= 0) & (q > 0)
            if not m.any():
                continue
            wsum = q[m].sum()
            if wsum <= 0:
                continue
            out[n] = float((v[m] * q[m]).sum() / wsum) / 100.0
        return out


class ValueMap:
    """A fixed-origin grid of (value, confidence), painted from camera observations.

    Fixed origin on purpose: ``/map`` resizes itself as SLAM explores (the
    ``StaticLayer: Resizing costmap`` lines in any run log), and following that is a
    bug farm. One generous grid allocated up front costs nothing -- 60 x 60 m at 0.2 m
    is 300 x 300 x 2 channels of float32, about 720 kB.

    Coarse on purpose too: this only ranks frontiers that are metres apart, so 0.2 m
    cells are plenty. The occupancy grid it raycasts against stays at its own (finer)
    resolution.
    """

    def __init__(
        self,
        size_m: float = 60.0,
        resolution: float = 0.2,
        origin: tuple[float, float] = (-30.0, -30.0),
        hfov_rad: float = math.pi / 2,
        n_strips: int = 4,
        max_range_m: float = 5.0,
        n_rays: int = 90,
    ):
        self.res = resolution
        self.origin = np.asarray(origin, dtype=float)
        self.n = int(round(size_m / resolution))
        self.hfov = hfov_rad
        self.n_strips = n_strips
        self.max_range = max_range_m
        self.n_rays = n_rays
        self.value = np.zeros((self.n, self.n), dtype=np.float32)
        self.conf = np.zeros((self.n, self.n), dtype=np.float32)
        self._observed: list[float] = []  # raw scores written, for read-time normalising

        # Cell-centre coordinates, built once. Indexing is [row=y, col=x].
        ax = self.origin[0] + (np.arange(self.n) + 0.5) * resolution
        ay = self.origin[1] + (np.arange(self.n) + 0.5) * resolution
        self._cx, self._cy = np.meshgrid(ax, ay)

    # ------------------------------------------------------------------ helpers
    def _idx(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (x, y) -> (row, col). Out-of-range indices are clipped by the caller."""
        ij = np.floor((np.atleast_2d(xy) - self.origin) / self.res).astype(int)
        return ij[:, 1], ij[:, 0]

    def clear(self) -> None:
        """Throw the map away.

        Called when ``map->odom`` jumps. Everything here is painted in the map frame, so
        a loop closure invalidates all of it at once -- the same failure that once
        painted 52 m^2 of free space lethal in the global costmap (see nav2_common.yaml).
        Cheap to accept: this map only orders frontiers, and a few observations rebuild
        it. The object goal is NOT handled this way; it is held in the odom frame so a
        correction moves it instead of ruining it.
        """
        self.value[:] = 0.0
        self.conf[:] = 0.0

    def transform(self, cx: float, cy: float, ctheta: float) -> None:
        """Rigidly move the painted map by a map-frame correction, instead of binning it.

        Clearing on every loop closure sounds safe and is not: measured over the
        2026-10-01 run, slam_toolbox moved map->odom past the threshold four times in 25
        minutes, so the map never got to accumulate -- 36,758 cell-writes went in and
        107 cells came out. The correction is known, so apply it.

        The honest caveat: a loop closure is not a rigid motion of the world. It
        redistributes error across the pose graph, so distant, old paint is moved
        approximately at best. It is still far better than zero, and the recent paint --
        the part that actually decides where to go next -- moves almost exactly right.
        Nearest-neighbour resampling, because the grid is 0.2 m and only ranks frontiers.
        """
        if abs(cx) < 1e-6 and abs(cy) < 1e-6 and abs(ctheta) < 1e-6:
            return
        # For each destination cell, find where it came from: source = C^-1 * dest.
        c, s_ = math.cos(-ctheta), math.sin(-ctheta)
        dx, dy = self._cx - cx, self._cy - cy
        sx = c * dx - s_ * dy
        sy = s_ * dx + c * dy
        si = np.floor((sx - self.origin[0]) / self.res).astype(int)
        sj = np.floor((sy - self.origin[1]) / self.res).astype(int)
        ok = (si >= 0) & (si < self.n) & (sj >= 0) & (sj < self.n)
        v = np.zeros_like(self.value)
        k = np.zeros_like(self.conf)
        v[ok] = self.value[sj[ok], si[ok]]
        k[ok] = self.conf[sj[ok], si[ok]]
        self.value, self.conf = v, k

    # ------------------------------------------------------------------ raycast
    def visible_range(
        self,
        cam_xy: np.ndarray,
        cam_yaw: float,
        occ: np.ndarray,
        occ_origin: tuple[float, float],
        occ_res: float,
        occupied_min: int = 50,
        unknown_is_blocking: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """How far the camera can see in each bearing. Returns (bearings, r_max).

        Marches the occupancy grid from the camera along ``n_rays`` bearings spanning the
        HFOV, stopping at the first occupied cell -- and at the first UNKNOWN cell too,
        which is what makes a frontier the far edge of its own fan rather than something
        the paint skips over.

        ``occ`` is the ROS OccupancyGrid convention: -1 unknown, 0..100 occupancy.
        """
        bearings = cam_yaw + np.linspace(-self.hfov / 2, self.hfov / 2, self.n_rays)
        steps = np.arange(1, int(self.max_range / occ_res) + 1) * occ_res
        # (rays, steps) sample points along every ray at once
        px = cam_xy[0] + np.cos(bearings)[:, None] * steps[None, :]
        py = cam_xy[1] + np.sin(bearings)[:, None] * steps[None, :]
        ci = np.floor((px - occ_origin[0]) / occ_res).astype(int)
        cj = np.floor((py - occ_origin[1]) / occ_res).astype(int)
        h, w = occ.shape
        inside = (ci >= 0) & (ci < w) & (cj >= 0) & (cj < h)
        vals = np.full(ci.shape, -1, dtype=np.int16)
        vals[inside] = occ[cj[inside], ci[inside]]
        blocked = vals >= occupied_min
        if unknown_is_blocking:
            blocked |= vals < 0
        blocked |= ~inside
        # First blocked step per ray; rays that never block see to max_range.
        any_block = blocked.any(axis=1)
        first = np.where(any_block, blocked.argmax(axis=1), len(steps) - 1)
        r_max = steps[first]
        return bearings, r_max

    # -------------------------------------------------------------------- paint
    def paint(
        self,
        cam_xy: np.ndarray,
        cam_yaw: float,
        strip_scores: np.ndarray,
        occ: np.ndarray,
        occ_origin: tuple[float, float],
        occ_res: float,
    ) -> int:
        """Write one observation into the map. Returns how many cells were touched.

        ``strip_scores`` is left-to-right **in the image**. That ordering matters and is
        easy to get backwards: image x grows rightward, bearing grows counter-clockwise
        (REP-103), so the LEFTMOST strip is the LARGEST bearing. Getting this wrong
        mirrors the value map about the optical axis and is invisible until the robot
        reliably walks away from the thing it is looking for.
        """
        strip_scores = np.asarray(strip_scores, dtype=float)
        assert len(strip_scores) == self.n_strips

        bearings, r_max = self.visible_range(cam_xy, cam_yaw, occ, occ_origin, occ_res)

        # Only the window within max_range can be touched; slicing it keeps the whole
        # update at a few thousand cells no matter how big the grid is.
        pad = int(math.ceil(self.max_range / self.res)) + 1
        r0, c0 = self._idx(cam_xy)
        r0, c0 = int(r0[0]), int(c0[0])
        rs, re = max(0, r0 - pad), min(self.n, r0 + pad + 1)
        cs, ce = max(0, c0 - pad), min(self.n, c0 + pad + 1)
        if rs >= re or cs >= ce:
            return 0

        dx = self._cx[rs:re, cs:ce] - cam_xy[0]
        dy = self._cy[rs:re, cs:ce] - cam_xy[1]
        rng = np.hypot(dx, dy)
        dbear = np.arctan2(dy, dx) - cam_yaw
        dbear = (dbear + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]

        in_fov = np.abs(dbear) <= self.hfov / 2
        # Line of sight: compare each cell's range against r_max at its own bearing.
        bin_i = np.clip(
            ((dbear + self.hfov / 2) / self.hfov * (self.n_rays - 1)).astype(int),
            0, self.n_rays - 1,
        )
        in_los = rng <= r_max[bin_i]
        mask = in_fov & in_los & (rng > 1e-6)
        if not mask.any():
            return 0

        # Leftmost image strip = largest bearing; see the docstring.
        strip_i = np.clip(
            ((self.hfov / 2 - dbear) / (self.hfov / self.n_strips)).astype(int),
            0, self.n_strips - 1,
        )
        s_curr = strip_scores[strip_i]

        # Confidence: 1 on the optical axis, 0 at the edge of the view (VLFM IV-B).
        c_curr = np.cos(dbear / (self.hfov / 2) * (math.pi / 2)) ** 2

        v_prev = self.value[rs:re, cs:ce]
        c_prev = self.conf[rs:re, cs:ce]
        denom = c_curr + c_prev
        safe = mask & (denom > 1e-9)
        v_new = np.where(safe, (c_curr * s_curr + c_prev * v_prev) / np.where(safe, denom, 1.0), v_prev)
        # Confidence is biased toward the higher of the two, so a place seen head-on
        # once is not talked down by later glimpses from the edge of the view.
        c_new = np.where(safe, (c_curr**2 + c_prev**2) / np.where(safe, denom, 1.0), c_prev)
        self.value[rs:re, cs:ce] = v_new
        self.conf[rs:re, cs:ce] = c_new

        self._observed.extend(float(s) for s in strip_scores)
        if len(self._observed) > 20000:
            del self._observed[:10000]
        return int(safe.sum())

    # ------------------------------------------------------------------- sample
    def score(
        self,
        points_xy: np.ndarray,
        radius_m: float = 0.5,
        min_conf: float = 0.05,
        neutral: float = 0.0,
    ) -> np.ndarray:
        """Confidence-weighted value around each point, normalised to [0, 1].

        A radius rather than one cell because a frontier cell jitters as the map grows;
        a weighted mean rather than a max because a max chases single-cell noise.

        Points whose neighbourhood was never looked at come back as ``neutral``, which
        is 0 -- the same as "looked at, nothing there". The camera is 90 deg and the
        LiDAR is 360, so plenty of mapped frontiers have never been in frame, and this
        says the VLM simply has no opinion about them: they compete on distance, which
        is the M4 behaviour. (A positive neutral would instead make "never looked at"
        actively preferable to "looked at and empty", which is a different and much
        pushier exploration policy than this one wants.)
        """
        pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
        out = np.full(len(pts), neutral)
        if len(pts) == 0:
            return out
        k = int(math.ceil(radius_m / self.res))
        rows, cols = self._idx(pts)
        for n, (r, c) in enumerate(zip(rows, cols)):
            rs, re = max(0, r - k), min(self.n, r + k + 1)
            cs, ce = max(0, c - k), min(self.n, c + k + 1)
            if rs >= re or cs >= ce:
                continue
            w = self.conf[rs:re, cs:ce]
            tot = w.sum()
            if tot <= min_conf:
                continue
            out[n] = float((self.value[rs:re, cs:ce] * w).sum() / tot)
        return self._normalise(out, neutral)

    MIN_SPREAD = 0.05
    """Below this, the observed scores are not a spread -- they are all noise.

    Measured over the 2026-10-01 run looking for a toilet: 66% of observed values were
    exactly 0.000, p95 was 0.016, and the single best was 0.523. Stretching a 0.016-wide
    band onto [0, 1] magnifies the noise 60x and paints confident-looking value across
    rooms the target is nowhere near -- which is exactly what the first rendering showed.
    0.05 is half the detection threshold: if nothing in the run has even half-way
    resembled the target, the honest answer is that there is no opinion to give.
    """

    def _normalise(self, raw: np.ndarray, neutral: float) -> np.ndarray:
        """Stretch the run's own observed spread onto [0, 1], if there is one.

        SigLIP's sigmoid is calibrated as "is this caption the match for this image out
        of everything on the web", which puts even a correct, frame-filling match low
        (0.39 on a strip, measured 2026-10-01) and almost everything else at 0.000. Used
        raw against the decision layer's 0.1/m distance cost, every frontier would look
        identical, so some stretching is needed. Percentile ends rather than min/max, so
        one lucky cell cannot define the scale -- and a floor under the spread, so a run
        that has seen nothing does not invent a landscape out of rounding.

        Returning ``neutral`` for everything is not a failure mode: it hands the choice
        back to distance, which is the M4 behaviour and a perfectly good explorer.
        """
        if len(self._observed) < 20:
            return np.full_like(raw, neutral)
        lo, hi = np.percentile(self._observed, [5, 95])
        if hi - lo < self.MIN_SPREAD:
            return np.full_like(raw, neutral)
        scaled = (raw - lo) / (hi - lo)
        # Untouched points keep meaning "no opinion" rather than being stretched too.
        return np.where(raw == neutral, neutral, np.clip(scaled, 0.0, 1.0))

    # -------------------------------------------------------------- publishing
    def as_occupancy(self) -> tuple[np.ndarray, np.ndarray]:
        """(value, confidence) as int8 arrays for nav_msgs/OccupancyGrid.

        0..100 where observed, -1 where never seen -- which Foxglove draws as
        transparent, so the two layers read against ``/map`` instead of hiding it.
        Publishing confidence as well as value is what makes the occlusion cut
        checkable: if the fan is bleeding through a wall, it shows up there first.
        """
        seen = self.conf > 1e-6
        val = np.full(self.value.shape, -1, dtype=np.int8)
        cnf = np.full(self.conf.shape, -1, dtype=np.int8)
        if seen.any():
            v = self.value[seen]
            lo, hi = ((0.0, 0.0) if len(self._observed) < 20
                      else np.percentile(self._observed, [5, 95]))
            if hi - lo < self.MIN_SPREAD:
                # Same floor as score(): with no real spread, drawing one would be a
                # picture of rounding error. Show it flat and let confidence carry the
                # information about where the robot has actually looked.
                val[seen] = 0
            else:
                val[seen] = np.clip((v - lo) / (hi - lo) * 100, 0, 100).astype(np.int8)
            cnf[seen] = np.clip(self.conf[seen] * 100, 0, 100).astype(np.int8)
        return val, cnf
