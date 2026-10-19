"""MID-360 height map for the MuJoCo wall-crossing evaluator (``wallcross_mujoco.py --mid360``).

A numpy re-implementation of the Isaac observation term ``mid360_map`` of the
``Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360`` task:

* sensor  -- ``tasks/wallcross/mid360/rolling_livox_sensor.py`` (+ ``patterns.py``, IsaacLab ``RayCaster``)
* map     -- ``tasks/wallcross/mid360/lidar_elevation_map.py`` (``LidarElevationMap``)
* config  -- ``tasks/wallcross/mid360/__init__.py``

Sensor, as Isaac casts it (RayCaster with ``ray_alignment="base"``):

    window w (one per control step, advancing by one, wrapping at 200 = 800000 / 4000):
        rows w*4000 + 0, 2, 4, ..., 3998 of scan_patterns/mid360.npy ([theta, phi], rad)
    d_sensor = (cos th cos ph, sin th cos ph, sin ph), normalised
    d_world  = R(base quat) @ R(mount quat) @ d_sensor
    origin   = base pos + R(base quat) @ mount pos            (same for every ray)
    hit      = first intersection with the static world (Isaac: /World/ground = terrain mesh incl. the wall;
               here: every geom of group 0 = the floor plane and the wall box -- the robot's geoms are in
               groups 2/3 and are not hit), max distance 20 m, miss = inf

Map (per control step, after the physics):

    1. motion compensation of the held grid (bilinear resample onto the new pose, +dz for measured cells,
       flat_fill for cells that come from outside the old window or were never measured)
    2. bin this step's hits into the 29 x 21 grid around the base (heading = horizontal x_b - z_b),
       value = base_z - hit_z - 0.273175, min per cell (= highest return), unobserved cells keep the held value
    3. output clipped to (-1, 5) (ObsTerm.clip); the held grid itself is not clipped

Window index: Isaac shares one global index between all envs and never resets it with the episode, so the
phase of the scan sequence at the start of an episode is arbitrary. The evaluator picks it per trial
(``window0``); Isaac's sensor starts at ``rolling_window_start // 4000 - 1`` and advances before the
first cast, so the first map of a fresh sim uses window 0.

``min_range`` (0.2 m) only acts inside the noise model, exactly as in Isaac (the noise-free map keeps
returns closer than 0.2 m).
"""
from __future__ import annotations

import math
import os

import numpy as np

import mujoco

REPO = "/home/tanaka/isaacsim/unitree_rl_lab_multitask"
PATTERN = os.path.join(REPO, "source/unitree_rl_lab/unitree_rl_lab/tasks/wallcross/mid360/scan_patterns/mid360.npy")

# tasks/wallcross/mid360/__init__.py
MOUNT_POS = np.array([0.287, 0.0, -0.137])
_S = math.sqrt(0.5)
MOUNT_QUAT = np.array([0.0, _S, _S, 0.0])          # (w, x, y, z)
SAMPLES_PER_STEP = 4000
DOWNSAMPLE = 2
MIN_RANGE = 0.2
MAX_DISTANCE = 20.0
RESOLUTION = 0.05
SIZE = (1.4, 1.0)
OFFSET = 0.273175
FLAT_FILL = 0.32 - OFFSET
CLIP = (-1.0, 5.0)
NUM_X = round(SIZE[0] / RESOLUTION) + 1            # 29
NUM_Y = round(SIZE[1] / RESOLUTION) + 1            # 21
NUM_CELLS = NUM_X * NUM_Y                          # 609
X0 = -SIZE[0] / 2
Y0 = -SIZE[1] / 2
_UNOBSERVED = 1.0e4

# cell centres (yaw-aligned base frame), ix-major (idx = ix * 21 + iy) -- torch.linspace + meshgrid "ij"
CELL_X = np.repeat(np.linspace(X0, X0 + SIZE[0], NUM_X), NUM_Y)
CELL_Y = np.tile(np.linspace(Y0, Y0 + SIZE[1], NUM_Y), NUM_X)

# MID360_NOISE_CFG (weak / nominal / strong), degrees where the cfg says degrees
NOISE = {
    "probability": np.array([0.60, 0.30, 0.10]),
    "range_std": np.array([0.005, 0.01, 0.02]),
    "tilt_step_std": np.radians([0.25, 0.5, 1.0]),
    "tilt_episode_std": np.radians([0.125, 0.25, 0.5]),
    "outlier_prob": np.array([0.0025, 0.005, 0.015]),
    "outlier_range": np.array([0.075, 0.15, 0.30]),
    "odom_xy_step_std": np.array([0.005, 0.01, 0.025]),
    "odom_xy_bias_std": np.array([0.005, 0.01, 0.025]),
    "odom_yaw_step_std": np.radians([0.25, 0.5, 1.5]),
    "odom_yaw_bias_std": np.radians([0.25, 0.5, 1.5]),
}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat(q):
    """lidar_elevation_map._yaw_from_quat: heading of horizontal (x_b - z_b)."""
    w, x, y, z = q
    fx = (1.0 - 2.0 * (y * y + z * z)) - 2.0 * (x * z + w * y)
    fy = 2.0 * (x * y + w * z) - 2.0 * (y * z - w * x)
    return math.atan2(fy, fx)


def load_windows(path=PATTERN):
    """(200, 2000, 3) unit ray directions in the *base* frame (mount rotation applied), as RollingLivoxSensor."""
    seq = np.load(path)                                 # float32 (800000, 2)
    th = seq[:, 0].astype(np.float32)
    ph = seq[:, 1].astype(np.float32)
    d = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], axis=1).astype(np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ quat_to_mat(MOUNT_QUAT).T
    stride = SAMPLES_PER_STEP
    nwin = len(d) // stride
    return np.ascontiguousarray(d[: nwin * stride].reshape(nwin, stride, 3)[:, ::DOWNSAMPLE, :])


def bilinear_zero_pad(img, sx, sy):
    """torch grid_sample(mode=bilinear, padding=zeros, align_corners=True) on an (C, NUM_X, NUM_Y) image at
    fractional pixel coordinates (sx along NUM_X, sy along NUM_Y)."""
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = sx - x0
    fy = sy - y0
    out = np.zeros((img.shape[0], sx.shape[0]))
    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < NUM_X) & (yi >= 0) & (yi < NUM_Y)
            v = np.zeros((img.shape[0], sx.shape[0]))
            v[:, ok] = img[:, xi[ok], yi[ok]]
            out += v * (wx * wy)
    return out


class Mid360Map:
    """One robot's MID-360 sensor + LidarElevationMap. Call ``reset`` at the start of a trial and ``step``
    once per control step (after the physics, i.e. when the observation is built)."""

    def __init__(self, m, base_body, noise=False, motion_compensation=True):
        self.m = m
        self.base = base_body
        self.windows = load_windows()
        self.nwin, self.nray = self.windows.shape[:2]
        self.noise = noise
        self.motion_compensation = motion_compensation
        # only static world geoms of group 0 (floor, wall). go2.xml puts the robot in groups 2 (visual) and 3.
        robot_groups = {int(m.geom_group[g]) for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == base_body}
        assert 0 not in robot_groups, f"robot geoms in group 0: {robot_groups}"
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(self.nray, dtype=np.int32)
        self._dist = np.zeros(self.nray, dtype=np.float64)
        self.reset(0, np.random.default_rng(0))

    # ------------------------------------------------------------------------------------------
    def reset(self, window0, rng):
        self.win = (int(window0) - 1) % self.nwin       # advanced before the first cast
        self.rng = rng
        self.hold = np.full(NUM_CELLS, FLAT_FILL)
        self.measured = np.zeros(NUM_CELLS, dtype=bool)
        self.pose_valid = False
        self.prev_xy = np.zeros(2)
        self.prev_yaw = 0.0
        self.prev_z = 0.0
        self.unobserved = np.ones(NUM_CELLS, dtype=bool)
        if self.noise:
            p = NOISE["probability"] / NOISE["probability"].sum()
            c = int(rng.choice(3, p=p))
            self.cond = c
            self.episode_tilt = rng.standard_normal(2) * NOISE["tilt_episode_std"][c]
            self.odom_bias_xy = rng.standard_normal(2) * NOISE["odom_xy_bias_std"][c]
            self.odom_bias_yaw = float(rng.standard_normal() * NOISE["odom_yaw_bias_std"][c])

    # ------------------------------------------------------------------------------------------
    def cast(self, d):
        """Advance the scan window and cast. Returns (origin (3,), hits (nray, 3) with inf for misses)."""
        self.win = (self.win + 1) % self.nwin
        R = d.xmat[self.base].reshape(3, 3)
        origin = d.xpos[self.base] + R @ MOUNT_POS
        dirs = self.windows[self.win] @ R.T
        mujoco.mj_multiRay(self.m, d, origin, dirs.reshape(-1), self.geomgroup, 1, -1,
                           self._geomid, self._dist, None, self.nray, MAX_DISTANCE)
        hit = (self._geomid >= 0) & (self._dist >= 0) & (self._dist <= MAX_DISTANCE)
        hits = np.full((self.nray, 3), np.inf)
        hits[hit] = origin + dirs[hit] * self._dist[hit, None]
        return origin, hits

    def step(self, d, hits=None, origin=None):
        """Update with this step's returns and return the 609-vector the policy sees (clipped).

        ``hits``/``origin`` may be passed in (cross-check with hits recorded in Isaac); then the scan window
        is not advanced here."""
        if hits is None:
            origin, hits = self.cast(d)
        root = d.xpos[self.base].copy()
        quat = d.xquat[self.base].copy()
        return self.update(root, quat, hits, origin)

    def update(self, root, quat, hits, origin):
        finite = np.isfinite(hits).all(axis=1)
        if self.noise:
            hits, finite = self._perturb(hits, origin, finite)
        yaw = yaw_from_quat(quat)
        self._advance_hold(root[:2], root[2], yaw)

        rel = hits - root
        c, s = math.cos(yaw), math.sin(yaw)
        with np.errstate(invalid="ignore"):
            bx = c * rel[:, 0] + s * rel[:, 1]
            by = -s * rel[:, 0] + c * rel[:, 1]
        valid = finite & np.isfinite(bx) & np.isfinite(by)
        ix = np.zeros(len(bx), dtype=np.int64)
        iy = np.zeros(len(by), dtype=np.int64)
        ix[valid] = np.round((bx[valid] - X0) / RESOLUTION).astype(np.int64)     # numpy & torch: half to even
        iy[valid] = np.round((by[valid] - Y0) / RESOLUTION).astype(np.int64)
        valid &= (ix >= 0) & (ix < NUM_X) & (iy >= 0) & (iy < NUM_Y)
        grid = np.full(NUM_CELLS, _UNOBSERVED)
        h = root[2] - hits[valid, 2] - OFFSET
        np.minimum.at(grid, ix[valid] * NUM_Y + iy[valid], h)
        unobserved = grid >= _UNOBSERVED * 0.5
        grid = np.where(unobserved, self.hold, grid)
        self.hold = grid
        self.measured |= ~unobserved
        self.unobserved = unobserved
        self.last_hits, self.last_valid = hits, valid
        return np.clip(grid, *CLIP)

    # ------------------------------------------------------------------------------------------
    def _advance_hold(self, root_xy, root_z, yaw):
        if not self.motion_compensation:
            return
        if self.pose_valid:
            cp, sp = math.cos(self.prev_yaw), math.sin(self.prev_yaw)
            delta = root_xy - self.prev_xy
            ux = cp * delta[0] + sp * delta[1]
            uy = -sp * delta[0] + cp * delta[1]
            step_yaw = yaw - self.prev_yaw
            if self.noise:
                ux, uy, step_yaw = self._corrupt_odometry(ux, uy, step_yaw)
            cd, sd = math.cos(step_yaw), math.sin(step_yaw)
            qx = cd * CELL_X - sd * CELL_Y + ux
            qy = sd * CELL_X + cd * CELL_Y + uy
            src_x = (qx - X0) / RESOLUTION
            src_y = (qy - Y0) / RESOLUTION
            img = np.stack([self.hold, self.measured.astype(np.float64), np.ones(NUM_CELLS)]).reshape(3, NUM_X, NUM_Y)
            sampled = bilinear_zero_pad(img, src_x, src_y)
            inside = sampled[2] > 0.999
            measured = inside & (sampled[1] > 0.5)
            self.hold = np.where(measured, sampled[0] + (root_z - self.prev_z), FLAT_FILL)
            self.measured = measured
        self.prev_xy = np.array(root_xy, dtype=np.float64)
        self.prev_yaw = float(yaw)
        self.prev_z = float(root_z)
        self.pose_valid = True

    # ------------------------------------------------------------------------------------------ noise
    def _corrupt_odometry(self, ux, uy, step_yaw):
        c = self.cond
        travelled = math.hypot(ux, uy)
        xy_err = self.odom_bias_xy + self.rng.standard_normal(2) * NOISE["odom_xy_step_std"][c]
        yaw_err = self.odom_bias_yaw + self.rng.standard_normal() * NOISE["odom_yaw_step_std"][c]
        return ux + travelled * xy_err[0], uy + travelled * xy_err[1], step_yaw + travelled * yaw_err

    def _perturb(self, hits, origin, finite):
        c = self.cond
        rel = hits - origin
        dist = np.where(finite, np.linalg.norm(np.where(finite[:, None], rel, 0.0), axis=1), 0.0)
        direction = np.where(finite[:, None], rel, 0.0) / np.maximum(dist, 1e-6)[:, None]
        n = len(dist)
        err = self.rng.standard_normal(n) * NOISE["range_std"][c]
        is_out = self.rng.random(n) < NOISE["outlier_prob"][c]
        out = (self.rng.random(n) * 2.0 - 1.0) * NOISE["outlier_range"][c]
        err = np.where(is_out, out, err)
        nd = dist + err
        finite = finite & (nd >= MIN_RANGE)
        rel = direction * np.maximum(nd, 0.0)[:, None]
        tilt = self.rng.standard_normal(2) * NOISE["tilt_step_std"][c] + self.episode_tilt
        ang = float(np.linalg.norm(tilt))
        if ang > 0:
            ax = np.array([tilt[0], tilt[1], 0.0]) / max(ang, 1e-9)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rt = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            rel = rel @ Rt.T
        out_hits = origin + rel
        out_hits[~finite] = np.inf
        return out_hits, finite
