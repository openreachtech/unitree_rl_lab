"""A Livox sensor whose scan window advances, unlike the ported ``LidarSensor``.

Not part of the OmniPerception port -- the files alongside this one are copies and stay
that way. This is the fix for the port's one substantive problem, and it lives here
rather than in the env config because it is sensor behaviour, not task configuration.

**What the port does.** ``patterns.livox_pattern`` slices ``samples`` rows out of the
recorded scan sequence starting at ``rolling_window_start``, and the RayCaster calls it
from ``_initialize_rays_impl`` -- once, at sim start. Nothing ever advances the start
index, so the window is frozen for the whole run. ``LidarSensor._update_dynamic_rays``
tries to compensate by rotating the frozen slice about the sensor's z axis each step,
but a rotation about z leaves every ray's elevation exactly where it was, so it cannot
recover what a frozen window is missing. (It also rotates in place by an angle that
grows with sensor time, which compounds into thousands of turns; see
``velocity_env_cfg_mid360.py``.)

**Why that matters.** A Livox scans non-repetitively: a pair of rotating prisms sweeps
the beam along a rosette, so any single frame covers a thin curve and coverage fills in
over time. For the MID-360 the elevation band cycles with a period of about 0.1 s. One
20 ms frame is genuinely only part of the band -- that much is faithful -- but the real
sensor then moves on, and the frozen window never does. Pinned at row 0 the MID-360 sits
forever on sensor-frame phi 23.8..52.2 deg of its -7.2..52.2 band.

**What this class does.** It consumes the recorded sequence the way the hardware does:
each sensor update takes the next ``pattern_cfg.samples`` points, wrapping at the end. At
the Go2's 0.02 s env step and the MID-360's 4,000 points per step that walks the
800,000-point (4 s) file in 200 windows, reproducing the 0.1 s elevation sweep.
``_update_dynamic_rays`` is overridden to do the advance, which also removes the spin --
the real sequence already moves the pattern, in the right way and along the right axis.

``pattern_cfg.downsample`` thins each window without slowing the advance: ``samples`` is
how much of the sequence a frame consumes (fixed by the sensor's point rate), while
``samples // downsample`` is how many rays are actually cast. Taking every n-th point of
the window rather than its first ``samples // downsample`` matters -- the rows are ordered
along the rosette, so truncating a window narrows its elevation band, while decimating it
keeps the band and just samples it more sparsely.

The whole sequence is converted to unit direction vectors once at init and kept on the
device (9.6 MB undecimated for the MID-360), so a step costs one indexed read and one
broadcast copy.
"""

from __future__ import annotations

import numpy as np
import os
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg

SCAN_PATTERN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_patterns")
"""Where the recorded sequences live. ``patterns.py`` falls back to the same directory;
every sensor type in its table maps to ``<sensor_type>.npy``."""


class RollingLivoxSensor(LidarSensor):
    """``LidarSensor`` with the scan window advancing once per sensor update."""

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()

        pattern_cfg = self.cfg.pattern_cfg
        if getattr(pattern_cfg, "use_simple_grid", False):
            # A static grid has no sequence to roll, and the base class does not call
            # _update_dynamic_rays for it either.
            self._scan_sequence = None
            return
        path = os.path.join(SCAN_PATTERN_DIR, f"{pattern_cfg.sensor_type}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No recorded scan sequence for '{pattern_cfg.sensor_type}' at {path}."
                " Copy it from OmniPerception's"
                " LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/."
            )
        sequence = np.load(path)  # (M, 2), columns [theta, phi] in radians

        # Same spherical convention as patterns.livox_pattern: x forward, y left, z up.
        theta = torch.from_numpy(sequence[:, 0]).to(self._device)
        phi = torch.from_numpy(sequence[:, 1]).to(self._device)
        directions = torch.stack(
            [torch.cos(theta) * torch.cos(phi), torch.sin(theta) * torch.cos(phi), torch.sin(phi)],
            dim=1,
        )
        directions = directions / directions.norm(dim=1, keepdim=True)
        # The base class bakes the mount rotation into ray_directions at init, so every
        # window has to carry it too. ray_starts needs no such treatment: the origin is
        # the same point for every window.
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        directions = quat_apply(offset_quat.repeat(len(directions), 1), directions)

        # `stride` is what a frame consumes of the sequence; `downsample` thins what is
        # actually cast within it. Keep them apart or a thinned pattern would also crawl
        # through the sequence more slowly, stretching the elevation sweep.
        stride = min(pattern_cfg.samples, len(directions))
        downsample = max(int(getattr(pattern_cfg, "downsample", 1)), 1)
        num_windows = len(directions) // stride
        if num_windows < 1:
            raise ValueError(
                f"Scan sequence for '{pattern_cfg.sensor_type}' has {len(directions)} points,"
                f" fewer than the {stride} a frame consumes; nothing to roll."
            )
        windows = directions[: num_windows * stride].view(num_windows, stride, 3)
        # Every downsample-th point of the window, matching patterns.livox_pattern's own
        # `torch.arange(0, len(theta), cfg.downsample)`, so window 0 reproduces exactly
        # what the base class built at init.
        self._scan_sequence = windows[:, ::downsample, :].contiguous()
        if self._scan_sequence.shape[1] != self.num_rays:
            raise ValueError(
                f"Rolled window has {self._scan_sequence.shape[1]} rays but the base class"
                f" allocated {self.num_rays}; samples/downsample must agree with the pattern."
            )
        self._num_windows = num_windows
        # Start where the port's frozen window would have, so window 0 of a run matches
        # the pattern LidarSensor would have used for the whole of it.
        self._window_index = (getattr(pattern_cfg, "rolling_window_start", 0) // stride) - 1

    def _update_dynamic_rays(self):
        """Advance to the next window instead of spinning the frozen one.

        Called once per ``_update_buffers_impl``, before the parent ray-casts. Every
        environment shares a window: they step in lockstep, and a real fleet's sensors
        would be free-running relative to each other, but nothing here reads across envs.

        Deliberately not reset with the episode -- the hardware's prisms keep turning
        across a reset, and the index is taken modulo the window count, so it cannot run
        away the way ``sensor_t`` does in the base class.
        """
        if self._scan_sequence is None:
            return
        self._window_index = (self._window_index + 1) % self._num_windows
        self.ray_directions[:] = self._scan_sequence[self._window_index]


@configclass
class RollingLivoxSensorCfg(LidarSensorCfg):
    """``LidarSensorCfg`` pointed at :class:`RollingLivoxSensor`.

    ``pattern_cfg.rolling_window_start`` becomes the *first* window rather than the only
    one; everything else means what it does on the base config.
    """

    class_type: type = RollingLivoxSensor
