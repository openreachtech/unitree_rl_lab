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

from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import convert_to_warp_mesh

from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg

SCAN_PATTERN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_patterns")
"""Where the recorded sequences live. ``patterns.py`` falls back to the same directory;
every sensor type in its table maps to ``<sensor_type>.npy``."""


class RollingLivoxSensor(LidarSensor):
    """``LidarSensor`` with the scan window advancing once per sensor update."""

    def _initialize_warp_meshes(self):
        """Optionally merge every mesh under the target prim into one raycast mesh.

        The stock ``RayCaster`` takes the *first* ``Mesh`` child of each entry in
        ``mesh_prim_paths`` -- correct for a generated terrain (one big mesh), useless
        for an imported scene like InteriorAgent's apartments, where the geometry is
        hundreds of furniture/wall meshes under one Xform. With
        ``cfg.combine_scene_meshes`` every ``UsdGeom.Mesh`` in the subtree is gathered
        (world-transformed, faces fan-triangulated -- scene assets carry quads/ngons,
        which the stock reader would misindex) into a single warp mesh, cached under
        the same key so any other RayCaster pointed at the path reuses it.
        """
        if not getattr(self.cfg, "combine_scene_meshes", False):
            return super()._initialize_warp_meshes()

        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            if mesh_prim_path in RayCaster.meshes:
                continue
            root = stage.GetPrimAtPath(mesh_prim_path)
            if not root.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            points_all, tris_all, base = [], [], 0
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() != "Mesh":
                    continue
                mesh = UsdGeom.Mesh(prim)
                pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                if pts.size == 0:
                    continue
                tm = np.array(omni.usd.get_world_transform_matrix(prim)).T
                pts = pts @ tm[:3, :3].T + tm[:3, 3]
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                tris = _fan_triangulate(counts, idx)
                if tris is None:
                    continue
                points_all.append(pts)
                tris_all.append(tris + base)
                base += len(pts)
            if not points_all:
                raise RuntimeError(f"No meshes found under: {mesh_prim_path}")
            points = np.concatenate(points_all).astype(np.float32)
            tris = np.concatenate(tris_all)
            # Double-side every face: scene assets are single-sided (walls carry
            # doubleSided=False with normals facing into their room), and the warp
            # raycast only hits front faces -- from the wrong side a wall was
            # INVISIBLE to the LiDAR while PhysX (double-sided trimesh colliders)
            # still blocked the robot: Nav2 walked it into "free" space that was
            # solid (observed 2026-09-25). Appending the flipped winding makes the
            # raycast see exactly what physics collides with.
            tris = np.concatenate([tris, tris[:, ::-1]])
            indices = tris.reshape(-1)
            RayCaster.meshes[mesh_prim_path] = convert_to_warp_mesh(points, indices, device=self.device)
            print(
                f"[RollingLivoxSensor] merged {len(points_all)} meshes under {mesh_prim_path}:"
                f" {len(points)} vertices, {len(indices) // 3} triangles (double-sided)."
            )

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

    combine_scene_meshes: bool = False
    """Merge every mesh under each ``mesh_prim_paths`` entry into one raycast target.

    For imported multi-mesh scenes (e.g. InteriorAgent apartments); leave False for
    generated terrains, whose single mesh the stock reader already handles."""


def _fan_triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray | None:
    """Faces of arbitrary vertex count -> (N, 3) triangle array (fan per face)."""
    if counts.size == 0 or indices.size == 0:
        return None
    if (counts == 3).all():
        return indices.reshape(-1, 3)
    tris = []
    ofs = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[ofs], indices[ofs + k], indices[ofs + k + 1]))
        ofs += c
    return np.asarray(tris, dtype=indices.dtype)
