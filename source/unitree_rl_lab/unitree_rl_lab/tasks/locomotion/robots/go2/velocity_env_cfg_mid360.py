"""Go2-Blind-GRU-Phase4 with a Livox MID-360, and the height map built from it.

The sensor is ``RollingLivoxSensor``: the OmniPerception ``LidarSensor``
(https://github.com/aCodeDog/OmniPerception, ported as-is into ``unitree_rl_lab.sensors``)
with the one fix described below. Each step it advances 4,000 rows through the real
MID-360 non-repetitive sequence (``sensors/scan_patterns/mid360.npy``) -- at the 0.02 s env
step that is the sensor's own 200k points/s -- and casts every 4th of them, so 1,000 rays
per step. See ``MID360_SAMPLES_PER_STEP`` / ``MID360_RAY_DOWNSAMPLE``.

The base task is untouched: the classes here extend ``RobotEnvCfgPhase4`` /
``RobotPlayEnvCfgPhase4`` and register separately as ``Go2-Blind-GRU-Mid360-Phase4``.

The returns are binned into a 609-cell height grid by ``mdp.LidarElevationMap``, the same
term the fan uses in ``velocity_env_cfg_lidar.py``; a cell with no return this step holds
what it last saw. See ``Mid360MapObsCfg`` below for the two settings that differ from the
fan's map -- no body exclusion, no noise model.

Nothing reads the ``mid360_map`` observation group. Like ``LidarMapObsCfg``, it exists so
the observation manager runs the term, which is what makes the sensor raycast and the held
grid advance. The policy and critic inputs are unchanged, so Go2-Blind-GRU-Phase4
checkpoints still load -- there is no mid360 experiment folder, so pass the base phase's
checkpoint explicitly::

    python scripts/rsl_rl/play.py --task Go2-Blind-GRU-Mid360-Phase4 --num_envs 4
        --checkpoint logs/rsl_rl/go2_blind_gru_phase4/<run>/model_7300.pt

The sensor sits at the real L1's pose -- nose tip, pitched nearly upside down; see
``GO2_L1_MOUNT`` / ``GO2_L1_ROT`` below.

Play draws the map and nothing else: 609 cell markers, **green measured this step, red
held from an earlier one**. Red is exactly where the real robot would have no current
data, so a wall's shadow should read as a solid red patch. The scanner's own RayCaster
markers -- the 1,000 raw returns, the ring that breathes at 10 Hz -- are off, because they
land on the same ground as the cell markers and make both unreadable; turn them back on
with ``_attach_mid360(..., show_raw_points=True)`` to watch the scan pattern itself.

Why the window has to roll
--------------------------
The port slices ``samples`` rows out of the recorded sequence at ``rolling_window_start``
and builds the pattern once in ``_initialize_rays_impl``, so the window is frozen for the
whole run. 4,000 rows is one 20 ms frame, and a real MID-360 genuinely covers only part of
its 59 deg elevation band in one frame -- that much is faithful -- but it then moves on,
cycling the whole band with a period of about 20,000 rows (0.1 s).

Pinned at row 0, the sim sits forever on the steepest quarter of the band (sensor phi
23.8..52.2 of its -7.2..52.2). Through the 164.9 deg mount that becomes a base-frame
elevation of -61.5..-11.1 deg -- *every* ray below horizontal, and on flat ground the
whole frame landing in an annulus of 0.15..1.39 m, median 0.38 m.

The damage is not the short reach; a metre is all this task needs. It is that there are no
near-horizontal rays, and an obstacle's height is only readable when some ray clears its
top edge. With the shallowest ray at -11.1 deg and the sensor 0.273 m up, a wall's top
first clears at ``(0.273 - h) / tan(11.1 deg)``:

    5 cm -> 1.14 m    10 cm -> 0.88 m    15 cm -> 0.63 m    20 cm -> 0.37 m    25 cm -> 0.12 m

Taller walls are seen *later*, which is backwards, and Phase 4's walls run to 25 cm. Worse,
only 29 in 4,000 of the window's rays are shallower than -12 deg (median -35.5), so at a usable ray
density the 25 cm figure is nearer 0.09 m. The wall is detected -- its lower face is hit
from 1.39 m -- but nothing says how high it is until the robot is on top of it.

``RollingLivoxSensor`` consumes the next 4,000 rows each step instead, so the band sweeps
its full -62.3..+19.7 deg every 0.1 s: five steps running all-down, horizon, all-down.
Note the near-horizontal rays then arrive in 10 Hz bursts rather than continuously -- fine
for a map that holds previous values, something to design around for a consumer of single
raw frames.

Rolling also retires the port's ``_update_dynamic_rays``, which rotated the frozen pattern
about base z by an angle growing as ``sensor_t * 0.1`` (quadratic in step count, and
``sensor_t`` never reset). That was upstream's stand-in for the window advance and could
not substitute for one: a rotation about z leaves each ray's z component untouched, so
every ray keeps its elevation exactly and the band never moves.

Remaining caveat
----------------
Rearward the real nose mount is blind behind the head and trunk; the RayCaster only sees
the static ground mesh, so the sim looks straight through the body and is more optimistic
than hardware there (same caveat as velocity_env_cfg_lidar.py).
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.sensors import LivoxPatternCfg, RollingLivoxSensorCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase4 import (
    RobotEnvCfgPhase4,
    RobotPlayEnvCfgPhase4,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    GO2_FLAT_SCAN_VALUE,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    GO2_HEIGHT_SCAN_CENTER_X,
    GO2_HEIGHT_SCAN_CENTER_Y,
    GO2_HEIGHT_SCAN_OFFSET,
    GO2_LIDAR_OFFSET_X,
    GO2_LIDAR_OFFSET_Y,
    GO2_LIDAR_OFFSET_Z,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
)

MID360_SAMPLES_PER_STEP = 4000
"""Sequence rows consumed per env step -- the window stride, not the ray count. 4,000 x
50 Hz = 200k points/s, the real MID-360's rate, which is what fixes this number: it walks
the 800,000-point file in 200 steps and repeats every 4 s. Change it and the elevation
sweep changes period with it."""

MID360_RAY_DOWNSAMPLE = 4
"""Cast every n-th point of the window: 4,000 / 4 = 1,000 rays per step, 50k points/s.

Thinning here rather than by shrinking ``samples`` keeps the window advancing at the
hardware's rate, so the 0.1 s elevation sweep is untouched -- measured against the full
4,000, each step's elevation band and median ground reach are the same to within a few
tenths of a degree and a few cm. Set to 1 for the sensor's full point rate."""

# ---------------------------------------------------------------------------
# Mount: the real L1 (utlidar) pose from go2_description.urdf's ``radar_joint``,
# ``xyz="0.28945 0 -0.046825" rpy="0 2.8782 0"`` -- the nose tip, pitched 164.9 deg
# so the sensor hangs nearly upside down looking out and down through the nose
# aperture. The translation is already in the repo as GO2_LIDAR_OFFSET_*.
#
# The sensor's own +z (the axis its 360 deg azimuth ring turns about) ends up at
# (0.26, 0, -0.97) in the base frame: pointing almost straight down, leaning 15 deg
# forward. So the band does not land as a horizon ring, and which part of it points
# where depends on the window -- see "Why the window has to roll" above.
# ---------------------------------------------------------------------------
GO2_L1_MOUNT = (GO2_LIDAR_OFFSET_X, GO2_LIDAR_OFFSET_Y, GO2_LIDAR_OFFSET_Z)
_L1_PITCH = 2.8782  # rad, from the URDF radar_joint rpy
GO2_L1_ROT = (math.cos(_L1_PITCH / 2), 0.0, math.sin(_L1_PITCH / 2), 0.0)
"""(w, x, y, z) quaternion of the L1 mount: a pure pitch of 2.8782 rad."""


def _mid360_scanner_cfg(debug_vis: bool) -> RollingLivoxSensorCfg:
    """A fresh sensor cfg per env-cfg instance, so play tweaks never leak into train."""
    return RollingLivoxSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RollingLivoxSensorCfg.OffsetCfg(pos=GO2_L1_MOUNT, rot=GO2_L1_ROT),
        # Bolted to the nose like the real L1: pitch and roll swing the pattern
        # with the body.
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=MID360_SAMPLES_PER_STEP,
            downsample=MID360_RAY_DOWNSAMPLE,
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
        min_range=0.2,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        update_frequency=50.0,
        debug_vis=debug_vis,
    )


# ---------------------------------------------------------------------------
# Height map. Same machinery as the fan's (``mdp.LidarElevationMap``): bin the
# returns into the body-centered grid, take the highest per cell, and let a cell
# with no return this step hold what it last saw. Only the sensor differs.
#
# Two things are set differently from velocity_env_cfg_lidar.py's fan map:
#
#   * **No body exclusion.** The fan sits on top of the trunk and cannot see under
#     itself, so its map cuts out a 0.80 x 0.60 m rectangle. The L1 is at the nose
#     pointing down and does reach under the body -- its steepest rays land 0.15 m
#     from the mount, inside the footprint -- so every cell stays. That also makes
#     the output line up cell-for-cell with the critic's top-down ``height_scan``:
#     both are the full 29 x 21, which is what a reconstruction loss would want.
#   * **No noise model.** ``LidarNoiseCfg`` was tuned against the fan; the MID-360's
#     error behaviour is its own. Clean returns for now -- pass a ``noise=`` to
#     ``_mid360_map_term`` when there is a model worth using.
#
# Grid, resolution, centre and offset are the project's existing ones, so this map
# is directly comparable with every other height grid in the repo.
# ---------------------------------------------------------------------------
MID360_MAP_CELLS = (round(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1) * (
    round(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1
)
"""609: the full 29 x 21 grid, nothing excluded."""


def _mid360_map_term(debug_vis: bool, debug_vis_env_index: int | None = None) -> ObsTerm:
    return ObsTerm(
        func=mdp.LidarElevationMap,
        params={
            "sensor_cfg": SceneEntityCfg("mid360_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "offset": GO2_HEIGHT_SCAN_OFFSET,
            "resolution": HEIGHT_SCAN_RESOLUTION,
            "size": HEIGHT_SCAN_SIZE,
            "scanner_offset_xy": (GO2_HEIGHT_SCAN_CENTER_X, GO2_HEIGHT_SCAN_CENTER_Y),
            # Negative rather than 0.0: an extent of exactly zero would still drop the
            # one cell sitting on the body origin.
            "exclude_half_extent_x": -1.0,
            "exclude_half_extent_y": -1.0,
            "lidar_offset": GO2_L1_MOUNT,
            # A full turn, so no cell is written off as out of field of view.
            "horizontal_fov": (-180.0, 180.0),
            "flat_fill": GO2_FLAT_SCAN_VALUE,
            "noise": None,
            "debug_vis": debug_vis,
            "debug_vis_env_index": debug_vis_env_index,
        },
        clip=(-1.0, 5.0),
        history_length=0,
    )


@configclass
class Mid360MapObsCfg(ObsGroup):
    """Display-only group: nothing reads it, it exists so the map term runs each step.

    Running is what makes the sensor raycast and the held grid advance, and in play it is
    also what draws the markers -- green measured this step, red held from an earlier one.
    The policy and critic are untouched, so Go2-Blind-GRU-Phase4 checkpoints still load.
    """

    height_scan = _mid360_map_term(debug_vis=False)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _attach_mid360(
    cfg: RobotEnvCfgPhase4,
    debug_vis: bool,
    show_raw_points: bool = False,
    keep_lidar_map: bool = False,
) -> None:
    """Bolt the MID-360 and its height map onto a blind-phase cfg.

    ``debug_vis`` draws the map: green measured this step, red held from an earlier one.

    ``show_raw_points`` turns on the scanner's own RayCaster markers -- the 1,000 raw
    returns of this step, the ring that breathes at 10 Hz. Off by default: they land on
    the same ground as the map's cell markers and make both hard to read. Worth turning on
    to watch the scan pattern itself rather than the map it builds.

    ``keep_lidar_map`` decides what happens to the fan-built map the play configs of this
    lineage add through ``apply_lidar_view``. Dropped by default, for two reasons: its 388
    green/red spheres per robot sit on top of this map's 609 and neither is then readable,
    and both terms publish their unobserved rates to the same ``env.lidar_map_unobserved_*``
    attributes, so whichever runs second wins. Set True to keep it and read the mid360
    map's diagnostics with care. Training configs have no ``lidar_map`` group to begin
    with, so it is a no-op there.
    """
    cfg.scene.mid360_scanner = _mid360_scanner_cfg(debug_vis=show_raw_points)
    cfg.scene.mid360_scanner.update_period = cfg.decimation * cfg.sim.dt
    cfg.observations.mid360_map = Mid360MapObsCfg()
    if debug_vis:
        # env_index=None draws every environment; only affordable at play sizes.
        cfg.observations.mid360_map.height_scan = _mid360_map_term(
            debug_vis=True, debug_vis_env_index=None
        )
    if not keep_lidar_map and getattr(cfg.observations, "lidar_map", None) is not None:
        # ObservationManager skips a group set to None.
        cfg.observations.lidar_map = None


@configclass
class RobotEnvCfgMid360Phase4(RobotEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        _attach_mid360(self, debug_vis=False)


@configclass
class RobotPlayEnvCfgMid360Phase4(RobotPlayEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        _attach_mid360(self, debug_vis=True)
