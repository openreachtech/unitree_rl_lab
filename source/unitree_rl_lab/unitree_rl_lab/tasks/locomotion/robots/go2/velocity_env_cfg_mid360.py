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
what it last saw. The grid keeps every cell -- no body exclusion, because this mount does
see under the trunk. Measurement noise is ``MID360_NOISE_CFG``: range error along the ray,
sensor tilt and outliers, drawn weak/nominal/strong at 60/30/10 per episode. Both are
explained where they are defined below, along with the error sources still unmodelled.

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

Self-occlusion
--------------
The robot blocks its own rays, which the stock IsaacLab ``RayCaster`` cannot do -- it casts
against one static mesh with its transform baked at init, so the body is transparent and a
ray fired past a leg still reports the ground behind it. ``MID360_DYNAMIC_MESH`` (on by
default) swaps in ``OccludedRollingLivoxSensor``, which carries the robot's collision
geometry as a second warp mesh moving with the bodies and drops any ray the body catches. In
play that shows up as red (held) cells sweeping with the gait where a leg crosses the field
of view. See ``sensors/robot_occluder.py`` and ``MID360_DYNAMIC_MESH`` for what it costs.

Note this is the *sensor's own* robot only. Other robots and any other dynamic object are
still invisible -- at ``env_spacing`` 2.5 m against a 1.4 x 1.0 m map window that does not
matter here, and Phase 4's walls are baked into the static terrain mesh, so they occlude
correctly either way.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.sensors import (
    LivoxPatternCfg,
    OccludedRollingLivoxSensorCfg,
    RollingLivoxSensorCfg,
)
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.lidar_elevation_map import LidarNoiseCfg
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

MID360_OCCLUDER_MODE = "hit"
"""What a ray that lands on the robot reports. Needs ``MID360_DYNAMIC_MESH``.

``"hit"`` returns the point on the robot's own surface, so the leg or belly the beam
actually struck appears in the height map as if it were terrain. That is what the raw point
cloud off a real MID-360 contains -- the laser does hit the legs and the light does come
back -- so this is the right setting when the height map on hardware is built straight from
the raw cloud.

``"drop"`` turns the same ray into a miss, which models the self-filter that mapping
pipelines normally apply (the robot's own collision shapes are used to delete its points
before they reach the grid). Pick it if ``deploy/``'s height-map publisher will do that
filtering; otherwise the sim would be cleaner than the robot.

The difference is not subtle, measured walking at 0.6 m/s with noise off, as the fraction of
cells the map places above the true ground:

    band from the L1     >5cm    >15cm    max
    0.00-0.14 m   hit   48.6%    23.1%   43.0 cm
                  drop   7.3%     0.5%   17.7 cm
    overall       hit   13.9%     3.0%
                  drop   6.8%     0.2%

The near field is where it lands, because that is where the robot's own body is. 43 cm is
taller than any Phase 4 wall, and it moves with the gait.
"""

MID360_DYNAMIC_MESH = True
"""Whether the robot's own body blocks its rays -- see ``sensors/robot_occluder.py``.

On, the sensor is ``OccludedRollingLivoxSensor``: the robot's collision geometry becomes a
second warp mesh that moves with the bodies each step, and a ray that would have gone through
a leg is dropped rather than reporting the ground behind it. Off, it is the plain
``RollingLivoxSensor`` and the robot is transparent, which is what the stock IsaacLab
``RayCaster`` gives you and what every other LiDAR task in this repo still does.

Default on because transparency is not a small error here. The L1 sits at the nose and looks
down and back; the front hips are 9.6 cm behind it, so much of the map's rear half is reached
by rays that pass right by a leg. Left transparent, those cells come back *measured* and
correct where the hardware would have no data at all -- so the map's unobserved pattern is
wrong, and wrong in a gait-correlated way, which is exactly the structure a height-map
encoder would otherwise learn to rely on. Measured standing still on flat ground, the body
takes 5.1 percentage points of the returns (1,000 rays -> about 51 blocked per step); walking
and stepping over Phase 4's walls swings the legs further and takes more.

The cost is real: at 4,096 environments the occluder mesh is 18 bodies x 4,096 = 2.66M
vertices and 4.90M triangles, whose BVH is refit every step. Measured against the same task
without it, a training iteration goes 1.479 s -> 1.852 s (+25%), of which the dynamic mesh is
+0.365 s -- about twice what the MID-360's own raycast costs. Narrow
``occluder_body_names`` to buy some back: ``["base", "F[LR]_.*"]`` keeps the parts that
actually shadow a forward-looking mount and roughly halves the triangle count."""

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


def _mid360_scanner_cfg(
    debug_vis: bool, dynamic_mesh: bool = MID360_DYNAMIC_MESH
) -> RollingLivoxSensorCfg:
    """A fresh sensor cfg per env-cfg instance, so play tweaks never leak into train.

    ``dynamic_mesh`` picks the sensor class and nothing else: mount, scan sequence, ray count,
    range and noise are identical either way, so the two differ in exactly one respect. See
    ``MID360_DYNAMIC_MESH``.
    """
    cfg_class = OccludedRollingLivoxSensorCfg if dynamic_mesh else RollingLivoxSensorCfg
    # Only the occluding config has these fields; the plain one would reject them.
    occluder_kwargs = {"occluder_mode": MID360_OCCLUDER_MODE} if dynamic_mesh else {}
    return cfg_class(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=cfg_class.OffsetCfg(pos=GO2_L1_MOUNT, rot=GO2_L1_ROT),
        # Bolted to the nose like the real L1: pitch and roll swing the pattern
        # with the body.
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=MID360_SAMPLES_PER_STEP,
            downsample=MID360_RAY_DOWNSAMPLE,
        ),
        **occluder_kwargs,
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
# One thing is set differently from velocity_env_cfg_lidar.py's fan map:
#
#   * **No body exclusion.** The fan sits on top of the trunk and cannot see under
#     itself, so its map cuts out a 0.80 x 0.60 m rectangle. The L1 is at the nose
#     pointing down and does reach under the body -- its steepest rays land 0.15 m
#     from the mount, inside the footprint -- so every cell stays. That also makes
#     the output line up cell-for-cell with the critic's top-down ``height_scan``:
#     both are the full 29 x 21, which is what a reconstruction loss would want.
#
# Grid, resolution, centre and offset are the project's existing ones, so this map
# is directly comparable with every other height grid in the repo.
# ---------------------------------------------------------------------------
MID360_MAP_CELLS = (round(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1) * (
    round(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1
)
"""609: the full 29 x 21 grid, nothing excluded."""

MID360_NOISE_CFG = LidarNoiseCfg()
"""Measurement noise. Same magnitudes and the same weak/nominal/strong 60/30/10 draw as
the fan's ``GO2_LIDAR_NOISE_CFG``, but a separate instance so tuning one does not move the
other. The ramp fields are inert at their defaults (``start_iteration == full_iteration``
means full magnitude from step 0), matching every other task in the repo; set them to fade
the noise in over training. ``scale=0.0`` gives a clean reference run.

The nominal ``range_std`` of 2 cm happens to sit right on the MID-360's own range
precision, so the Position term is roughly calibrated for this sensor -- confirm against
the datasheet before leaning on it.

Three of the six augmentations in the reference paper (see ``mdp/lidar_elevation_map.py``)
stay off because this sensor produces them from geometry rather than from a model:
*Pruning* -- the 1,000-ray budget over 609 cells, plus the 10 Hz elevation sweep, already
leaves whole annuli unmeasured for several steps at a time, which is a more realistic
temporally-correlated dropout than random patch removal; *Height* and *Robot Pose* -- a held
cell carries whatever the terrain looked like when a beam last landed there, and is
transported forward on simulated odometry rather than re-measured.

Deliberately **not** modelled, and worth knowing before trusting a sim-to-real number:

  * *Incidence-angle dropout.* The mount is 0.273 m up and rays meet the ground at
    11..62 deg; a real return weakens and vanishes at grazing incidence, while the
    raycast always hits. This is the largest remaining gap, and it correlates with
    range -- the sim is most optimistic exactly where the map is thinnest.
  * *Blind zone.* ``min_range`` is a dead field on the ported sensor (only read inside
    ``_apply_noise``, which is off), and the steepest rays land 0.15 m from the mount --
    possibly inside the hardware's near cutoff.
  * *Reflectivity dropout.* Dark, wet and specular surfaces return nothing; there is no
    material information in the raycast to key off.
  * *Motion distortion.* A frame's points are acquired over 20 ms while the body moves,
    but every ray here is cast from one pose.

Self-occlusion used to be on this list and no longer is: ``MID360_DYNAMIC_MESH`` puts the
robot's own collision geometry in the way of its rays.

**Outliers are mis-scaled for this sensor, and it shows in play.** ``outlier_range`` is
0.15 / 0.30 / 0.60 m, but this mount's ground returns run 0.15..1.6 m with a median near
0.4 m, so a strong outlier is larger than the whole measurement. ``_perturb`` moves a point
along its own ray, and the distance is clamped at zero, so a short outlier lands somewhere
between the ground and the sensor -- at the limit, on the sensor itself. Every ray leaves
the L1 at the nose, and the map keeps the *highest* return per cell (an ``amin`` on these
inverted heights), so the near-side outliers always win their cell and the far-side ones
never do. The result is a one-sided conical spray of raised cells with its apex at the
robot's face, held in place by the previous-value fill.

Measured against the true terrain, cell for cell, over 4 environments:

    cells more than 5 cm above ground, by distance from the L1 mount
    0.00-0.15 m   46.3% with noise, 4.5% without
    0.15-0.30 m   12.9% / 1.1%
    0.30-0.45 m    6.9% / 4.6%
    0.60-0.80 m    2.3% / 1.9%

The radial gradient is entirely the noise: without it the near field is no worse than the
far field. The innermost band is worst because it is also the blind disc -- the steepest ray
leaves at -62.3 deg from 0.273 m up, so nothing lands within 0.143 m of the mount and a
spike there is never overwritten. Self-occlusion is not the cause and in fact reduces it
(7.1% of cells over 5 cm with the occluder, 9.1% without).

Making the magnitude proportional to the measured distance was considered and **rejected**:
a wild reading from a faulty unit does not know how far away the target was, and scaling the
error by range would model only the mixed-pixel half of the phenomenon. The absolute
magnitude stays.

What that leaves is the near-face spray, which is a real consequence of the model and will
reappear whenever ``scale`` is non-zero. Two levers that do not touch the absolute-magnitude
decision: lower ``outlier_range`` / ``outlier_prob``, or stop treating a short outlier as a
point at the sensor -- ``rel = direction * (distance + error).clamp(min=0.0)`` in ``_perturb``
puts it exactly on the mount, and no LiDAR reports a range below its own minimum. Dropping
those as non-returns instead would keep faults arbitrary while removing the one artifact that
is physically impossible.
"""


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
            "noise": MID360_NOISE_CFG,
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
    dynamic_mesh: bool = MID360_DYNAMIC_MESH,
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

    ``dynamic_mesh`` decides whether the robot blocks its own rays; see
    ``MID360_DYNAMIC_MESH`` for what it costs and why it defaults on. In play it is the one
    switch worth flipping back and forth: with it on, a leg crossing the field of view leaves
    a red (held) streak in the map that sweeps with the gait; with it off that same streak is
    green and confidently wrong.
    """
    cfg.scene.mid360_scanner = _mid360_scanner_cfg(
        debug_vis=show_raw_points, dynamic_mesh=dynamic_mesh
    )
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
