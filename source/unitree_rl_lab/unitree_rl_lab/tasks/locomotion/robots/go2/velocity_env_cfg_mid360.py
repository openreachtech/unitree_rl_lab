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
The stock IsaacLab ``RayCaster`` cannot model it -- it casts against one static mesh with
its transform baked at init, so the body is transparent and a ray fired past a leg still
reports the ground behind it. ``MID360_DYNAMIC_MESH`` swaps in
``OccludedRollingLivoxSensor``, which carries the robot's collision geometry as a second warp
mesh moving with the bodies, so the body catches the rays it should.

**Off by default**, because the deployed stack removes the robot's own returns before the
height map is built -- it knows where its links are from forward kinematics and deletes the
points inside them. Train against a transparent body and sim matches that filtered cloud;
model the occlusion and sim carries self-hits the hardware never delivers. See
``MID360_DYNAMIC_MESH`` for the argument the other way and what it costs.

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
from unitree_rl_lab.tasks.locomotion.mdp.lidar_elevation_map import (
    LidarNoiseCfg,
    LidarNoiseConditionCfg,
)
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

MID360_RAY_DOWNSAMPLE = 2
"""Cast every n-th point of the window: 4,000 / 2 = 2,000 rays per step, 100k points/s.

Thinning here rather than by shrinking ``samples`` keeps the window advancing at the
hardware's rate, so the 0.1 s elevation sweep is untouched -- measured against the full
4,000, each step's elevation band and median ground reach are the same to within a few
tenths of a degree and a few cm. Set to 1 for the sensor's full point rate.

Raised 4 -> 2 on 2026-09-18, to put the point budget on the same footing as the deployed
pipeline. That pipeline builds each map from **one 10 Hz scan with no accumulation**
(~20,000 points) -- deliberately, because what accumulation buys on this sensor is RKO-LIO
pose error, which costs the map more than the extra density gains it. This map refreshes at
50 Hz and holds cells no beam reached, so five steps span the same 0.1 s: 5 x 2,000 = 10,000
rays against the hardware's ~20,000, where a downsample of 4 gave only 5,000.

Cost, estimated rather than re-measured: the 4,096-env benchmark put this sensor's own
raycast at +0.173 s per iteration over the top-down baseline at 1,000 rays, so doubling the
rays should add about that much again (~12% of an iteration)."""

MID360_MIN_RANGE = 0.2
"""Closest range the sensor reports (m). Fed to the scanner and to the height map's
noise model, which drops a perturbed return shorter than this as a non-return rather
than placing it on the mount -- see ``MID360_NOISE_CFG``."""

MID360_OCCLUDER_MODE = "hit"
"""What a ray that lands on the robot reports. Inert unless ``MID360_DYNAMIC_MESH`` is on,
which by default it is not -- the body is transparent and no ray lands on the robot at all.

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

MID360_DYNAMIC_MESH = False
"""Whether the robot's own body blocks its rays -- see ``sensors/robot_occluder.py``.

Off, the sensor is the plain ``RollingLivoxSensor``: the robot is transparent, a ray fired
past a leg reports the ground behind it, and the map has no self-hits in it. That is what the
stock IsaacLab ``RayCaster`` gives you and what every other LiDAR task in this repo does.

On, it is ``OccludedRollingLivoxSensor``: the robot's collision geometry becomes a second warp
mesh that moves with the bodies each step, and a ray the body catches reports the body
(``MID360_OCCLUDER_MODE = "hit"``) or nothing at all (``"drop"``).

**Default off because the real robot does not deliver self-hits either.** The deployed
height-map publisher computes its own link poses and deletes the points that fall inside them
before anything downstream sees the cloud, so what the policy is trained on should be the
filtered cloud -- and a transparent body produces exactly that, for free. Modelling the
occlusion and leaving it on would put returns off the robot's own legs into the map that
hardware never produces, and ``"hit"`` puts them there large: measured walking at 0.6 m/s with
noise off, 48.6% of cells within 0.14 m of the L1 sit more than 5 cm above true ground, up to
43 cm -- taller than any Phase 4 wall, and moving with the gait.

The argument for turning it on is the *unobserved* pattern rather than the values. The L1 sits
at the nose looking down and back; the front hips are 9.6 cm behind it, so much of the map's
rear half is reached by rays that graze a leg. Transparent, those cells come back measured and
correct where hardware -- after its self-filter -- would have no data at all, so sim's holes
are in the wrong places, in a gait-correlated way. Standing still on flat ground the body
accounts for 5.1% of the returns (about 51 of 1,000 rays per step); walking swings the legs
further and takes more. ``MID360_OCCLUDER_MODE = "drop"`` is the setting that reproduces that
hole pattern without inventing the self-hits; turn both on if the deploy-side filter turns out
to punch the same holes.

The cost, if you do: at 4,096 environments the occluder mesh is 18 bodies x 4,096 = 2.66M
vertices and 4.90M triangles, whose BVH is refit every step. Measured against the same task
without it, a training iteration goes 1.479 s -> 1.852 s (+25%), of which the dynamic mesh is
+0.365 s -- about twice what the MID-360's own raycast costs. Narrow ``occluder_body_names``
to buy some back: ``["base", "F[LR]_.*"]`` keeps the parts that actually shadow a
forward-looking mount and roughly halves the triangle count."""

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
        min_range=MID360_MIN_RANGE,
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

MID360_NOISE_CFG = LidarNoiseCfg(
    weak=LidarNoiseConditionCfg(
        probability=0.60, range_std=0.005, tilt_step_std=0.25, tilt_episode_std=0.125,
        outlier_prob=0.0025, outlier_range=0.075,
        odom_xy_step_std=0.005, odom_xy_bias_std=0.005,
        odom_yaw_step_std=0.25, odom_yaw_bias_std=0.25,
    ),
    nominal=LidarNoiseConditionCfg(
        probability=0.30, range_std=0.01, tilt_step_std=0.5, tilt_episode_std=0.25,
        outlier_prob=0.005, outlier_range=0.15,
        odom_xy_step_std=0.01, odom_xy_bias_std=0.01,
        odom_yaw_step_std=0.5, odom_yaw_bias_std=0.5,
    ),
    strong=LidarNoiseConditionCfg(
        probability=0.10, range_std=0.02, tilt_step_std=1.0, tilt_episode_std=0.5,
        outlier_prob=0.015, outlier_range=0.30,
        odom_xy_step_std=0.025, odom_xy_bias_std=0.025,
        odom_yaw_step_std=1.5, odom_yaw_bias_std=1.5,
    ),
)
"""Measurement noise, weak/nominal/strong drawn 60/30/10 per episode. A separate instance
from the fan's ``GO2_LIDAR_NOISE_CFG``, which still carries ``LidarNoiseCfg``'s class
defaults -- the magnitudes above override them and move this sensor only. The ramp fields
are inert at their defaults (``start_iteration == full_iteration`` means full magnitude from
step 0); ``scale=0.0`` gives a clean reference run.

Halved 2026-09-18: every magnitude in all three conditions is exactly half what it was, and
the mix stays 60/30/10. This is a training decision, not a calibration -- see below for what
the hardware actually measures, which is lower still. Noise here is a robustness budget; the
policy should not be tuned to the one unit, one surface and one posture that got measured.

What the hardware measures
==========================
A stationary Go2 on flat ground, 299 s of rosbag (``bag_sport_0910_2012``) through the
deployed MID-360 + RKO-LIO + ``heightmap_generator`` stack, 2,941 published maps. The robot
does not move, so each cell's variation over time *is* the pipeline's noise, measured where
it matters -- at the map the policy reads, not at the raw range:

    per-cell std          2.12 mm mean, 2.00 mm median, 4.33 mm worst
    per-cell peak-to-peak 19.9 mm mean
    valid_ratio           ~1.0 on all 170 cells outside the self-crop
    distance dependence   weak; 1.5-3 mm across the whole 1.6 x 1.0 m grid

Two things to carry away from that. First, **the delivered map is very quiet** -- quieter
than any condition here, including weak. Second, that is an aggregation result rather than a
clean sensor: the deployed stack voxel-filters at 3 cm into 10 cm cells and averages roughly
a hundred points per cell, which is how a ~2 cm per-point range error becomes a 2 mm cell.
This map bins ~2,000 rays into 609 cells at 5 cm and takes an ``amin``, so most cells see one
to three returns, the per-ray error passes through nearly intact, and ``amin`` over a couple
of noisy samples is biased on top. The same per-ray number therefore lands very differently
in the two pipelines, and the magnitudes here have to be read as *map* noise, not as the
sensor's per-point precision.

Which terms that measurement can and cannot constrain:

  * ``range_std`` and ``tilt_step_std`` -- constrained, and both sit above what it found.
    Deliberately: a tilt about the mount lifts a cell by ``r sin(theta)``, so at this grid's
    0.7 m reach nominal's 0.5 deg is 6 mm, three times the measured mean. The bag is
    stationary and RKO-LIO's gravity direction is best observed exactly then, so nothing in
    it covers the walking case.
  * ``outlier_prob`` / ``outlier_range`` -- the measurement says nothing gross ever reaches
    the deployed map: across ~500k cell observations the largest excursion any cell made was
    20 mm. Kept anyway, at half the old rate, because the term models an equipment fault
    rather than the sensor's noise floor, and one 300 s bag from a healthy unit cannot bound
    how often a faulty one misbehaves. Making the magnitude proportional to range was
    considered for this and **rejected**: a wild reading from a broken unit does not know how
    far the target was.
  * ``tilt_episode_std`` -- not constrained. A fixed mounting error is constant in time and
    contributes exactly zero to a temporal std; this bag cannot see it at all.
  * ``odom_*`` -- not constrained. ``_corrupt_odometry`` scales with distance travelled and
    the robot was stationary, so none of these terms were active during the measurement.

The near-face spray, and why the legs were never the cause
==========================================================
``_perturb`` used to write ``(distance + error).clamp(min=0.0)``, so a short outlier landed
between the ground and the sensor -- at the limit exactly on the mount. Every ray leaves the
L1 at the nose and the map keeps the highest return per cell (an ``amin`` on these inverted
heights), so those always won their cell, and the blind disc inside 0.143 m is never
refreshed, so nothing cleared them: a one-sided cone of raised cells with its apex at the
robot's face. It now takes ``min_range`` (``MID360_MIN_RANGE``) and reports such a ray as a
non-return instead, which is what the hardware does -- no LiDAR publishes a range below its
own minimum.

This is worth stating plainly because it is easy to attribute to self-occlusion and it is
not: measured cell-for-cell against true terrain, the innermost band ran 46.3% of cells more
than 5 cm high *with* noise and 4.5% without, and turning the occluder on *reduced* the
overall rate (7.1% against 9.1%) rather than causing it. Making the body transparent does
nothing to the cone either way. Outliers are kept at a rate that still produces it -- half
the old one -- so the ``min_range`` dropout is the only thing holding it down.

Both halves checked back in sim, same protocol as the bag
=========================================================
Standing still on a flat *generated* tile, trained Phase 4 policy, 8 envs x 500 steps (10 s),
per-cell std over the frames a beam actually landed in that cell. Left column is this config;
right is the same run with the ``min_range`` dropout disabled, i.e. the old clamp:

                                    min_range 0.2    old clamp     bag
    per-cell std, mean                   4.96 mm       6.47 mm    2.12 mm
    per-cell std, median                 4.11          4.13       2.00
    per-cell std, p95                    9.34         19.29          -
    per-cell peak-to-peak, mean         35.4          46.5       19.9

    near-face spray, cells reading >5 cm above the true flat ground
      0.000-0.143 m (the blind disc)     3.45%        13.16%
      0.143-0.30 m                       2.20%         5.58%
      overall                            0.61%         1.58%
    same, >15 cm
      0.000-0.143 m                      0.00%         6.01%
      overall                            0.00%         0.29%

So the cone is a 13% effect in the blind disc even at these halved magnitudes, and the
dropout takes it to 3.5% and removes every spike over 15 cm. Note the median std barely
moves between the two columns -- the fix is entirely in the tail, which is what an outlier
artifact should look like. Against the bag, this config sits at about 2x the delivered
hardware noise, which is the intended margin.

Two measurement notes, both of which cost a run to find:

  * Do **not** use ``terrain_type="plane"`` for anything at this scale. That one enormous
    quad raycasts in float32 to about a centimetre, and it put +-40 mm of spread on a
    nominally flat ground -- which reads exactly like sensor noise and is ten times what is
    being measured. A generated flat tile is a finite mesh and comes out at 1.4 mm.
  * Compute the statistic over the frames a cell was actually measured. A held cell repeats
    its last value, and a cell no beam has ever reached holds ``flat_fill``, a convention
    rather than a measurement; including those frames measures the fill policy. It matters
    here because only ~25% of cells are measured per step, against the bag's ~100%.

Three of the six augmentations in the reference paper (see ``mdp/lidar_elevation_map.py``)
stay off because this sensor produces them from geometry rather than from a model:
*Pruning* -- the ray budget over 609 cells, plus the 10 Hz elevation sweep, already leaves
whole annuli unmeasured for several steps at a time, which is a more realistic
temporally-correlated dropout than random patch removal; *Height* and *Robot Pose* -- a held
cell carries whatever the terrain looked like when a beam last landed there, and is
transported forward on simulated odometry rather than re-measured.

Deliberately **not** modelled, and worth knowing before trusting a sim-to-real number:

  * *Incidence-angle dropout.* The mount is 0.273 m up and rays meet the ground at
    11..62 deg; a real return weakens and vanishes at grazing incidence, while the
    raycast always hits. This is the largest remaining gap, and it correlates with
    range -- the sim is most optimistic exactly where the map is thinnest. Note the
    measurement above found ``valid_ratio`` ~1.0 everywhere outside the self-crop, so on
    flat ground at rest it costs nothing; a sloped or distant surface is another matter.
  * *Reflectivity dropout.* Dark, wet and specular surfaces return nothing; there is no
    material information in the raycast to key off.
  * *Motion distortion.* A frame's points are acquired over 20 ms while the body moves,
    but every ray here is cast from one pose. The deployed stack deskews with RKO-LIO.

Self-occlusion is deliberately *not* on this list. The body is transparent by default and
that is the intended match to hardware, whose publisher filters its own points out; see
``MID360_DYNAMIC_MESH`` for the switch and the reasoning.
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
            "min_range": MID360_MIN_RANGE,
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
    ``MID360_DYNAMIC_MESH`` for what it costs and why it defaults off. In play it is the one
    switch worth flipping back and forth: off, the map under the robot is green everywhere,
    matching the self-filtered cloud the hardware publishes; on, a leg crossing the field of
    view leaves a streak that sweeps with the gait -- raised cells under ``"hit"``, red (held)
    ones under ``"drop"``.
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
