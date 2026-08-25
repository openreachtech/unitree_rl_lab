"""LiDAR fan and the height grid it builds, for the blind lineage's play configs.

Nothing here defines a task. ``robots/go2/velocity_env_cfg_blind.py`` pulls
``GO2_LIDAR_SCANNER_CFG``, ``LidarMapObsCfg`` and ``PLAY_LIDAR_HEIGHT_SCAN_CFG`` out of
this module to hang the fan-built map off the Go2-Blind-GRU play environments, where it
is drawn but never fed to the policy. The eventual use is the other way round: this map,
noise and all, is what a height-map encoder will be trained to clean up while the blind
policy stays frozen.

See ``mdp/lidar_elevation_map.py`` for why the map is built from a fan rather than a
top-down raycast, and for which of the reference paper's augmentations that makes
unnecessary. ``scripts/tools/check_lidar_map_coverage.py`` predicts coverage
analytically -- run it before changing any fan parameter.

Watch the markers in play: green is measured this step, red is held from an earlier one.
At the settings below, 35.8% of cells are unobserved on flat ground (48.6% beyond 0.50 m)
and 48.7% in front of a 25 cm wall -- so the wall accounts for only 12.9 of those points.
This fan is deliberately sparse, and the unobserved-rate diagnostics have to be read
against the flat-ground baseline rather than against zero.
"""

from __future__ import annotations

import copy
import math


from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.lidar_elevation_map import LidarNoiseCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    _cropped_grid_pattern_num_points,
    GO2_HEIGHT_SCAN_CENTER_X,
    GO2_HEIGHT_SCAN_CENTER_Y,
    GO2_HEIGHT_SCAN_OFFSET,
    GO2_NOMINAL_BASE_Z,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    ObservationsCfgGo2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# ---------------------------------------------------------------------------
# Mount. Base frame; the front-hip line, on top of the trunk.
# ---------------------------------------------------------------------------
GO2_LIDAR_MOUNT = (0.1934, 0.0, 0.20)
"""LiDAR position relative to the base origin. x/y is the midpoint of the front hip
joints, measured off the Go2 USD (FL/FR_hip at x=+0.19340, y=+-0.04650, z=0). z puts
the sensor ~11 cm clear of the trunk, whose USD bound tops out at +0.0888."""

GO2_LIDAR_GROUND_HEIGHT = GO2_NOMINAL_BASE_Z + GO2_LIDAR_MOUNT[2]  # 0.52 m
"""Height above flat ground at nominal stance. Sets every shadow length below."""

# ---------------------------------------------------------------------------
# Body exclusion, widened from the top-down tasks' 0.30/0.20 and kept local to
# this module: changing GO2_BODY_HALF_EXTENT_* globally would resize every Go2
# task's observation and invalidate their checkpoints.
#
# The near field is a genuine blind cone for a fan fired from one point, so the
# grid may as well stop where the fan does. Pushing the nearest kept cell from
# 0.157 m out to 0.257 m raises the steepest beam it has to reach from 73.2 to
# 63.7 deg, and the vertical span shrinks with it -- 29 channels become 24.
# Widening it further keeps paying, but at close to one observed cell per ray
# saved, and the cells being sold are where the next footfall lands.
# ---------------------------------------------------------------------------
GO2_LIDAR_BODY_HALF_EXTENT_X = 0.40
GO2_LIDAR_BODY_HALF_EXTENT_Y = 0.30

LIDAR_EXTERO_DIM = _cropped_grid_pattern_num_points(
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    (GO2_HEIGHT_SCAN_CENTER_X, GO2_HEIGHT_SCAN_CENTER_Y),
    (GO2_LIDAR_BODY_HALF_EXTENT_X, GO2_LIDAR_BODY_HALF_EXTENT_Y),
)
"""388 cells (609 minus the 17x13 exclusion), against the top-down tasks' 492."""

GO2_FLAT_SCAN_VALUE = GO2_NOMINAL_BASE_Z - GO2_HEIGHT_SCAN_OFFSET
"""Grid value meaning "flat ground at nominal stance"; what a cell holds before
the fan has ever reached it. Non-zero because ``GO2_HEIGHT_SCAN_OFFSET`` is the
ground-to-LiDAR distance of the *old* mount rather than the base height."""

# ---------------------------------------------------------------------------
# Fan geometry, derived from the grid it has to cover rather than from a
# datasheet -- swap in the real sensor's numbers once it is chosen.
#
# Horizontal: a full turn, so every cell of the grid is at least nominally in view.
# Two caveats come with that and neither is modelled here:
#
#   * On the real robot the rear is a permanent blind spot -- the mount clears the
#     trunk by 11 cm, so the line of sight astern grazes the trunk top and does not
#     reach the ground for metres. RayCaster only supports static meshes, so it never
#     hits the robot and the sim sees rearward freely. The sim is therefore *more*
#     optimistic than the hardware behind the robot, and the held value in those
#     cells is closer to the truth than the measurement is.
#   * The rear corners sit 1.02 m out against the forward corners' 0.71 m, and the
#     angular steps below were sized for 0.71 m. Keeping them means the rear is
#     sampled about 1.4x more coarsely than the front.
#
# Vertical: the span is set by what must be *seen*, not by flat ground. The top of a
# 25 cm obstacle at the forward far edge sits only 20.8 deg below horizontal, well
# outside the 36 deg that flat ground alone would ask for -- clip the FOV there and
# every far wall top falls outside the fan, which is precisely the reading the policy
# needs to decide whether to step over or climb. The span is *not* re-widened to
# 14.8 deg for the 1.02 m rear corner: measured, that buys 0.7 points of flat-ground
# coverage for 208 more rays, because azimuth is what limits the rear, not elevation.
# ---------------------------------------------------------------------------
_FAR_EDGE_RANGE = math.hypot(HEIGHT_SCAN_SIZE[0] / 2 - GO2_LIDAR_MOUNT[0], HEIGHT_SCAN_SIZE[1] / 2)
"""Horizontal distance to the farthest forward grid corner (0.712 m)."""

LIDAR_H_FOV = (-180.0, 180.0)
"""A full turn. ``lidar_pattern`` drops the duplicate ray where the ends meet, so this
gives 45 azimuths at the resolution below, not 46."""

LIDAR_H_RES = 8.0
"""deg. Deliberately coarser than the 4.0 deg that would put one beam per cell at the
forward far edge (0.05 / 0.712 rad): at 8 deg the arc there is 9.9 cm, and 14.3 cm at
the 1.02 m rear corner, so most far cells get no return even on flat ground.

This is the fan's binding limit -- azimuth, not elevation, is what leaves the far band
sparse, which is why widening the vertical span for the rear corner was measured and
rejected above.

It is a decision to model a sparse sensor rather than a dense one. Defensible on
realism: a real scanning LiDAR is sparse at range too, the pattern is fixed in the body
frame so the gaps sweep across the terrain as the robot advances, and the held previous
value covers them the way an accumulated elevation map does.

The cost is diagnostic, and worth knowing before reading the logs. Measured on flat
ground -- where there is no occlusion at all -- 35.8% of cells go unobserved, rising to
48.6% beyond 0.50 m. So the unobserved rate does not separate "occluded" from "not
sampled": in front of a 25 cm wall it reads 48.7%, of which only 12.9 points are the
shadow. Always compare against the flat-ground baseline, never against zero. For
reference, 4 deg over the forward half alone reads 1.0% flat and resolves 63 shadow
cells against this configuration's 50."""

LIDAR_V_FOV = (-64.0, -20.0)
"""deg below horizontal, rounded outward from the geometry: flat ground at the nearest
observed cell (0.257 m -> 63.7 deg) through the top of a 25 cm obstacle at the
farthest (0.712 m -> 20.8 deg)."""

LIDAR_CHANNELS = 24
"""1.91 deg spacing, still about one cell per beam radially: on flat ground at the far
edge 1 deg moves the landing point 2.6 cm. Note this is only true near the grid edge --
the same 1.91 deg is 2 cm at the nearest cell and 13 cm out at 20 deg, which is why the
farthest cell always sets the requirement and the near field ends up oversampled.

Two things brought this down from the 33 the first pass needed: raising the mount from
0.42 m to 0.52 m (steeper look-down, so both a narrower span and less ground per
degree) and widening the exclusion (nearest cell further out, so the steep end of the
span is no longer needed).

The shallow end earns its keep on elevated surfaces rather than on the floor: the
-20 deg beam reaches 1.43 m over flat ground -- past even the rear corner -- but lands
at 0.74 m on a 25 cm top. Clip the span at the flat-ground limit and every distant wall
top drops out of view, which is the one reading needed to choose between stepping over
and climbing."""

GO2_LIDAR_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=GO2_LIDAR_MOUNT),
    # "base", not "yaw": the sensor is bolted to the trunk, so pitch and roll swing
    # the whole fan. The top-down height_scanner uses "yaw" because its grid is
    # meant to stay gravity-aligned; this one must not.
    ray_alignment="base",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=LIDAR_CHANNELS,
        vertical_fov_range=LIDAR_V_FOV,
        horizontal_fov_range=LIDAR_H_FOV,
        horizontal_res=LIDAR_H_RES,
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
"""1080 rays (24 x 45) against the 609 of the top-down grid."""


GO2_LIDAR_NOISE_CFG = LidarNoiseCfg()
"""Measurement noise, 60/30/10 weak/nominal/strong per episode. Turn the whole thing
down or off with ``scale``; see LidarNoiseCfg for what each condition covers and which
of the paper's augmentations the fan already produces without a model."""


def _lidar_height_scan(
    debug_vis: bool = False,
    debug_vis_env_index: int | None = 0,
    noise: LidarNoiseCfg | None = GO2_LIDAR_NOISE_CFG,
) -> ObsTerm:
    """The policy's height-grid term, sourced from the fan.

    Same 492 values, order, units and clip as ``POLICY_HEIGHT_SCAN_CFG``, so this is
    a drop-in swap -- the critic keeps the clean top-down scan as privileged input.
    """
    return ObsTerm(
        func=mdp.LidarElevationMap,
        params={
            "sensor_cfg": SceneEntityCfg("lidar_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "offset": GO2_HEIGHT_SCAN_OFFSET,
            "resolution": HEIGHT_SCAN_RESOLUTION,
            "size": HEIGHT_SCAN_SIZE,
            "scanner_offset_xy": (GO2_HEIGHT_SCAN_CENTER_X, GO2_HEIGHT_SCAN_CENTER_Y),
            "exclude_half_extent_x": GO2_LIDAR_BODY_HALF_EXTENT_X,
            "exclude_half_extent_y": GO2_LIDAR_BODY_HALF_EXTENT_Y,
            "lidar_offset": GO2_LIDAR_MOUNT,
            "horizontal_fov": LIDAR_H_FOV,
            "flat_fill": GO2_FLAT_SCAN_VALUE,
            "noise": noise,
            "debug_vis": debug_vis,
            "debug_vis_env_index": debug_vis_env_index,
        },
        clip=(-1.0, 5.0),
        history_length=0,
    )


def lidar_noise_only(condition: str, base: LidarNoiseCfg = GO2_LIDAR_NOISE_CFG) -> LidarNoiseCfg:
    """Force every episode onto one noise condition, for looking at it in isolation.

    The normal mix draws weak/nominal/strong per episode at 60/30/10, so watching any one
    of them means waiting for it to come up and then knowing which you got. This pins the
    draw instead: the named condition keeps its magnitudes and takes all the probability,
    the other two go to zero.
    """
    names = ("weak", "nominal", "strong")
    if condition not in names:
        raise ValueError(f"condition must be one of {names}, got {condition!r}")
    cfg = copy.deepcopy(base)
    for name in names:
        getattr(cfg, name).probability = 1.0 if name == condition else 0.0
    return cfg


def play_lidar_height_scan(noise: LidarNoiseCfg | None = GO2_LIDAR_NOISE_CFG) -> ObsTerm:
    """The visualising height-grid term, with the noise model of the caller's choosing."""
    return _lidar_height_scan(debug_vis=True, debug_vis_env_index=None, noise=noise)


LIDAR_HEIGHT_SCAN_CFG = _lidar_height_scan()
PLAY_LIDAR_HEIGHT_SCAN_CFG = _lidar_height_scan(debug_vis=True, debug_vis_env_index=None)


@configclass
class LidarMapObsCfg(ObsGroup):
    """Display-only group: nothing reads it, it exists so the fan term runs.

    The policy and critic keep the top-down scan, so this changes no network shape and
    the top-down tasks' checkpoints still load. The observation manager computes every
    group each step, which is what gives the term its chance to draw itself and to
    publish the unobserved-rate diagnostics; the runner simply carries the extra entry
    along unused.
    """

    height_scan = LIDAR_HEIGHT_SCAN_CFG

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
