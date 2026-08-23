"""Go2 envs whose policy height grid comes from a LiDAR fan, not a top-down raycast.

The fan feeds a display-only observation group, not the policy: these tasks are
Phase 1 and Phase 4 with the fan built alongside so it can be looked at, so the
networks keep the top-down scan and those tasks' checkpoints load unchanged. Wiring
the fan into the policy is a later step. See ``mdp/lidar_elevation_map.py`` for why
the fan exists at all.

``Go2-v3-Lidar-Phase1``  Phase 1's flat terrain. Flat ground casts no shadows, so
                         every gap there is the fan's own sparsity -- the baseline
                         everything else is read against, and the check that the frame
                         transform, cell binning and sign convention are right.
``Go2-v3-Lidar``         Phase 4's wall terrain, where the shadows appear.

``scripts/tools/check_lidar_map_coverage.py`` predicts the coverage analytically,
without launching Isaac -- use it before changing any fan parameter. At the settings
below it reports 35.8% of cells unobserved on flat ground and 48.7% in front of a
25 cm wall, so the wall accounts for only 12.9 of those points: this fan is sparse
enough that density gaps, not occlusion, dominate what is missing. That is a
deliberate choice of a sparse sensor over a dense one (see ``LIDAR_H_RES``), and it
means the unobserved-rate logs have to be read against the flat-ground baseline
rather than against zero.

Watch the markers -- green is measured this step, red is held from an earlier one.
Neither variant draws the top-down scan's own markers, so what is on screen is only
the grid the policy is fed:

    ./unitree_rl_lab.sh -p scripts/rsl_rl/play.py --task Go2-v3-Lidar-Phase1 \
        --checkpoint logs/rsl_rl/go2_v3_phase1/<run>/model_499.pt

Note the fan's grid is cropped by the wider exclusion below (388 cells against the
policy's 492), so the drawn cells stop further from the robot than the policy's own
scan would.
"""

from __future__ import annotations

import copy
import math

import isaaclab.terrains as terrain_gen

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
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


def _lidar_height_scan(debug_vis: bool = False, debug_vis_env_index: int | None = 0) -> ObsTerm:
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
            "debug_vis": debug_vis,
            "debug_vis_env_index": debug_vis_env_index,
        },
        clip=(-1.0, 5.0),
        history_length=0,
    )


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


@configclass
class ObservationsCfgLidar(ObservationsCfgGo2):
    """Phase 4's groups untouched, plus the fan-built grid alongside for inspection."""

    lidar_map: LidarMapObsCfg = LidarMapObsCfg()


@configclass
class CurriculumCfgLidar(CurriculumCfg):
    """Unobserved-cell rates, split by band.

    Flat terrain has no occlusion, so whatever these read there is the fan's density
    shortfall; the rise above that baseline on obstacle terrain is the shadow. Read
    them as a pair, never the aggregate alone.
    """

    lidar_unobserved = CurrTerm(func=mdp.lidar_map_unobserved_rate)
    lidar_unobserved_near = CurrTerm(func=mdp.lidar_map_unobserved_near)
    lidar_unobserved_mid = CurrTerm(func=mdp.lidar_map_unobserved_mid)
    lidar_unobserved_far = CurrTerm(func=mdp.lidar_map_unobserved_far)


# ---------------------------------------------------------------------------
# Inspection terrain: one obstacle type per column, heights pinned so the fan's
# behaviour can be attributed to the geometry rather than to which difficulty row
# the robot happened to spawn on. Columns map to sub-terrains in declaration order
# and rows to difficulty (TerrainGenerator._generate_curriculum_terrains), so
# collapsing each range to a single value makes every row in a column identical.
#
# The three heights bracket what the mount can and cannot hide behind. Shadow length
# is distance x h / (0.52 - h):
#   10 cm box   -> 0.24x  a box barely shadows anything from 0.52 m up
#   15 cm step  -> 0.41x
#   20 cm wall  -> 0.63x  the deepest shadow of the three, still under 1x
# ---------------------------------------------------------------------------
INSPECT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=3,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Column 1: 10 cm boxes.
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.10, 0.10),
            platform_width=2.0,
        ),
        # Column 2: inverted pyramid stairs, 15 cm rise. step_width matches Phase 3's.
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.15, 0.15),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # Column 3: 20 cm free-standing walls. Thickness pinned too -- it normally
        # thins with difficulty, which would vary the shadow along the column.
        "wall": terrains.MeshThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=(0.20, 0.20),
            wall_thickness_range=(0.10, 0.10),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)
"""3x3 patches: boxes / stairs / walls, one type per column, uniform down each."""


@configclass
class RobotSceneCfgLidarPhase1(RobotSceneCfgPhase1):
    """Phase 1's flat scene plus the fan. The top-down height_scanner stays: the critic
    still uses it, and it is the ground truth the fan-built grid is compared against."""

    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotEnvCfgLidarPhase1(RobotEnvCfgPhase1):
    """Flat-terrain null test. Same observation layout as Phase 1, so its checkpoints load."""

    scene: RobotSceneCfgLidarPhase1 = RobotSceneCfgLidarPhase1(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgLidar = ObservationsCfgLidar()
    curriculum: CurriculumCfgLidar = CurriculumCfgLidar()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RobotPlayEnvCfgLidarPhase1(RobotEnvCfgLidarPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.lidar_map.height_scan = PLAY_LIDAR_HEIGHT_SCAN_CFG
        # Only the fan-built grid on screen: the top-down scanner's own markers off, and
        # the magenta body-footprint overlay that POLICY_HEIGHT_SCAN_CFG draws off too.
        self.scene.height_scanner.debug_vis = False
        self.observations.policy.height_scan = copy.deepcopy(self.observations.policy.height_scan)
        self.observations.policy.height_scan.params["debug_vis_excluded_body"] = False


@configclass
class RobotSceneCfgLidar(RobotSceneCfgPhase4):
    """Phase 4's scene plus the fan. The top-down height_scanner stays: the critic
    still uses it, and it is the ground truth the fan-built grid is compared against."""

    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotEnvCfgLidar(RobotEnvCfgPhase4):
    scene: RobotSceneCfgLidar = RobotSceneCfgLidar(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgLidar = ObservationsCfgLidar()
    curriculum: CurriculumCfgLidar = CurriculumCfgLidar()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RobotSceneCfgLidarInspect(RobotSceneCfgLidar):
    terrain = RobotSceneCfgLidar().terrain.replace(
        terrain_generator=INSPECT_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class RobotEnvCfgLidarInspect(RobotEnvCfgLidar):
    """Phase 4's MDP on the inspection terrain. Play-shaped by default -- there is
    nothing to train here, the terrain exists to be looked at."""

    scene: RobotSceneCfgLidarInspect = RobotSceneCfgLidarInspect(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.lidar_map.height_scan = PLAY_LIDAR_HEIGHT_SCAN_CFG
        # Only the fan-built grid on screen; see RobotPlayEnvCfgLidarPhase1.
        self.scene.height_scanner.debug_vis = False
        self.observations.policy.height_scan = copy.deepcopy(self.observations.policy.height_scan)
        self.observations.policy.height_scan.params["debug_vis_excluded_body"] = False


@configclass
class RobotPlayEnvCfgLidar(RobotEnvCfgLidar):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.lidar_map.height_scan = PLAY_LIDAR_HEIGHT_SCAN_CFG
        # See RobotPlayEnvCfgLidarPhase1: only the fan-built grid is drawn.
        self.scene.height_scanner.debug_vis = False
        self.observations.policy.height_scan = copy.deepcopy(self.observations.policy.height_scan)
        self.observations.policy.height_scan.params["debug_vis_excluded_body"] = False
