"""MID-360 height map for the wall crossing, built the way the robot builds it.

The sensor and the map are copies from ``ort/feat/mid360`` @ 7886126 (tak): the OmniPerception
``LidarSensor`` with ``RollingLivoxSensor`` (the real MID-360 scan sequence, 4,000 rows per 0.02 s
step, every 2nd cast = 2,000 rays) and ``LidarElevationMap`` (29 x 21 cells of 5 cm around the body,
highest return per cell, a cell no ray reached holds its last value carried with the body, noise
weak/nominal/strong 60/30/10 per episode). Copied rather than imported so the shared locomotion code
stays as it is; the files beside this one say what changed.

What differs from the locomotion setup, and why:

* **Mount**: the real robot's, under the chin -- ``heightmap_mid360_live.launch.py`` of
  ``unitree_go2_locomotion_heightmap`` (porizou, 2026-09-17): base_link -> livox_frame
  x 0.287, y 0, z -0.137, quaternion (x, y, z, w) = (0.70710678, 0.70710678, 0, 0), i.e. upside
  down and turned 90 deg. The locomotion task uses the old L1 pose (pitch 164.9 deg) instead.
* **Heading**: body x minus body z, flattened (see ``lidar_elevation_map._yaw_from_quat``). The
  Euler yaw used everywhere else flips by 180 deg once the stance leans back past vertical.
* **No body exclusion**: every one of the 609 cells is kept. In the stance the wall comes within
  0.4 m of the base, which the real pipeline's under-robot crop would delete.

Values are ``base z - terrain z - 0.273``, the locomotion convention and what
``heightmap_generator`` publishes (``-(h - base.z) - height_offset``).
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from .lidar_elevation_map import LidarElevationMap, LidarNoiseCfg, LidarNoiseConditionCfg
from .patterns_cfg import LivoxPatternCfg
from .rolling_livox_sensor import RollingLivoxSensor, RollingLivoxSensorCfg

MID360_MOUNT_POS = (0.287, 0.0, -0.137)
_S = math.sqrt(0.5)
MID360_MOUNT_ROT = (0.0, _S, _S, 0.0)
"""(w, x, y, z). ROS (x, y, z, w) = (0.70710678, 0.70710678, 0, 0) from the live launch."""

MID360_SAMPLES_PER_STEP = 4000
MID360_RAY_DOWNSAMPLE = 2
MID360_MIN_RANGE = 0.2

MAP_RESOLUTION = 0.05
MAP_SIZE = (1.4, 1.0)
MAP_OFFSET = 0.273175           # GO2_HEIGHT_SCAN_OFFSET = 0.32 - 0.046825
MAP_FLAT_FILL = 0.32 - MAP_OFFSET
MID360_MAP_DIM = 29 * 21

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
"""ort/feat/mid360's MID360_NOISE_CFG (halved on 2026-09-18), unchanged."""


def mid360_scanner_cfg(debug_vis: bool = False) -> RollingLivoxSensorCfg:
    return RollingLivoxSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RollingLivoxSensorCfg.OffsetCfg(pos=MID360_MOUNT_POS, rot=MID360_MOUNT_ROT),
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360", samples=MID360_SAMPLES_PER_STEP, downsample=MID360_RAY_DOWNSAMPLE
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
        min_range=MID360_MIN_RANGE,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        update_frequency=50.0,
        debug_vis=debug_vis,
    )


def mid360_map_term(noise: bool = True, debug_vis: bool = False) -> ObsTerm:
    return ObsTerm(
        func=LidarElevationMap,
        params={
            "sensor_cfg": SceneEntityCfg("mid360_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "offset": MAP_OFFSET,
            "resolution": MAP_RESOLUTION,
            "size": MAP_SIZE,
            "scanner_offset_xy": (0.0, 0.0),
            "exclude_half_extent_x": -1.0,
            "exclude_half_extent_y": -1.0,
            "lidar_offset": MID360_MOUNT_POS,
            "horizontal_fov": (-180.0, 180.0),
            "flat_fill": MAP_FLAT_FILL,
            "noise": MID360_NOISE_CFG if noise else None,
            "min_range": MID360_MIN_RANGE,
            "debug_vis": debug_vis,
            "debug_vis_env_index": 0,
        },
        clip=(-1.0, 5.0),
        history_length=0,
    )
