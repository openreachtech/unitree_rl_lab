"""G1-FPM: perceptive locomotion with foothold-gated height patches (Heyrman et al., "Watch
Your Step", 2026, doc/papers/Watch_Your_Step_Anticipated_Perception_for_Humanoid_Locomotion.md).

One task, trained from scratch (no phases), on top of the G1-GRU env stack:

* **Perception.** The head MID-360 (``mid360_joint`` of g1_29dof_rev_1_0.urdf: upside down,
  ~1.25 m up) is raycast every step (``RollingLivoxSensor``, 2000 rays), binned into an
  odom-aligned 2.8 m x 2.8 m map with the sensor / odometry noise of feat/mid360
  (``OdomElevationMap``). The sensor cannot see the ground within ~0.95 m of the robot, so
  everything under the feet is what was seen earlier. Never-seen cells read as flat ground at
  the lower foot's sole.
* **Footholds.** ``FootholdPatches``: the FPM predicts the next two footholds from
  proprioception and the command; 7 x 7 cells of the map around each go to the actor
  (102 values). Pinned under the base until the policy walks (80 % time-outs, >= 500
  iterations), then refit every 500.
* **Network.** TCN actor (proprioception, 2 s) with the patches beside it; MLP critic over
  3 steps of privileged obs + the clean 1.6 x 1.0 m map + the patches (``G1FpmPPORunnerCfg``).
* **Terrain.** 7 columns x 20 rows: rough | ridge slope | ridge stairs | boxes | concentric
  gaps | stepping stones | square walls. Stock distance curriculum on every column, promoting at 3 m instead of half a tile.
* **Rewards.** GRU-Phase3's, with yaw tracking 1.0, action rate -0.02, joint acc -1e-7, no
  base-height term, foot clearance from the highest ground near each foot.
* **Assist.** An upward force on the torso, 50 % of body weight, fading to 0 at mean terrain
  level 8 (of 20).
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.sensors import LivoxPatternCfg, RollingLivoxSensorCfg
from unitree_rl_lab.tasks.locomotion import mdp

from .gru_env_cfg import (
    FEET,
    FOOT_SCAN_SENSORS,
    PHASE2_TERRAIN_CFG,
    GruObservationsCfg,
    GruPhase1EnvCfg,
    GruPhase3RewardsCfg,
    GruSceneCfg,
)

NOMINAL_HEIGHT = 0.78
"""Pelvis height at nominal stance; map values are ``root_z - terrain_z - NOMINAL_HEIGHT``."""
SOLE_HEIGHT = 0.035
"""ankle_roll_link origin above the sole (URDF contact spheres at z = -0.03, r = 0.005)."""

# MID-360 on torso_link: URDF mid360_joint xyz, rpy = (pi, 0.0511, 0) as (w, x, y, z)
MID360_MOUNT = (0.0002835, 0.00003, 0.428434)
MID360_ROT = (0.0, math.cos(0.05112069379091391 / 2), 0.0, -math.sin(0.05112069379091391 / 2))

MID360_NOISE_CFG = mdp.LidarNoiseCfg(
    weak=mdp.LidarNoiseConditionCfg(
        probability=0.60, range_std=0.005, tilt_step_std=0.25, tilt_episode_std=0.125,
        outlier_prob=0.0025, outlier_range=0.075,
        odom_xy_step_std=0.005, odom_xy_bias_std=0.005, odom_yaw_step_std=0.25, odom_yaw_bias_std=0.25,
    ),
    nominal=mdp.LidarNoiseConditionCfg(
        probability=0.30, range_std=0.01, tilt_step_std=0.5, tilt_episode_std=0.25,
        outlier_prob=0.005, outlier_range=0.15,
        odom_xy_step_std=0.01, odom_xy_bias_std=0.01, odom_yaw_step_std=0.5, odom_yaw_bias_std=0.5,
    ),
    strong=mdp.LidarNoiseConditionCfg(
        probability=0.10, range_std=0.02, tilt_step_std=1.0, tilt_episode_std=0.5,
        outlier_prob=0.015, outlier_range=0.30,
        odom_xy_step_std=0.025, odom_xy_bias_std=0.025, odom_yaw_step_std=1.5, odom_yaw_bias_std=1.5,
    ),
)  # fmt: skip
"""feat/mid360's MID360_NOISE_CFG (Go2, fitted to a MID-360 rosbag with 2x margin)."""

# ---------------------------------------------------------------------------
# Terrain: 7 columns x 20 rows, one sub-terrain per column in definition order
# ---------------------------------------------------------------------------
FPM_TERRAIN_CFG = PHASE2_TERRAIN_CFG.replace(
    num_cols=7,
    num_rows=20,
    sub_terrains={
        "random_rough": PHASE2_TERRAIN_CFG.sub_terrains["random_rough"].replace(proportion=1.0),
        "ridge_slope": mdp.HfRidgeSlopeTerrainCfg(proportion=1.0, slope_range=(0.0, 0.5), border_width=0.25),
        "ridge_stairs": mdp.MeshRidgeStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.05, 0.25), step_width_range=(0.35, 0.275)
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.30), platform_width=2.0
        ),
        "gap": mdp.MeshGapWithFloorTerrainCfg(proportion=1.0, gap_width_range=(0.1, 0.5)),
        "stepping_stones": mdp.MeshSteppingStonesTerrainCfg(
            proportion=1.0, stone_width_range=(0.5, 0.3), gap_width_range=(0.05, 0.2)
        ),
        # square rings: a wall every 1.25 m whichever way the robot walks, so no special rules
        "wall": mdp.MeshThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.03),
            wall_spacing=1.25,
            platform_width=2.0,
        ),
    },
)
HOLE_COLS = (4, 5)


@configclass
class FpmSceneCfg(GruSceneCfg):
    lidar = RollingLivoxSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RollingLivoxSensorCfg.OffsetCfg(pos=MID360_MOUNT, rot=MID360_ROT),
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=4000, downsample=2),
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
        min_range=0.2,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        update_frequency=50.0,
        debug_vis=False,
    )
    # the critic's clean map: 1.6 x 1.0 m, 5 cm, body-centred, yaw-aligned (33 x 21 = 693)
    map_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


FOOTHOLD_PATCHES = ObsTerm(
    func=mdp.FootholdPatches,
    clip=(-1.0, 1.0),
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "feet_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
        "contact_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
        "command_name": "base_velocity",
        "patch_size": 7,
        "bootstrap_footholds": ((0.0, 0.1), (0.0, -0.1)),
        "num_steps_per_env": 24,
        "bootstrap_iterations": 500,
        "refit_interval": 500,
        "walk_threshold": 0.8,  # first refit once 80% of episodes reach the time limit
        "walk_check_iterations": 50,
        "buffer_steps": 150,
        "min_air_time": 0.05,
        "train": True,
        "fit": {"learning_rate": 5e-3, "epochs": 25, "batch_size": 1024, "huber_delta": 0.01},
        "map_params": {
            "sensor_cfg": SceneEntityCfg("lidar"),
            "offset": NOMINAL_HEIGHT,
            "resolution": 0.05,
            "size": 2.8,
            "sole_height": SOLE_HEIGHT,
            "noise": MID360_NOISE_CFG,
            "min_range": 0.2,
            "debug_vis": False,
        },
        "debug_vis": False,
    },
)


@configclass
class FpmObservationsCfg(GruObservationsCfg):
    """policy: proprioception (TCN window) | foothold_patches: current only, actor and critic |
    critic: privileged, stacked 3 by the runner | height_map_clean: current only."""

    @configclass
    class FootholdPatchesCfg(ObsGroup):
        foothold_patches = FOOTHOLD_PATCHES

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightMapCleanCfg(ObsGroup):
        height_map_clean = ObsTerm(
            func=mdp.height_scan_root,
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("map_scanner"), "offset": NOMINAL_HEIGHT},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    foothold_patches: FootholdPatchesCfg = FootholdPatchesCfg()
    height_map_clean: HeightMapCleanCfg = HeightMapCleanCfg()


@configclass
class FpmRewardsCfg(GruPhase3RewardsCfg):
    track_ang_vel_z = GruPhase3RewardsCfg().track_ang_vel_z.replace(weight=1.0)
    action_rate = GruPhase3RewardsCfg().action_rate.replace(weight=-0.02)
    joint_acc = GruPhase3RewardsCfg().joint_acc.replace(weight=-1.0e-7)
    base_height = None
    feet_clearance = GruPhase3RewardsCfg().feet_clearance.replace(
        func=mdp.foot_clearance_local_max_reward,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "radius": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "sensor_cfgs": FOOT_SCAN_SENSORS,
        },
    )


@configclass
class FpmEnvCfg(GruPhase1EnvCfg):
    scene: FpmSceneCfg = FpmSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: FpmObservationsCfg = FpmObservationsCfg()
    rewards: FpmRewardsCfg = FpmRewardsCfg()

    def __post_init__(self):
        self.scene.terrain.terrain_generator = FPM_TERRAIN_CFG.replace()  # a copy: play cfgs edit it
        self.scene.terrain.max_init_terrain_level = 2
        super().__post_init__()  # turns on the generator's curriculum mode (terrain_levels)
        self.scene.map_scanner.update_period = self.decimation * self.sim.dt
        self.scene.lidar.update_period = self.decimation * self.sim.dt

        # -- terminations (Phase 2 + holes); no termination penalty
        self.terminations.base_height = self.terminations.base_height.replace(
            func=mdp.root_height_below_minimum_terrain,
            params={"minimum_height": 0.2, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=mdp.terrain_out_of_bounds, params={"distance_buffer": 3.0}, time_out=True
        )
        self.terminations.fell_into_hole = DoneTerm(
            func=mdp.fell_into_hole, params={"terrain_types": HOLE_COLS, "nominal_height": NOMINAL_HEIGHT}
        )

        # -- commands: vx up to 1.0 (the map is ~1 m of look-ahead); level curriculum from small
        cmd = self.commands.base_velocity
        cmd.limit_ranges = cmd.limit_ranges.replace(lin_vel_x=(-0.5, 1.0))

        # -- curriculum: the stock rule on every column, promoting at 3 m instead of 4 (half a
        #    tile); logging of the FPM and the assist
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel_walls,
            params={
                "fail_termination_names": ("bad_orientation", "base_height", "fell_into_hole"),
                "up_distance": 3.0,  # past the first obstacle on every terrain
            },
        )
        self.curriculum.fpm_val_error_1 = CurrTerm(func=mdp.fpm_val_error_1)
        self.curriculum.fpm_val_error_2 = CurrTerm(func=mdp.fpm_val_error_2)
        self.curriculum.fpm_label_rate = CurrTerm(func=mdp.fpm_label_rate)
        self.curriculum.fpm_walk_rate = CurrTerm(func=mdp.fpm_walk_rate)
        self.curriculum.assist_fraction = CurrTerm(func=mdp.assist_fraction)

        # -- assistive upward force, through the push term (one force write per body)
        self.events.push_robot.params["assist_max_fraction"] = 0.5
        self.events.push_robot.params["assist_cutoff_level"] = 8.0


@configclass
class FpmPlayEnvCfg(FpmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 5 columns (no rough / slope) x 3 rows (bottom / mid / top); robots stay on their tile
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k not in ("random_rough", "ridge_slope")}
        gen.num_cols = len(gen.sub_terrains)
        gen.num_rows = 3
        self.terminations.fell_into_hole.params["terrain_types"] = (2, 3)
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 45
        self.scene.terrain.max_init_terrain_level = 2
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.events.push_robot.params["assist_max_fraction"] = 0.0
        patches = self.observations.foothold_patches.foothold_patches
        patches.params["train"] = False
        patches.params["debug_vis"] = True
        patches.params["map_params"]["debug_vis"] = True
