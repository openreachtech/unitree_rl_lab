"""Blind Phase 4, VLFM flavor: robustness to low obstacles, at nav speeds.

Phase 4 trains the hardest thing this lineage does -- clearing an isolated 25 cm wall.
That is not what the VLFM exploration run needs. There the robot walks a furnished
apartment at Nav2's pace and keeps brushing things: chair legs, thresholds, rug edges,
the lip of a sofa base. Observed 2026-09-30 in kujiale, a light contact was enough to
put it on the floor and end the run (every reset teleports the robot home, which
invalidates the SLAM map, so play_ros2.py stops rather than overlay a second floor
plan).

So this config keeps Phase 4's rewards and its wall types but retunes both axes toward
that job:

* **Terrain: 4 columns x 10 rows -- flat, rough, wall, floating wall.** Phase 4 is
  walls only (2:1 solid to floating, no flat share). Half this grid is ground the robot
  will actually spend its time on, which is also where a fall costs a run. ``rough``
  reuses Phase 2's exact settings rather than inventing a third roughness, so the
  terrain the policy already learned is the terrain it keeps.

* **Walls 5-15 cm, not 5-25 cm.** Apartment obstacles are low. 25 cm walls buy a skill
  the exploration run never exercises while making the curriculum harder to ratchet
  through, and the curriculum collapsing is a documented failure mode of this phase
  (see the fixed-height note in velocity_env_cfg_blind_phase4.py).

* **Top speed 1.2 -> 1.0 m/s.** The Nav2 overlay (go2_nav_bringup/params/nav2_go2.yaml)
  caps vx at 1.0, so everything above that is envelope the deployed stack never
  commands. Sampling it only dilutes the range that matters.

Resume from a Phase 4 checkpoint:

    --task Go2-Blind-GRU-Phase4-VLFM --resume --previous-task Go2-Blind-GRU-Phase4
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase4 import (
    RewardsCfgPhase4,
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# Walls top out here instead of Phase 4's 0.25 m. Also feeds the clearance reward below,
# which converts obstacle height to a 0-1 ratio against this number.
VLFM_MAX_WALL_HEIGHT = 0.15
VLFM_WALL_HEIGHT_RANGE = (0.05, VLFM_MAX_WALL_HEIGHT)

PHASE4_VLFM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # Equal proportions over 4 columns put one sub-terrain in each: TerrainGenerator
    # assigns sub-terrains to columns by proportion and difficulty to rows. Rows stay at
    # 10 to keep the 10 levels the curriculum ratchets through.
    num_cols=4,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Flat ground is half the point. The apartment floor is flat, and a policy that
        # only ever trains on obstacles has no reason to be stable without one.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # Phase 2's settings verbatim -- same noise range, step and border.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # Phase 4's two wall types, heights capped at 0.15 m. Thickness still narrows
        # 0.15 -> 0.05 m with difficulty, and spacing stays 0.60 m: the walls get thinner
        # and harder to feel, which is the discrimination worth keeping, while the height
        # stays in the range an apartment actually contains.
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=VLFM_WALL_HEIGHT_RANGE,
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
        "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=VLFM_WALL_HEIGHT_RANGE,
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase4VLFM(RobotSceneCfgPhase4):
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=PHASE4_VLFM_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class RewardsCfgPhase4VLFM(RewardsCfgPhase4):
    """Phase 4's rewards, with the clearance target rescaled to the lower walls.

    ``calf_flexion_clearance`` divides the measured obstacle height by
    ``max_obstacle_height`` to get a 0-1 ratio, then scales the target calf-flexion
    angle by it. Left at Phase 4's 0.25 m while the tallest wall here is 0.15 m, the
    target would never exceed 60% of its range and the policy would be under-asked to
    bend on the very obstacles this config exists to teach.
    """

    calf_flexion_clearance = RewardsCfgPhase4().calf_flexion_clearance.replace(
        params={
            **RewardsCfgPhase4().calf_flexion_clearance.params,
            "max_obstacle_height": VLFM_MAX_WALL_HEIGHT,
        }
    )


@configclass
class RobotEnvCfgPhase4VLFM(RobotEnvCfgPhase4):
    """Phase 4 retuned for the VLFM exploration run."""

    scene: RobotSceneCfgPhase4VLFM = RobotSceneCfgPhase4VLFM(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase4VLFM = RewardsCfgPhase4VLFM()

    def __post_init__(self):
        super().__post_init__()
        # Command envelope, capped at what the deployed Nav2 overlay actually sends.
        # Lateral is reduced but NOT removed: Nav2 runs MPPI in DiffDrive and never
        # commands vy, but the policy rejects sideways disturbances with the same
        # machinery it strafes with, and a blind robot in a furnished room needs that.
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        )


# ---------------------------------------------------------------------------
# Play: one column per sub-terrain, four difficulty rows, so a row shows flat, rough,
# solid wall and floating tread at the same difficulty side by side.
# ---------------------------------------------------------------------------
PLAY_TERRAIN_CFG_PHASE4_VLFM = PHASE4_VLFM_TERRAIN_CFG.replace(
    num_rows=4,
    num_cols=4,
    sub_terrains={
        name: (
            cfg.replace(wall_height_range=(0.0625, 0.1875), wall_thickness_range=(0.05, 0.05))
            if name.endswith("wall")
            else cfg.replace()
        )
        for name, cfg in PHASE4_VLFM_TERRAIN_CFG.sub_terrains.items()
    },
)
"""4 x 4: every sub-terrain at each of four difficulties.

Rows are difficulty, derived by TerrainGenerator as (row + jitter) / num_rows with the
jitter uniform on [0, 1), so a row is a band and an exact height cannot be requested.
wall_height_range is set so the four bands *centre* on 7.5 / 10 / 12.5 / 15 cm: with
(0.0625, 0.1875) over 4 rows they span 6.25-8.75, 8.75-11.25, 11.25-13.75 and
13.75-16.25 cm.

Thickness is pinned at 5 cm rather than narrowing with difficulty, so height is the only
thing that changes down a wall column -- the training mix's hardest thickness, so every
row here is a thin wall.
"""


@configclass
class RobotPlayEnvCfgPhase4VLFM(RobotEnvCfgPhase4VLFM):
    scene: RobotSceneCfgPhase4VLFM = RobotSceneCfgPhase4VLFM(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # .replace() rather than mutating the module-level cfg's sub_terrains dict, which
        # is shared with the training config.
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG_PHASE4_VLFM.copy()
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
