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

* **Terrain: 5 columns x 10 rows -- flat, rough, wall, floating wall, posts.** Phase 4
  is walls only (2:1 solid to floating, no flat share). Two of these columns are ground
  the robot will actually spend its time on, which is also where a fall costs a run.
  ``rough`` reuses Phase 2's exact settings rather than inventing a third roughness, so
  the terrain the policy already learned is the terrain it keeps. ``posts`` was added
  2026-10-05, after the policy kept hooking a foot on chair legs in kujiale -- see its
  comment for why neither a wall nor a push teaches that.

* **Walls 5-15 cm, not 5-25 cm.** Apartment obstacles are low. 25 cm walls buy a skill
  the exploration run never exercises while making the curriculum harder to ratchet
  through, and the curriculum collapsing is a documented failure mode of this phase
  (see the fixed-height note in velocity_env_cfg_blind_phase4.py).

* **Stronger disturbance.** ``push_robot`` goes from +-0.5 m/s every 5-10 s to +-0.8 m/s
  every 3-6 s, and gains a +-0.8 rad/s yaw component. See the comment in ``__post_init__``.

* **Top speed 1.2 -> 1.0 m/s.** The Nav2 overlay (go2_nav_bringup/params/nav2_go2.yaml)
  caps vx at 1.0, so everything above that is envelope the deployed stack never
  commands. Sampling it only dilutes the range that matters.

Resume from a Phase 4 checkpoint:

    --task Go2-Blind-GRU-Phase4-VLFM --resume --previous-task Go2-Blind-GRU-Phase4
"""

from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.trimesh.utils import make_cylinder
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

# Post thickness, metres. A chair leg is at the bottom of this range and a table leg or
# a sofa foot near the top; one range covers the lot, and a blind policy has to cope
# with all of it on the same floor.
POST_RADIUS_RANGE = (0.01, 0.15)


def make_random_post(radius: float, height: float, center, max_yx_angle: float = 0.0, degrees: bool = True):
    """``make_cylinder`` with the radius drawn per post instead of fixed.

    The repeated-objects terrain randomises two of the three things that matter here by
    itself -- the tilt (``max_yx_angle`` is already a per-object cap that each post
    samples under) and the length (``abs_height_noise`` is drawn per object) -- but the
    radius it passes is a single interpolated number shared by every post in the tile.
    A floor of identical columns is not what the robot trips over, so the incoming
    ``radius`` is discarded and a fresh one drawn from ``POST_RADIUS_RANGE``.

    Signature must match ``make_cylinder``: the terrain function calls it with keyword
    arguments built from the cylinder ObjectCfg.
    """
    del radius  # intentionally ignored -- see above
    return make_cylinder(
        radius=float(np.random.uniform(*POST_RADIUS_RANGE)),
        height=height,
        center=center,
        max_yx_angle=max_yx_angle,
        degrees=degrees,
    )


PHASE4_VLFM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # Equal proportions over 5 columns put one sub-terrain in each: TerrainGenerator
    # assigns sub-terrains to columns by proportion and difficulty to rows. Rows stay at
    # 10 to keep the 10 levels the curriculum ratchets through.
    num_cols=5,
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
        # Chair legs. A wall is crossed by all four feet at the same place, so it
        # teaches a symmetric step-over; what put the robot on the floor in kujiale was
        # ONE swing leg catching a chair leg while the other three stood on clear
        # ground, which yaws the body instead of pitching it. No wall produces that, and
        # neither does push_robot (that drives the trunk and leaves the legs free), so
        # before this column the failure was not in the training distribution at all.
        #
        # Tilted, not upright: the posts lean up to max_yx_angle, which gives glancing
        # contacts and overhangs rather than a clean vertical face. The mesh helper puts
        # each cylinder's CENTRE at z=0, so half is buried and the post stays rooted when
        # it leans -- above-ground height is length/2, hence the doubled numbers below.
        # It also builds them with 4-5 sections, so they come out as square or pentagonal
        # prisms, with the edges a real chair leg has.
        #
        # DIFFICULTY MOVES DENSITY ONLY. Length, radius and tilt are drawn per post over
        # their full range at every level, because a furnished room does not get its
        # furniture thinner as you walk further in -- what changes is how much of it
        # there is. Tying the three to the curriculum would also mean the easy rows never
        # show a thin leg and the hard rows never show a thick one, when both have to be
        # handled on the same floor.
        #
        #   length  0.20-0.60 m  -> 0.10-0.30 m above ground, via abs_height_noise about
        #                           a fixed 0.40 m. The top end sits just under the
        #                           ~0.32 m trunk: a post the trunk cannot clear is a
        #                           wall, and this grid already has two columns of those.
        #   radius  0.01-0.15 m  -> make_random_post, since the stock terrain shares one
        #                           interpolated radius across every post in a tile.
        #   tilt    0-45 deg     -> max_yx_angle is ALREADY a per-post cap that each post
        #                           samples under, so holding it equal at both ends gives
        #                           a fresh random lean every time.
        "posts": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            # No border_width: unlike the wall terrains, which define that field
            # themselves, the repeated-objects family inherits plain SubTerrainBaseCfg
            # and would reject it. platform_width is the only clearance it has -- it
            # keeps the spawn point free of posts.
            platform_width=1.5,
            object_type=make_random_post,
            # Drawn per post and added to the 0.40 m below, giving 0.20-0.60 m.
            abs_height_noise=(-0.20, 0.20),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                # radius is ignored by make_random_post; kept because ObjectCfg requires it
                num_objects=30, height=0.40, radius=0.05, max_yx_angle=45.0,
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=180, height=0.40, radius=0.05, max_yx_angle=45.0,
            ),
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
        # Harder shoves, more often, and now about the yaw axis too. The kujiale runs
        # kept ending with the robot down -- brushing furniture, losing it mid-turn --
        # and a policy that only ever recovers from a gentle nudge every 5-10 s has not
        # been asked to do much. Yaw is new and deliberate: MPPI runs in DiffDrive now,
        # so turning in place is the only way the robot changes heading, and that is
        # exactly when it was going over.
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-0.8, 0.8),
        }
        # Command envelope, capped at what the deployed Nav2 overlay actually sends.
        # Lateral is reduced but NOT removed: Nav2 runs MPPI in DiffDrive and never
        # commands vy, but the policy rejects sideways disturbances with the same
        # machinery it strafes with, and a blind robot in a furnished room needs that.
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        )


# ---------------------------------------------------------------------------
# Play: one column per sub-terrain, three difficulty rows, so a row shows flat, rough,
# solid wall, floating tread and posts at the same difficulty side by side.
# ---------------------------------------------------------------------------
PLAY_TERRAIN_CFG_PHASE4_VLFM = PHASE4_VLFM_TERRAIN_CFG.replace(
    num_rows=3,
    num_cols=5,
    # Without this the generator samples BOTH the sub-terrain and the difficulty of
    # every tile at random (TerrainGenerator._generate_random_terrains), which is what a
    # play grid must not do -- types end up mixed down a column and a row is not one
    # difficulty. curriculum=True switches to the deterministic layout this grid is for:
    # column index picks the sub-terrain off the cumulative proportions, row index sets
    # the difficulty.
    #
    # The base RobotEnvCfg.__post_init__ does set this flag, but on whatever generator is
    # attached at the time; RobotPlayEnvCfgPhase4VLFM then swaps the whole generator for
    # this one, so the value has to be part of this config rather than inherited.
    curriculum=True,
    sub_terrains={
        name: (
            cfg.replace(wall_height_range=(0.025, 0.175), wall_thickness_range=(0.05, 0.05))
            if name.endswith("wall")
            else cfg.replace()
        )
        for name, cfg in PHASE4_VLFM_TERRAIN_CFG.sub_terrains.items()
    },
)
"""5 x 3: every sub-terrain at each of three difficulties.

Column order follows the sub_terrains dict, the proportions all being equal:
flat | random_rough | thin_wall | floating_thin_wall | posts. Row 0 is the easiest.

Rows are difficulty, derived by TerrainGenerator as (row + jitter) / num_rows with the
jitter uniform on [0, 1), so a row is a band and an exact height cannot be requested.
wall_height_range is set so the three bands *centre* on 5 / 10 / 15 cm -- the ends of
the training range and its middle: with (0.025, 0.175) over 3 rows they span 2.5-7.5,
7.5-12.5 and 12.5-17.5 cm.

Thickness is pinned at 5 cm rather than narrowing with difficulty, so height is the only
thing that changes down a wall column -- the training mix's hardest thickness, so every
row here is a thin wall.

The posts column changes only in density down the rows (30 -> 180 per 8x8 m tile); each
post's length, radius and lean are drawn fresh at every level, so even the front row
shows the full variety.
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
