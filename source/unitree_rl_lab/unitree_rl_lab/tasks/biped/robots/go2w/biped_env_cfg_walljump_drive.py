"""Drive on four wheels and jump walls with one policy.

``Go2w-Biped-WallJump`` made into a driving task with the wall jump inside it, trained on from its
policy (which itself started from ``Go2w-Biped-Switch``'s four-wheel driving). Two kinds of tile:

* **Rough ground** (column 0, a quarter of the robots): four-wheel driving at the full command
  (fore-aft and yaw, no lateral -- the wheels do not roll sideways), the stance and the jump never
  called for. Keeps the driving from being forgotten.
* **A wall** (columns 1-3, 5 / 15 / 25 cm thick, 5 cm -> 1 m over 50 rows): drive at the wall on four
  wheels; 0.8-1.2 m from its inner face the hind stance is switched on; once it has been up for 1 s
  and is 0.3-0.7 m from the face (or 4 s into the stance) it is released and the jump fires; after
  the landing the robot drives on, forward, on four wheels. In deployment the two switches are two
  key presses.

Rewards: four-wheel driving (``Go2w-Biped-Switch``'s ``wheels_*`` terms) whenever the robot is on its
wheels and not in the crossing (:func:`~...mdp.while_driving`); the stance's terms while it is on;
the crossing's (``Go2w-Biped-WallJump``'s) between the jump and the success. A fall costs 20 at any
point; the four-wheel orientation limit applies outside the stance, its release grace and the
crossing. The episode no longer ends at the success: driving on is part of it.

    python scripts/rsl_rl/train.py --task Go2w-Biped-WallJump-Drive --headless \\
        --resume --load_run <walljump run> --checkpoint model_3800.pt
"""

from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_walljump import (
    NUM_RINGS,
    NUM_ROWS,
    PLAY_WALL_HEIGHTS,
    WALL_HEIGHT_RANGE,
    WALL_THICKNESSES,
    FALL_TERMS,
    RobotEnvCfgBipedWallJump,
    WallJumpCommandsCfg,
    WallJumpRewardsCfg,
    WallJumpSceneCfg,
    WallJumpTerminationsCfg,
)
from unitree_rl_lab.tasks.biped.terrains import CIRCULAR_WALLS, ROUGH, WALL_TERRAIN_CFG
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg import LEG_JOINT_NAMES

ROUGH_COLUMN = 0
EPISODE_LENGTH_S = 14.0
APPROACH_SPEED = (0.5, 0.8)
DRIVE_RANGES = mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0))
"""Driving on the rough column: Go2w-Biped-Switch's ranges."""


def _terrain(row_heights=None) -> terrain_gen.TerrainGeneratorCfg:
    walls = {
        f"wall_{round(t * 100)}cm": CIRCULAR_WALLS.replace(
            wall_thickness=t, wall_height_range=WALL_HEIGHT_RANGE, num_rings=NUM_RINGS, row_heights=row_heights
        )
        for t in WALL_THICKNESSES
    }
    # Equal proportions: the dict order is the column order, rough first.
    return WALL_TERRAIN_CFG.replace(
        num_rows=len(row_heights) if row_heights else NUM_ROWS,
        num_cols=1 + len(WALL_THICKNESSES),
        sub_terrains={"rough": ROUGH, **walls},
    )


@configclass
class DriveSceneCfg(WallJumpSceneCfg):
    terrain = WallJumpSceneCfg().terrain.replace(terrain_generator=_terrain(), max_init_terrain_level=25)


@configclass
class DriveCommandsCfg(WallJumpCommandsCfg):
    base_velocity = mdp.TerrainTypeVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=APPROACH_SPEED, lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        column_ranges={str(ROUGH_COLUMN): DRIVE_RANGES},
    )
    walljump = mdp.WallJumpCommandCfg(stance_on_gap_range=(0.8, 1.2), trigger_max_time_s=4.0)

    def __post_init__(self):
        # WallJumpCommandsCfg's post-init rewrites base_velocity's ranges for a UniformLevel
        # command; the terrain-type command above carries its own.
        handstand = self.handstand
        handstand.stance_front_probability = None
        handstand.balance_stance_population = False
        handstand.stance = mdp.STANCE_HIND


def _driving(term, weight: float, **term_params) -> RewTerm:
    return RewTerm(func=mdp.while_driving, weight=weight, params={"term": term, "term_params": term_params})


@configclass
class DriveRewardsCfg(WallJumpRewardsCfg):
    # Four-wheel driving (Go2w-Biped-Switch's wheels_* terms), outside the stance and the crossing.
    wheels_track_lin_vel_xy = _driving(mdp.track_lin_vel_xy_yaw_frame_exp, 1.5, command_name="base_velocity", std=math.sqrt(0.25))
    wheels_track_ang_vel_z = _driving(mdp.track_ang_vel_z_exp, 1.0, command_name="base_velocity", std=math.sqrt(0.25))
    wheels_lin_vel_z = _driving(mdp.lin_vel_z_l2, -2.0)
    wheels_ang_vel_xy = _driving(mdp.ang_vel_xy_l2, -0.05)
    wheels_flat_orientation = _driving(mdp.flat_orientation_l2, -2.5)
    wheels_joint_pos = _driving(
        mdp.joint_position_penalty,
        -0.7,
        asset_cfg=SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        stand_still_scale=5.0,
        velocity_threshold=0.3,
    )
    wheels_undesired_contacts = _driving(
        mdp.undesired_contacts,
        -1.0,
        threshold=1.0,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
    )
    termination_penalty = RewTerm(
        func=mdp.terminated_by, weight=-1000.0, params={"term_keys": [*FALL_TERMS, "bad_orientation"]}
    )


@configclass
class DriveTerminationsCfg(WallJumpTerminationsCfg):
    wall_success_done = None
    bad_orientation = DoneTerm(func=mdp.bad_orientation_driving, params={"limit_angle": 0.8})


@configclass
class RobotEnvCfgBipedWallJumpDrive(RobotEnvCfgBipedWallJump):
    scene: DriveSceneCfg = DriveSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: DriveCommandsCfg = DriveCommandsCfg()
    rewards: DriveRewardsCfg = DriveRewardsCfg()
    terminations: DriveTerminationsCfg = DriveTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S


@configclass
class RobotPlayEnvCfgBipedWallJumpDrive(RobotEnvCfgBipedWallJumpDrive):
    """3 rows (50 / 75 / 100 cm) x 4 columns (rough, 5 / 15 / 25 cm thick)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.terrain.terrain_generator = _terrain(PLAY_WALL_HEIGHTS)
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.spread_over_rows = EventTerm(func=mdp.spread_over_rows, mode="startup")
