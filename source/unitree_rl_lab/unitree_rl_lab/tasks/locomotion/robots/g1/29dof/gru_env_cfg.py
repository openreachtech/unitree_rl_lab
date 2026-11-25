"""G1 GRU tasks: G1-MLP-Phase1 with a recurrent policy, then rough terrain, then stairs.

G1-GRU-Phase1 (flat): G1-MLP-Phase1 with these changes, for SymmetricGruPPORunnerCfg
  * no stacked history in either group; the GRU hidden state carries the past
  * critic also sees the terrain and contacts at the feet (after Lee et al. 2020,
    as feat/mid360's Go2 blind critic): per-foot height grid and terrain normal,
    foot contact force, knee contact. Phase 2 needs them; having them from Phase 1
    keeps the critic's input, and so its checkpoint, the same across phases.

G1-GRU-Phase2 (terrain), from the Phase1 checkpoint:
  * rough / slope down / slope up / boxes, 10 difficulty rows, terrain curriculum
  * base height (reward and termination) and foot clearance measured from the
    terrain, not z = 0
  * vertical velocity and tilt penalties relaxed (as feat/mid360 Go2 Phase 2)
  * episodes end (as time-outs) near the terrain edge, where the ray casters miss

G1-GRU-Phase3 (stairs, walls), from the Phase2 checkpoint: see GruPhase3EnvCfg.

The actor stays blind (proprioception only); deploy.yaml layout is unchanged except
for history_length.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .mlp_phase1_env_cfg import MlpPhase1EnvCfg, MlpPhase1ObservationsCfg, MlpPhase1RewardsCfg
from .velocity_env_cfg import RobotSceneCfg

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
KNEES = ["left_knee_link", "right_knee_link"]
FOOT_SCAN_SENSORS = [SceneEntityCfg("foot_scan_left"), SceneEntityCfg("foot_scan_right")]


def _foot_scanner(body: str) -> RayCasterCfg:
    """7 x 5 rays at 5 cm around one sole: 0.1 m behind to 0.2 m ahead, +-0.1 m to the side,
    in the foot's yaw frame, cast down from 20 m above."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body,
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class GruSceneCfg(RobotSceneCfg):
    foot_scan_left = _foot_scanner(FEET[0])
    foot_scan_right = _foot_scanner(FEET[1])


@configclass
class GruObservationsCfg(MlpPhase1ObservationsCfg):
    @configclass
    class CriticCfg(MlpPhase1ObservationsCfg.CriticCfg):
        foot_height_scan = ObsTerm(
            func=mdp.foot_height_scan, params={"sensor_cfgs": FOOT_SCAN_SENSORS}, clip=(-1.0, 1.0)
        )
        foot_terrain_normal = ObsTerm(func=mdp.foot_terrain_normal, params={"sensor_cfgs": FOOT_SCAN_SENSORS})
        foot_contact_force = ObsTerm(
            func=mdp.contact_force_magnitude,
            scale=0.01,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)},
        )
        knee_contact = ObsTerm(
            func=mdp.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=KNEES, preserve_order=True)},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class GruPhase1EnvCfg(MlpPhase1EnvCfg):
    scene: GruSceneCfg = GruSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: GruObservationsCfg = GruObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # no stacked history: the GRU hidden state carries the past
        self.observations.policy.history_length = 0
        self.observations.critic.history_length = 0
        for sensor in (self.scene.foot_scan_left, self.scene.foot_scan_right):
            sensor.update_period = self.decimation * self.sim.dt


@configclass
class GruPhase1PlayEnvCfg(GruPhase1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ---------------------------------------------------------------------------
# Phase 2: terrain
# ---------------------------------------------------------------------------

# 4 columns x 10 rows: TerrainGenerator gives each sub-terrain columns by proportion
# and sets difficulty by row, so equal proportions put one type per column.
PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=2.0
        ),
    },
)


@configclass
class GruPhase2RewardsCfg(MlpPhase1RewardsCfg):
    # stepping onto a box or a slope needs vertical base velocity and some body tilt
    base_linear_velocity = MlpPhase1RewardsCfg().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = MlpPhase1RewardsCfg().flat_orientation_l2.replace(weight=-1.0)

    # heights from the terrain under the robot instead of z = 0
    base_height = MlpPhase1RewardsCfg().base_height.replace(
        func=mdp.base_height_terrain_l2,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    feet_clearance = MlpPhase1RewardsCfg().feet_clearance.replace(
        params={
            **MlpPhase1RewardsCfg().feet_clearance.params,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )


@configclass
class GruPhase2EnvCfg(GruPhase1EnvCfg):
    rewards: GruPhase2RewardsCfg = GruPhase2RewardsCfg()

    def __post_init__(self):
        self.scene.terrain.terrain_generator = PHASE2_TERRAIN_CFG.replace()  # a copy: play cfgs edit it
        self.scene.terrain.max_init_terrain_level = 2
        super().__post_init__()  # turns on the generator's curriculum mode (terrain_levels)
        # z = 0 is the top of the slope-up pyramids; their bottom sinks to -0.68 m
        self.terminations.base_height = self.terminations.base_height.replace(
            func=mdp.root_height_below_minimum_terrain,
            params={"minimum_height": 0.2, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        # at 2 m/s a robot can walk past the 20 m border within an episode; off the mesh
        # the ray casters return inf, so end the episode before that
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=mdp.terrain_out_of_bounds, params={"distance_buffer": 3.0}, time_out=True
        )


@configclass
class GruPhase2PlayEnvCfg(GruPhase2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ---------------------------------------------------------------------------
# Phase 3: stairs and walls (terrain after tsubame_isaac_env's Anaguma Pos-Target Phase 3)
# ---------------------------------------------------------------------------

_STAIRS = dict(
    proportion=1.0,
    # 5 -> 20 cm risers, 35 -> 27.5 cm treads: row 13 of 20 is a 15 cm x 30 cm stair
    step_height_range=(0.05, 0.20),
    step_width_range=(0.35, 0.275),
    # the generated platform comes out narrower (one more step fits inside): 3.0 leaves
    # 2.4-2.8 m, room for the +-0.5 m spawn noise
    platform_width=3.0,
    border_width=1.0,
)

# 5 columns x 20 rows, one sub-terrain per column in definition order (proportions equal)
PHASE3_TERRAIN_CFG = PHASE2_TERRAIN_CFG.replace(
    num_cols=5,
    num_rows=20,
    sub_terrains={
        "random_rough": PHASE2_TERRAIN_CFG.sub_terrains["random_rough"].replace(proportion=1.0),
        "stairs_down": mdp.MeshPyramidStairsVariableWidthCfg(**_STAIRS),
        "stairs_up": mdp.MeshInvertedPyramidStairsVariableWidthCfg(**_STAIRS),
        "stairs_up_front": mdp.MeshInvertedPyramidStairsVariableWidthCfg(**_STAIRS),
        # 1 cm thick rings at 1.65 m and 3.22 m from the spawn, 5 -> 25 cm high
        "wall": mdp.MeshCircularWallTerrainCfg(
            proportion=1.0, wall_height_range=(0.05, 0.25), wall_thickness_range=(0.01, 0.01)
        ),
    },
)
PHASE3_STAIRS_FRONT_COLS = (3,)
PHASE3_WALL_COLS = (4,)


@configclass
class GruPhase3RewardsCfg(GruPhase2RewardsCfg):
    # climbing leans the torso; do not penalize tilt against gravity
    flat_orientation_l2 = None


@configclass
class GruPhase3EnvCfg(GruPhase2EnvCfg):
    """Phase 2 on stairs and walls. Changes from Phase 2:

    * terrain: rough | stairs down | stairs up | stairs up, spawned facing the steps |
      1 cm circular walls; 20 rows
    * forward-only commands (0.3-1.0 m/s) on the front-stairs and wall columns
    * wall rows ratchet on distance from spawn (got past the first ring or not)
    * no tilt penalty
    * commands start at their full range (the level curriculum restarts from the
      initial ranges on every run, which cost Phase 2 ~800 iterations)
    """

    rewards: GruPhase3RewardsCfg = GruPhase3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = PHASE3_TERRAIN_CFG.replace()  # a copy: play cfgs edit it
        self.scene.terrain.terrain_generator.curriculum = True  # set on Phase 2's generator by super()
        self.scene.terrain.max_init_terrain_level = 3

        cmd = self.commands.base_velocity
        cmd.ranges = cmd.limit_ranges.replace()
        cmd.forward_only_types = PHASE3_STAIRS_FRONT_COLS + PHASE3_WALL_COLS
        cmd.forward_speed_range = (0.3, 1.0)

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel_walls,
            params={"distance_types": PHASE3_WALL_COLS, "promote_distance": 2.3, "demote_distance": 2.0},
        )
        # after reset_base (full yaw circle), so it overrides only the front-stairs column
        self.events.stairs_facing = EventTerm(
            func=mdp.reset_root_facing_axes, mode="reset", params={"terrain_types": PHASE3_STAIRS_FRONT_COLS}
        )


@configclass
class GruPhase3PlayEnvCfg(GruPhase3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.max_init_terrain_level = 9


__all__ = [
    "GruPhase1EnvCfg",
    "GruPhase1PlayEnvCfg",
    "GruPhase2EnvCfg",
    "GruPhase2PlayEnvCfg",
    "GruPhase3EnvCfg",
    "GruPhase3PlayEnvCfg",
]
