"""Blind Phase 4: stepping over short free-standing walls.

Continual learning on top of Phase 3, following ``Go2-v3-Phase4``. Where Phase 3 is
stairs -- a continuous surface the robot climbs -- these are isolated walls it has to
lift a leg over and clear, with nothing to stand on in between. That is the hardest
thing this blind lineage is asked to do: a wall is invisible to proprioception until a
foot hits it, so the GRU has to infer it from the contact and the stumble.

The terrain is the v3 phase's two wall types without its flat share, at 2:1 rather than
its 4:5 -- see the note on num_cols below for why the floating tread is dialled back.

Two things carried over from the v3 phase, both load-bearing there:

  * the wall height scales with difficulty (0.05 -> 0.25 m) rather than being fixed. A
    first attempt at a fixed 0.20 m saw terrain_levels collapse and never recover -- the
    robot never got past the wall at all, so the curriculum had nothing to ratchet on.
  * the gait fix: ``joint_deviation_hips`` penalises hip deviation during the swing
    phase specifically, after MuJoCo testing found the policy crossing walls with a
    wide-legged, hip-swinging gait rather than a natural stride. A blanket joint_pos
    penalty could not separate hip abduction from the thigh/calf motion the climb needs.

``calf_flexion_clearance`` reads the top-down ``height_scanner``. That is privileged
information, which is fine in a reward -- rewards only exist during training -- and does
not make the policy sighted: the actor still sees 45 proprioceptive numbers and nothing
else.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    GO2_LIDAR_SCANNER_CFG,
    ObservationsCfgGo2LidarView,
    apply_lidar_view,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase3 import (
    RewardsCfgPhase3BalanceMatched,
    RobotEnvCfgPhase3BalanceMatched,
    RobotSceneCfgPhase3Balance,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    lidar_noise_only,
    play_lidar_height_scan,
)

PHASE4_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # 3 columns, 2:1 -- solid wall in columns 0-1, floating in column 2. Weighted toward
    # the solid wall because Phase 3 here is plain stairs: unlike the v3 lineage this
    # follows, this policy has never met a floating tread, and the v3 mix put half the
    # grid on one. Rows stay at 10 for the difficulty ratchet.
    num_cols=3,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Height scales with difficulty for the reason in the module docstring; thickness
        # narrows 0.15 -> 0.05 m over the same range, walls a fixed 0.60 m apart.
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=2.0,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
        # The same wall hollowed out: no solid body, just the tread hovering at
        # wall_height with an open gap underneath, so a trailing leg can be caught by
        # something it cannot feel from above.
        "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase4(RobotSceneCfgPhase3Balance):
    terrain = RobotSceneCfgPhase3Balance().terrain.replace(
        terrain_generator=PHASE4_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class RewardsCfgPhase4(RewardsCfgPhase3BalanceMatched):
    """Phase 3's rewards plus the v3 Phase 4 gait fix.

    ``undesired_contacts`` goes back to -1.0 from Phase 3's -0.3: on stairs a knee
    brushing a riser is part of climbing, but on an isolated wall it means the leg failed
    to clear and the robot is dragging itself over.

    The two posture penalties go the other way, back to the values Go2-v3-Phase4 uses.
    Phase 3 here inherits them from RewardsCfgPhase3BalanceMatched, which raised them to
    -0.5/-0.5 only so the Balance and Goal reward designs could be compared on equal
    terms -- a measurement decision with no bearing on this phase. Clearing a wall means
    pitching the body and moving it vertically, which is exactly what these two charge
    for, so the looser v3 values apply.
    """

    undesired_contacts = RewardsCfgPhase3BalanceMatched().undesired_contacts.replace(weight=-1.0)
    base_linear_velocity = RewardsCfgPhase3BalanceMatched().base_linear_velocity.replace(weight=-0.2)
    flat_orientation_l2 = RewardsCfgPhase3BalanceMatched().flat_orientation_l2.replace(weight=-0.3)
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_swing_gated_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
            ),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
        },
    )
    calf_flexion_clearance = RewTerm(
        func=mdp.calf_flexion_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "calf_asset_cfg": SceneEntityCfg(
                "robot", joint_names=["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_flex": 0.05,
            "max_flex": 0.4,
            "max_obstacle_height": 0.25,
            "roughness_ref": 0.05,
        },
    )


@configclass
class RobotEnvCfgPhase4(RobotEnvCfgPhase3BalanceMatched):
    """Phase 4: short free-standing walls to step over."""

    scene: RobotSceneCfgPhase4 = RobotSceneCfgPhase4(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase4 = RewardsCfgPhase4()


# ---------------------------------------------------------------------------
# Play: one column per wall type at five difficulties, so a row shows the same wall
# height as a solid and as a floating tread side by side.
# ---------------------------------------------------------------------------
PLAY_TERRAIN_CFG_PHASE4 = PHASE4_TERRAIN_CFG.replace(
    num_rows=4,
    num_cols=2,
    sub_terrains={
        name: cfg.replace(
            wall_height_range=(0.075, 0.275),
            wall_thickness_range=(0.05, 0.05),
        )
        for name, cfg in PHASE4_TERRAIN_CFG.sub_terrains.items()
    },
)
"""2 x 4: solid wall and floating tread side by side at each of four heights.

Rows are difficulty, derived by TerrainGenerator as (row + jitter) / num_rows with the
jitter uniform on [0, 1), so a row is a band and an exact height cannot be requested.
wall_height_range is set so the four bands *centre* on 10 / 15 / 20 / 25 cm: with
(0.075, 0.275) over 4 rows they span 7.5-12.5, 12.5-17.5, 17.5-22.5 and 22.5-27.5 cm.

Thickness is pinned at 5 cm rather than narrowing with difficulty, so height is the only
thing that changes down a column. That is the training mix's hardest thickness (it
narrows 15 -> 5 cm with difficulty), so every row here is a thin wall. Over two columns the training mix's 2:1 puts one wall
type in each, so both are visible in the same row."""


@configclass
class RobotSceneCfgPlayPhase4(RobotSceneCfgPhase4):
    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotPlayEnvCfgPhase4(RobotEnvCfgPhase4):
    scene: RobotSceneCfgPlayPhase4 = RobotSceneCfgPlayPhase4(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfgGo2LidarView = ObservationsCfgGo2LidarView()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # .replace() rather than mutating PHASE4_TERRAIN_CFG's sub_terrains dict, which is
        # a module-level object shared with the training config.
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG_PHASE4.copy()
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt
        apply_lidar_view(self)


@configclass
class RobotPlayEnvCfgPhase4NoiseWeak(RobotPlayEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("weak"))


@configclass
class RobotPlayEnvCfgPhase4NoiseNominal(RobotPlayEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("nominal"))


@configclass
class RobotPlayEnvCfgPhase4NoiseStrong(RobotPlayEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("strong"))
