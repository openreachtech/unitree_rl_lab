import copy

import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg, TerminationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    lidar_noise_only,
    play_lidar_height_scan,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    GO2_LIDAR_SCANNER_CFG,
    ObservationsCfgGo2LidarView,
    apply_lidar_view,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase2 import (
    CommandsCfgPhase2,
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)

PHASE3_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # 6 columns so the 1:2:3 proportions below land one sub-terrain per column exactly:
    # flat gets column 0, pyramid 1-2, inverted pyramid 3-5. Rows stay at 10 for the 10
    # terrain levels the curriculum ratchets through.
    num_cols=6,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=2.0,
            step_height_range=(0.05, 0.25),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0,
            step_height_range=(0.05, 0.25),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# Tread narrows from 0.25 m (difficulty 0 / row 0) to 0.19 m (difficulty 1 /
# row 9) instead of being 0.19 m at every level -- the same difficulty-driven
# interpolation step_height_range already uses, applied to tread width too
# (see terrains.MeshPyramidStairsVariableWidthCfg). A robot promoted through
# terrain_levels sees an easy, wide-tread staircase before it ever reaches
# the narrow-tread rows. Used by Phase3-stairfocus below (sandbox Try-10
# result: terrain_levels 4.869, softer initial collapse on first stairs
# contact, faster recovery than the fixed-width PHASE3_TERRAIN_CFG).
# ===========================================================================
# Phase 3 with floating treads mixed in: 5 columns of 10 difficulty rows, laid out
# 1 : 2 : 2 by the 20 / 40 / 40 proportions below.
#
#   col 0      floating pyramid stairs      ascending, open risers
#   col 1-2    inverted pyramid stairs      descending, solid -- the old default
#   col 3-4    floating inverted stairs     descending, open risers
#
# Weighted toward descending because that is where this lineage struggled, and toward
# floating because an open riser is the case a blind policy cannot feel ahead of time:
# there is nothing under the tread to catch a trailing leg against, so the terrain has
# to be seen rather than probed. That makes it the sharper test of whether a height map
# earns its place -- which is the question the four perceptive arms are asking.
#
# Step geometry matches the fixed-width stairs it sits beside (height 0.05 -> 0.25,
# tread 0.27 -> 0.23 with difficulty) so a row means the same thing across columns.
# ===========================================================================
PHASE3_TERRAIN_CFG_FLOATING = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=5,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "floating_pyramid_stairs": terrains.MeshFloatingPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.40,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "floating_pyramid_stairs_inv": terrains.MeshFloatingInvertedPyramidStairsTerrainCfg(
            proportion=0.40,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""5 x 10: floating | inverted x2 | floating inverted x2, rows ascending in difficulty."""


PHASE3_TERRAIN_CFG_VARIABLE_WIDTH = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=20,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        # 1段ごとに登らせる想定（段差、踏み幅広め）
        "pyramid_stairs_wide": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 1段ごとに登らせる想定（段差、踏み幅広め）
        "pyramid_stairs_inv_wide": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.40,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase3(RobotSceneCfgPhase2):
    # terrain_type back to "generator": Phase 1 switched the shared importer to a bare
    # "plane", and a generator attached to a plane-type importer is ignored outright --
    # the patches are never built and terrain_levels never appears.
    terrain = RobotSceneCfgPhase2().terrain.replace(
        terrain_type="generator",
        terrain_generator=PHASE3_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class CommandsCfgPhase3(CommandsCfgPhase2):
    base_velocity = CommandsCfgPhase2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RewardsCfgPhase3(RewardsCfgGo2):
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-1.0)
    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-0.5)
    joint_torques = RewardsCfgGo2().joint_torques.replace(weight=-1e-4)
    action_rate = RewardsCfgGo2().action_rate.replace(weight=-0.05)
    feet_air_time = RewardsCfgGo2().feet_air_time.replace(
        weight=0.07,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.35,
        },
    )
    air_time_variance = RewardsCfgGo2().air_time_variance.replace(weight=-0.2)
    wild_foot_clearance = RewardsCfgGo2().wild_foot_clearance.replace(weight=0.4)
    # foot_clearance_terrain_adaptive = RewardsCfgGo2().foot_clearance_terrain_adaptive.replace(weight=0.4)
    # forward_command_progress = RewardsCfgGo2().forward_command_progress.replace(weight=1.0)


@configclass
class RobotEnvCfgPhase3(RobotEnvCfgPhase2):
    """Phase 3: inverted pyramid stairs."""

    scene: RobotSceneCfgPhase3 = RobotSceneCfgPhase3(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase3 = CommandsCfgPhase3()
    rewards: RewardsCfgPhase3 = RewardsCfgPhase3()


# =============================================================================
# Phase3-balance: promoted from sandbox Try-13. Flattened into its own class in the v1
# lineage so that a sibling stair-focused variant's later reward changes could not leak
# into this validated recipe; that sibling is not carried here, but the class stays flat
# so it still diffs against the v1 original.
# Swaps the fixed-target foot-clearance reward for mdp.adaptive_foot_clearance_reward
# (obstacle-aware lookahead + terrain-roughness gate), so the clearance target
# collapses to a natural ~3 cm lift on flat ground and only scales up toward
# real riser height near a stair. Trades peak terrain_levels (~4.9, measured
# on the original easier terrain before it was hardened) for a natural
# flat-ground gait and much higher episode survivability (time_out 91.5% vs
# ~79-80%, bad_orientation 3.7% vs ~11-15%).
#
# Try-4 -> Try-11: feet_slide strengthened -0.1 -> -0.2 (stock isaaclab_tasks
# reward, previously inherited unchanged from the base RewardsCfg despite all
# the clearance-reward work) to discourage foot-dragging/scuffing, on top of
# adaptive_foot_clearance_reward's lift incentive.
#
# Try-11 -> Try-13: wild_foot_clearance's max_clearance raised 0.20 -> 0.23 m
# -- 0.20 m was capping the adaptive clearance target 1 cm *below* a 0.21 m
# stair riser, a plausible direct cause of toe-catching on the edge. MuJoCo
# confirmed improvement (fewer bad_orientation falls) with no flat-ground
# regression. Try-12 additionally added landing_stability_reward alongside
# this same clearance raise, but that caused idle foot-fidgeting and stair
# refusal (landing_stability is farmable by standing still and tapping a
# foot, with nothing in Balance to make standing still costly) -- dropped,
# not part of this promotion. Terrain is unchanged by this promotion -- still
# whatever RobotSceneCfgPhase3Balance below already uses, no floating-stairs
# terrain.
# =============================================================================


@configclass
class RewardsCfgPhase3Balance(RewardsCfgPhase3):
    """Preserves the exact Try-13 recipe (Try-1/2's relaxed penalties +
    Try-4's terrain-adaptive clearance + Try-11's strengthened feet_slide +
    Try-13's raised clearance cap) that produced its validated result."""

    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.3)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.2)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.3)
    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)
    forward_command_progress = RewardsCfgPhase3().forward_command_progress.replace(weight=0.8)
    feet_air_time = RewardsCfgPhase3().feet_air_time.replace(weight=0.1)
    wild_foot_clearance = RewardsCfgPhase3().wild_foot_clearance.replace(
        func=mdp.adaptive_foot_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_clearance": 0.03,
            "max_clearance": 0.23,
            "roughness_ref": 0.05,
        },
    )
    # Strengthened from the base RewardsCfg's -0.1 (see promotion note above).
    feet_slide = RewardsCfgPhase3().feet_slide.replace(weight=-0.2)


@configclass
class RobotSceneCfgPhase3Balance(RobotSceneCfgPhase3):
    terrain = RobotSceneCfgPhase3().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_VARIABLE_WIDTH,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgPhase3Balance(RobotEnvCfgPhase3):
    """Phase 3 - balance: natural flat-ground gait + terrain_levels >= 4.5 (sandbox Try-13 result: 5.049).

    Extends RobotEnvCfgPhase3 directly. Uses the variable-width terrain
    (PHASE3_TERRAIN_CFG_VARIABLE_WIDTH) and the independent
    RewardsCfgPhase3Balance above.
    """

    scene: RobotSceneCfgPhase3Balance = RobotSceneCfgPhase3Balance(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase3Balance = RewardsCfgPhase3Balance()


@configclass
class RobotPlayEnvCfgPhase3Balance(RobotEnvCfgPhase3Balance):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RewardsCfgPhase3BalanceMatched(RewardsCfgPhase3Balance):
    """Phase3-balance with the three posture penalties held at -0.5 / -0.5 / -0.7.

    These values come from a comparison against a goal-directed reward variant (since
    removed): Balance's own tuning relaxes them further, to -0.3 / -0.2 / -0.3, and
    holding them equal was what let the two designs be compared on their designs rather
    than on their posture budgets. Balance won that comparison and became the default,
    and these values came with it. Whether the looser Balance values are better on their
    own terms has not been measured -- Phase 4 puts them back to -0.2 / -0.3 for wall
    crossing, where pitching and lifting the body is the task.

    Kept as a subclass rather than edited into RewardsCfgPhase3Balance so that class stays
    a faithful port of the v1 original and still diffs against it.
    """

    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.5)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.5)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.7)


@configclass
class RobotEnvCfgPhase3BalanceMatched(RobotEnvCfgPhase3Balance):
    rewards: RewardsCfgPhase3BalanceMatched = RewardsCfgPhase3BalanceMatched()


# ===========================================================================
# The phase's single play config, plus the noise-pinned variants that subclass it -- the
# same shape Phase 2 and Phase 4 use, so `--task Go2-Blind-GRU-Phase3` draws the fan-built
# map like the other phases do rather than only the noise tasks doing so.
#
# 2 columns (pyramid | inverted pyramid) x 3 rows of ascending step height. Rows are
# difficulty under TerrainGenerator's curriculum mode, which derives it as
# (row + jitter) / num_rows with the jitter uniform on [0, 1) -- so a row is a band, not
# a single height, and an exact 5/10/15 cm cannot be asked for. step_height_range is set
# so the three bands *centre* on those values instead: with (0.025, 0.175) over 3 rows,
# row 0 spans 2.5-7.5 cm, row 1 spans 7.5-12.5 cm and row 2 spans 12.5-17.5 cm.
#
# Built from PHASE3_TERRAIN_CFG_VARIABLE_WIDTH, which is what the Phase 3 default trains
# on, keeping only its two fixed-width stair types -- the wide variants and flat are
# dropped so each column is one thing to look at.
# ===========================================================================
PLAY_TERRAIN_CFG_PHASE3 = PHASE3_TERRAIN_CFG_VARIABLE_WIDTH.replace(
    num_rows=3,
    num_cols=2,
    sub_terrains={
        name: cfg.replace(proportion=1.0, step_height_range=(0.025, 0.175))
        for name, cfg in PHASE3_TERRAIN_CFG_VARIABLE_WIDTH.sub_terrains.items()
        if name in ("pyramid_stairs", "pyramid_stairs_inv")
    },
)
"""2 x 3 tiles: pyramid | inverted pyramid, rows centred on 5 / 10 / 15 cm steps."""


@configclass
class RobotSceneCfgPlayPhase3(RobotSceneCfgPhase3Balance):
    """The Phase 3 default's scene plus the LiDAR fan, on the stepped play terrain."""

    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3BalanceMatched):
    scene: RobotSceneCfgPlayPhase3 = RobotSceneCfgPlayPhase3(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfgGo2LidarView = ObservationsCfgGo2LidarView()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = copy.deepcopy(PLAY_TERRAIN_CFG_PHASE3)
        # Spread the spawn over all three rows; the training value exceeds num_rows - 1.
        self.scene.terrain.max_init_terrain_level = 2
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt
        apply_lidar_view(self)


# The same three noise conditions Phase 2 has, pinned one per task rather than drawn
# 60/30/10. Only the drawn map differs -- the policy never sees it, so the gait is
# identical across all three and the red cells, which are geometry, should be too.
@configclass
class RobotPlayEnvCfgPhase3NoiseWeak(RobotPlayEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("weak"))


@configclass
class RobotPlayEnvCfgPhase3NoiseNominal(RobotPlayEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("nominal"))


@configclass
class RobotPlayEnvCfgPhase3NoiseStrong(RobotPlayEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.observations.lidar_map.height_scan = play_lidar_height_scan(lidar_noise_only("strong"))
