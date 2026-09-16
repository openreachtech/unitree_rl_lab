"""Phase 3: stairs.

The A2 counterpart of ``robots/go2/velocity_env_cfg_blind_phase3.py``, carrying over
that lineage's Phase3-balance recipe -- the one that traded peak terrain level for a
natural flat-ground gait and much higher episode survivability. The reward *weights* are
unitless trade-offs and come over unchanged; every length is Go2's x1.29 and every gait
time x1.14, so a riser is the same fraction of leg length to climb as it was there.

Go2's module also defines a floating-tread terrain that only its perceptive arms use.
This lineage is blind-only, so that variant is not carried here.

The registered ``A2-Blind-GRU-Phase3`` is ``RobotEnvCfgPhase3BalanceMatched``, matching
Go2.
"""

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg import A2_TERRAIN_TILE
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg_blind import (
    A2_GAIT_PERIOD,
    A2_TROT_OFFSET,
    RewardsCfgA2,
)
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg_blind_phase2 import (
    CommandsCfgPhase2,
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)

# Go2's step_height_range (0.05, 0.25) x1.29. A 0.32 m riser is 58% of A2's 0.55 m leg,
# the same fraction 0.25 m is of Go2's 0.426 m.
A2_STEP_HEIGHT_RANGE = (0.065, 0.32)

PHASE3_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=A2_TERRAIN_TILE,
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
            step_height_range=A2_STEP_HEIGHT_RANGE,
            # Go2's 0.23 / 2.0 / 1.0, all x1.29.
            step_width=0.30,
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=3.0,
            step_height_range=A2_STEP_HEIGHT_RANGE,
            step_width=0.30,
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
    },
)


# Tread narrows with difficulty instead of being fixed -- the same difficulty-driven
# interpolation step_height_range already uses, applied to tread width too (see
# terrains.MeshPyramidStairsVariableWidthCfg). A robot promoted through terrain_levels
# meets a wide-tread staircase before it ever reaches the narrow-tread rows. Go2's
# (0.27, 0.23) and (0.60, 0.30) x1.29.
PHASE3_TERRAIN_CFG_VARIABLE_WIDTH = terrain_gen.TerrainGeneratorCfg(
    size=A2_TERRAIN_TILE,
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
            step_height_range=A2_STEP_HEIGHT_RANGE,
            step_width_range=(0.77, 0.39),
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=A2_STEP_HEIGHT_RANGE,
            step_width_range=(0.35, 0.30),
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
        # 1段ごとに登らせる想定（段差、踏み幅広め）
        "pyramid_stairs_inv_wide": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=A2_STEP_HEIGHT_RANGE,
            step_width_range=(0.77, 0.39),
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.40,
            step_height_range=A2_STEP_HEIGHT_RANGE,
            step_width_range=(0.35, 0.30),
            platform_width=2.6,
            border_width=1.3,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase3(RobotSceneCfgPhase2):
    terrain = RobotSceneCfgPhase2().terrain.replace(
        terrain_type="generator",
        terrain_generator=PHASE3_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class CommandsCfgPhase3(CommandsCfgPhase2):
    # Go2's Phase 3 limits x1.14 on the linear axes, same as Phase 2's.
    base_velocity = CommandsCfgPhase2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.37, 1.37), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RewardsCfgPhase3(RewardsCfgA2):
    flat_orientation_l2 = RewardsCfgA2().flat_orientation_l2.replace(weight=-1.0)
    base_linear_velocity = RewardsCfgA2().base_linear_velocity.replace(weight=-0.5)
    # Go2 halves its torque penalty for this phase (-2e-4 -> -1e-4); the A2 weight is the
    # same halving applied to the /11.8-scaled base value.
    joint_torques = RewardsCfgA2().joint_torques.replace(weight=-8.5e-6)
    action_rate = RewardsCfgA2().action_rate.replace(weight=-0.05)
    feet_air_time = RewardsCfgA2().feet_air_time.replace(
        weight=0.07,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            # Go2's 0.35 x1.14.
            "threshold": 0.40,
        },
    )
    air_time_variance = RewardsCfgA2().air_time_variance.replace(weight=-0.2)
    wild_foot_clearance = RewardsCfgA2().wild_foot_clearance.replace(weight=0.4)


@configclass
class RobotEnvCfgPhase3(RobotEnvCfgPhase2):
    """Phase 3: pyramid and inverted pyramid stairs."""

    scene: RobotSceneCfgPhase3 = RobotSceneCfgPhase3(num_envs=4096, env_spacing=3.2)
    commands: CommandsCfgPhase3 = CommandsCfgPhase3()
    rewards: RewardsCfgPhase3 = RewardsCfgPhase3()


# =============================================================================
# Phase3-balance, carried over from the Go2 lineage's promoted recipe. It swaps the
# fixed-target foot-clearance reward for mdp.adaptive_foot_clearance_reward
# (obstacle-aware lookahead + terrain-roughness gate), so the clearance target collapses
# to a natural walking lift on flat ground and only scales up toward real riser height
# near a stair. On Go2 that traded peak terrain_levels for a natural flat-ground gait and
# much higher episode survivability (time_out 91.5% vs ~79-80%, bad_orientation 3.7% vs
# ~11-15%), with feet_slide strengthened to -0.2 against foot-dragging and the clearance
# cap raised so it sits *above* the tallest riser rather than 1 cm below it.
#
# The A2 cap keeps that relationship: 0.30 m against a 0.32 m tallest riser is Go2's
# 0.23 against 0.25, x1.29. Every other length here is x1.29 and every period x1.14; the
# weights are unchanged.
# =============================================================================
@configclass
class RewardsCfgPhase3Balance(RewardsCfgPhase3):
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
            "period": A2_GAIT_PERIOD,
            "offset": A2_TROT_OFFSET,
            # Go2's 0.15 / 0.03 / 0.23 / 0.05, all x1.29.
            "lookahead_distance": 0.19,
            "natural_clearance": 0.039,
            "max_clearance": 0.30,
            "roughness_ref": 0.065,
        },
    )
    # Strengthened from the base RewardsCfg's -0.1 (see the promotion note above).
    feet_slide = RewardsCfgPhase3().feet_slide.replace(weight=-0.2)


@configclass
class RobotSceneCfgPhase3Balance(RobotSceneCfgPhase3):
    terrain = RobotSceneCfgPhase3().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_VARIABLE_WIDTH,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgPhase3Balance(RobotEnvCfgPhase3):
    """Phase 3 - balance: natural flat-ground gait, variable-width stairs."""

    scene: RobotSceneCfgPhase3Balance = RobotSceneCfgPhase3Balance(num_envs=4096, env_spacing=3.2)
    rewards: RewardsCfgPhase3Balance = RewardsCfgPhase3Balance()


@configclass
class RewardsCfgPhase3BalanceMatched(RewardsCfgPhase3Balance):
    """Phase3-balance with the three posture penalties held at -0.5 / -0.5 / -0.7.

    On Go2 these came out of a comparison against a goal-directed reward variant: holding
    the posture budgets equal was what let the two designs be compared on their designs.
    Balance won and these values came with it. Phase 4 puts them back to -0.2 / -0.3 for
    wall crossing, where pitching and lifting the body is the task. Kept as a subclass so
    ``RewardsCfgPhase3Balance`` stays a faithful port of the Go2 original.
    """

    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.5)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.5)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.7)


@configclass
class RobotEnvCfgPhase3BalanceMatched(RobotEnvCfgPhase3Balance):
    rewards: RewardsCfgPhase3BalanceMatched = RewardsCfgPhase3BalanceMatched()


# ===========================================================================
# Play: 2 columns (pyramid | inverted pyramid) x 3 rows of ascending step height.
#
# Rows are difficulty under TerrainGenerator's curriculum mode, which derives it as
# (row + jitter) / num_rows with the jitter uniform on [0, 1) -- so a row is a band, not
# a single height, and an exact value cannot be asked for. step_height_range is set so
# the three bands *centre* on Go2's 5 / 10 / 15 cm x1.29, i.e. 6.5 / 13 / 19.4 cm: with
# (0.032, 0.226) over 3 rows, row 0 spans 3.2-9.7, row 1 spans 9.7-16.1 and row 2 spans
# 16.1-22.6 cm.
#
# Built from PHASE3_TERRAIN_CFG_VARIABLE_WIDTH, which is what the Phase 3 default trains
# on, keeping only its two fixed-width stair types -- the wide variants and flat are
# dropped so each column is one thing to look at.
# ===========================================================================
PLAY_TERRAIN_CFG_PHASE3 = PHASE3_TERRAIN_CFG_VARIABLE_WIDTH.replace(
    num_rows=3,
    num_cols=2,
    sub_terrains={
        name: cfg.replace(proportion=1.0, step_height_range=(0.032, 0.226))
        for name, cfg in PHASE3_TERRAIN_CFG_VARIABLE_WIDTH.sub_terrains.items()
        if name in ("pyramid_stairs", "pyramid_stairs_inv")
    },
)
"""2 x 3 tiles: pyramid | inverted pyramid, rows centred on 6.5 / 13 / 19.4 cm steps."""


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3BalanceMatched):
    scene: RobotSceneCfgPhase3Balance = RobotSceneCfgPhase3Balance(num_envs=32, env_spacing=3.2)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = copy.deepcopy(PLAY_TERRAIN_CFG_PHASE3)
        # Spread the spawn over all three rows; the training value exceeds num_rows - 1.
        self.scene.terrain.max_init_terrain_level = 2
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
