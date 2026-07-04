import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import TerminationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase2 import (
    CommandsCfgPhase2,
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)

PHASE3_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.23),
            step_width=0.19,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.60,
            step_height_range=(0.05, 0.23),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase3(RobotSceneCfgPhase2):
    terrain = RobotSceneCfgPhase2().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class CommandsCfgPhase3(CommandsCfgPhase2):
    base_velocity = CommandsCfgPhase2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
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


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# =============================================================================
# Phase3-stairfocus: promoted from sandbox Try-1 + Try-2.
# Maximizes terrain_levels (reached ~5.3-5.4 in the sandbox runs) by relaxing
# penalties that conflict with stair-climbing and adding a forward-progress
# term, at the cost of an exaggerated flat-ground gait (see Phase3-balance
# below for the fix).
# =============================================================================


@configclass
class RewardsCfgPhase3StairFocus(RewardsCfgPhase3):
    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.3)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.2)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.3)
    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)
    feet_air_time = RewardsCfgPhase3().feet_air_time.replace(weight=0.2)
    wild_foot_clearance = RewardsCfgPhase3().wild_foot_clearance.replace(
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "radius": 0.1,
            "target_clearance": 0.15,
        },
    )
    forward_command_progress = RewardsCfgPhase3().forward_command_progress.replace(weight=0.8)


@configclass
class TerminationsCfgPhase3StairFocus(TerminationsCfg):
    # Stairs require more sustained body pitch than flat/rough terrain.
    bad_orientation = TerminationsCfg().bad_orientation.replace(params={"limit_angle": 1.0})


@configclass
class RobotEnvCfgPhase3StairFocus(RobotEnvCfgPhase3):
    """Phase 3 - stair focus: maximize terrain_levels (sandbox result: ~5.3-5.4)."""

    rewards: RewardsCfgPhase3StairFocus = RewardsCfgPhase3StairFocus()
    terminations: TerminationsCfgPhase3StairFocus = TerminationsCfgPhase3StairFocus()


@configclass
class RobotPlayEnvCfgPhase3StairFocus(RobotEnvCfgPhase3StairFocus):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# =============================================================================
# Phase3-balance: promoted from sandbox Try-4.
# Swaps the fixed-target foot-clearance reward for mdp.adaptive_foot_clearance_reward
# (obstacle-aware lookahead + terrain-roughness gate), so the clearance target
# collapses to a natural ~3 cm lift on flat ground and only scales up toward
# real riser height near a stair. Trades peak terrain_levels (~4.9 vs
# StairFocus's ~5.3-5.4) for a natural flat-ground gait and much higher
# episode survivability (time_out 91.5% vs ~79-80%, bad_orientation 3.7% vs
# ~11-15%).
# =============================================================================


@configclass
class RewardsCfgPhase3Balance(RewardsCfgPhase3StairFocus):
    wild_foot_clearance = RewardsCfgPhase3StairFocus().wild_foot_clearance.replace(
        func=mdp.adaptive_foot_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_clearance": 0.03,
            "max_clearance": 0.20,
            "roughness_ref": 0.05,
        },
    )
    feet_air_time = RewardsCfgPhase3StairFocus().feet_air_time.replace(weight=0.1)


@configclass
class RobotEnvCfgPhase3Balance(RobotEnvCfgPhase3StairFocus):
    """Phase 3 - balance: natural flat-ground gait + terrain_levels >= 4.5 (sandbox result: 4.899)."""

    rewards: RewardsCfgPhase3Balance = RewardsCfgPhase3Balance()


@configclass
class RobotPlayEnvCfgPhase3Balance(RobotEnvCfgPhase3Balance):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
