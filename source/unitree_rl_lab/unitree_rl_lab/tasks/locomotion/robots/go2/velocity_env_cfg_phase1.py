import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotSceneCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    CommandsCfgGo2,
    RobotEnvCfgGo2,
)

PHASE1_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class RobotSceneCfgPhase1(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_generator=PHASE1_TERRAIN_CFG,
        max_init_terrain_level=0,
    )


@configclass
class CommandsCfgPhase1(CommandsCfgGo2):
    base_velocity = CommandsCfgGo2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RobotEnvCfgPhase1(RobotEnvCfgGo2):
    """Phase 1: flat terrain."""

    scene: RobotSceneCfgPhase1 = RobotSceneCfgPhase1(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase1 = CommandsCfgPhase1()


@configclass
class RobotPlayEnvCfgPhase1(RobotEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Raw grid markers duplicate the magenta excluded-cell overlay and hide its shape.
        # Off here so only the exclusion coverage is visible; flip back on to see the full grid.
        self.scene.height_scanner.debug_vis = False
