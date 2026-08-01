import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)

# Mirrors the Go2 Phase 2 mix so the two robots can be compared on equal terrain.
# Note that ``boxes`` is demanding for a wheeled base: the wheels are 0.086 m in
# radius, and the step between adjacent grid cells approaches or exceeds that at
# higher difficulty, so those cells cannot be rolled over and have to be crossed
# by lifting a wheel with the leg. If terrain levels stall, this is the first
# thing to reconsider -- lowering ``grid_height_range`` below the wheel radius
# turns the boxes back into something the wheels can simply drive across.
PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.50,
            grid_width=0.45,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase2(RobotSceneCfgPhase1):
    terrain = RobotSceneCfgPhase1().terrain.replace(
        terrain_generator=PHASE2_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class CommandsCfgPhase2(CommandsCfgPhase1):
    """Phase 2 commands: slower than Phase 1 to suit the rough terrain.

    ``ang_vel_z`` stays at Phase 1's value because no curriculum widens it, so
    its ``ranges`` entry is the actual training range and ``limit_ranges`` has to
    repeat it -- ``limit_ranges`` is what ``export_deploy_cfg`` publishes as the
    deploy joystick range.
    """

    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: rough terrain and boxes."""

    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
