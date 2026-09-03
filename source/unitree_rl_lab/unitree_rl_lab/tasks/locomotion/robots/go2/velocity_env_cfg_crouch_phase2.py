"""Go2-Crouch-Phase2: Phase1's crouch-height tracking, now on rough ground and boxes.

Same base_height command/curriculum/reward as Phase1 (inherited unchanged) -- only the
terrain and base_velocity's speed ceiling change:

- Terrain: a generated grid, 2 columns x 10 rows (terrain_levels difficulty). Column 1 is
  random rough ground, column 2 is boxes -- see PHASE2_TERRAIN_CFG. terrain_levels_vel is
  re-enabled here (Phase1 set it to None for its infinite plane; a real generated grid needs
  it, same as the base CurriculumCfg).
- base_velocity: lin_vel_x/lin_vel_y limits both raised to +-1.0 m/s (Phase1 had lin_vel_x
  at +-1.0 already but lin_vel_y at +-0.8 -- widened to match).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_crouch_phase1 import (
    CommandsCfgCrouchPhase1,
    RobotEnvCfgCrouchPhase1,
    RobotSceneCfgCrouchPhase1,
)

PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=2,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.5,
            grid_width=0.45,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0,
        ),
    },
)


@configclass
class RobotSceneCfgCrouchPhase2(RobotSceneCfgCrouchPhase1):
    terrain = RobotSceneCfgCrouchPhase1().terrain.replace(
        terrain_type="generator",
        terrain_generator=PHASE2_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class CommandsCfgCrouchPhase2(CommandsCfgCrouchPhase1):
    base_velocity = CommandsCfgCrouchPhase1().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class CurriculumCfgCrouchPhase2(CurriculumCfg):
    """Real generated terrain grid (unlike Phase1's infinite plane) -- terrain_levels_vel
    is back on (inherited from CurriculumCfg), plus crouch_depth_levels for base_height."""

    crouch_depth_levels = CurrTerm(func=mdp.crouch_depth_levels)


@configclass
class RobotEnvCfgCrouchPhase2(RobotEnvCfgCrouchPhase1):
    """Phase 2: rough terrain and boxes."""

    scene: RobotSceneCfgCrouchPhase2 = RobotSceneCfgCrouchPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgCrouchPhase2 = CommandsCfgCrouchPhase2()
    curriculum: CurriculumCfgCrouchPhase2 = CurriculumCfgCrouchPhase2()


@configclass
class RobotPlayEnvCfgCrouchPhase2(RobotEnvCfgCrouchPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_height.ranges = self.commands.base_height.limit_ranges
