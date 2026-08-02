import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import terrains
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase3 import (
    RobotEnvCfgPhase3,
    RobotSceneCfgPhase3,
)

# Mirrors Go2's Phase4: terrain-swap-only on top of Phase3 (stairs -> a free-
# standing wall to step/drive over), including Go2's full wall_height_range
# ceiling of 0.25 m -- kept aligned with Go2 rather than capped to the wheel
# radius, same call as Phase3's step_height_range (see that module's docstring).
PHASE4_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.90,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.03),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase4(RobotSceneCfgPhase3):
    terrain = RobotSceneCfgPhase3().terrain.replace(
        terrain_generator=PHASE4_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgPhase4(RobotEnvCfgPhase3):
    """Phase 4: Phase3's rewards/commands/terminations, unchanged -- only the
    terrain mix changes (stairs -> thin_wall). See module docstring."""

    scene: RobotSceneCfgPhase4 = RobotSceneCfgPhase4(num_envs=4096, env_spacing=2.5)


@configclass
class RobotPlayEnvCfgPhase4(RobotEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 1
        # Wall only -- no flat column.
        self.scene.terrain.terrain_generator.sub_terrains = {
            name: cfg for name, cfg in PHASE4_TERRAIN_CFG.sub_terrains.items() if name != "flat"
        }
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
