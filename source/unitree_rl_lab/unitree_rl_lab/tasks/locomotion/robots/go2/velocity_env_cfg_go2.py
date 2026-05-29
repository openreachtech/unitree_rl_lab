import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import (
    CRITIC_HEIGHT_SCAN_CFG,
    CurriculumCfgGo2,
    apply_manual_curriculum_level,
    apply_play_velocity_ranges,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg

# Terrain layout for manual curriculum (columns are assigned by proportion).
GO2_CURRICULUM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic: base fields plus privileged ``height_scan`` (enabled per curriculum level)."""

    height_scan = CRITIC_HEIGHT_SCAN_CFG


@configclass
class ObservationsCfgGo2(ObservationsCfg):
    """Go2 observations: policy unchanged; extended critic for privileged training."""

    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class RewardsCfgGo2(RewardsCfg):
    """Go2-specific reward tuning."""

    track_ang_vel_z = RewardsCfg().track_ang_vel_z.replace(weight=1.0)


@configclass
class RobotSceneCfgGo2(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_generator=GO2_CURRICULUM_TERRAIN_CFG,
        max_init_terrain_level=0,
    )


@configclass
class RobotEnvCfgGo2(RobotEnvCfg):
    """Go2 velocity env with manual terrain curriculum."""
    curriculum_level: int = 1
    focus_spin_in_place: bool = False

    scene: RobotSceneCfgGo2 = RobotSceneCfgGo2(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgGo2 = ObservationsCfgGo2()
    rewards: RewardsCfgGo2 = RewardsCfgGo2()
    curriculum: CurriculumCfgGo2 = CurriculumCfgGo2()

    def __post_init__(self):
        super().__post_init__()
        apply_manual_curriculum_level(self)


@configclass
class RobotPlayEnvCfgGo2(RobotEnvCfgGo2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        apply_play_velocity_ranges(self)
