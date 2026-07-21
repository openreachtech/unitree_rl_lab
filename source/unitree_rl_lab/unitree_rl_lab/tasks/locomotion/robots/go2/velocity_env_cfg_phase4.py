import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase3 import (
    RobotEnvCfgPhase3BalanceFloating,
    RobotSceneCfgPhase3Balance,
)

# =============================================================================
# Phase 4: continual learning on top of Phase3-balance-floating -- dedicated
# terrain mix for stepping over short free-standing walls.
#
# Phase3-balance-floating already climbs stairs, including floating (open-riser)
# ones; that behavior is expected to carry over via the checkpoint (trained
# with --previous-task pointed at the promoted balance-floating checkpoint, not
# from scratch), not by keeping stairs in this phase's terrain mix. This phase
# mirrors the stairfocus/balance pattern of specializing the mix on one
# obstacle: 10% flat (keeps flat-ground gait honest) + 90% thin_wall.
# =============================================================================
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
        # Short free-standing wall to step over. First attempt at a fixed 0.20 m
        # height went nowhere -- terrain_levels collapsed and never recovered
        # (robot never got past the wall) -- so height now scales with
        # difficulty too, 0.05 m (easy) -> 0.25 m (hard), same convention as
        # step_height_range on the stair terrains; thickness still narrows
        # 0.15 m (easy) -> 0.03 m (hard); walls spaced a fixed 0.60 m apart.
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.90,
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
class RobotEnvCfgPhase4(RobotEnvCfgPhase3BalanceFloating):
    """Phase 4: Phase3-balance-floating's rewards/terminations/commands, unchanged --
    only the terrain mix changes (adds thin_wall). See module docstring."""

    scene: RobotSceneCfgPhase4 = RobotSceneCfgPhase4(num_envs=4096, env_spacing=2.5)


@configclass
class RobotPlayEnvCfgPhase4(RobotEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
