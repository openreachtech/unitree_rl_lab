import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    GO2_LIDAR_SCANNER_CFG,
    ObservationsCfgGo2LidarView,
    apply_lidar_view,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)

PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # 2 columns, one sub-terrain each: TerrainGenerator assigns sub-terrains to columns
    # by proportion and difficulty to rows, so 50/50 over 2 columns puts rough in column
    # 0 and boxes in column 1. Rows stay at 10 to keep the 10 terrain levels the
    # curriculum ratchets through.
    num_cols=2,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.50,
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
    # terrain_type back to "generator": Phase 1 switched the shared importer to a bare
    # "plane", and a generator attached to a plane-type importer is ignored outright --
    # the patches are never built and terrain_levels never appears.
    terrain = RobotSceneCfgPhase1().terrain.replace(
        terrain_type="generator",
        terrain_generator=PHASE2_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class CommandsCfgPhase2(CommandsCfgPhase1):
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RewardsCfgPhase2(RewardsCfgGo2):
    """Posture penalties relaxed for box terrain.

    All three penalise exactly what crossing a box requires:
      base_linear_velocity  vertical base velocity -- stepping up and down a 0.05-0.15 m
                            box *is* vertical base velocity.
      flat_orientation_l2   the horizontal components of projected gravity, i.e. any body
                            tilt -- but the body has to pitch to get a leg up.
      joint_pos             deviation from the default stance, which every non-flat
                            foothold is.

    Values follow this repo's own stair phases, which hit the same wall: Phase 3 relaxed
    base_linear_velocity to -0.5 and flat_orientation_l2 to -1.0, and Phase3Balance went
    on to -0.2/-0.3/-0.3 (see velocity_env_cfg_blind_phase3.py's Try-1/2 note). Boxes are
    milder than stairs, so these sit at the Phase 3 end rather than Phase3Balance's.

    Measured against the un-relaxed weights over 4000 iterations from the same Phase 1
    checkpoint: mean episode length 972 against 888, i.e. about 9% fewer falls, at the
    same terrain level (5.79 vs 5.67 -- within single-seed noise). The gain is stability,
    not reach.
    """

    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-1.0)
    joint_pos = RewardsCfgGo2().joint_pos.replace(weight=-0.4)


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: rough terrain and boxes."""

    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()
    # Back to the base curriculum, which has terrain_levels. Phase 1 sets it to None
    # because its scene is a bare plane, and inheriting that here is silently harmful:
    # RobotEnvCfg.__post_init__ reads terrain_levels to decide whether to switch the
    # generator into curriculum mode, so a None here means the patches are laid out
    # randomly instead of by ascending difficulty, and every robot starts on terrain of
    # arbitrary difficulty with no ratchet. Phase 3 and its variants inherit this.
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotSceneCfgPlayPhase2(RobotSceneCfgPhase2):
    """Phase 2's scene plus the LiDAR fan, play only."""

    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    scene: RobotSceneCfgPlayPhase2 = RobotSceneCfgPlayPhase2(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfgGo2LidarView = ObservationsCfgGo2LidarView()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 2 columns (rough | boxes) x 5 rows. Rows are difficulty under
        # TerrainGenerator's curriculum mode -- difficulty = (row + jitter)/num_rows --
        # so row 0 is easiest and row 4 hardest, in order rather than sampled.
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 2
        # Spread the spawn over every row. The training value caps it at 2, which would
        # leave the top two difficulties empty and unwatchable.
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt
        apply_lidar_view(self)
