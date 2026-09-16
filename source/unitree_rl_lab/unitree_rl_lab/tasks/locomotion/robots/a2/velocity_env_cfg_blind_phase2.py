"""Phase 2: random rough ground and boxes.

The A2 counterpart of ``robots/go2/velocity_env_cfg_blind_phase2.py``. Every terrain
length is Go2's x1.29, so a box is the same fraction of leg length to climb as it was
there, and the Go2 lineage's measured tuning carries over rather than being re-guessed.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg import A2_TERRAIN_TILE, CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg_blind import RewardsCfgA2
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg_blind_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)

PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=A2_TERRAIN_TILE,
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
            # Go2's (0.01, 0.06) / 0.01 / 0.25, all x1.29.
            noise_range=(0.013, 0.077),
            noise_step=0.013,
            border_width=0.32,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.50,
            # Go2's 0.45 / (0.05, 0.15) / 2.0, all x1.29. A 0.19 m box is 35% of A2's
            # 0.55 m leg, the same fraction 0.15 m is of Go2's 0.426 m.
            grid_width=0.58,
            grid_height_range=(0.065, 0.19),
            platform_width=2.6,
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
    # Go2's Phase 2 limits x1.14 on the linear axes (1.2 -> 1.37, 0.7 -> 0.8).
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.37, 1.37), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RewardsCfgPhase2(RewardsCfgA2):
    """Posture penalties relaxed for box terrain, at the Go2 lineage's measured values.

    All three penalise exactly what crossing a box requires:
      base_linear_velocity  vertical base velocity -- stepping up and down a box *is*
                            vertical base velocity.
      flat_orientation_l2   the horizontal components of projected gravity, i.e. any body
                            tilt -- but the body has to pitch to get a leg up.
      joint_pos             deviation from the default stance, which every non-flat
                            foothold is.

    The weights are unitless trade-offs between reward terms, not lengths, so they carry
    over from Go2 unscaled. On Go2 these were measured against the un-relaxed weights over
    4000 iterations from the same Phase 1 checkpoint: mean episode length 972 against 888,
    about 9% fewer falls, at the same terrain level. The gain is stability, not reach.
    """

    base_linear_velocity = RewardsCfgA2().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfgA2().flat_orientation_l2.replace(weight=-1.0)
    joint_pos = RewardsCfgA2().joint_pos.replace(weight=-0.4)


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: rough terrain and boxes."""

    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=4096, env_spacing=3.2)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()
    # Back to the base curriculum, which has terrain_levels. Phase 1 sets it to None
    # because its scene is a bare plane, and inheriting that here is silently harmful:
    # RobotEnvCfg.__post_init__ reads terrain_levels to decide whether to switch the
    # generator into curriculum mode, so a None here means the patches are laid out
    # randomly instead of by ascending difficulty, and every robot starts on terrain of
    # arbitrary difficulty with no ratchet. Phase 3 and 4 inherit this.
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=32, env_spacing=3.2)

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
