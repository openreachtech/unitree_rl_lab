import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import RewardsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase2 import (
    CommandsCfgPhase2,
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)

# Mirrors Go2's Phase3 stair mix (flat/pyramid_stairs/pyramid_stairs_inv),
# including Go2's full step_height_range=(0.05, 0.25) ceiling -- by request,
# kept at Go2's original difficulty rather than capped to the 0.086 m wheel
# radius (see Phase1 ActionsCfg's comment). Risers above the wheel radius can't
# be rolled onto and have to be crossed by lifting a wheel with the leg, which
# is the same mechanism that stalled Phase2's terrain_levels curriculum on
# 0.15 m boxes -- expect the same plateau risk here, now deliberately, as a way
# to see how far the wheel-lift behavior actually gets pushed.
#
# Go2's variable-tread-width and floating-stairs refinements
# (PHASE3_TERRAIN_CFG_VARIABLE_WIDTH / PHASE3_TERRAIN_CFG_BALANCE_FLOATING) are
# sandbox fixes for leg-specific failure modes -- toe-catching on a narrow
# tread, discrete foot placement across an open riser -- that don't have a
# wheeled equivalent yet, so this starts from Go2's plain PHASE3_TERRAIN_CFG
# instead of that lineage.
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
            step_height_range=(0.05, 0.25),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.60,
            step_height_range=(0.05, 0.25),
            step_width=0.23,
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
    """Phase 3 commands: lin_vel_y starts narrower than Phase2 (stairs are less
    forgiving of lateral drift), mirroring Go2's own Phase2->Phase3 change.

    lin_vel_x/lin_vel_y limit_ranges match Go2 Phase3's exactly (1.2, 0.7)
    rather than carrying over Go2W's own higher Phase1/2 ceiling (1.5, 0.8).
    That ceiling is a wheel-speed premium justified by rollable terrain (flat
    ground, boxes near the wheel radius); Phase3's step_height_range now goes
    up to Go2's full 0.25 m, so most of this terrain isn't rollable at all and
    has to be crossed the same wheel-lift-via-leg way Go2's legs cross a
    stair -- there's no basis for the wheeled robot to go faster than the
    legged one on ground neither can roll over. ang_vel_z stays at Go2W's own
    1.0 (that gap vs. Go2's 1.2 predates Phase3, unrelated to this terrain
    change).
    """

    base_velocity = CommandsCfgPhase2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class RewardsCfgPhase3(RewardsCfg):
    """Applies the same *relative* loosening Go2's own Phase3 applies to its
    base weights, scaled onto Go2W's base weights rather than copying Go2's
    absolute numbers -- Go2W's base already differs from Go2's in several of
    these (e.g. action_rate -0.01 vs Go2's -0.1), so reusing Go2's Phase3
    constants directly would change the ratio, not just the terrain.
    """

    flat_orientation_l2 = RewardsCfg().flat_orientation_l2.replace(weight=-1.0)  # -2.5 * 0.4
    base_linear_velocity = RewardsCfg().base_linear_velocity.replace(weight=-0.5)  # -2.0 * 0.25
    joint_torques = RewardsCfg().joint_torques.replace(weight=-1e-4)  # legs only; -2e-4 * 0.5
    action_rate = RewardsCfg().action_rate.replace(weight=-0.005)  # -0.01 * 0.5


@configclass
class RobotEnvCfgPhase3(RobotEnvCfgPhase2):
    """Phase 3: stairs, riser height matching Go2's full 0.05-0.25 m range."""

    scene: RobotSceneCfgPhase3 = RobotSceneCfgPhase3(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase3 = CommandsCfgPhase3()
    rewards: RewardsCfgPhase3 = RewardsCfgPhase3()


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 2
        # Just the two pyramid types, one column each -- no flat column to burn
        # a difficulty row on.
        self.scene.terrain.terrain_generator.sub_terrains = {
            name: cfg for name, cfg in PHASE3_TERRAIN_CFG.sub_terrains.items() if name != "flat"
        }
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
