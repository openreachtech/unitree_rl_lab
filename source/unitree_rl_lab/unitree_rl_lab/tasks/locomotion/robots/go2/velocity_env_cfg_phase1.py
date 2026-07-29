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
        # CommandsCfgGo2 drops rel_standing_envs to 0.01 for every Go2 v3 task -- only 1%
        # of envs ever get a near-zero command, so "stand still" was almost never
        # practiced. MuJoCo deploy testing showed this checkpoint flattening/flapping its
        # legs on flat ground at zero command (not seen on the older blind v1 policy,
        # which trains at the un-touched base CommandsCfg's rel_standing_envs=0.1).
        # Promoted from sandbox/try1.py after MuJoCo confirmed the flat-idle flapping is
        # resolved with this alone -- Phase1 has no active terrain-height reward term to
        # also gate (see RewardsCfgGo2: wild_foot_clearance / foot_clearance_terrain_adaptive
        # / forward_command_progress are all weight=0.0 by default and untouched here).
        rel_standing_envs=0.1,
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
        # Raw ray-hit markers (red) show true sensor geometry, never the noise applied
        # downstream to the policy's observation -- off so only the noisy overlay is visible.
        self.scene.height_scanner.debug_vis = False
        # Temporary: overlay a noisy-height sample (orange) so the configured height-scan
        # noise applied to the policy's observation is visible in the viewport.
        self.observations.policy.height_scan.params["debug_vis_noise"] = True
