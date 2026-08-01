import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import (
    CommandsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
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
class CommandsCfgPhase1(CommandsCfg):
    """Go2W velocity commands.

    The wheels top out at 30.1 rad/s over a 0.086 m radius, so ~2.6 m/s is the
    hardware ceiling; 2.0 m/s leaves headroom for the tracking reward to stay
    reachable. This ceiling is tied to the wheel action scale in ``ActionsCfg``
    (currently 8.0) and the two must be raised together, or the policy has to
    emit actions far out in the tail of its output distribution to comply.

    ``lin_vel_x``/``lin_vel_y`` start small and are widened toward
    ``limit_ranges`` by the ``lin_vel_cmd_levels`` curriculum. Nothing widens
    ``ang_vel_z``, so its ``ranges`` value is the real training range and
    ``limit_ranges`` must repeat it: ``limit_ranges`` is what
    ``export_deploy_cfg`` publishes as the joystick range in ``deploy.yaml``, so
    any gap there would let the robot be driven outside what it was trained on.
    """

    base_velocity = CommandsCfg().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.5, 1.5)
        ),
    )


@configclass
class RobotEnvCfgPhase1(RobotEnvCfg):
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
