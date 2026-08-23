import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotSceneCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    CommandsCfgGo2,
    RobotEnvCfgGo2,
)


@configclass
class RobotSceneCfgPhase1(RobotSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
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
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
