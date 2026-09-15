import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg, RobotSceneCfg
from isaaclab.sensors import RayCasterCfg

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    GO2_LIDAR_SCANNER_CFG,
    ObservationsCfgGo2LidarView,
    apply_lidar_view,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    foot_ring_sensor,
    CommandsCfgGo2,
    RobotEnvCfgGo2,
)


@configclass
class RobotSceneCfgPhase1(RobotSceneCfg):
    # Lee et al. 2020's per-foot terrain rings, feeding the privileged critic terms in
    # velocity_env_cfg_blind.py. Declared here so Phase 2 and 3 inherit them.
    foot_scan_FL_foot: RayCasterCfg = foot_ring_sensor("FL_foot")
    foot_scan_FR_foot: RayCasterCfg = foot_ring_sensor("FR_foot")
    foot_scan_RL_foot: RayCasterCfg = foot_ring_sensor("RL_foot")
    foot_scan_RR_foot: RayCasterCfg = foot_ring_sensor("RR_foot")

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
class CurriculumCfgPhase1(CurriculumCfg):
    """No terrain-level ratchet: the scene is an infinite plane, not a generated grid,
    so there are no levels for ``mdp.terrain_levels_vel`` to read. Setting it to None
    also flips off ``terrain_generator.curriculum`` in ``RobotEnvCfg.__post_init__``."""

    terrain_levels = None


@configclass
class RobotEnvCfgPhase1(RobotEnvCfgGo2):
    """Phase 1: flat terrain."""

    scene: RobotSceneCfgPhase1 = RobotSceneCfgPhase1(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase1 = CommandsCfgPhase1()
    curriculum: CurriculumCfgPhase1 = CurriculumCfgPhase1()


@configclass
class RobotSceneCfgPlayPhase1(RobotSceneCfgPhase1):
    """Phase 1's scene plus the LiDAR fan, play only."""

    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class RobotPlayEnvCfgPhase1(RobotEnvCfgPhase1):
    scene: RobotSceneCfgPlayPhase1 = RobotSceneCfgPlayPhase1(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfgGo2LidarView = ObservationsCfgGo2LidarView()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt
        apply_lidar_view(self)
