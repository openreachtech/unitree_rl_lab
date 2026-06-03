from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import POLICY_HEIGHT_SCAN_CFG, apply_play_velocity_ranges
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ObservationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import ObservationsCfgGo2, RobotEnvCfgGo2


@configclass
class PolicyCfgGo2V2(ObservationsCfg.PolicyCfg):
    """Policy observations with exteroceptive height scan (LiDAR / ray grid)."""

    height_scan = POLICY_HEIGHT_SCAN_CFG


@configclass
class ObservationsCfgGo2V2(ObservationsCfgGo2):
    """Go2 v2: policy and critic both use height_scan."""

    policy: PolicyCfgGo2V2 = PolicyCfgGo2V2()


@configclass
class RobotEnvCfgGo2V2(RobotEnvCfgGo2):
    """Go2 velocity env with height_scan in the policy observation vector."""

    observations: ObservationsCfgGo2V2 = ObservationsCfgGo2V2()


@configclass
class RobotPlayEnvCfgGo2V2(RobotEnvCfgGo2V2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.height_scanner.debug_vis = True
        apply_play_velocity_ranges(self)


@configclass
class Go2VelocityV2PPORunnerCfg(BasePPORunnerCfg):
    """PPO runner for Go2 velocity v2 (policy includes ~187-dim height_scan)."""

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
