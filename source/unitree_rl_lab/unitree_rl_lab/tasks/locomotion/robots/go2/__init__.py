import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v1",
    entry_point=f"{__name__}.velocity_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:RobotEnvCfgGo2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:RobotPlayEnvCfgGo2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v2",
    entry_point=f"{__name__}.velocity_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2_v2:RobotEnvCfgGo2V2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2_v2:RobotPlayEnvCfgGo2V2",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2_v2:Go2VelocityV2PPORunnerCfg",
    },
)
