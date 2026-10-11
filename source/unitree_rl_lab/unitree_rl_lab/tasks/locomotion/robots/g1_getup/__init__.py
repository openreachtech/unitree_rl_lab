import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-GetUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-23dof-StandUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:StandUpPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-23dof-GetUp-HoSTModel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotHoSTModelEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotHoSTModelPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-23dof-GetUp-HoSTURDF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotHoSTURDFEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotHoSTURDFPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpPPORunnerCfg",
    },
)
