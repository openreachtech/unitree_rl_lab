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

gym.register(
    id="Unitree-G1-29dof-GetUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUp29PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-GetUp-HoST-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29HoSTEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29HoSTPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUp29HoSTPPORunnerCfg",
    },
)

# from scratch in one run: HoST's rewards, then the posture terms faded in (staged.py)
gym.register(
    id="Unitree-G1-23dof-GetUp-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotStagedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpStagedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-GetUp-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29StagedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUp29HoSTPPORunnerCfg",
    },
)
