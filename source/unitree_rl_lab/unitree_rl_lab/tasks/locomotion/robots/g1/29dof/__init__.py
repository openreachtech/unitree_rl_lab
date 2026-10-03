import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
gym.register(
    id="Unitree-G1-29dof-Velocity-Gait",
    entry_point="unitree_rl_lab.tasks.locomotion.gait_env:VelocityManagerBasedRLGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
gym.register(
    id="Unitree-G1-29dof-Velocity-Gait-Stairs",
    entry_point="unitree_rl_lab.tasks.locomotion.gait_env:VelocityManagerBasedRLGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitStairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
# --- 条件B: モデル生成の粗い基準軌道で誘導する設定 -------------------
gym.register(
    id="Unitree-G1-29dof-Velocity-GaitRef",
    entry_point="unitree_rl_lab.tasks.locomotion.gait_env:VelocityManagerBasedRLGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitRefEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitRefPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
gym.register(
    id="Unitree-G1-29dof-Velocity-GaitRef-Stairs",
    entry_point="unitree_rl_lab.tasks.locomotion.gait_env:VelocityManagerBasedRLGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitRefStairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gait_env_cfg:GaitRefStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)