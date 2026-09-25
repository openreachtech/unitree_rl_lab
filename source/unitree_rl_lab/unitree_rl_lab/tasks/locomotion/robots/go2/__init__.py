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
    entry_point=f"{__name__}.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:RobotEnvCfgGo2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:RobotPlayEnvCfgGo2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Stage A of the running long jump task: flat-ground running with an
# extended forward-speed ceiling, used as the Net2Net base for
# Unitree-Go2-LongJump-v1 (Stage B). See longjump_base_env_cfg.py.
gym.register(
    id="Unitree-Go2-LongJump-Base-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.longjump_base_env_cfg:RobotEnvCfgLongJumpBase",
        "play_env_cfg_entry_point": f"{__name__}.longjump_base_env_cfg:RobotPlayEnvCfgLongJumpBase",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Stage B: jump-integrated env built on Stage A. See longjump_env_cfg.py.
gym.register(
    id="Unitree-Go2-LongJump-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.longjump_env_cfg:RobotEnvCfgLongJump",
        "play_env_cfg_entry_point": f"{__name__}.longjump_env_cfg:RobotPlayEnvCfgLongJump",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LongJumpPPORunnerCfg",
    },
)
