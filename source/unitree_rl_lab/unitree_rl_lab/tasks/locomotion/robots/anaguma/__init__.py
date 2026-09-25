import gymnasium as gym

# Stage A of the running long jump on Anaguma: flat-ground running, to be used as the
# Net2Net base for the jump-integrated Stage B. See longjump_base_env_cfg.py and
# project memory プロジェクト_Anaguma移植の基礎.md.
gym.register(
    id="Anaguma-LongJump-Base-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.longjump_base_env_cfg:RobotEnvCfgAnagumaLongJumpBase",
        "play_env_cfg_entry_point": f"{__name__}.longjump_base_env_cfg:RobotPlayEnvCfgAnagumaLongJumpBase",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Stage B: jump-integrated env built on Stage A, reached via Net2Net transplant.
# See longjump_env_cfg.py -- the height objective is pinned there rather than inherited
# from whatever Go2 is currently chasing.
gym.register(
    id="Anaguma-LongJump-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.longjump_env_cfg:RobotEnvCfgAnagumaLongJump",
        "play_env_cfg_entry_point": f"{__name__}.longjump_env_cfg:RobotPlayEnvCfgAnagumaLongJump",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LongJumpPPORunnerCfg",
    },
)
