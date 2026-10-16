import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Go2w-Biped-Front",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg_front:RobotEnvCfgBipedFront",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg_front:RobotPlayEnvCfgBipedFront",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2w-Biped-Hind",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg_hind:RobotEnvCfgBipedHind",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg_hind:RobotPlayEnvCfgBipedHind",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2w-Biped",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:RobotEnvCfgBiped",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:RobotPlayEnvCfgBiped",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
