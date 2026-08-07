import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Go2-Biped-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Biped-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Biped-Front",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg_front:RobotEnvCfgFront",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg_front:RobotPlayEnvCfgFront",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Multimode",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multimode_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.multimode_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg_multimode:BipedMultimodePPORunnerCfg"
        ),
    },
)
