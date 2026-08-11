import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Go2-Jump-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Jump-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
