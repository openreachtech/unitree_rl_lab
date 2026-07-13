import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Unitree-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v2-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v2-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v2-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Promoted from sandbox Try-4: terrain-adaptive foot clearance for a natural
# flat-ground gait, terrain_levels >= 4.5 (reached 4.899).
gym.register(
    id="Unitree-Go2-Velocity-v2-Phase3-balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3Balance",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3Balance",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Promoted from sandbox Try-1 + Try-2: maximizes terrain_levels (~5.3-5.4),
# exaggerated flat-ground gait as a tradeoff.
gym.register(
    id="Unitree-Go2-Velocity-v2-Phase3-stairfocus",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3StairFocus",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3StairFocus",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Balance rewards + terrain mix that includes floating inverted pyramid stairs.
gym.register(
    id="Unitree-Go2-Velocity-v2-Phase3-balance-floating",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3BalanceFloating",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3BalanceFloating",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
