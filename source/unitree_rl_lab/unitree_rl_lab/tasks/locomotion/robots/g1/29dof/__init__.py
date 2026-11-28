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
    id="G1-MLP-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mlp_phase1_env_cfg:MlpPhase1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.mlp_phase1_env_cfg:MlpPhase1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymmetricPPORunnerCfg",
    },
)

gym.register(
    id="G1-GRU-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymmetricGruPPORunnerCfg",
    },
)

gym.register(
    id="G1-GRU-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymmetricGruPPORunnerCfg",
    },
)

gym.register(
    id="G1-GRU-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gru_env_cfg:GruPhase3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymmetricGruPPORunnerCfg",
    },
)

gym.register(
    id="G1-FPM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fpm_env_cfg:FpmEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.fpm_env_cfg:FpmPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G1FpmPPORunnerCfg",
    },
)
