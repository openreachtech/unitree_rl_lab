import gymnasium as gym

# G1 get-up from lying on the back (HoST, policy_docs/g1_getup.md). Both train from scratch in one run:
# HoST's rewards until the robot gets up without the pulling force, then the posture terms are faded in
# (staged.py).
gym.register(
    id="Unitree-G1-23dof-GetUp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotStagedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpStagedPPORunnerCfg",
    },
)

# 29dof: the same with the URDF effort limits and a 500-iteration delay before the posture ramp
gym.register(
    id="Unitree-G1-29dof-GetUp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29StagedV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.getup_29dof_env_cfg:Robot29PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUp29HoSTPPORunnerCfg",
    },
)
