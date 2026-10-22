"""Go2 walk / fall / get-up in one mixture-of-experts policy."""

import gymnasium as gym

# =================================================================================================
# Unitree-Go2-Recovery-MoE-v0 -- walk on a velocity command, get knocked down, get up, walk on.
#
# Two experts, both initialised from trained blind policies: locomotion
# (Unitree-Go2-Velocity-v1-Phase3-balance) and get-up (Unitree-Go2-GetUp-v0). The gate's prior
# routes by trunk attitude. Build the starting checkpoint, then resume from it:
#
#   python scripts/rsl_rl/build_recovery_moe_checkpoint.py \
#       --locomotion-checkpoint <velocity run>/model_6497.pt \
#       --getup-checkpoint <get-up run>/model_2999.pt
#   python scripts/rsl_rl/train.py --task Unitree-Go2-Recovery-MoE-v0 --headless --resume
# =================================================================================================

gym.register(
    id="Unitree-Go2-Recovery-MoE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RecoveryMoEPPORunnerCfg",
    },
)


# Same environment and policy, played from a robot that is already lying down (no knockdowns):
#   python scripts/rsl_rl/play.py --task Unitree-Go2-Recovery-MoE-Fallen-v0 --checkpoint <run>/model_XXXX.pt
gym.register(
    id="Unitree-Go2-Recovery-MoE-Fallen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotFallenPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RecoveryMoEPPORunnerCfg",
    },
)


# Walking forward and knocked down every 5 s, too hard to catch:
#   python scripts/rsl_rl/play.py --task Unitree-Go2-Recovery-MoE-HardKnockdown-v0 --checkpoint <run>/model_XXXX.pt
gym.register(
    id="Unitree-Go2-Recovery-MoE-HardKnockdown-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotHardKnockdownPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RecoveryMoEPPORunnerCfg",
    },
)


# Get-up alone, on the same observation / action / deploy interface as the mixture above, so the
# same deploy state runs either one. Converted from Unitree-Go2-GetUp-v0:
#   python scripts/rsl_rl/build_recovery_moe_checkpoint.py --getup-only --getup-checkpoint <run>/model_2999.pt
#   python scripts/rsl_rl/export_policy.py --task Unitree-Go2-GetUp-Unified-v0 --checkpoint <converted>/model_0.pt
gym.register(
    id="Unitree-Go2-GetUp-Unified-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:GetUpEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:GetUpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GetUpUnifiedPPORunnerCfg",
    },
)
