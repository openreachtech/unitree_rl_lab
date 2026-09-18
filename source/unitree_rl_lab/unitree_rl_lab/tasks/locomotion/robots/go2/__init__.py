import gymnasium as gym

# Shared by every registration below. Module-qualified rather than an absolute string so
# train.py's task filter, which keys on the "locomotion." prefix, still matches.
_CFG = __name__
_RUNNER = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg"

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
    id="Go2-Blind-GRU-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 2: flat 10% / random rough 40% / boxes 50%.
gym.register(
    id="Go2-Blind-GRU-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 3: stairs. The hardest thing a blind policy is asked to cross here, and the
# one that most needs the GRU -- a step edge is only observable through contact.
gym.register(
    id="Go2-Blind-GRU-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotEnvCfgPhase3BalanceMatched",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 4: stepping over short free-standing walls, after Go2-v3-Phase4. Continual
# learning on top of Phase 3 -- stairs are a surface to climb, these are isolated walls
# to clear, invisible to a blind policy until a foot hits them. See
# velocity_env_cfg_blind_phase4.py.
#   --task Go2-Blind-GRU-Phase4 --resume --previous-task Go2-Blind-GRU-Phase3
gym.register(
    id="Go2-Blind-GRU-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)


# ===========================================================================
# Experimental: Go2-Blind-GRU-Phase4 plus a Livox MID-360, consuming the real scan
# sequence at the hardware's rate (4,000 rows/step = 200k pts/s at 50 Hz, of which
# every 2nd is cast). The body is transparent, matching the self-filtered cloud the
# deployed publisher produces. Display-only
# observation group; policy inputs unchanged, so Go2-Blind-GRU-Phase4 checkpoints
# still load. There is no mid360 experiment folder, so play needs that checkpoint
# passed explicitly:
#   python scripts/rsl_rl/play.py --task Go2-Blind-GRU-Mid360-Phase4 --num_envs 4 \
#       --checkpoint logs/rsl_rl/go2_blind_gru_phase4/<run>/model_7300.pt
# Green cells are measured this step, red are held from an earlier one. Flip
# MID360_DYNAMIC_MESH in velocity_env_cfg_mid360.py to make the body opaque and
# see the legs sweep streaks through the map.
# ===========================================================================
gym.register(
    id="Go2-Blind-GRU-Mid360-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_mid360:RobotEnvCfgMid360Phase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_mid360:RobotPlayEnvCfgMid360Phase4",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

