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

# Play-only: Phase 2 with the LiDAR noise pinned to one condition instead of drawn
# 60/30/10, so each can be inspected on its own. The policy is unchanged and never sees
# the map; only the drawn grid differs. See velocity_env_cfg_blind_phase2.py.
#   --task Go2-Blind-GRU-Phase2-Noise-Weak --checkpoint <phase2 checkpoint>
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase2-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2Noise{_level}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )

# Play-only: the Phase 3 default on a 2 x 3 stepped terrain (pyramid | inverted pyramid,
# rows centred on 5 / 10 / 15 cm), with the LiDAR noise pinned to one condition instead of
# drawn 60/30/10. The policy is unchanged and never sees the map; only the drawn grid
# differs. See velocity_env_cfg_blind_phase3.py.
#   --task Go2-Blind-GRU-Phase3-Noise-Weak --checkpoint <phase3 checkpoint>
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase3-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3Noise{_level}",
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

# The same three pinned-noise play variants Phase 2 and 3 have.
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase4-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4Noise{_level}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )



# ===========================================================================
# Perceptive arms: what a height map buys the blind policy, and how much of that
# depends on the map being clean. Four arms x four phases; see
# velocity_env_cfg_perceptive.py for what differs between them (only the actor's
# exteroceptive input, and the fan's presence in the scene).
#
# Phase 4 resumes from Phase 2, not Phase 3 -- stair training was measured to cost
# wall-crossing on this lineage.
#
#   python scripts/rsl_rl/train.py --task Go2-HM-Belief-Phase2 --resume \
#       --load_run <phase1 run> --checkpoint model_500.pt
# ===========================================================================
_PERCEPTIVE_RUNNER = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PerceptiveGruPPORunnerCfg"

gym.register(
    id="Go2-HM-Blind-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Blind-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Blind-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindPlayCfgPhase3",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Blind-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BlindPlayCfgPhase4",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)

gym.register(
    id="Go2-HM-Belief-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Belief-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Belief-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefPlayCfgPhase3",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Belief-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:BeliefPlayCfgPhase4",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)

gym.register(
    id="Go2-HM-Noisy-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Noisy-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Noisy-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyPlayCfgPhase3",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Noisy-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:NoisyPlayCfgPhase4",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)

gym.register(
    id="Go2-HM-Clean-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Clean-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Clean-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanPlayCfgPhase3",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
gym.register(
    id="Go2-HM-Clean-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_perceptive:CleanPlayCfgPhase4",
        "rsl_rl_cfg_entry_point": _PERCEPTIVE_RUNNER,
    },
)
