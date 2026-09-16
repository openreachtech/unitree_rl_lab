"""Unitree A2: the Go2 blind-GRU curriculum on a robot one size up.

Four phases, each resuming from the last, mirroring ``Go2-Blind-GRU-Phase1..4``:

  Phase 1   flat ground
  Phase 2   random rough + boxes
  Phase 3   stairs (the Phase3-balance recipe)
  Phase 4   free-standing walls to step over

The actor is proprioception-only throughout and the critic privileged; the runner puts a
GRU in front of both MLPs. Terrain and clearance lengths are Go2's x1.29 (leg length) and
gait times x1.14 (sqrt of it) -- see ``velocity_env_cfg.py`` for the full derivation.

  python scripts/rsl_rl/train.py --task A2-Blind-GRU-Phase1
  python scripts/rsl_rl/train.py --task A2-Blind-GRU-Phase2 --resume --previous-task A2-Blind-GRU-Phase1
  python scripts/rsl_rl/train.py --task A2-Blind-GRU-Phase3 --resume --previous-task A2-Blind-GRU-Phase2
  python scripts/rsl_rl/train.py --task A2-Blind-GRU-Phase4 --resume --previous-task A2-Blind-GRU-Phase3
"""

import gymnasium as gym

# Shared by every registration below. Module-qualified rather than an absolute string so
# train.py's task filter, which keys on the "locomotion." prefix, still matches.
_CFG = __name__
_RUNNER = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg"

gym.register(
    id="A2-Blind-GRU-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 2: random rough 50% / boxes 50%.
gym.register(
    id="A2-Blind-GRU-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 3: stairs. The hardest thing a blind policy is asked to cross here short of
# Phase 4, and the one that most needs the GRU -- a step edge is only observable through
# contact. Registered against the BalanceMatched variant, as Go2 is.
gym.register(
    id="A2-Blind-GRU-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotEnvCfgPhase3BalanceMatched",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 4: stepping over short free-standing walls. Continual learning on top of
# Phase 3 -- stairs are a surface to climb, these are isolated walls to clear, invisible
# to a blind policy until a foot hits them.
gym.register(
    id="A2-Blind-GRU-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)
