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
    id="Go2-Jump-60",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_jump:RobotEnvCfgJump",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_jump:RobotPlayEnvCfgJump",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Sideflip-Double",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_jump:RobotEnvCfgSideflipDouble",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_jump:RobotPlayEnvCfgSideflipDouble",
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


# =================================================================================================
# Go2-Multitask-Jump-Phase1 / Phase2 -- the same two-stage jump curriculum as Go2-Jump-Phase1 /
# Phase2, re-expressed on the unified 122/330-column observation shared with the locomotion side
# (unitree_rl_lab.tasks.multitask.obs_spec). Task definition is otherwise identical: same commands,
# rewards, terminations, events and assist-force curriculum.
#
# They exist so the acrobatics expert's input layout already matches the multi-task policy, making
# weight surgery unnecessary when the mixture of experts is assembled. Both phases move together --
# Phase 2 resumes from Phase 1, so a Phase 1 left on the old observation could not seed it.
#
# Bootstrap from the existing checkpoints rather than training from scratch. widen_checkpoint.py
# re-expresses a checkpoint on the unified observation by zero-filling the new columns, which leaves
# the network mathematically identical, so each phase restarts with its skill intact:
#
#   python scripts/rsl_rl/widen_checkpoint.py --previous-task Go2-Jump-Phase1 \
#       --task Go2-Multitask-Jump-Phase1
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Jump-Phase1 --resume
#
#   python scripts/rsl_rl/widen_checkpoint.py --previous-task Go2-Jump-Phase2 \
#       --task Go2-Multitask-Jump-Phase2
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Jump-Phase2 --resume
# =================================================================================================

gym.register(
    id="Go2-Multitask-Jump-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_multitask:RobotEnvCfgMultitaskPhase1",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_multitask:RobotPlayEnvCfgMultitaskPhase1",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Multitask-Jump-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg_multitask:RobotEnvCfgMultitaskPhase2",
        "play_env_cfg_entry_point": f"{__name__}.jump_env_cfg_multitask:RobotPlayEnvCfgMultitaskPhase2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
