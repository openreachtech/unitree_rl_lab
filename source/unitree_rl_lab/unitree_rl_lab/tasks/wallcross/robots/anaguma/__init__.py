"""Anaguma port of the hind-leg wall-crossing pipeline (walk -> hind stand -> wall cross).

Train the walking expert the same way as Go2's:
  python scripts/rsl_rl/train_and_aggregate.py --task Anaguma-Multitask-Gallop-Phase1 --max_iterations 1300
  python scripts/rsl_rl/train_and_aggregate.py --task Anaguma-Multitask-Gallop-Phase2 \
      --previous-task Anaguma-Multitask-Gallop-Phase1
"""

import gymnasium as gym

_PPO = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

gym.register(
    id="Anaguma-Multitask-Gallop-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:AnagumaWalkPhase1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:AnagumaWalkPhase1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Anaguma-Multitask-Gallop-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:AnagumaWalkPhase2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:AnagumaWalkPhase2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO,
    },
)

# Ordinary walking (trot) expert -- the walking expert the wall-crossing pipeline actually wants.
# The gallop tasks above copy Tamura's gallop expert and bound forward at every speed.
#   python scripts/rsl_rl/train_wallcross.py --task Anaguma-Multitask-Trot --num_envs 4096 --headless \
#       --deploy-keyboard-commands --max_iterations 3000
gym.register(
    id="Anaguma-Multitask-Trot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trot_env_cfg:AnagumaTrotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.trot_env_cfg:AnagumaTrotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO,
    },
)
