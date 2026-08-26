"""Go2 multi-task environment: running and acrobatics in one mixture-of-experts policy."""

import gymnasium as gym

# =================================================================================================
# Go2-Multitask-Phase1 -- the merged environment.
#
# Locomotion and acrobatics in one 20 s episode: full-range velocity commands (up to 3.5 m/s), with
# a jump/backflip/sideflip interrupting them three to four times per episode. The policy is a
# mixture of three experts -- locomotion, acrobatics, and one starting random to absorb the
# run/take-off and landing/run transitions that neither pre-trained policy has ever visited.
#
# Both source reward sets are carried over unchanged and switched by command state, because the
# value function is initialised from critics trained against those exact rewards. External
# assistance is off on both sides: each expert was weaned off it by its own curriculum.
#
# The experts must be loaded before training -- build the starting checkpoint from the two
# single-task runs, then resume from it:
#
#   python scripts/rsl_rl/build_moe_checkpoint.py \
#       --locomotion-task Go2-Multitask-Gallop-Phase2 \
#       --acrobatics-task Go2-Multitask-Jump-Phase2 \
#       --task Go2-Multitask-Phase1
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Phase1 --resume
# =================================================================================================

gym.register(
    id="Go2-Multitask-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe:RobotEnvCfgMoe",
        "play_env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe:RobotPlayEnvCfgMoe",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.multitask.agents.rsl_rl_ppo_cfg:MoEPPORunnerCfg",
    },
)
