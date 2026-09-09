"""Go2 multi-task environment: running and acrobatics in one mixture-of-experts policy."""

import gymnasium as gym

# =================================================================================================
# Go2-Multitask -- the merged environment.
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
#       --task Go2-Multitask-v1
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-v1 --resume
# =================================================================================================

gym.register(
    id="Go2-Multitask-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe:RobotEnvCfgMoe",
        "play_env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe:RobotPlayEnvCfgMoe",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.multitask.agents.rsl_rl_ppo_cfg:MoEPPORunnerCfg",
    },
)


# =================================================================================================
# Go2-Multitask-v2 -- locomotion, acrobatics and a bipedal stance in three experts.
#
# The two-expert policy above was renamed `Go2-Multitask-v1` so this one can be `-v2`: gymnasium
# reads a trailing `-v<N>` as its own version suffix and refuses to register a versioned id
# alongside an unversioned one of the same base name. Two versioned ids coexist, an unversioned and
# a versioned one do not -- and the refusal aborts the whole module, taking every task in this
# package down with it rather than just the one being registered.
#
# The third slot no longer holds a randomly initialised transition expert. Measured on the finished
# two-expert policy that expert's routing weight read 0.000 while running, 0.000 inside an
# acrobatic window and 0.001 in the hand-back bin where it had the most to contribute, so
# `Go2-Multitask-Biped` takes the slot and every expert starts from a trained policy.
#
#   python scripts/rsl_rl/build_moe_checkpoint.py \
#       --locomotion-task Go2-Multitask-Gallop-Phase2 \
#       --acrobatics-task Go2-Multitask-Jump-Phase2 \
#       --biped-task Go2-Multitask-Biped \
#       --task Go2-Multitask-v2
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-v2 --resume
#
# Watch the two take-off limits and the gate split. The bipedal stance is offered only below a
# commanded speed its own curriculum raises, for the same reason the acrobatics one exists: neither
# expert has ever started its skill from a moving robot.
# =================================================================================================

gym.register(
    id="Go2-Multitask-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe_v2:RobotEnvCfgMoeV2",
        "play_env_cfg_entry_point": f"{__name__}.multitask_env_cfg_moe_v2:RobotPlayEnvCfgMoeV2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.multitask.agents.rsl_rl_ppo_cfg:MoEV2PPORunnerCfg",
    },
)
