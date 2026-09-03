"""Go2 bipedal-stance tasks."""

import gymnasium as gym

# =================================================================================================
# Go2-Multitask-Handstand -- the bipedal expert.
#
# The robot rises onto its front legs, hind legs tucked, and walks there tracking a velocity command
# of up to 1 m/s. Same observation layout, control rate and actuator model as the locomotion and
# acrobatics experts, so it can join the mixture without weight surgery beyond the zero-padding
# widen. The `Go2-Multitask-` prefix names that layout, not this directory -- `Go2-Multitask-Jump-*`
# lives under `dynamic/` and `Go2-Multitask-Gallop-*` under `locomotion/` for the same reason.
#
# Trained from scratch: the front stance exists on `feat/biped` as `Go2-Biped-Front` and works, but
# that policy carries a jointly-trained state estimator and its own proprio-history layout, so its
# weights do not map onto this network. The recipe carries over; the checkpoint does not.
#
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Handstand
# =================================================================================================

gym.register(
    id="Go2-Multitask-Handstand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handstand_env_cfg_multitask:RobotEnvCfgHandstand",
        "play_env_cfg_entry_point": f"{__name__}.handstand_env_cfg_multitask:RobotPlayEnvCfgHandstand",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
