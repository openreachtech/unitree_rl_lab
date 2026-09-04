"""Go2 bipedal-stance tasks."""

import gymnasium as gym

# =================================================================================================
# Go2-Multitask-Biped-Front -- the bipedal expert.
#
# Named for the stance rather than the pose, so the mirror (`Go2-Multitask-Biped-Hind`) has an
# obvious name and the pair reads as one family -- the same shape as `feat/biped`'s own
# `Go2-Biped-Front`. The `Go2-Multitask-` prefix names the observation layout, not this directory;
# `Go2-Multitask-Jump-*` lives under `dynamic/` and `Go2-Multitask-Gallop-*` under `locomotion/`
# for the same reason.
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
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Biped-Front
# =================================================================================================

gym.register(
    id="Go2-Multitask-Biped-Front",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_front_env_cfg_multitask:RobotEnvCfgBipedFront",
        "play_env_cfg_entry_point": f"{__name__}.biped_front_env_cfg_multitask:RobotPlayEnvCfgBipedFront",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# =================================================================================================
# Go2-Multitask-Biped-Hind -- the mirror stance.
#
# Same skill on the other pair of legs, and the harder of the two: `feat/biped` reached a working
# front stance in three sandbox tries and needed seven for this one. Two of its problems were never
# solved there -- a sideways-drifting gait, and a front foot touching down during forward walking --
# so expect this to need a round of its own rather than to fall out of the mirror. See the config
# module's docstring for what is known before starting.
#
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Biped-Hind
#
# Starting from scratch is the default. Resuming from the front stance is worth trying instead --
# same network, same observation, and the `stance` column already means something to it -- but it
# has not been tested, and a policy that has learned "nose down" may fight "nose up" harder than a
# fresh one learns it.
# =================================================================================================

gym.register(
    id="Go2-Multitask-Biped-Hind",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_hind_env_cfg_multitask:RobotEnvCfgBipedHind",
        "play_env_cfg_entry_point": f"{__name__}.biped_hind_env_cfg_multitask:RobotPlayEnvCfgBipedHind",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
