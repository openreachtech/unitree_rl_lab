"""Unitree A2: Go2's jump task on a robot one size up, as the first rung of a height ladder.

``Go2-Jump-60`` (feat/jump) ported to A2. The intent is to raise the target in 10 cm steps
until ``Metrics/jump/max_height`` stops following it, which is what actually measures how
high A2 can jump; only the 0.60 m rung has been trained and verified so far, so only it is
registered. A2 is 2.49x Go2's mass with roughly 5x the joint torque, so the torque-to-weight
ratio is about double and 1 m should be in reach.

  python scripts/rsl_rl/train.py --task A2-Jump-Phase1
  python scripts/rsl_rl/train.py --task A2-Jump-60 --resume --previous-task A2-Jump-Phase1

Measured on the trained A2-Jump-60 policy, unaided (assist_scale 0.0), 256 randomised envs:
0.592 m above stance, absolute apex 1.008 m, 256/256 reaching the target and landing upright.
That is the target being hit, not A2's ceiling -- ``motion_progress`` is two-sided, so
exceeding 0.60 m is penalised too. Finding the ceiling means adding rungs, which is
``TARGET_HEIGHTS_CM`` in ``jump_heights.py`` plus a fresh assist calibration per rung.

Phase 1 trains nothing but a quiet stand; it exists so the jump rungs resume into the final
observation layout instead of spending their iterations learning to stand. See
``jump_env_cfg_jump.py`` for what each rung inherits from Go2-Jump-60 and why, and
``jump_env_cfg.py`` for the Go2 -> A2 scaling.
"""

import gymnasium as gym

# Relative, and from a module with no imports of its own: this file is imported by
# train.py/list_envs.py before the simulator app launches, so anything that reaches
# isaaclab from here fails on `pxr`.
from .jump_heights import TARGET_HEIGHTS_CM

# Module-qualified rather than an absolute string, so the "dynamic." prefix that
# scripts/list_envs.py and scripts/rsl_rl/train.py filter on still matches -- those import
# the task tree with .../tasks/ on sys.path, which makes this module "dynamic.robots.a2".
_CFG = __name__
_RUNNER = "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

gym.register(
    id="A2-Jump-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.jump_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{_CFG}.jump_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# One task per rung of the height ladder. The config classes are generated at import time
# from the same tuple (see jump_env_cfg_jump._build_height_variant), so adding a height
# means editing TARGET_HEIGHTS_CM and nothing else.
for _height_cm in TARGET_HEIGHTS_CM:
    gym.register(
        id=f"A2-Jump-{_height_cm}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.jump_env_cfg_jump:RobotEnvCfgJump{_height_cm}",
            "play_env_cfg_entry_point": f"{_CFG}.jump_env_cfg_jump:RobotPlayEnvCfgJump{_height_cm}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )
