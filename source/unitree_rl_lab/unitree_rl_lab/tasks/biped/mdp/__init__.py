"""MDP terms for bipedal stances.

The handstand command, the stance rewards and the upright gate are ported from
``feat/multi-task`` (``tasks/biped/mdp``); the locomotion terms come along for the velocity command,
``energy`` and the generic penalties.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .gating import resolve_nested_term_params, stance_aware, when_stance, when_upright  # noqa: F401
from .handstand import *  # noqa: F401, F403
from .stance_rewards import *  # noqa: F401, F403
