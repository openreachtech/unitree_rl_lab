"""MDP terms for bipedal stances.

Star-imported by ``..multitask.mdp`` the same way ``dynamic.mdp`` and ``locomotion.mdp`` are, so a
multi-task config reaches these through one ``mdp`` namespace. Nothing here imports ``multitask``:
the dependency runs one way, from the merged environment toward the skills it is built out of.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .handstand import *  # noqa: F401, F403
from .stance_rewards import *  # noqa: F401, F403
