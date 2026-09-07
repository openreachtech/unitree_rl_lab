"""MDP terms for bipedal stances.

Star-imported by ``..multitask.mdp`` the same way ``dynamic.mdp`` and ``locomotion.mdp`` are, so a
multi-task config reaches these through one ``mdp`` namespace. Nothing here imports ``multitask``:
the dependency runs one way, from the merged environment toward the skills it is built out of.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculums import handstand_takeoff_speed_levels  # noqa: F401
from .handstand import *  # noqa: F401, F403
from .rewards import stance_aware  # noqa: F401
from .symmetry import mirror_left_right  # noqa: F401
from .stance_rewards import *  # noqa: F401, F403
