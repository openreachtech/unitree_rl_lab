"""MDP terms for the hind-stance wall crossing task.

Star-imports the bipedal namespace (which itself carries Isaac Lab's), so a config reaches the
inherited terms and these through one ``mdp``, the way ``biped.mdp`` does for its own.
"""

from unitree_rl_lab.tasks.multitask.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
