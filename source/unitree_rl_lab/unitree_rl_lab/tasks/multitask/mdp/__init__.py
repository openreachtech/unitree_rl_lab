"""MDP terms available to the multi-task environment.

Re-exports every source task family so a single config can mix them: locomotion, the dynamic
acrobatics, and the bipedal stances. The three packages were checked for overlapping public names
(observations, rewards, curriculums, commands) and share none, so the star imports cannot shadow
each other's terms -- only the identical Isaac Lab base terms that all of them already re-export.

The multi-task terms are imported last and by name. They do not replace anything above: the merged
environment keeps the source tasks' reward functions untouched and wraps them with ``gated`` /
``gated_termination`` instead, so the pre-trained critics stay meaningful.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from unitree_rl_lab.tasks.biped.mdp import *  # noqa: F401, F403
from unitree_rl_lab.tasks.dynamic.mdp import *  # noqa: F401, F403
from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .commands import MultiTriggerJumpCommand, MultiTriggerJumpCommandCfg  # noqa: F401
from .curriculums import takeoff_speed_levels  # noqa: F401
from .events import assert_observation_layout  # noqa: F401
from .gating import (  # noqa: F401
    GATE_ACROBATICS,
    GATE_ACROBATICS_STANDING,
    GATE_HANDSTAND,
    GATE_HANDSTAND_UPRIGHT,
    GATE_LOCOMOTION,
    GATE_STANDING,
    gate_mask,
)
from .rewards import gated, gated_termination, resolve_gated_term_params  # noqa: F401
