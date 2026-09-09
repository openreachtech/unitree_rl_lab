# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whether a curriculum may restore its persisted state.

Several curricula persist their progress to a JSON file, because rsl_rl checkpoints save only
network weights: without the file, every ``--resume`` silently restarts the tow-assist decay, the
velocity ratchet, or the jump assist from its initial value. That part works.

The trap is the other direction. A *completed* run leaves its terminal state behind, and a later
**fresh** run of the same task reads it as its own starting point. A curriculum that has finished
decaying reads back as "already finished" -- so the new run begins where the old one ended, with
none of the help the curriculum exists to provide.

That cost two full Go2-Multitask-Gallop retrains (roughly seven hours). Both phases ran with
``tow_assist`` at 0.000 from iteration 0, inherited from the previous lineage's completed state
file. Without the EFGCL tow the robot never reached the speeds where bounding pays off; it settled
into a pronk, and above 2.5 m/s it simply stood still (measured: 0.00 m/s achieved against 2.5,
3.0 and 3.4 m/s commanded, while the previous lineage's policy runs 3.27 m/s under the *same*
physics). The metric it showed up in -- ``Curriculum/tow_assist: 0.000`` -- reads exactly like a
curriculum that has graduated, which is why it survived a review.

So a state file is resume state, never task configuration: it is restored only when this process
was launched to continue the same task's own lineage. ``--previous-task`` does not qualify even
though it implies ``--resume``: it seeds this task from *another* task's weights, which makes it a
new lineage for this task's curricula, and each phase is meant to start its own assist at full
scale (the working lineage did exactly that -- Phase 1 and Phase 2 both from 1.000).

Anything that is not training -- play, and the measurement scripts -- never restores, which is the
safe direction: the value that gets used is then the one written in the task config.
"""

from __future__ import annotations

import os

_RESUME_ENV_VAR = "UNITREE_RL_LAB_CURRICULUM_RESUME"


def mark_curriculum_resume(is_resume: bool) -> None:
    """Declare whether this process continues the current task's own lineage.

    Called by ``train.py`` before the environment is built. Left unset by every other entry point,
    which is why :func:`should_restore` treats "unset" as "do not restore".
    """
    os.environ[_RESUME_ENV_VAR] = "1" if is_resume else "0"


def should_restore(state_file: str | None) -> bool:
    """Whether ``state_file`` should be read back into a curriculum on startup."""
    if state_file is None or not os.path.isfile(state_file):
        return False
    return os.environ.get(_RESUME_ENV_VAR) == "1"
