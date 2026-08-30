"""Go2w-v1-Phase5-Adjust -- a standing "polish" task, not a curriculum phase.

Context (2026-08-29): Go2w-v1-Phase5's own trained checkpoint reliably trembles and
creeps forward while meant to be holding position under a zero command (confirmed on
plain flat-ground standing, not just post-climb goal arrival). Adding
``mdp.wheel_vel_without_cmd_penalty`` (weight -0.001) fixes this -- but only when
applied as a short refinement on top of an already-competent checkpoint (Try31,
~1500 iterations on top of Go2w-v1-Phase5-Try30). Baking the same term into the base
Phase2->Phase5 training recipe itself and training continuously from scratch for
3000 iterations instead produced a slow terrain_levels decline (peak ~5.0 at
iteration 3628, down to 2.8 by the end) that a 3-way ablation
(Go2w-v1-Phase5-Try32/33/34) traced specifically to this combination -- see
velocity_env_cfg_phase5.py's own RewardsCfgPhase5 docstring and sandbox/SUMMARY.md
for the full record.

Given that, this term does not belong in ``RewardsCfgPhase5`` itself. Instead: this
task exists purely to apply it as a targeted, short adjustment pass, every time
Go2w-v1-Phase5's own checkpoint has just been (re)trained and needs its zero-command
stillness polished. Not a phase in the Phase1->Phase5 curriculum sequence -- there is
no "Phase6" here, and no other task resumes from this one. Run for ~1000 iterations
against Go2w-v1-Phase5's own latest checkpoint:

    python scripts/rsl_rl/train.py --task Go2w-v1-Phase5-Adjust --resume \\
        --previous-task Go2w-v1-Phase5 --num_envs 4096 --headless --max_iterations 1000
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase5 import (
    RewardsCfgPhase5,
    RobotEnvCfgPhase5,
    RobotPlayEnvCfgPhase5,
)


@configclass
class RewardsCfgPhase5Adjust(RewardsCfgPhase5):
    """The default Phase5 reward set plus wheel_vel_without_cmd -- see this file's
    module docstring for why this lives here rather than in RewardsCfgPhase5
    itself."""

    wheel_vel_without_cmd = RewTerm(
        func=mdp.wheel_vel_without_cmd_penalty,
        weight=-0.001,
        params={"command_name": "base_velocity"},
    )


@configclass
class RobotEnvCfgPhase5Adjust(RobotEnvCfgPhase5):
    """Default Go2w-v1-Phase5 with only wheel_vel_without_cmd added -- see this
    file's module docstring."""

    rewards: RewardsCfgPhase5Adjust = RewardsCfgPhase5Adjust()


@configclass
class RobotPlayEnvCfgPhase5Adjust(RobotPlayEnvCfgPhase5):
    """Same wall-only, pinned-30/40/50/60cm inspection layout as the default
    Go2w-v1-Phase5 Play view, with wheel_vel_without_cmd added to match."""

    rewards: RewardsCfgPhase5Adjust = RewardsCfgPhase5Adjust()
