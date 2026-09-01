"""One Play task per motion, so the assist force can be watched on its own.

The five motions differ only in which hips the pulse lifts and which way the rotation target
points, and that is precisely what is hard to check from aggregate numbers. Watching all five at
once does not work either: the command term picks a motion per environment, so any single run is a
mixture, and a motion whose force is mis-aimed looks like a handful of bad episodes rather than a
broken configuration.

Each task below enables exactly one motion and holds the assist at full strength, so every
environment on screen is performing the same move under the same force. What the run shows is the
*launch mechanics* -- where the robot is pushed, which way it rotates, how high it goes -- not what
the policy has learned; for that, set ``INSPECT_ASSIST_SCALE`` to 0.0 (or pass a task whose play cfg
does, like ``Go2-Multitask-Jump-Phase2``).

These derive from the promoted default (``Go2-Multitask-Jump-Phase2``), not a sandbox copy, so a
change to the real task's assist configuration shows up here without anything being kept in sync.

Checkpoints resolve from ``INSPECT_EXPERIMENT`` rather than from the task's own name, since these
tasks are never trained. Override per run with ``--experiment_name``, or point at a file directly
with ``--checkpoint``.

    python scripts/rsl_rl/play.py --task Go2-Multitask-Jump-Inspect-Handspring --num_envs 16 --real-time
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg_multitask import (
    MultitaskCommandsCfgPhase2,
    RobotPlayEnvCfgMultitaskPhase2,
)

INSPECT_ASSIST_SCALE = 1.0
"""Full assist: the point is to see the force, not the policy. Set to 0.0 to see the policy alone."""

INSPECT_EXPERIMENT = "go2_multitask_jump_phase2"
"""Which run's checkpoints these tasks load. These tasks are never trained, so they have no log
directory of their own; without this, play.py would look under a folder named after the inspect
task and find nothing. Override with ``--experiment_name`` when inspecting a different run."""


@configclass
class InspectPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = INSPECT_EXPERIMENT


def _only(motion: str) -> dict:
    """Enable flags with exactly one motion turned on."""
    names = ("jump", "backflip", "sideflip", "handspring", "sideflip_right")
    assert motion in names, motion
    return {f"enable_{name}": (name == motion) for name in names}


@configclass
class _InspectBase(RobotPlayEnvCfgMultitaskPhase2):
    def __post_init__(self):
        super().__post_init__()
        # The parent play cfg zeroes the assist to show what the policy can do unaided. These tasks
        # want the opposite, so this has to run after it.
        self.commands.jump.initial_assist_scale = INSPECT_ASSIST_SCALE
        self.commands.jump.state_file = None


@configclass
class CommandsCfgJumpOnly(MultitaskCommandsCfgPhase2):
    jump = MultitaskCommandsCfgPhase2().jump.replace(**_only("jump"))


@configclass
class CommandsCfgBackflipOnly(MultitaskCommandsCfgPhase2):
    jump = MultitaskCommandsCfgPhase2().jump.replace(**_only("backflip"))


@configclass
class CommandsCfgSideflipLeftOnly(MultitaskCommandsCfgPhase2):
    jump = MultitaskCommandsCfgPhase2().jump.replace(**_only("sideflip"))


@configclass
class CommandsCfgHandspringOnly(MultitaskCommandsCfgPhase2):
    jump = MultitaskCommandsCfgPhase2().jump.replace(**_only("handspring"))


@configclass
class CommandsCfgSideflipRightOnly(MultitaskCommandsCfgPhase2):
    jump = MultitaskCommandsCfgPhase2().jump.replace(**_only("sideflip_right"))


@configclass
class InspectJump(_InspectBase):
    commands: CommandsCfgJumpOnly = CommandsCfgJumpOnly()


@configclass
class InspectBackflip(_InspectBase):
    commands: CommandsCfgBackflipOnly = CommandsCfgBackflipOnly()


@configclass
class InspectSideflipLeft(_InspectBase):
    commands: CommandsCfgSideflipLeftOnly = CommandsCfgSideflipLeftOnly()


@configclass
class InspectHandspring(_InspectBase):
    commands: CommandsCfgHandspringOnly = CommandsCfgHandspringOnly()


@configclass
class InspectSideflipRight(_InspectBase):
    commands: CommandsCfgSideflipRightOnly = CommandsCfgSideflipRightOnly()


INSPECT_TASKS = {
    "Jump": ("InspectJump", "4 hips lifted evenly, no rotation, target height 0.20 m"),
    "Backflip": ("InspectBackflip", "front hips lifted (FR/FL), pitch -1 turn"),
    "Handspring": ("InspectHandspring", "rear hips lifted (RR/RL), pitch +1 turn"),
    "SideflipLeft": ("InspectSideflipLeft", "right hips lifted (FR/RR), roll -1 turn"),
    "SideflipRight": ("InspectSideflipRight", "left hips lifted (FL/RL), roll +1 turn"),
}
