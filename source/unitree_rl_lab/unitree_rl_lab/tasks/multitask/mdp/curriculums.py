"""Take-off speed curriculum for the merged multi-task environment."""

from __future__ import annotations

import json
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def takeoff_speed_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "jump",
    increase_threshold: float = 0.6,
    decrease_threshold: float = 0.4,
    step: float = 0.1,
    minimum_episodes: int = 1024,
    maximum_speed: float = 3.5,
    state_file: str | None = None,
) -> torch.Tensor:
    """Raise the commanded speed at which an acrobatic move may be triggered, as flips keep landing.

    Neither pre-trained expert has ever taken off with forward speed, and a failed landing at speed
    ends the episode -- so the reward signal is thinnest exactly where the new skill lives. Starting
    with flips restricted to near-stationary environments puts the acrobatics expert back in the
    regime it was trained in, and moves the ground under it only once it is succeeding.

    This also drives the reward definition. The three standing-assumption reward terms are gated on
    commanded speed (see ``..gating.GATE_ACROBATICS_STANDING``), so at a low limit they apply to
    essentially every flip -- making the task identical to ``Go2-Multitask-Jump-Phase2`` -- and
    retire themselves as the limit rises. One knob moves the task and its reward together.

    The ratchet is deliberately two-way. This repository has already paid for a one-way one:
    ``lin_vel_cmd_levels`` records a run where the commanded speed climbed past what the robot could
    do, the exponential tracking reward went numerically flat, the policy collapsed to standing
    still, and 2600 further iterations could not recover because nothing could lower the range
    again. ``decrease_threshold`` gives this one a way back down.

    Args:
        env_ids: Environments that just reset; only their episodes are counted.
        increase_threshold: Flip success rate above which the limit steps up.
        decrease_threshold: Flip success rate below which it steps back down, never below the
            command's configured starting limit.
        step: Size of one step, in m/s.
        minimum_episodes: Completed attempts to accumulate before each decision, so the rate is
            not read off a handful of samples.
        maximum_speed: Ceiling, normally the commanded speed ceiling of the locomotion task.
        state_file: Where to persist the limit. rsl_rl checkpoints hold only network weights, so
            without this every ``--resume`` silently restarts the curriculum from its initial
            limit -- the same trap the jump assist curriculum documents.
    """
    command = env.command_manager.get_term(command_name)

    if not hasattr(command, "takeoff_speed_limit"):
        raise TypeError(
            f"Command {command_name!r} is a {type(command).__name__}, which has no take-off speed"
            " gate. Use MultiTriggerJumpCommand."
        )

    if not hasattr(command, "_takeoff_seen"):
        command._takeoff_seen = 0
        command._takeoff_seen_successes = 0
        command._takeoff_floor = float(command.takeoff_speed_limit)
        if state_file is not None and os.path.isfile(state_file):
            with open(state_file) as f:
                command.takeoff_speed_limit = float(json.load(f)["takeoff_speed_limit"])

    # Read the command's per-attempt tally rather than sampling `success` at reset time. That
    # sample only ever catches attempts still in the air -- `success` cannot turn true before
    # `minimum_landing_time_s` -- so the rate it produced was structurally far below the increase
    # threshold and the limit could never rise off its starting value.
    attempts = command.total_attempts - command._takeoff_seen
    successes = command.total_successes - command._takeoff_seen_successes

    if attempts >= minimum_episodes:
        rate = successes / max(attempts, 1)
        if rate >= increase_threshold:
            command.takeoff_speed_limit = min(command.takeoff_speed_limit + step, maximum_speed)
        elif rate < decrease_threshold:
            command.takeoff_speed_limit = max(command.takeoff_speed_limit - step, command._takeoff_floor)
        command._takeoff_seen = command.total_attempts
        command._takeoff_seen_successes = command.total_successes

        if state_file is not None:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump({"takeoff_speed_limit": command.takeoff_speed_limit}, f)

    return torch.tensor(command.takeoff_speed_limit, device=env.device)
