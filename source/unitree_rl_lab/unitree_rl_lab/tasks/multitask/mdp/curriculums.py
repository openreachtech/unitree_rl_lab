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
    velocity_command_name: str = "base_velocity",
    increase_threshold: float = 0.6,
    decrease_threshold: float = 0.4,
    max_velocity_error: float | None = None,
    step: float = 0.1,
    minimum_attempts: int = 1024,
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
        env_ids: Environments that just reset.
        velocity_command_name: Unused by the error measurement -- the jump command supplies that --
            but kept so the take-off gate and the curriculum name the same velocity term.
        increase_threshold: Flip success rate above which the limit steps up.
        decrease_threshold: Flip success rate below which it steps back down, never below the
            command's configured starting limit.
        max_velocity_error: Velocity-tracking error above which the limit may not rise, and below
            which it must fall. ``None`` (the default) judges on flip success alone.

            Success alone is not enough to promote on, and this environment showed why: the limit
            climbed 0.3 -> 3.5 m/s in 225 iterations while tracking error went 0.40 -> 1.49, against
            0.30-0.40 for the locomotion expert by itself. Flips kept landing, so the curriculum
            kept promoting, and the policy settled on buying flip success by giving up running.

            Measured over the right window, though: the error comes from
            ``MultiTriggerJumpCommand.locomotion_error``, which ignores the steps spent mid-flip.
            Judging on the unrestricted error instead makes the gate charge for flipping itself, and
            the limit oscillates 0.3 -> 0.5 -> 0.3 without ever settling.
        step: Size of one step, in m/s.
        minimum_attempts: Completed attempts to accumulate before each decision.

            Sized in *attempts*, not episodes, and the distinction is easy to get wrong: this was
            copied from ``assist_force_decay``, where one episode is one attempt and 1024 of them
            take hundreds of iterations. Here each 20 s episode contains three to five flips across
            thousands of environments, so 1024 attempts accumulated in about seven iterations --
            32 promotions in 225. The limit reached its ceiling long before the policy could
            consolidate anything at the speeds it passed through.
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
        command._takeoff_error_sum = 0.0
        command._takeoff_error_count = 0
        if state_file is not None and os.path.isfile(state_file):
            with open(state_file) as f:
                command.takeoff_speed_limit = float(json.load(f)["takeoff_speed_limit"])

    # Locomotion quality over the same window as the flip statistics, so both describe the same
    # stretch of training. error_vel_xy is accumulated per episode by the velocity command term, so
    # sampling it at reset gives one figure per finished episode.
    if max_velocity_error is not None and len(env_ids) > 0:
        # The jump command's own figure, not the velocity command's error_vel_xy. That one
        # accumulates on every step including mid-flip, where the robot cannot follow a ground
        # velocity command by construction -- so it charged the gate for flipping, and raising the
        # limit (more flips -> more error) pulled the limit straight back down. See
        # MultiTriggerJumpCommand.locomotion_error.
        command._takeoff_error_sum += float(command.locomotion_error[env_ids].sum().item())
        command._takeoff_error_count += len(env_ids)

    # Read the command's per-attempt tally rather than sampling `success` at reset time. That
    # sample only ever catches attempts still in the air -- `success` cannot turn true before
    # `minimum_landing_time_s` -- so the rate it produced was structurally far below the increase
    # threshold and the limit could never rise off its starting value.
    attempts = command.total_attempts - command._takeoff_seen
    successes = command.total_successes - command._takeoff_seen_successes

    if attempts >= minimum_attempts:
        rate = successes / max(attempts, 1)
        error = (
            command._takeoff_error_sum / max(command._takeoff_error_count, 1)
            if max_velocity_error is not None
            else 0.0
        )
        running_well = max_velocity_error is None or error <= max_velocity_error

        if rate >= increase_threshold and running_well:
            command.takeoff_speed_limit = min(command.takeoff_speed_limit + step, maximum_speed)
        elif rate < decrease_threshold or not running_well:
            command.takeoff_speed_limit = max(command.takeoff_speed_limit - step, command._takeoff_floor)

        command._takeoff_seen = command.total_attempts
        command._takeoff_seen_successes = command.total_successes
        command._takeoff_error_sum = 0.0
        command._takeoff_error_count = 0

        if state_file is not None:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump({"takeoff_speed_limit": command.takeoff_speed_limit}, f)

    return torch.tensor(command.takeoff_speed_limit, device=env.device)
