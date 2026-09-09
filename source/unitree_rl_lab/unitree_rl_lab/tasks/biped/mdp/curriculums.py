"""Take-off speed curriculum for the bipedal stance."""

from __future__ import annotations

import json
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from unitree_rl_lab.utils.curriculum_state import should_restore

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def handstand_takeoff_speed_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "handstand",
    jump_command_name: str = "jump",
    increase_threshold: float = 0.6,
    decrease_threshold: float = 0.4,
    max_velocity_error: float | None = 0.6,
    step: float = 0.1,
    minimum_attempts: int = 1024,
    maximum_speed: float = 1.0,
    state_file: str | None = None,
) -> torch.Tensor:
    """Raise the commanded speed at which a bipedal stance may be entered, as rises keep landing.

    The same shape as the acrobatics take-off curriculum, for the same reason. Neither pre-trained
    skill has ever started from a moving robot: the bipedal expert's episodes all begin at rest and
    rise from there, so a stance commanded at 1 m/s in the merged environment is a state it has
    never seen. Starting with the stance restricted to near-stationary environments puts it back in
    the regime it was trained in and moves the ground under it only once it is succeeding.

    Two-way. This repository has already paid for a one-way ratchet: ``lin_vel_cmd_levels`` records
    a run where the commanded speed climbed past what the robot could do, the tracking reward went
    numerically flat, the policy collapsed to standing still, and 2600 further iterations could not
    recover because nothing could lower the range again.

    Args:
        increase_threshold: Fraction of windows in which the stance was reached, above which the
            limit steps up.
        decrease_threshold: Fraction below which it steps back down, never past the configured
            starting limit.
        max_velocity_error: Velocity-tracking error above which the limit may not rise. Read from
            the acrobatics command's ``locomotion_error``, which excludes both the steps spent
            mid-flip *and* the steps spent on two legs -- neither can follow a ground velocity
            command by construction, and charging this gate for them makes the limit fall every
            time the other skill fires.
        minimum_attempts: Completed windows to accumulate before each decision. A stance fires at
            most once per episode and only in a share of them, so these accumulate roughly twenty
            times slower than the acrobatics' -- which is the point. That curriculum's own note
            records 32 promotions in 225 iterations from deciding too often.
        maximum_speed: Ceiling. Kept at the acrobatics' own ceiling so the two skills are offered
            over the same range of commanded speeds.
        state_file: Where to persist the limit. rsl_rl checkpoints hold only network weights, so a
            resume without this restarts the curriculum at its initial limit.
    """
    command = env.command_manager.get_term(command_name)

    if not hasattr(command, "_takeoff_seen"):
        command._takeoff_seen = 0
        command._takeoff_seen_successes = 0
        command._takeoff_floor = float(command.takeoff_speed_limit)
        if should_restore(state_file):
            with open(state_file) as f:
                command.takeoff_speed_limit = float(json.load(f)["takeoff_speed_limit"])

    attempts = command.attempts - command._takeoff_seen
    successes = command.successes - command._takeoff_seen_successes
    if attempts < minimum_attempts:
        return torch.tensor(command.takeoff_speed_limit, device=env.device)

    rate = successes / max(attempts, 1)
    running_well = True
    if max_velocity_error is not None:
        try:
            error = float(env.command_manager.get_term(jump_command_name).locomotion_error.mean())
            running_well = error <= max_velocity_error
        except (KeyError, ValueError, AttributeError):
            running_well = True

    if rate >= increase_threshold and running_well:
        command.takeoff_speed_limit = min(command.takeoff_speed_limit + step, maximum_speed)
    elif rate < decrease_threshold or not running_well:
        command.takeoff_speed_limit = max(command.takeoff_speed_limit - step, command._takeoff_floor)

    command._takeoff_seen = command.attempts
    command._takeoff_seen_successes = command.successes

    if state_file is not None:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({"takeoff_speed_limit": command.takeoff_speed_limit}, f)

    return torch.tensor(command.takeoff_speed_limit, device=env.device)
