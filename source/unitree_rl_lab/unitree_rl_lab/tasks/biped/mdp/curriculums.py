"""Curriculum terms for the biped tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def assist_force_decay(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    success_threshold: float = 0.60,
    decay_step: float = 0.01,
    minimum_episodes: int = 1024,
    front_contact_fraction_threshold: float = 0.10,
) -> torch.Tensor:
    """Decay a :class:`~unitree_rl_lab.tasks.biped.mdp.commands.ForwardAssistVelocityCommand`'s
    EFGCL assist force once a sufficiently large batch of recently-finished episodes
    clears ``success_threshold``. Success requires *both* reaching ``time_out``
    (didn't fall) *and* keeping front-foot ground contact below
    ``front_contact_fraction_threshold`` for that episode -- "didn't fall" alone lets
    the robot pass by leaning on its front leg, never exercising the behavior the
    assist force's up-component exists to fix. Mirrors `feat/jump`'s
    ``assist_force_decay`` curriculum term (same success-rate-gated decay
    mechanism, applied to a continuous walking skill instead of a one-shot motion).
    """
    command = env.command_manager.get_term(command_name)
    if len(env_ids) == 0:
        return torch.tensor(command.assist_scale, device=env.device)

    survived = env.termination_manager.time_outs[env_ids]
    clean_gait = command.front_contact_fraction[env_ids] < front_contact_fraction_threshold
    successes = survived & clean_gait
    command.curriculum_episode_count += len(env_ids)
    command.curriculum_success_count += int(successes.sum().item())

    if command.curriculum_episode_count >= minimum_episodes:
        success_rate = command.curriculum_success_count / command.curriculum_episode_count
        command.curriculum_success_rate = success_rate
        if success_rate >= success_threshold:
            command.assist_scale = max(0.0, command.assist_scale - decay_step)
        command.curriculum_episode_count = 0
        command.curriculum_success_count = 0
        command.save_curriculum_state()

    return torch.tensor(command.assist_scale, device=env.device)
