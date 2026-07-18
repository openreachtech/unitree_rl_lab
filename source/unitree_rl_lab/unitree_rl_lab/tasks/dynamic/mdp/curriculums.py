from __future__ import annotations

from collections.abc import Sequence

import torch


def assist_force_decay(
    env,
    env_ids: Sequence[int],
    command_name: str = "jump",
    success_threshold: float = 0.60,
    decay_step: float = 0.01,
    minimum_episodes: int = 1024,
) -> torch.Tensor:
    """Decay assistance after a sufficiently large successful episode batch."""
    command = env.command_manager.get_term(command_name)
    if len(env_ids) == 0:
        return torch.tensor(command.assist_scale, device=env.device)

    motion_codes = command.motion_code[env_ids]
    episode_counts = torch.bincount(motion_codes, minlength=4)
    success_counts = torch.bincount(
        motion_codes,
        weights=command.success[env_ids].float(),
        minlength=4,
    ).long()
    command.curriculum_episode_count_by_motion += episode_counts
    command.curriculum_success_count_by_motion += success_counts

    enabled_motion_codes = []
    if command.cfg.enable_jump:
        enabled_motion_codes.append(command.MOTION_JUMP)
    if command.cfg.enable_backflip:
        enabled_motion_codes.append(command.MOTION_BACKFLIP)
    if command.cfg.enable_sideflip:
        enabled_motion_codes.append(command.MOTION_SIDEFLIP)

    enough_episodes = all(
        command.curriculum_episode_count_by_motion[code] >= minimum_episodes
        for code in enabled_motion_codes
    )
    if enough_episodes:
        success_rates = [
            (
                command.curriculum_success_count_by_motion[code]
                / command.curriculum_episode_count_by_motion[code]
            ).item()
            for code in enabled_motion_codes
        ]
        command.curriculum_success_rate = min(success_rates)
        if all(rate >= success_threshold for rate in success_rates):
            command.assist_scale = max(0.0, command.assist_scale - decay_step)
        command.curriculum_episode_count_by_motion.zero_()
        command.curriculum_success_count_by_motion.zero_()

    return torch.tensor(command.assist_scale, device=env.device)
