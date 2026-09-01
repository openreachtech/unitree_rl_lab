from __future__ import annotations

import json
import os
from collections.abc import Sequence

import torch


def _save_curriculum_state(command) -> None:
    """Persist assist-force curriculum state so it survives a training process restart."""
    if command.cfg.state_file is None:
        return
    state = {
        "assist_scale": command.assist_scale_by_motion.tolist(),
        "curriculum_success_rate": command.curriculum_success_rate,
        "curriculum_episode_count_by_motion": command.curriculum_episode_count_by_motion.tolist(),
        "curriculum_success_count_by_motion": command.curriculum_success_count_by_motion.tolist(),
    }
    os.makedirs(os.path.dirname(command.cfg.state_file), exist_ok=True)
    tmp_path = command.cfg.state_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, command.cfg.state_file)


def _largest_remaining_crutch(command) -> torch.Tensor:
    """The largest assist scale still in use, across the motions this task has enabled.

    Over *enabled* motions specifically. ``assist_scale_by_motion`` is indexed by motion code, so it
    carries a slot for code 0 -- "no motion" -- plus a slot for every motion the command class knows
    about, whether or not this task turned it on. Those slots never decay, because nothing is ever
    assigned to them, so a plain ``.max()`` returns the initial 1.0 for the life of the run. Measured
    during the promotion run (see the sandbox SUMMARY.md): every one of the five motions decayed
    to 0.38-0.60 while
    ``Curriculum/assist_force`` sat at exactly 1.000 throughout -- the same flat line that, one run
    earlier, meant the decay really was deadlocked.
    """
    enabled = [
        code
        for code, metric_name in command._motion_metric_names.items()
        if getattr(command.cfg, "enable_" + metric_name.removeprefix("success_"), False)
    ]
    if not enabled:
        return command.assist_scale_by_motion.max()
    return command.assist_scale_by_motion[enabled].max()


def assist_force_decay(
    env,
    env_ids: Sequence[int],
    command_name: str = "jump",
    success_threshold: float = 0.60,
    decay_step: float = 0.01,
    minimum_episodes: int = 1024,
) -> torch.Tensor:
    """Decay each motion's assistance once that motion, on its own, lands reliably.

    Every decision here is per motion. The previous version kept one scale for all of them and
    stepped it only when *every* enabled motion cleared ``success_threshold`` -- which turned a
    single failing motion into a deadlock for the whole task, because full assist means the policy
    never has to contribute, so the motion that is failing has no pressure to improve and the gate
    it is blocking never opens. The five-motion expert sat in exactly that state for a full
    run: jump 0.97, backflip 1.00, sideflip 0.99, handspring 1.00, right sideflip 0.000, and
    ``assist_scale`` pinned at 1.000 throughout. Nothing in the aggregate showed it -- ``success``
    read 0.804 -- and in Play, where the assist is switched off, the policy could not jump at all,
    because across the entire run the external force had done every bit of the work.

    The episode-count requirement is per motion for the same reason. Requiring all of them to reach
    ``minimum_episodes`` before any decision let a rarely-sampled motion hold back the rest even
    when nothing was wrong with them.
    """
    command = env.command_manager.get_term(command_name)
    if len(env_ids) == 0:
        return _largest_remaining_crutch(command)

    motion_codes = command.motion_code[env_ids]
    # Length comes from the command's own motion table. It was hardcoded to 4, which stayed correct
    # only until motions were added -- and then failed as a tensor-shape error inside this function,
    # nowhere near the file that changed.
    slots = command.curriculum_episode_count_by_motion.numel()
    episode_counts = torch.bincount(motion_codes, minlength=slots)
    success_counts = torch.bincount(
        motion_codes,
        weights=command.success[env_ids].float(),
        minlength=slots,
    ).long()
    command.curriculum_episode_count_by_motion += episode_counts
    command.curriculum_success_count_by_motion += success_counts

    # Derived from the command's motion table and the matching enable flag, so a new motion is
    # picked up here without this list having to be edited too. Listing them by hand is what left
    # the two newest motions ungated: the decay would have advanced on the other three alone,
    # declaring success while two motions were still being assisted.
    enabled_motion_codes = [
        code
        for code, metric_name in command._motion_metric_names.items()
        if getattr(command.cfg, "enable_" + metric_name.removeprefix("success_"), False)
    ]

    decided = False
    rates: list[float] = []
    for code in enabled_motion_codes:
        episodes = command.curriculum_episode_count_by_motion[code]
        if episodes < minimum_episodes:
            continue
        rate = (command.curriculum_success_count_by_motion[code] / episodes).item()
        rates.append(rate)
        if rate >= success_threshold:
            command.assist_scale_by_motion[code] = max(
                0.0, command.assist_scale_by_motion[code].item() - decay_step
            )
        # Counters reset only for the motion that was judged, so the others keep accumulating
        # toward their own decision rather than being cleared underneath them.
        command.curriculum_episode_count_by_motion[code] = 0
        command.curriculum_success_count_by_motion[code] = 0
        decided = True

    if decided:
        # Reported as the worst motion, which is what the whole task is gated on in practice.
        command.curriculum_success_rate = min(rates)
        _save_curriculum_state(command)

    return _largest_remaining_crutch(command)
