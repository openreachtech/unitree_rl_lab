"""Initialise the mixture-of-experts heads from single-task checkpoints.

Expert 0 is loaded from a locomotion run and expert 1 from an acrobatics run; expert 2 and the gates
keep their random initialisation. If a checkpoint was trained on a narrower observation than the
unified one, its input layer is widened here (see :mod:`.weight_surgery`) so old checkpoints can
still be used -- an exactly equivalent network, with the new columns inert until training gives them
weight.

Action noise is carried over too, and that is not a detail. ``rsl_rl`` defaults ``init_noise_std`` to
1.0, while the checkpoints in this repository converged to roughly 0.20 (jump) and 0.59 (speed).
Starting a fine-tune at 1.0 would sample actions with several times the noise these policies operate
under and destroy their behaviour on the first step, which is precisely what initialising from them
was meant to avoid.
"""

from __future__ import annotations

import torch
from rsl_rl.networks import MLP

from ..moe_actor import EXPERT_ACROBATICS, EXPERT_LOCOMOTION, MoEActorCritic
from .weight_surgery import ColumnMap, _first_linear_key, expand_state_dict


def _head_params(
    state_dict: dict[str, torch.Tensor], prefix: str, target_in_dim: int, sources: dict[int, ColumnMap]
) -> tuple[dict[str, torch.Tensor], bool]:
    """Extract one head's MLP parameters, widening the input layer if the checkpoint is narrower.

    Returns the parameters and whether widening was applied.
    """
    source_in_dim = state_dict[_first_linear_key(state_dict, prefix)].shape[1]
    if source_in_dim == target_in_dim:
        return {
            key[len(prefix) + 1 :]: value.clone()
            for key, value in state_dict.items()
            if key.startswith(f"{prefix}.")
        }, False
    mapping = sources.get(source_in_dim)
    if mapping is None:
        raise ValueError(
            f"Checkpoint's {prefix} input width is {source_in_dim}, which matches neither the"
            f" unified width {target_in_dim} nor any known single-task layout"
            f" {sorted(sources)}. Check that the checkpoint belongs to this task family."
        )
    return expand_state_dict(state_dict, target_in_dim, mapping, prefix), True


def _load_expert(expert: MLP, params: dict[str, torch.Tensor], label: str) -> None:
    missing, unexpected = expert.load_state_dict(params, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Expert {label} does not match the checkpoint's architecture."
            f" missing={list(missing)} unexpected={list(unexpected)}."
            " The expert hidden dimensions must equal the ones the checkpoint was trained with."
        )


def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} is not an rsl_rl checkpoint (no 'model_state_dict').")
    return checkpoint["model_state_dict"]


def initialize_experts(
    policy: MoEActorCritic,
    locomotion_checkpoint: str | None = None,
    acrobatics_checkpoint: str | None = None,
    noise_std_mode: str = "min",
    policy_sources: dict[int, ColumnMap] | None = None,
    critic_sources: dict[int, ColumnMap] | None = None,
) -> dict[str, str]:
    """Load pre-trained weights into experts 0 and 1 of both heads, in place.

    Args:
        policy: The mixture-of-experts policy to initialise.
        locomotion_checkpoint: ``model_*.pt`` from the locomotion run, or None to leave expert 0
            randomly initialised.
        acrobatics_checkpoint: ``model_*.pt`` from the acrobatics run, or None to leave expert 1
            randomly initialised.
        policy_sources: Maps a narrower actor width to the column mapping that widens it, for
            checkpoints predating the unified observation. Supplied by the caller because the
            layout belongs to the task, not to the network. Omit to require an exact width.
        critic_sources: The same for the critic.
        noise_std_mode: How to combine the checkpoints' action-noise parameters into the single
            shared one. ``"min"`` (default) takes the elementwise minimum, keeping the sampled
            action closest to what both pre-trained policies were tuned for -- the acrobatics
            skill is the more fragile of the two under added noise. ``"locomotion"`` /
            ``"acrobatics"`` take one checkpoint's value; ``"keep"`` leaves the configured
            ``init_noise_std`` untouched.

    Returns:
        A short report per loaded head, for logging.
    """
    report: dict[str, str] = {}
    noise_stds: dict[str, torch.Tensor] = {}

    for label, path, index in (
        ("locomotion", locomotion_checkpoint, EXPERT_LOCOMOTION),
        ("acrobatics", acrobatics_checkpoint, EXPERT_ACROBATICS),
    ):
        if path is None:
            report[label] = "random init (no checkpoint given)"
            continue

        state_dict = _load_checkpoint(path)
        notes = []
        for head_name, head, sources in (
            ("actor", policy.actor, policy_sources or {}),
            ("critic", policy.critic, critic_sources or {}),
        ):
            target_in_dim = head.experts[index][0].in_features
            params, widened = _head_params(state_dict, head_name, target_in_dim, sources)
            _load_expert(head.experts[index], params, f"{label}/{head_name}")
            notes.append(f"{head_name}{' (widened)' if widened else ''}")

        if "std" in state_dict:
            noise_stds[label] = state_dict["std"].clone()
        elif "log_std" in state_dict:
            noise_stds[label] = state_dict["log_std"].exp()

        report[label] = f"expert {index} <- {path} [{', '.join(notes)}]"

    report["action_noise"] = _apply_noise_std(policy, noise_stds, noise_std_mode)
    return report


def _apply_noise_std(
    policy: MoEActorCritic, noise_stds: dict[str, torch.Tensor], mode: str
) -> str:
    """Set the shared action-noise parameter from the loaded checkpoints."""
    if mode == "keep" or not noise_stds:
        current = policy.std if hasattr(policy, "std") else policy.log_std.exp()
        return f"kept configured init_noise_std (mean {current.mean().item():.3f})"

    if mode == "min":
        combined = torch.stack(list(noise_stds.values())).min(dim=0).values
        source = f"elementwise min of {sorted(noise_stds)}"
    elif mode in noise_stds:
        combined = noise_stds[mode]
        source = mode
    else:
        raise ValueError(
            f"noise_std_mode={mode!r} is not available; loaded checkpoints provide"
            f" {sorted(noise_stds)} (or use 'min' / 'keep')."
        )

    with torch.no_grad():
        if hasattr(policy, "std"):
            policy.std.copy_(combined.to(policy.std.device))
        else:
            policy.log_std.copy_(combined.clamp_min(1e-7).log().to(policy.log_std.device))
    return f"{source} (mean {combined.mean().item():.3f})"
