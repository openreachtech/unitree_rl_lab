"""Widen a single-task checkpoint's input layer onto the unified observation layout.

``Go2-Gallop-Phase2`` and ``Go2-Jump-Phase2`` were trained on 117- and 47-column observations
respectively; the multi-task environment serves a 122-column superset. A linear layer's output is
``y = sum_j W[:, j] * x[j] + b``, so a column of ``W`` that is zero contributes nothing regardless
of what arrives on that input. Placing the original weights at the columns their inputs moved to and
zeroing the rest therefore yields a network that computes **exactly** the original function -- no
approximation, no retraining. Only the first layer changes; every later layer and the action noise
parameter are copied verbatim.

This is used to bootstrap the single-task re-training runs on the unified observation (the widened
columns start inert and the run makes them useful), not to initialise the mixture-of-experts
directly -- by then the experts are already the right width.
"""

from __future__ import annotations

import re
import torch

ColumnMap = list[tuple[int, int, int]]
"""``(source_start, source_end, target_start)`` contiguous copy instructions."""


def expand_linear_input(
    weight: torch.Tensor, target_in_dim: int, mapping: ColumnMap
) -> torch.Tensor:
    """Widen a linear layer's weight matrix from ``weight.shape[1]`` to ``target_in_dim`` columns.

    Args:
        weight: Original weight of shape ``(out_features, source_in_dim)``.
        target_in_dim: Input width of the widened layer.
        mapping: Column mapping, e.g. from :func:`..obs_spec.source_to_unified_map`.

    Returns:
        The widened weight. Columns not named by ``mapping`` are zero.
    """
    source_in_dim = weight.shape[1]
    covered = sum(end - start for start, end, _ in mapping)
    if covered != source_in_dim:
        raise ValueError(
            f"Mapping covers {covered} source columns but the weight has {source_in_dim}. Every"
            " original column must be placed, otherwise the widened layer is not equivalent."
        )

    expanded = torch.zeros(weight.shape[0], target_in_dim, dtype=weight.dtype, device=weight.device)
    for source_start, source_end, target_start in mapping:
        width = source_end - source_start
        target_end = target_start + width
        if target_end > target_in_dim:
            raise ValueError(
                f"Mapping writes columns [{target_start}, {target_end}) past the target width"
                f" {target_in_dim}."
            )
        destination = expanded[:, target_start:target_end]
        if destination.any():
            raise ValueError(
                f"Mapping writes columns [{target_start}, {target_end}) twice; overlapping"
                " destinations would silently drop weights."
            )
        destination.copy_(weight[:, source_start:source_end])
    return expanded


def _first_linear_key(state_dict: dict[str, torch.Tensor], prefix: str) -> str:
    """Find the input layer's weight key under ``prefix`` (``actor`` or ``critic``).

    ``rsl_rl``'s ``MLP`` is an ``nn.Sequential``, so its layers are numbered and the input layer is
    the lowest-numbered ``Linear``.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for key in state_dict if (m := pattern.match(key)))
    if not indices:
        raise KeyError(f"No '{prefix}.<n>.weight' entries found in the checkpoint.")
    return f"{prefix}.{indices[0]}.weight"


def expand_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_in_dim: int,
    mapping: ColumnMap,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Return the parameters of ``prefix``'s MLP with its input layer widened.

    Keys are returned relative to the MLP itself (``"0.weight"``, ``"0.bias"``, ...) so the result
    can be loaded straight into an :class:`rsl_rl.networks.MLP`.
    """
    first_key = _first_linear_key(state_dict, prefix)
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(f"{prefix}."):
            continue
        local_key = key[len(prefix) + 1 :]
        out[local_key] = (
            expand_linear_input(value, target_in_dim, mapping) if key == first_key else value.clone()
        )
    if not out:
        raise KeyError(f"Checkpoint contains no parameters under prefix {prefix!r}.")
    return out


def scatter_observation(
    source_obs: torch.Tensor, target_dim: int, mapping: ColumnMap
) -> torch.Tensor:
    """Place a source-layout observation into a unified-layout vector, zero-filling the rest.

    The counterpart to :func:`expand_state_dict`, and the only way to check that a widening was
    exact: feed the original network ``source_obs`` and the widened network
    ``scatter_observation(source_obs, ...)`` and the outputs must be bit-identical. That check earns
    its keep because the failure it catches is silent -- putting a block in the wrong history slot
    produces a network that acts on stale observations and raises nothing.
    """
    scattered = torch.zeros(
        *source_obs.shape[:-1], target_dim, dtype=source_obs.dtype, device=source_obs.device
    )
    for source_start, source_end, target_start in mapping:
        width = source_end - source_start
        scattered[..., target_start : target_start + width] = source_obs[..., source_start:source_end]
    return scattered
