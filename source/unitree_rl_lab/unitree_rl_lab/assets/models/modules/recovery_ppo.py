"""MoEPPO with gate statistics split by trunk attitude, for the walk / fall / get-up mixture."""

from __future__ import annotations

import torch

from ..recovery_moe import RECOVERY_EXPERT_NAMES
from .moe_ppo import MoEPPO


class RecoveryMoEPPO(MoEPPO):
    """:class:`.MoEPPO` whose gate statistics describe attitude rather than a command flag.

    The parent splits routing weight on the jump command's ``enabled`` column, which this mixture
    does not have. Here the question the logs have to answer is "does the locomotion expert drive
    while upright, and does the get-up expert take over once the robot is down", so the split is on
    projected gravity's z: upright (< -0.8), tilted (> -0.5), inverted (> 0).
    """

    @torch.no_grad()
    def _gate_statistics(self, stride: int = 8) -> dict[str, float]:
        storage = getattr(self, "storage", None)
        if storage is None or getattr(storage, "observations", None) is None:
            return {}
        gating = getattr(self.policy.actor, "gating", None)
        if gating is None or self.gravity_z_index is None:
            return {}

        obs = self.policy.get_actor_obs(storage.observations)
        flat = obs.reshape(-1, obs.shape[-1])[::stride]
        weights = gating(flat)
        gravity_z = flat[:, self.gravity_z_index]
        conditions = {"upright": gravity_z < -0.8, "tilted": gravity_z > -0.5, "inverted": gravity_z > 0.0}

        stats: dict[str, float] = {}
        for label, mask in conditions.items():
            stats[f"gate/fraction_{label}"] = mask.float().mean().item()
        for index in range(weights.shape[-1]):
            label = RECOVERY_EXPERT_NAMES[index] if index < len(RECOVERY_EXPERT_NAMES) else f"expert{index}"
            stats[f"gate/{label}"] = weights[:, index].mean().item()
            for suffix, mask in conditions.items():
                if mask.any():
                    stats[f"gate/{label}_{suffix}"] = weights[mask, index].mean().item()
        return stats
