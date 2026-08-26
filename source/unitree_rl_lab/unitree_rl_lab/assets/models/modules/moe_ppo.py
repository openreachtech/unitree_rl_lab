"""PPO with per-parameter-group learning-rate scales, for fine-tuning a mixture of experts.

Fine-tuning pre-trained experts needs a gentler step than training the randomly initialised
transition expert and the gate. PyTorch supports that natively through parameter groups, but
``rsl_rl``'s adaptive KL schedule overwrites every group's ``lr`` on each update::

    for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate      # rsl_rl/algorithms/ppo.py

so any per-group value set up front is erased before it is ever used. Rather than fork ``update()``
-- a long method that would then have to be kept in sync with upstream -- the scaling is applied
inside the optimizer's ``step()``, where it cannot be clobbered. The adaptive schedule keeps writing
the shared base rate and each group multiplies by its own factor on the way through.

Note that scaling *gradients* instead would not work here: Adam normalises by the gradient's own
running magnitude, so a uniformly scaled gradient produces the same step. The scale has to be on the
learning rate.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from typing import Any

DEFAULT_LR_SCALES: dict[str, float] = {
    # Pre-trained experts: keep their specialisation instead of overwriting it.
    "actor_pretrained": 0.1,
    "critic_pretrained": 0.1,
    # Randomly initialised transition expert and the gates: nothing to preserve.
    "actor_new": 1.0,
    "critic_new": 1.0,
    "actor_gating": 1.0,
    "critic_gating": 1.0,
    # Action noise parameter and anything else not in a head.
    "other": 1.0,
}


class ScaledAdam(torch.optim.Adam):
    """Adam that multiplies each group's learning rate by that group's ``lr_scale`` at step time.

    The scale is applied and reverted around :meth:`step`, so an outside caller that writes ``lr``
    (the adaptive KL schedule) sees exactly the value it wrote and its bookkeeping stays intact.
    """

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        saved = [group["lr"] for group in self.param_groups]
        for group in self.param_groups:
            group["lr"] = group["lr"] * group.get("lr_scale", 1.0)
        try:
            return super().step(closure)
        finally:
            for group, lr in zip(self.param_groups, saved):
                group["lr"] = lr


class MoEPPO(PPO):
    """PPO that gives each mixture-of-experts parameter group its own learning-rate scale.

    Args:
        lr_scales: Per-group multipliers, keyed by the names returned by
            ``MoEActorCritic.parameter_groups()``. Missing groups default to
            :data:`DEFAULT_LR_SCALES`.
        actor_warmup_iterations: Number of initial updates during which every ``actor_*`` group is
            held at scale 0. The value function is initialised from critics trained on different
            reward functions and has never seen the transition states at all, so its first
            advantage estimates are noise -- and applying that noise to a good actor at the
            adaptive schedule's starting rate undoes the expert initialisation. Freezing the actor
            briefly lets the critic catch up first. Costs nothing beyond the scale mechanism that
            already exists here.
    """

    def __init__(
        self,
        policy,
        lr_scales: dict[str, float] | None = None,
        actor_warmup_iterations: int = 0,
        gravity_z_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)

        if not hasattr(policy, "parameter_groups"):
            raise TypeError(
                "MoEPPO requires a policy exposing parameter_groups(); got"
                f" {type(policy).__name__}. Use MoEActorCritic, or plain PPO."
            )

        self.lr_scales = {**DEFAULT_LR_SCALES, **(lr_scales or {})}
        self.actor_warmup_iterations = int(actor_warmup_iterations)
        self.gravity_z_index = gravity_z_index
        self._iteration = 0

        named_groups = policy.parameter_groups()
        unknown = set(named_groups) - set(self.lr_scales)
        if unknown:
            raise ValueError(f"No learning-rate scale configured for parameter groups: {sorted(unknown)}")

        # Empty groups are dropped: torch rejects a param group with no parameters.
        self._group_names = [name for name, params in named_groups.items() if params]
        self.optimizer = ScaledAdam(
            [
                {"params": named_groups[name], "lr_scale": self.lr_scales[name], "name": name}
                for name in self._group_names
            ],
            lr=self.learning_rate,
        )
        self._apply_lr_scales()

    def _apply_lr_scales(self) -> None:
        """Refresh each group's scale for the current iteration."""
        warming_up = self._iteration < self.actor_warmup_iterations
        for group in self.optimizer.param_groups:
            name = group["name"]
            scale = self.lr_scales[name]
            group["lr_scale"] = 0.0 if (warming_up and name.startswith("actor_")) else scale

    @torch.no_grad()
    def _gate_statistics(self, stride: int = 8) -> dict[str, float]:
        """Mean routing weight per expert, split by whether a move is commanded.

        The gate decides which expert drives the robot, and nothing else in the logs reveals it --
        a policy whose locomotion has quietly degraded looks identical to one whose gate has
        started routing running states to the wrong expert. Splitting on the jump command's
        ``enabled`` flag gives the two numbers that matter: who drives while running, and who
        drives at a take-off.

        Reads the rollout before ``PPO.update`` clears it, and subsamples by ``stride`` because a
        mean over a twelfth of ~98k samples is already far more precise than the quantity needs.
        """
        storage = getattr(self, "storage", None)
        if storage is None or getattr(storage, "observations", None) is None:
            return {}
        gating = getattr(self.policy.actor, "gating", None)
        if gating is None:
            return {}

        obs = self.policy.get_actor_obs(storage.observations)
        flat = obs.reshape(-1, obs.shape[-1])[::stride]
        weights = gating(flat)
        enabled = flat[:, gating.prior_index] > 0.5

        # Splitting on the command flag only covers ``command_duration_s`` (0.5 s), while a flip
        # runs ~1.2 s -- so who drives during flight and landing, the part where a flip is actually
        # lost, went unmeasured. Projected gravity answers it straight from the observation: its z
        # component is about -1 upright and turns positive once the trunk passes horizontal. No
        # observation change needed, and it reads the physical state rather than a timer.
        conditions = {"commanded": enabled, "idle": ~enabled}
        if self.gravity_z_index is not None:
            gravity_z = flat[:, self.gravity_z_index]
            conditions["tilted"] = gravity_z > -0.5
            conditions["inverted"] = gravity_z > 0.0

        names = ["locomotion", "acrobatics", "transition"]
        stats: dict[str, float] = {"gate/commanded_fraction": enabled.float().mean().item()}
        for label, mask in conditions.items():
            stats[f"gate/fraction_{label}"] = mask.float().mean().item()
        for index in range(weights.shape[-1]):
            label = names[index] if index < len(names) else f"expert{index}"
            stats[f"gate/{label}"] = weights[:, index].mean().item()
            for suffix, mask in conditions.items():
                if mask.any():
                    stats[f"gate/{label}_{suffix}"] = weights[mask, index].mean().item()
        return stats

    def update(self) -> dict[str, float]:
        self._apply_lr_scales()
        gate_stats = self._gate_statistics()
        loss_dict = super().update()
        loss_dict.update(gate_stats)
        self._iteration += 1
        # Surfaces the warm-up boundary and the effective rates in TensorBoard, so a run that
        # behaves oddly early on can be checked against the schedule rather than guessed at.
        for group in self.optimizer.param_groups:
            loss_dict[f"lr_{group['name']}"] = self.learning_rate * group["lr_scale"]
        return loss_dict
