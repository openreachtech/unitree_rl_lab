"""Mixture-of-experts actor-critic for the Go2 multi-task policy.

Three experts are blended by a soft gate: expert 0 is initialised from the locomotion policy,
expert 1 from the acrobatics policy, and expert 2 starts random and exists to absorb the transitions
(run -> take-off, landing -> run) that neither pre-trained policy has ever visited. Every expert runs
on every step and their outputs are averaged by the gate weights, so the action stays continuous
across a transition instead of jumping the way a hard switch would.

Both the actor and the critic use this structure. The critic is never exported (Isaac Lab's exporter
only walks ``policy.actor``), so its three experts cost nothing at deployment and buy the ability to
initialise the value function from the pre-trained critics instead of from noise.

The gate's command prior
------------------------
A randomly initialised gate outputs near-equal weights, which at step zero would hand the
environment the *average* of a walking action, an acrobatic action, and noise -- destroying both
pre-trained skills before training has produced anything. "Which expert handles a commanded flip" is
not a fact worth discovering by gradient descent, so it is written directly into the gate as a fixed
additive prior on the logits, keyed off the jump command's ``enabled`` flag. At initialisation the
gate's own MLP contributes exactly zero (its last layer is zero-initialised), so routing starts
essentially hard and each expert sees the action distribution it was trained on. The MLP output is
*added*, so training can override the prior and learn genuine blends where they help.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict
from typing import Any

EXPERT_LOCOMOTION = 0
EXPERT_ACROBATICS = 1
EXPERT_BIPED = 2

_PRETRAINED = (EXPERT_LOCOMOTION, EXPERT_ACROBATICS, EXPERT_BIPED)
"""Expert slots initialised from a trained policy, and so fine-tuned rather than learned."""
"""Slot 2 held a transition expert that started from random weights and was meant to earn its place
from the gate. Measured on the finished two-expert policy it never did: its routing weight read
0.000 while running, 0.000 inside an acrobatic window, and 0.001 in the hand-back bin where it had
the most to contribute. The bipedal policy takes the slot instead."""

NUM_EXPERTS = 3


class Gating(nn.Module):
    """Soft gate over the experts, biased at initialisation by the commanded motion.

    Args:
        obs_dim: Width of the observation this gate reads.
        num_experts: Number of experts to weight.
        hidden_dims: Hidden layer sizes of the gate's MLP.
        activation: Activation of the gate's MLP.
        prior_index: Column of the 0/1 "an acrobatic move is commanded" flag.
        biped_prior_index: Column of the 0/1 "a bipedal stance is commanded" flag. ``None`` leaves
            the bipedal expert unfavoured, which is the two-flag gate reduced to the one-flag one.
        prior_scale: Logit offset applied by the prior. With three experts, 5.0 puts 98.7% of
            the weight on the prior's expert at initialisation (``e^5 / (e^5 + 2)``); the 1.3%
            leaking to the others perturbs the action by well under a milliradian, since a freshly
            initialised expert's output is small. 0.0 disables the prior entirely.
    """

    def __init__(
        self,
        obs_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        prior_index: int,
        prior_scale: float,
        biped_prior_index: int | None = None,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, num_experts, hidden_dims, activation)
        # Zero last layer => the gate contributes nothing at initialisation and the prior decides
        # routing alone. Gradients still flow, so this is a starting point and not a constraint.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.prior_index = int(prior_index)
        self.biped_prior_index = -1 if biped_prior_index is None else int(biped_prior_index)
        self.prior_scale = float(prior_scale)

        # One logit row per commanded regime. Nothing commanded routes to locomotion, an acrobatic
        # move to the acrobatics expert, a bipedal stance to the bipedal one. The environment keeps
        # the two flags mutually exclusive, so no row ever has to describe both at once.
        #
        # The prior exists because the mixture cannot bootstrap without it: at a uniform 1/3 each,
        # the blend of a walking action and an acrobatic one leaves the ground in neither, so no
        # attempt succeeds, so nothing ever rewards the gate for routing correctly. A 2000-iteration
        # run with the prior off measured a jump height of exactly zero throughout.
        prior_off = torch.zeros(1, num_experts)
        prior_acro = torch.zeros(1, num_experts)
        prior_biped = torch.zeros(1, num_experts)
        prior_off[0, EXPERT_LOCOMOTION] = 1.0
        prior_acro[0, EXPERT_ACROBATICS] = 1.0
        prior_biped[0, EXPERT_BIPED if num_experts > EXPERT_BIPED else EXPERT_LOCOMOTION] = 1.0
        self.register_buffer("prior_off", prior_off)
        self.register_buffer("prior_on", prior_acro)
        self.register_buffer("prior_biped", prior_biped)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return per-expert weights of shape ``(batch, num_experts)`` summing to one."""
        acro = obs[:, self.prior_index : self.prior_index + 1]
        if self.biped_prior_index >= 0:
            biped = obs[:, self.biped_prior_index : self.biped_prior_index + 1]
        else:
            biped = torch.zeros_like(acro)
        idle = (1.0 - acro).clamp(min=0.0) * (1.0 - biped).clamp(min=0.0)
        prior = idle * self.prior_off + acro * self.prior_on + biped * self.prior_biped
        return torch.softmax(self.mlp(obs) + self.prior_scale * prior, dim=-1)


class MixtureOfExperts(nn.Module):
    """Weighted sum of ``num_experts`` MLPs sharing one input.

    Kept free of dicts and optionals so ``torch.jit.script`` can trace it -- Isaac Lab's exporter
    scripts whatever it finds at ``policy.actor``.
    """

    def __init__(
        self,
        obs_dim: int,
        out_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        gating: Gating,
        num_experts: int = NUM_EXPERTS,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [MLP(obs_dim, out_dim, hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gating = gating

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        weights = self.gating(obs)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(obs))
        stacked = torch.stack(outputs, dim=1)  # (batch, num_experts, out_dim)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)

    def __getitem__(self, index: int) -> nn.Module:
        """Expose the experts' shared layer structure by index.

        Isaac Lab's ONNX exporter builds its dummy input with
        ``torch.zeros(1, self.actor[0].in_features)``, which assumes the actor is an
        ``nn.Sequential``. Every expert here takes the same input, so indexing the mixture returns
        the corresponding layer of the first expert and the exporter reads the right width. Without
        it, ``export_policy_as_onnx`` fails with "'MixtureOfExperts' object is not subscriptable".
        """
        return self.experts[0][index]


class MoEActorCritic(ActorCritic):
    """:class:`~rsl_rl.modules.ActorCritic` with mixture-of-experts actor and critic heads.

    Everything except the two heads -- the action distribution, the noise parameter, the optional
    observation normalizers, checkpoint I/O -- is inherited unchanged, so this stays compatible with
    ``OnPolicyRunner`` and with the JIT/ONNX exporter.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_experts: int = NUM_EXPERTS,
        gating_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        gating_activation: str = "elu",
        gating_prior_scale: float = 5.0,
        actor_prior_index: int = 0,
        critic_prior_index: int = 0,
        actor_biped_prior_index: int | None = None,
        critic_biped_prior_index: int | None = None,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )
        if self.state_dependent_std:
            raise NotImplementedError(
                "MoEActorCritic does not support state_dependent_std; the experts were pre-trained"
                " with a scalar noise parameter."
            )

        num_actor_obs = sum(obs[group].shape[-1] for group in self.obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in self.obs_groups["critic"])

        # Replace the plain MLP heads built by the base class. Separate gates: the action and the
        # value do not have to route the same way -- a value can be predictable from state in a
        # regime where the action is still a blend.
        self.actor = MixtureOfExperts(
            num_actor_obs,
            num_actions,
            actor_hidden_dims,
            activation,
            Gating(
                num_actor_obs,
                num_experts,
                gating_hidden_dims,
                gating_activation,
                actor_prior_index,
                gating_prior_scale,
                actor_biped_prior_index,
            ),
            num_experts,
        )
        self.critic = MixtureOfExperts(
            num_critic_obs,
            1,
            critic_hidden_dims,
            activation,
            Gating(
                num_critic_obs,
                num_experts,
                gating_hidden_dims,
                gating_activation,
                critic_prior_index,
                gating_prior_scale,
                critic_biped_prior_index,
            ),
            num_experts,
        )
        print(f"Actor MoE ({num_experts} experts): {self.actor.experts[0]}")
        print(f"Critic MoE ({num_experts} experts): {self.critic.experts[0]}")

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the groups that get independent learning-rate scales.

        The pre-trained experts are separated from the randomly initialised transition expert so the
        former can be fine-tuned gently while the latter learns at full rate, and the actor is
        separated from the critic so the actor can be held still (scale 0) while the value function
        catches up -- see :class:`.ppo_moe.MoEPPO`.
        """
        groups: dict[str, list[nn.Parameter]] = {
            "actor_pretrained": [],
            "actor_acrobatics": [],
            "actor_new": [],
            "actor_gating": list(self.actor.gating.parameters()),
            "critic_pretrained": [],
            "critic_acrobatics": [],
            "critic_new": [],
            "critic_gating": list(self.critic.gating.parameters()),
            "other": [],
        }
        # Every expert is initialised from a trained policy now, so none of them wants the full
        # learning rate a randomly initialised head would.
        #
        # The acrobatics expert gets its own group so it can be protected separately. It is the one
        # that measurably erodes: probed at a fixed condition, the merged policy's per-attempt flip
        # success ran 0.30 at merge, 0.45 at iteration 1500, and 0.08 by 3000, with the backflip
        # specifically going 0.485 -> 0.000 -- while the locomotion top speed stopped improving at
        # iteration 1000 and the bipedal stance held its success rate throughout. Splitting the
        # group changes nothing on its own: the default below matches ``actor_pretrained``.
        for index, expert in enumerate(self.actor.experts):
            name = "actor_acrobatics" if index == EXPERT_ACROBATICS else "actor_pretrained"
            groups[name if index in _PRETRAINED else "actor_new"].extend(expert.parameters())
        for index, expert in enumerate(self.critic.experts):
            name = "critic_acrobatics" if index == EXPERT_ACROBATICS else "critic_pretrained"
            groups[name if index in _PRETRAINED else "critic_new"].extend(expert.parameters())

        assigned = {id(p) for params in groups.values() for p in params}
        groups["other"] = [p for p in self.parameters() if id(p) not in assigned]
        return groups
