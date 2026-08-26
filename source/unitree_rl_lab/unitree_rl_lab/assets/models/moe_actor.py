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
EXPERT_TRANSITION = 2
NUM_EXPERTS = 3


class Gating(nn.Module):
    """Soft gate over the experts, biased at initialisation by the commanded motion.

    Args:
        obs_dim: Width of the observation this gate reads.
        num_experts: Number of experts to weight.
        hidden_dims: Hidden layer sizes of the gate's MLP.
        activation: Activation of the gate's MLP.
        prior_index: Column of the 0/1 "a motion is commanded" flag in the observation.
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
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, num_experts, hidden_dims, activation)
        # Zero last layer => the gate contributes nothing at initialisation and the prior decides
        # routing alone. Gradients still flow, so this is a starting point and not a constraint.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.prior_index = int(prior_index)
        self.prior_scale = float(prior_scale)

        # Logit rows selected by the command flag: no motion commanded -> locomotion expert,
        # motion commanded -> acrobatics expert. The transition expert is never favoured by the
        # prior; it has to earn its weight from the gate MLP.
        prior_off = torch.zeros(1, num_experts)
        prior_on = torch.zeros(1, num_experts)
        prior_off[0, EXPERT_LOCOMOTION] = 1.0
        prior_on[0, EXPERT_ACROBATICS] = 1.0
        self.register_buffer("prior_off", prior_off)
        self.register_buffer("prior_on", prior_on)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return per-expert weights of shape ``(batch, num_experts)`` summing to one."""
        enabled = obs[:, self.prior_index : self.prior_index + 1]
        prior = (1.0 - enabled) * self.prior_off + enabled * self.prior_on
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
            "actor_new": [],
            "actor_gating": list(self.actor.gating.parameters()),
            "critic_pretrained": [],
            "critic_new": [],
            "critic_gating": list(self.critic.gating.parameters()),
            "other": [],
        }
        pretrained = (EXPERT_LOCOMOTION, EXPERT_ACROBATICS)
        for index, expert in enumerate(self.actor.experts):
            groups["actor_pretrained" if index in pretrained else "actor_new"].extend(expert.parameters())
        for index, expert in enumerate(self.critic.experts):
            groups["critic_pretrained" if index in pretrained else "critic_new"].extend(expert.parameters())

        assigned = {id(p) for params in groups.values() for p in params}
        groups["other"] = [p for p in self.parameters() if id(p) not in assigned]
        return groups
