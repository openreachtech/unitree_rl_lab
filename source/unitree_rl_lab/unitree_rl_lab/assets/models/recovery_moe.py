"""Mixture-of-experts actor-critic for the Go2 walk / fall / get-up policy.

Two experts share one observation: expert 0 is initialised from a blind locomotion policy, expert 1
from the blind get-up policy. Unlike the multi-task mixture in :mod:`.moe_actor`, nothing *commands*
a fall -- it happens to the robot -- so the gate's routing prior cannot key off a command flag. It
keys off the trunk's attitude instead, read straight from projected gravity: upright routes to
locomotion, tilted or inverted to get-up. The gate's own MLP is added on top (zero-initialised), so
training can move the boundary wherever the experts actually hand over best -- e.g. a robot lying
prone with its legs folded is "upright" by gravity but still needs the get-up expert.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks import MLP
from tensordict import TensorDict
from typing import Any

from .moe_actor import MixtureOfExperts, MoEActorCritic

EXPERT_WALK = 0
EXPERT_GETUP = 1
RECOVERY_EXPERT_NAMES = ("locomotion", "getup")


class AttitudeGating(nn.Module):
    """Soft gate whose prior routes by how upright the trunk is.

    Args:
        obs_dim: Width of the observation this gate reads.
        num_experts: Number of experts to weight. Experts beyond the first two get no prior.
        hidden_dims: Hidden layer sizes of the gate's MLP.
        activation: Activation of the gate's MLP.
        gravity_z_index: Column of projected gravity's z component (-1 upright, +1 inverted).
        prior_scale: Logit offset applied by the prior. 0.0 disables it.
        upright_threshold: ``-g_z`` at which the prior is split evenly between the two experts.
            0.8 is about 37 degrees of tilt -- past what the locomotion expert recovers from on its
            own, short of anything the get-up expert would not recognise as a fall.
        upright_width: Softness of that boundary in ``-g_z`` units. Wider than the observation's
            +-0.05 gravity noise so the prior does not chatter while standing near the threshold.
    """

    def __init__(
        self,
        obs_dim: int,
        num_experts: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        gravity_z_index: int,
        prior_scale: float,
        upright_threshold: float = 0.8,
        upright_width: float = 0.08,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, num_experts, hidden_dims, activation)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.gravity_z_index = int(gravity_z_index)
        self.prior_scale = float(prior_scale)
        self.upright_threshold = float(upright_threshold)
        self.upright_width = float(upright_width)

        prior_walk = torch.zeros(1, num_experts)
        prior_getup = torch.zeros(1, num_experts)
        prior_walk[0, EXPERT_WALK] = 1.0
        prior_getup[0, EXPERT_GETUP] = 1.0
        self.register_buffer("prior_walk", prior_walk)
        self.register_buffer("prior_getup", prior_getup)

    def upright(self, obs: torch.Tensor) -> torch.Tensor:
        """Soft "trunk is upright" indicator in ``[0, 1]``, shape ``(batch, 1)``."""
        neg_gz = -obs[:, self.gravity_z_index : self.gravity_z_index + 1]
        return torch.sigmoid((neg_gz - self.upright_threshold) / self.upright_width)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        upright = self.upright(obs)
        prior = upright * self.prior_walk + (1.0 - upright) * self.prior_getup
        return torch.softmax(self.mlp(obs) + self.prior_scale * prior, dim=-1)


class RecoveryMoEActorCritic(MoEActorCritic):
    """:class:`.MoEActorCritic` with attitude-keyed gates on both heads.

    The critic keeps the mixture structure but starts from random weights: the two source critics
    were trained on different reward functions and different privileged observations, so there is
    no inherited value worth placing. ``MoEPPO.actor_warmup_iterations`` holds the actor still
    while it catches up.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_experts: int = 2,
        gating_hidden_dims: tuple[int, ...] | list[int] = (128,),
        gating_activation: str = "elu",
        gating_prior_scale: float = 5.0,
        actor_gravity_z_index: int = 5,
        critic_gravity_z_index: int = 8,
        upright_threshold: float = 0.8,
        upright_width: float = 0.08,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            num_experts=num_experts,
            gating_hidden_dims=gating_hidden_dims,
            gating_activation=gating_activation,
            gating_prior_scale=gating_prior_scale,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )
        num_actor_obs = sum(obs[group].shape[-1] for group in self.obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in self.obs_groups["critic"])

        def gate(obs_dim: int, gravity_z_index: int) -> AttitudeGating:
            return AttitudeGating(
                obs_dim,
                num_experts,
                gating_hidden_dims,
                gating_activation,
                gravity_z_index,
                gating_prior_scale,
                upright_threshold,
                upright_width,
            )

        self.actor = MixtureOfExperts(
            num_actor_obs, num_actions, actor_hidden_dims, activation, gate(num_actor_obs, actor_gravity_z_index), num_experts
        )
        self.critic = MixtureOfExperts(
            num_critic_obs, 1, critic_hidden_dims, activation, gate(num_critic_obs, critic_gravity_z_index), num_experts
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Actor experts are pre-trained and fine-tuned gently; the critic is learned from scratch."""
        groups: dict[str, list[nn.Parameter]] = {
            "actor_pretrained": [p for expert in self.actor.experts for p in expert.parameters()],
            "actor_gating": list(self.actor.gating.parameters()),
            "critic_new": [p for expert in self.critic.experts for p in expert.parameters()],
            "critic_gating": list(self.critic.gating.parameters()),
        }
        assigned = {id(p) for params in groups.values() for p in params}
        groups["other"] = [p for p in self.parameters() if id(p) not in assigned]
        return groups
