# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO actor-critic whose actor is Lee et al. 2020's privileged teacher.

Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning Quadrupedal Locomotion over
Challenging Terrain", Science Robotics 2020 (Table S5, Teacher).

The paper's teacher encodes privileged ``xt`` to a 64-d latent, concatenates
proprioception ``ot``, and outputs actions. Here the same split is trained with
PPO: the actor is that teacher; the critic uses the same encoder-concat MLP as a
value head. No imitation / distillation. Feedforward (not recurrent).

Export (JIT/ONNX) copies ``policy.actor``. That module's ``forward`` therefore
takes ``concat(ot, xt)`` — not proprioception alone. This policy is sim-only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import MLP

from unitree_rl_lab.tasks.locomotion.agents.actor_critic_base import ActorCriticPpoBase, register_policy_class


class PrivilegedMlp(nn.Module):
    """``xt`` encoder, concat ``ot``, then an MLP.

    isaaclab_rl's exporter deep-copies ``policy.actor`` and probes
    ``actor[0].in_features`` for the dummy ONNX input, so this module reports
    ``in_features = num_proprio + num_privileged`` and ``forward`` consumes that
    concatenated vector.
    """

    def __init__(self, encoder: MLP, mlp: MLP, num_proprio: int, num_privileged: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.num_proprio = num_proprio
        self.in_features = num_proprio + num_privileged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ot = x[..., : self.num_proprio]
        xt = x[..., self.num_proprio :]
        return self.mlp(torch.cat([self.encoder(xt), ot], dim=-1))

    def __getitem__(self, idx: int) -> nn.Module:
        if idx == 0:
            return self
        return self.mlp[idx]


class ActorCriticTeacher(ActorCriticPpoBase):
    """Privileged teacher actor (Table S5) and matching MLP critic, trained with PPO.

    ``concat(ot, xt)`` → encode ``xt`` (72 → 64), concat ``ot``, then the actor /
    critic MLP. Encoder weights are not shared between actor and critic.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        privileged_encoder_hidden_dims: tuple[int] | list[int] = [72],
        privileged_latent_dim: int = 64,
        **kwargs: dict[str, Any],
    ) -> None:
        self._warn_unexpected_kwargs("ActorCriticTeacher", kwargs)
        super().__init__()
        num_actor_obs, num_privileged_obs = self._count_group_obs(obs, obs_groups)
        self.num_actor_obs = num_actor_obs

        actor_encoder = MLP(
            num_privileged_obs,
            privileged_latent_dim,
            privileged_encoder_hidden_dims,
            activation,
            last_activation=activation,
        )
        actor_mlp = (
            MLP(privileged_latent_dim + num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
            if state_dependent_std
            else MLP(privileged_latent_dim + num_actor_obs, num_actions, actor_hidden_dims, activation)
        )
        self.actor = PrivilegedMlp(actor_encoder, actor_mlp, num_actor_obs, num_privileged_obs)
        print(f"Actor privileged encoder: {actor_encoder}")
        print(f"Actor MLP: {actor_mlp}")

        critic_encoder = MLP(
            num_privileged_obs,
            privileged_latent_dim,
            privileged_encoder_hidden_dims,
            activation,
            last_activation=activation,
        )
        critic_mlp = MLP(privileged_latent_dim + num_actor_obs, 1, critic_hidden_dims, activation)
        self.critic = PrivilegedMlp(critic_encoder, critic_mlp, num_actor_obs, num_privileged_obs)
        print(f"Critic privileged encoder: {critic_encoder}")
        print(f"Critic MLP: {critic_mlp}")

        self._init_normalizers(
            actor_obs_normalization, critic_obs_normalization, num_actor_obs, num_privileged_obs
        )
        self._init_gaussian_noise(num_actions, init_noise_std, noise_std_type, state_dependent_std)

    def _concat_obs(self, obs: TensorDict) -> torch.Tensor:
        ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
        xt = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return torch.cat([ot, xt], dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(self._concat_obs(obs))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._concat_obs(obs)
        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        return self.actor(features)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self.critic(self._concat_obs(obs))


register_policy_class("ActorCriticTeacher", ActorCriticTeacher)
