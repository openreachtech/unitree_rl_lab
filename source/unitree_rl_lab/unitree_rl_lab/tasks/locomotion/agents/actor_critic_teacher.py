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
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticTeacher(nn.Module):
    """Privileged teacher actor (Table S5) and matching MLP critic, trained with PPO.

    ``xt`` (critic / privileged group) → MLP 72 → 64, concat ``ot`` (policy group),
    then the actor / critic MLP. The actor therefore sees terrain and contact;
    this policy is for simulation comparison, not onboard deployment.
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
        if kwargs:
            print(
                "ActorCriticTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTeacher module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_privileged_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTeacher module only supports 1D observations."
            num_privileged_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        trunk_dim = privileged_latent_dim + num_actor_obs

        self.actor_privileged_encoder = MLP(
            num_privileged_obs,
            privileged_latent_dim,
            privileged_encoder_hidden_dims,
            activation,
            last_activation=activation,
        )
        if self.state_dependent_std:
            self.actor = MLP(trunk_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(trunk_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor privileged encoder: {self.actor_privileged_encoder}")
        print(f"Actor MLP: {self.actor}")

        self.critic_privileged_encoder = MLP(
            num_privileged_obs,
            privileged_latent_dim,
            privileged_encoder_hidden_dims,
            activation,
            last_activation=activation,
        )
        self.critic = MLP(trunk_dim, 1, critic_hidden_dims, activation)
        print(f"Critic privileged encoder: {self.critic_privileged_encoder}")
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_privileged_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _privileged_features(self, obs: TensorDict, encoder: MLP) -> torch.Tensor:
        ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
        xt = self.critic_obs_normalizer(self.get_privileged_obs(obs))
        return torch.cat([encoder(xt), ot], dim=-1)

    def _update_distribution(self, features: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(self._privileged_features(obs, self.actor_privileged_encoder))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._privileged_features(obs, self.actor_privileged_encoder)
        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        return self.actor(features)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self.critic(self._privileged_features(obs, self.critic_privileged_encoder))

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_privileged_obs(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_privileged_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def register_actor_critic_teacher() -> None:
    """Make ``eval("ActorCriticTeacher")`` work inside RSL-RL's OnPolicyRunner."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticTeacher = ActorCriticTeacher


register_actor_critic_teacher()
