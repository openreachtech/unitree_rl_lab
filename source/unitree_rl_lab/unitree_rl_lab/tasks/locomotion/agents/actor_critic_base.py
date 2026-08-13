# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PPO actor-critic helpers for the TCN student and privileged teacher."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization


class ActorCriticPpoBase(nn.Module):
    """Gaussian policy boilerplate shared by :class:`ActorCriticTcn` and :class:`ActorCriticTeacher`."""

    is_recurrent: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _count_group_obs(self, obs: TensorDict, obs_groups: dict[str, list[str]]) -> tuple[int, int]:
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, f"{type(self).__name__} only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, f"{type(self).__name__} only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        return num_actor_obs, num_critic_obs

    def _init_normalizers(
        self,
        actor_obs_normalization: bool,
        critic_obs_normalization: bool,
        num_actor_obs: int,
        num_critic_obs: int,
    ) -> None:
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

    def _init_gaussian_noise(
        self,
        num_actions: int,
        init_noise_std: float,
        noise_std_type: str,
        state_dependent_std: bool,
    ) -> None:
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        if state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

    @staticmethod
    def _warn_unexpected_kwargs(cls_name: str, kwargs: dict[str, Any]) -> None:
        if kwargs:
            print(f"{cls_name}.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _update_distribution(self, features: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            mean = self.actor(features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def register_policy_class(name: str, cls: type) -> None:
    """Make ``eval(name)`` work inside RSL-RL's OnPolicyRunner."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    setattr(on_policy_runner, name, cls)
