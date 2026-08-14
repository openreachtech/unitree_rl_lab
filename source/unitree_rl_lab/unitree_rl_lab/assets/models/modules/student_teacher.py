# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TCN student + frozen privileged teacher for rsl-rl Distillation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacher as RslStudentTeacher
from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict
from torch.distributions import Normal

from unitree_rl_lab.assets.models.student_actor import TcnMemory
from unitree_rl_lab.assets.models.teacher_actor import PrivilegedMlp


class TcnStudentTeacher(RslStudentTeacher):
    """Lee et al. 2020 student (TCN-100) + frozen teacher actor.

    Named ``TcnStudentTeacher`` so rsl-rl's default ``StudentTeacher`` class_name
    still resolves to the stock MLP. Builds the TCN / ``PrivilegedMlp`` nets
    directly (does not construct the parent's throwaway MLPs).

        student → TCN-100 over proprioception ``ot``, then MLP
        teacher → ``PrivilegedMlp`` (``xt`` encoder concat ``ot``)

    Observation layout (same as ``ActorCriticTeacher``):

        policy / student obs = ``ot``
        teacher obs          = ``concat(ot, xt)``  (policy + critic groups)

    Loading a PPO ``ActorCriticTeacher`` checkpoint copies ``actor.*`` into
    ``teacher`` (same ``PrivilegedMlp`` keys). A previous distillation
    checkpoint resumes the student.
    """

    is_recurrent = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        teacher_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        tcn_history_length: int = 100,
        tcn_channels: int = 34,
        tcn_kernel_size: int = 5,
        tcn_latent_dim: int = 64,
        tcn_concat_current_obs: bool = True,
        privileged_encoder_hidden_dims: tuple[int] | list[int] = [72],
        privileged_latent_dim: int = 64,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "TcnStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        nn.Module.__init__(self)

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "TcnStudentTeacher only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "TcnStudentTeacher only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.num_proprio = num_student_obs
        self.num_privileged = num_teacher_obs - num_student_obs
        self.tcn_latent_dim = int(tcn_latent_dim)
        assert self.num_privileged > 0, (
            f"teacher obs ({num_teacher_obs}) must be proprio+privileged, "
            f"got privileged width {self.num_privileged}."
        )

        self.memory_s = TcnMemory(
            num_student_obs,
            history_length=tcn_history_length,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            latent_dim=tcn_latent_dim,
            concat_current_obs=tcn_concat_current_obs,
        )
        self.student = MLP(self.memory_s.rnn.output_size, num_actions, student_hidden_dims, activation)
        print(f"Student TCN: {self.memory_s.rnn}")
        print(f"Student MLP: {self.student}")

        teacher_encoder = MLP(
            self.num_privileged,
            privileged_latent_dim,
            privileged_encoder_hidden_dims,
            activation,
            last_activation=activation,
        )
        teacher_mlp = MLP(
            privileged_latent_dim + num_student_obs,
            num_actions,
            teacher_hidden_dims,
            activation,
        )
        self.teacher = PrivilegedMlp(teacher_encoder, teacher_mlp, num_student_obs, self.num_privileged)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        print(f"Teacher privileged encoder: {teacher_encoder}")
        print(f"Teacher MLP: {teacher_mlp}")

        self.student_obs_normalization = student_obs_normalization
        self.student_obs_normalizer = (
            EmpiricalNormalization(num_student_obs) if student_obs_normalization else nn.Identity()
        )
        self.teacher_obs_normalization = teacher_obs_normalization
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _student_features(self, obs: TensorDict) -> torch.Tensor:
        ot = self.student_obs_normalizer(self.get_student_obs(obs))
        return self.memory_s(ot).squeeze(0)

    def _update_distribution(self, features: torch.Tensor) -> None:
        mean = self.student(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        features = self._student_features(obs)
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, return_latent: bool = False):
        features = self._student_features(obs)
        action = self.student(features)
        if return_latent:
            return action, features[..., : self.tcn_latent_dim]
        return action

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self.teacher_obs_normalizer(self.get_teacher_obs(obs))
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def evaluate_latent(self, obs: TensorDict) -> torch.Tensor:
        """Teacher privileged embedding ``l̄_t`` (supervision for the TCN latent)."""
        teacher_obs = self.teacher_obs_normalizer(self.get_teacher_obs(obs))
        with torch.no_grad():
            return self.teacher.encoder(teacher_obs[..., self.num_proprio :])

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        self.memory_s.reset(dones, hidden_states[0])

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_s.hidden_state, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.memory_s.detach_hidden_state(dones)
