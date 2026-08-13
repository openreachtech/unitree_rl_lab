# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO actor-critic whose actor is the TCN student encoder from Lee et al. 2020.

Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning Quadrupedal Locomotion over
Challenging Terrain", Science Robotics 2020 (Table S5, TCN-N student).

The paper trains that TCN by imitating a privileged teacher. This module keeps the
same encoder and trains it with PPO instead: the actor is the TCN over a proprioceptive
history. The critic is either a feedforward MLP on the critic observation, or (when
``use_privileged_encoder`` is set) the paper's teacher MLP: privileged ``xt`` encoded
to a latent, concatenated with current proprioception ``ot``, then a value head.
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories


class CausalConv1d(nn.Module):
    """1D convolution that does not look into the future (left-padded, then cropped)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TcnEncoder(nn.Module):
    """Fully convolutional TCN encoder from Lee et al. 2020 Table S5.

    Three dilated causal convolutions (dilation 1, 2, 4) interleaved with stride-2
    convolutions that downsample the time axis. Filter size is 5; each conv is followed
    by ReLU. The resulting feature map is flattened and projected to ``latent_dim``.
    """

    def __init__(
        self,
        in_channels: int,
        history_length: int,
        channels: int = 34,
        kernel_size: int = 5,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.history_length = history_length
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            CausalConv1d(in_channels, channels, kernel_size, dilation=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size, dilation=4),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, history_length)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, latent_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows of shape (N, C, T) to (N, latent_dim)."""
        return self.fc(self.conv(x).reshape(x.shape[0], -1))


class TcnHistory(nn.Module):
    """TCN encoder over a packed observation history.

    RSL-RL's GRU path (rollout storage, ONNX/JIT export, deploy ``OrtRunner``) expects
    ``(x, h) -> (y, h)``. Here ``h`` is not an RNN hidden state: it is the last
    ``history_length`` observations, packed. That lets the existing GRU plumbing carry
    the TCN's context window without a custom runtime.
    """

    def __init__(
        self,
        input_size: int,
        history_length: int = 100,
        channels: int = 34,
        kernel_size: int = 5,
        latent_dim: int = 64,
        concat_current_obs: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = history_length
        self.hidden_size = history_length * input_size
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.concat_current_obs = concat_current_obs
        self.output_size = latent_dim + input_size if concat_current_obs else latent_dim
        self.encoder = TcnEncoder(
            in_channels=input_size,
            history_length=history_length,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the TCN over a sequence.

        Args:
            x: ``(seq_len, batch, input_size)``.
            hidden: ``(num_layers, batch, history_length * input_size)`` packed history
                from *before* ``x[0]`` (oldest first).

        Returns:
            features: ``(seq_len, batch, output_size)``.
            new_hidden: packed history after consuming ``x``.
        """
        seq_len, batch, channels = x.shape
        hist = hidden.reshape(batch, self.history_length, channels)
        seq = x.transpose(0, 1)
        stream = torch.cat([hist, seq], dim=1)
        windows = stream.unfold(dimension=1, size=self.history_length, step=1)
        windows = windows[:, 1 : seq_len + 1]
        latent = self.encoder(windows.reshape(batch * seq_len, channels, self.history_length))
        latent = latent.view(batch, seq_len, -1)
        if self.concat_current_obs:
            features = torch.cat([latent, seq], dim=-1)
        else:
            features = latent
        new_hidden = stream[:, seq_len : seq_len + self.history_length].reshape(1, batch, self.hidden_size)
        return features.transpose(0, 1), new_hidden


class TcnMemory(nn.Module):
    """``Memory``-compatible wrapper around :class:`TcnHistory`.

    ``self.rnn`` is the name isaaclab_rl's exporter looks up (``policy.memory_a.rnn``);
    the module itself is :class:`TcnHistory`, not an RNN.
    """

    def __init__(
        self,
        input_size: int,
        history_length: int = 100,
        channels: int = 34,
        kernel_size: int = 5,
        latent_dim: int = 64,
        concat_current_obs: bool = True,
    ) -> None:
        super().__init__()
        self.rnn = TcnHistory(
            input_size=input_size,
            history_length=history_length,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            concat_current_obs=concat_current_obs,
        )
        self.hidden_state: HiddenState = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to TCN memory during policy update")
            out, _ = self.rnn(input, hidden_state)
            return unpad_trajectories(out, masks)
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(
                self.rnn.num_layers,
                input.shape[0],
                self.rnn.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
        out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.hidden_state = None
        elif self.hidden_state is not None:
            self.hidden_state[..., dones == 1, :] = 0.0


class ActorCriticTcn(nn.Module):
    """TCN actor (student) with an MLP critic.

    Actor: TCN over proprioceptive history, concatenated with the current observation,
    then an MLP. Critic defaults to a feedforward MLP on the critic observation. With
    ``use_privileged_encoder``, the critic matches the paper's teacher: encode privileged
    ``xt`` to a latent, concatenate current proprioception ``ot``, then a value MLP.
    """

    is_recurrent: bool = True

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
        tcn_history_length: int = 100,
        tcn_channels: int = 34,
        tcn_kernel_size: int = 5,
        tcn_latent_dim: int = 64,
        tcn_concat_current_obs: bool = True,
        use_privileged_encoder: bool = False,
        privileged_encoder_hidden_dims: tuple[int] | list[int] = [72],
        privileged_latent_dim: int = 64,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticTcn.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTcn module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_privileged_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTcn module only supports 1D observations."
            num_privileged_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        self._hidden_c: HiddenState = None

        self.memory_a = TcnMemory(
            num_actor_obs,
            history_length=tcn_history_length,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            latent_dim=tcn_latent_dim,
            concat_current_obs=tcn_concat_current_obs,
        )
        actor_input_dim = self.memory_a.rnn.output_size
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor TCN: {self.memory_a.rnn}")
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.use_privileged_encoder = use_privileged_encoder
        if use_privileged_encoder:
            self.privileged_encoder = MLP(
                num_privileged_obs,
                privileged_latent_dim,
                privileged_encoder_hidden_dims,
                activation,
                last_activation=activation,
            )
            self.critic = MLP(privileged_latent_dim + num_actor_obs, 1, critic_hidden_dims, activation)
            print(f"Privileged encoder: {self.privileged_encoder}")
        else:
            self.privileged_encoder = None
            self.critic = MLP(num_privileged_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

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
        self.memory_a.reset(dones)
        if dones is None:
            self._hidden_c = None
        elif self._hidden_c is not None:
            self._hidden_c[..., dones == 1, :] = 0.0

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

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        if masks is None:
            self._ensure_dummy_hidden_c(out_mem)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs).squeeze(0)
        self._ensure_dummy_hidden_c(out_mem)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        if self.use_privileged_encoder:
            ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
            xt = self.critic_obs_normalizer(self.get_privileged_obs(obs))
            latent = self.privileged_encoder(xt)
            values = self.critic(torch.cat([latent, ot], dim=-1))
        else:
            xt = self.critic_obs_normalizer(self.get_critic_obs(obs))
            values = self.critic(xt)
        if masks is not None:
            values = unpad_trajectories(values, masks).squeeze(0)
        return values

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.get_privileged_obs(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self._hidden_c

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_privileged_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    def _ensure_dummy_hidden_c(self, features: torch.Tensor) -> None:
        """Keep a GRU-shaped dummy critic hidden state so rollout storage stays symmetric."""
        if self._hidden_c is not None:
            return
        batch = features.shape[0]
        self._hidden_c = torch.zeros(1, batch, 1, device=features.device, dtype=features.dtype)


def register_actor_critic_tcn() -> None:
    """Make ``eval("ActorCriticTcn")`` work inside RSL-RL's OnPolicyRunner, and let the
    isaaclab_rl GRU exporter accept :class:`TcnHistory` (same ``h_in`` / ``h_out`` I/O).
    """
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticTcn = ActorCriticTcn

    try:
        from isaaclab_rl.rsl_rl import exporter
    except ImportError:
        return

    def _alias_tcn_as_gru(policy: object):
        rnn = getattr(getattr(policy, "memory_a", None), "rnn", None)
        if rnn is None or type(rnn).__name__ != "TcnHistory":
            return None, None
        orig_cls = type(rnn)
        rnn.__class__ = type("GRU", (orig_cls,), {})
        return rnn, orig_cls

    orig_onnx_init = exporter._OnnxPolicyExporter.__init__
    orig_jit_init = exporter._TorchPolicyExporter.__init__

    def _patched_init(orig_init):
        def init(self, policy, *args, **kwargs):
            rnn, orig_cls = _alias_tcn_as_gru(policy)
            try:
                orig_init(self, policy, *args, **kwargs)
            finally:
                if rnn is not None and orig_cls is not None:
                    rnn.__class__ = orig_cls

        return init

    if getattr(exporter._OnnxPolicyExporter.__init__, "_tcn_patched", False):
        return
    exporter._OnnxPolicyExporter.__init__ = _patched_init(orig_onnx_init)
    exporter._OnnxPolicyExporter.__init__._tcn_patched = True
    exporter._TorchPolicyExporter.__init__ = _patched_init(orig_jit_init)
    exporter._TorchPolicyExporter.__init__._tcn_patched = True


register_actor_critic_tcn()
