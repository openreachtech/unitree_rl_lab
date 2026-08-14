"""TCN-100 student encoder from Lee et al. 2020, Table S5.

Used by PPO (``ActorCriticTcn``) and by teacher-student distillation
(``StudentTeacher``). Distillation imitates the privileged teacher (action +
latent matching); PPO can train the same encoder directly.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.networks import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories

from unitree_rl_lab.tasks.locomotion.agents.actor_critic_base import ActorCriticPpoBase, register_policy_class


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

    ``self.rnn`` is the name isaaclab_rl's exporter looks up (``policy.memory_a.rnn`` /
    ``policy.memory_s.rnn``); the module itself is :class:`TcnHistory`, not an RNN.
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

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.hidden_state = hidden_state
        elif self.hidden_state is not None:
            self.hidden_state[..., dones == 1, :] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is None:
            return
        if dones is None:
            self.hidden_state = self.hidden_state.detach()
        else:
            self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()


class ActorCriticTcn(ActorCriticPpoBase):
    """TCN actor (student) with an MLP critic, trained with PPO.

    Actor: TCN over proprioceptive history, concatenated with the current observation,
    then an MLP. Critic defaults to a feedforward MLP on the critic observation. With
    ``use_privileged_encoder``, the critic matches the paper's teacher: encode privileged
    ``xt`` to a latent, concatenate current proprioception ``ot``, then a value MLP.
    Distillation uses the same TCN via ``StudentTeacher``, not this PPO wrapper.
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
        self._warn_unexpected_kwargs("ActorCriticTcn", kwargs)
        super().__init__()
        num_actor_obs, num_privileged_obs = self._count_group_obs(obs, obs_groups)
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
        if state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor TCN: {self.memory_a.rnn}")
        print(f"Actor MLP: {self.actor}")

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

        self._init_normalizers(
            actor_obs_normalization, critic_obs_normalization, num_actor_obs, num_privileged_obs
        )
        self._init_gaussian_noise(num_actions, init_noise_std, noise_std_type, state_dependent_std)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        if dones is None:
            self._hidden_c = None
        elif self._hidden_c is not None:
            self._hidden_c[..., dones == 1, :] = 0.0

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
        out_mem = self.memory_a(ot, masks, hidden_state).squeeze(0)
        if masks is None:
            self._ensure_dummy_hidden_c(out_mem)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
        out_mem = self.memory_a(ot).squeeze(0)
        self._ensure_dummy_hidden_c(out_mem)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        if self.use_privileged_encoder:
            ot = self.actor_obs_normalizer(self.get_actor_obs(obs))
            xt = self.critic_obs_normalizer(self.get_critic_obs(obs))
            values = self.critic(torch.cat([self.privileged_encoder(xt), ot], dim=-1))
        else:
            values = self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs)))
        if masks is not None:
            values = unpad_trajectories(values, masks).squeeze(0)
        return values

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self._hidden_c

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
    register_policy_class("ActorCriticTcn", ActorCriticTcn)

    try:
        from isaaclab_rl.rsl_rl import exporter
    except ImportError:
        return

    def _alias_tcn_as_gru(policy: object):
        for mem_name in ("memory_a", "memory_s"):
            rnn = getattr(getattr(policy, mem_name, None), "rnn", None)
            if rnn is not None and type(rnn).__name__ == "TcnHistory":
                orig_cls = type(rnn)
                rnn.__class__ = type("GRU", (orig_cls,), {})
                return rnn, orig_cls
        return None, None

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
