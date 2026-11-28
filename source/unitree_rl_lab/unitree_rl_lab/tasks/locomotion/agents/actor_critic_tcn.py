"""TCN の actor-critic (Lee et al. 2020, Table S5 の TCN-N を PPO で直接学習する)。

``doc/papers/Learning_Quadrupedal_Locomotion_over_Challenging_Terrain.md`` の student の
エンコーダを、蒸留せずに PPO で学習する。unitree_rl_lab ``feat/go2w`` の
``assets/models/student_actor.py`` からの移植::

    actor :  固有感覚の履歴 (N x 45)          -> TCN -> 潜在 64 ─┐
                                   現在の固有感覚 45 ──────────────┴-> MLP -> 行動
    critic:  critic 観測の直近 M ステップ (M x 129) を平らに並べる  -> MLP -> 価値

GRU 版 (``ActorCriticRecurrent``) の ``memory_a`` / ``memory_c`` を :class:`TcnMemory` に
置き換えただけで、actor と critic がそれぞれ自分の観測の記憶を持つ点は同じ。違いは記憶の
長さが窓 (N ステップ) で切れること。critic は特権情報を見るので長い記憶は要らず、TCN にすると
更新の GPU メモリが倍 (15.4 GB 対 7.4 GB) になるので、短い履歴 + MLP にしている。

**窓を GRU の「隠れ状態」として持ち回す** (:class:`TcnHistory`)。入出力が GRU と同じ
``(x, h) -> (y, h)`` なので、rsl_rl の再帰方策の経路 (保存・ミニバッチ・
:class:`~unitree_rl_lab.tasks.locomotion.agents.ppo_recurrent_symmetry.PPORecurrentSymmetry` の左右反転) が
そのまま使える。``h`` は直近 N ステップの観測を古い順に並べて平らにしたもの。エピソードの
終わりで 0 に戻す (GRU と同じ) ので、エピソード開始直後の窓は前が 0 で埋まる。

更新では、軌跡の始まりの窓と観測の列から全ステップの窓を組み立て直す (``unfold``)。
``PPORecurrentSymmetry`` は隠れ状態を rollout の最初のステップだけ保存する (リセット直後は 0)。
ステップごとに保存すると窓の分だけ重く、100 x 129 の critic では 10 GB を超えるため。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.networks import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories


class CausalConv1d(nn.Module):
    """未来を見ない 1D 畳み込み (左だけ詰めて長さを保つ)。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TcnEncoder(nn.Module):
    """Lee et al. 2020 Table S5 の TCN-N エンコーダ。

    dilation 1 / 2 / 4 の因果的な畳み込みと、時間を半分にする stride 2 の畳み込みを交互に 3 回。
    フィルタ幅 5、各層の後に ELU。最後に平らにして ``tanh(latent_dim)``。

    論文は ReLU だが、ReLU 版は学習で最後の層のユニットが全部止まった (dying ReLU、Pos-Target TCN の
    model_1000 で稼働 0 %、潜在がほぼ 0 で実質 MLP)。ELU は負側も勾配が残る。
    """

    def __init__(
        self, in_channels: int, history_length: int, channels: int = 34, kernel_size: int = 5, latent_dim: int = 64
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for dilation in (1, 2, 4):
            layers += [
                CausalConv1d(c_in, channels, kernel_size, dilation=dilation),
                nn.ELU(),
                nn.Conv1d(channels, channels, kernel_size, stride=2, padding=kernel_size // 2),
                nn.ELU(),
            ]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, history_length)).numel()
        self.fc = nn.Sequential(nn.Linear(n_flat, latent_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(バッチ, チャンネル, 時間) -> (バッチ, latent_dim)。"""
        return self.fc(self.conv(x).flatten(1))


def sliding_windows(
    x: torch.Tensor, hidden: torch.Tensor | None, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``x``: (時間, バッチ, 入力)、``hidden``: (1, バッチ, N x 入力) = ``x[0]`` より前の N ステップ (古い順)。

    返り値: 各時刻の窓 (バッチ, 時間, 入力, N) (時刻 t の窓は ``x[t]`` で終わる N ステップ)、
    ``x`` をバッチ先頭にしたもの (バッチ, 時間, 入力)、新しい ``hidden``。
    """
    seq_len, batch, c = x.shape
    if hidden is None:
        hidden = x.new_zeros(1, batch, n * c)
    seq = x.transpose(0, 1)
    stream = torch.cat([hidden.reshape(batch, n, c), seq], dim=1)  # (バッチ, N + 時間, 入力)
    windows = stream.unfold(1, n, 1)[:, 1:]  # 時刻 t の窓 = stream[t+1 : t+1+N]
    new_hidden = stream[:, seq_len:].reshape(1, batch, n * c)
    return windows, seq, new_hidden


class TcnHistory(nn.Module):
    """窓を隠れ状態として持つ TCN。GRU と同じ ``(x, h) -> (y, h)``。

    ``x``: (時間, バッチ, 入力)。``h``: (1, バッチ, N x 入力)、``x[0]`` より前の N ステップ (古い順)。
    出力: (時間, バッチ, latent_dim [+ 入力])、新しい ``h``。
    """

    def __init__(
        self,
        input_size: int,
        history_length: int = 100,
        channels: int = 34,
        kernel_size: int = 5,
        latent_dim: int = 64,
        concat_current_obs: bool = True,
        bf16: bool = True,
        history_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = history_length
        # Only the first ``history_dims`` inputs go into the window (proprioception); the rest
        # (e.g. height patches) reach the actor MLP as the current value only.
        self.history_dims = input_size if history_dims is None else history_dims
        self.hidden_size = history_length * self.history_dims
        self.num_layers = 1
        self.concat_current_obs = concat_current_obs
        self.bf16 = bf16
        """畳み込みエンコーダだけ bf16 で計算する (重みと出力は fp32)。rollout と更新で同じ経路を通る。
        fp32 との方策の KL 1.6e-10、更新は 1.7 倍速い。"""
        self.output_size = latent_dim + (input_size if concat_current_obs else 0)
        self.encoder = TcnEncoder(self.history_dims, history_length, channels, kernel_size, latent_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch, _ = x.shape
        n, c = self.history_length, self.history_dims
        windows, _, new_hidden = sliding_windows(x[..., :c], hidden, n)
        seq = x.transpose(0, 1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.bf16 and x.is_cuda):
            latent = self.encoder(windows.reshape(batch * seq_len, c, n))
        latent = latent.float().view(batch, seq_len, -1)
        features = torch.cat([latent, seq], dim=-1) if self.concat_current_obs else latent
        return features.transpose(0, 1), new_hidden


class HistoryStack(nn.Module):
    """直近 N ステップの観測を平らに並べるだけの記憶 (MLP 用)。:class:`TcnHistory` と同じ ``(x, h) -> (y, h)``。

    出力: (時間, バッチ, N x 入力)。古い順に並べた窓を平らにしたもの (最後の ``入力`` 個が現在の観測)。
    """

    def __init__(self, input_size: int, history_length: int = 5, history_dims: int | None = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = history_length
        # Only the first ``history_dims`` inputs are stacked; the rest (e.g. the height map)
        # is appended once, current value only.
        self.history_dims = input_size if history_dims is None else history_dims
        self.hidden_size = history_length * self.history_dims
        self.num_layers = 1
        self.output_size = self.hidden_size + input_size - self.history_dims

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch, _ = x.shape
        c = self.history_dims
        windows, seq, new_hidden = sliding_windows(x[..., :c], hidden, self.history_length)  # (バッチ, 時間, 入力, N)
        flat = windows.transpose(2, 3).reshape(batch, seq_len, self.hidden_size)  # 時間の古い順に観測を並べる
        flat = torch.cat([flat, x.transpose(0, 1)[..., c:]], dim=-1)
        return flat.transpose(0, 1), new_hidden


class TcnMemory(nn.Module):
    """rsl_rl の ``Memory`` と同じ使い方の窓の記憶 (``self.rnn`` が :class:`TcnHistory` か :class:`HistoryStack`)。"""

    def __init__(self, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn
        self.hidden_state: HiddenState = None

    def forward(
        self, input: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        if masks is not None:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to the TCN memory during the policy update")
            out, _ = self.rnn(input, hidden_state)
            return unpad_trajectories(out, masks)
        out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.hidden_state = hidden_state
        elif self.hidden_state is not None:
            self.hidden_state[..., dones == 1, :] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        # 窓は観測そのもので勾配を持たない (rollout は inference_mode)。rsl_rl の呼び出しに合わせるだけ。
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()


def _history_dims(obs: TensorDict, groups: list[str], history_groups: list[str] | None) -> int | None:
    """Width of the leading groups that go through the memory; the groups after them are current-only.

    ``history_groups`` must be a prefix of ``groups`` (the memory slices the first columns)."""
    if history_groups is None:
        return None
    if list(groups[: len(history_groups)]) != list(history_groups):
        raise ValueError(f"history groups {history_groups} must lead the observation groups {groups}")
    return sum(obs[g].shape[-1] for g in history_groups)


class ActorCriticTcn(ActorCriticRecurrent):
    """GRU 版の ``memory_a`` / ``memory_c`` を TCN に置き換えた actor-critic。モジュールの docstring 参照。"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        state_dependent_std: bool = False,
        tcn_history_length: int = 100,
        tcn_channels: int = 34,
        tcn_kernel_size: int = 5,
        tcn_latent_dim: int = 64,
        tcn_concat_current_obs: bool = True,
        tcn_bf16: bool = True,
        critic_history_length: int = 5,
        actor_history_groups: list[str] | None = None,
        critic_history_groups: list[str] | None = None,
        update_chunks: int = 1,
        **kwargs: Any,
    ) -> None:
        if state_dependent_std:
            raise ValueError("ActorCriticTcn does not support state_dependent_std.")
        for k in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            kwargs.pop(k, None)
        self.update_chunks = update_chunks
        """``PPORecurrentSymmetry`` がミニバッチを何個に分けて勾配を足し合わせるか (更新は同じ、GPU メモリのため)。"""
        # GRU 版を作ってから、記憶と MLP だけ差し替える (正規化・行動の分布などは GRU 版のまま)。
        super().__init__(
            obs, obs_groups, num_actions,
            actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims, activation=activation,
            state_dependent_std=False, rnn_type="gru", rnn_hidden_dim=8, rnn_num_layers=1, **kwargs,
        )  # fmt: skip
        tcn = dict(
            history_length=tcn_history_length,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            latent_dim=tcn_latent_dim,
            concat_current_obs=tcn_concat_current_obs,
            bf16=tcn_bf16,
        )
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        actor_dims = _history_dims(obs, obs_groups["policy"], actor_history_groups)
        critic_dims = _history_dims(obs, obs_groups["critic"], critic_history_groups)
        self.memory_a = TcnMemory(TcnHistory(num_actor_obs, history_dims=actor_dims, **tcn))
        self.actor = MLP(self.memory_a.rnn.output_size, num_actions, actor_hidden_dims, activation)
        # 直近 critic_history_length ステップの critic 観測を並べて MLP に入れる
        self.memory_c = TcnMemory(HistoryStack(num_critic_obs, critic_history_length, history_dims=critic_dims))
        self.critic = MLP(self.memory_c.rnn.output_size, 1, critic_hidden_dims, activation)
        print(f"Actor TCN: {self.memory_a.rnn}\nActor MLP: {self.actor}")
        print(f"Critic memory: {self.memory_c.rnn}\nCritic MLP: {self.critic}")


class TcnPolicyJitExporter(nn.Module):
    """デプロイ用の TorchScript 方策 (1 ステップずつ観測を入れ、行動を返す)。

    GRU の書き出し (isaaclab_rl の ``export_policy_as_jit``) と同じく、記憶を ``hidden_state``
    バッファに持つ: 直近 N ステップの観測 (古い順、平らにしたもの)。ros2_ws の制御側は
    ``hidden_state`` があれば再帰方策とみなし、エピソードの開始で 0 にする (学習と同じく前が 0 で
    埋まる)。観測の幅は ``input_dim`` バッファで読める (最初の層の幅は TCN の潜在 + 観測なので)。
    畳み込みは fp32 で計算する (学習の bf16 とは方策の KL 1.6e-10 の差)。

    ``torch.jit.script`` ではなく ``trace`` で書き出す: この環境の torch は ``CausalConv1d`` の
    ``F.pad`` を script にすると segfault する。分岐が無いので trace で同じになり、バッファへの
    書き込みも trace に残る。0 に戻すのは制御側がバッファを直接 ``zero_()`` する。
    """

    def __init__(self, policy: ActorCriticTcn) -> None:
        super().__init__()
        import copy

        tcn = policy.memory_a.rnn
        self.history_length = int(tcn.history_length)
        self.input_size = int(tcn.input_size)
        self.history_dims = int(tcn.history_dims)
        self.concat_current_obs = bool(tcn.concat_current_obs)
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.encoder = copy.deepcopy(tcn.encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.register_buffer("hidden_state", torch.zeros(1, 1, self.history_length * self.history_dims))
        self.register_buffer("input_dim", torch.tensor(self.input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (1, 観測) -> (1, 行動)。"""
        x = self.normalizer(x)
        n, c = self.history_length, self.history_dims
        window = torch.cat([self.hidden_state.view(1, n, c)[:, 1:], x[:, :c].view(1, 1, c)], dim=1)
        self.hidden_state[:] = window.reshape(1, 1, n * c).detach()
        latent = self.encoder(window.transpose(1, 2))
        features = torch.cat([latent, x], dim=-1) if self.concat_current_obs else latent
        return self.actor(features)

    def export(self, path: str, filename: str = "policy.pt") -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        with torch.no_grad():
            traced = torch.jit.trace(self, torch.zeros(1, self.input_size), check_trace=False)
        traced.hidden_state.zero_()
        traced.save(os.path.join(path, filename))


class _TcnOnnxPolicy(nn.Module):
    """ONNX form of :class:`TcnPolicyJitExporter`: the window goes in and out explicitly,
    ``(obs, h_in) -> (actions, h_out)``, as the GRU policies' ``h_in`` / ``h_out``."""

    def __init__(self, jit_like: TcnPolicyJitExporter) -> None:
        super().__init__()
        self.p = jit_like

    def forward(self, x: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.p
        x = p.normalizer(x)
        n, c = p.history_length, p.history_dims
        window = torch.cat([h_in.view(1, n, c)[:, 1:], x[:, :c].view(1, 1, c)], dim=1)
        latent = p.encoder(window.transpose(1, 2))
        features = torch.cat([latent, x], dim=-1) if p.concat_current_obs else latent
        return p.actor(features), window.reshape(1, 1, n * c)


def export_tcn_policy(policy: ActorCriticTcn, path: str) -> None:
    """Write ``policy.pt`` (TorchScript, window in a buffer) and ``policy.onnx`` (window as
    ``h_in`` / ``h_out``, shape (1, 1, N x history_dims)) to ``path``."""
    import os

    jit = TcnPolicyJitExporter(policy)
    jit.export(path, filename="policy.pt")
    onnx_model = _TcnOnnxPolicy(TcnPolicyJitExporter(policy)).to("cpu").eval()
    x = torch.zeros(1, jit.input_size)
    h = torch.zeros(1, 1, jit.history_length * jit.history_dims)
    torch.onnx.export(
        onnx_model, (x, h), os.path.join(path, "policy.onnx"),
        input_names=["obs", "h_in"], output_names=["actions", "h_out"], opset_version=18, dynamo=False,
    )  # fmt: skip
