"""PPO + 対称性のデータ拡張を、再帰 (GRU / LSTM) 方策でも使えるようにしたもの。

rsl_rl 3.1.2 の :class:`rsl_rl.algorithms.PPO` は、``symmetry_cfg`` があって方策が
``ActorCriticRecurrent`` だと例外で止まる (最新の 5.5.2 も同じ)。止まる理由は 2 つ:

1. 再帰のミニバッチは [時間, 軌跡, ...] の形で、rsl_rl は反転分を先頭 (時間) の次元に
   つなげてしまう。
2. 反転した観測の列を GRU に通すと隠れ状態も別物になる。ミニバッチは収集中に保存した
   隠れ状態から始まるが、反転側の初期隠れ状態はどこにも無い。

ここでは次のようにする。損失の作り方 (確率比の分母とアドバンテージに元のサンプルの値を
使う、Mittal et al. の式 6) と ``symmetry_cfg`` の意味は rsl_rl と同じ。

* 収集中: 各ステップで、反転した観測でも actor / critic の RNN を回して反転側の隠れ状態を
  進める (行動の計算には使わない)。終了した環境はゼロに戻す。
* 隠れ状態 (元・反転側とも) は **rollout の最初のステップの分だけ保存する**。更新で使うのは
  軌跡の始まりの隠れ状態だけで、始まりは rollout の最初のステップか、リセット直後 (0) の
  どちらかだから。rsl_rl はステップごとに保存するが、TCN (``ActorCriticTcn``) の隠れ状態は
  窓 (100 ステップ x 観測) なので、ステップごとだと 10 GB を超える。リセット直後が 0 という
  前提が崩れていないかは収集中に見ていて、崩れていれば更新の前に例外で止める。
* 更新: 反転分を軌跡の次元 (1 次元目) につなげる。初期隠れ状態は [元, 反転側]、マスクと
  確率比の分母・アドバンテージ・リターン・価値の目標は複製する。
* ミラー損失 (反転した観測での行動の平均と、元の行動の平均を反転したものの MSE) は毎回計算して
  ``Loss/symmetry`` に記録する。``use_mirror_loss`` が True のときだけ損失に足す。

あわせて rsl_rl 本家で後から直されたバグを 2 つ直す:

* #206: ``compute_returns`` の最後の価値の評価で critic の隠れ状態が 1 ステップ余計に進む
  (次の収集の最初の観測を 2 回読んだことになる)。評価の前後で隠れ状態を戻す。
* #249: 代理損失の ``torch.squeeze()`` が、ミニバッチの次元が 1 のとき潰しすぎる。
  ``squeeze(-1)`` にする。

RND とマルチ GPU の対称化は扱わない (使うと例外)。

地図入りの方策 (``ActorCriticPerceptiveRecurrent``) にも使える。反転側の RNN には、方策と同じく
固有感覚だけを入れる (``_memory_input``)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.utils import string_to_callable


def _as_tuple(h) -> tuple[torch.Tensor, ...]:
    return h if isinstance(h, tuple) else (h,)


def _from_tuple(hs: tuple[torch.Tensor, ...]):
    return hs[0] if len(hs) == 1 else tuple(hs)


class PPORecurrentSymmetry(PPO):
    """再帰方策用の対称性データ拡張つき PPO。設定は rsl_rl の ``symmetry_cfg`` と同じ。"""

    policy: ActorCriticRecurrent

    def __init__(self, policy, symmetry_cfg: dict | None = None, **kwargs):
        if symmetry_cfg is None:
            raise ValueError("PPORecurrentSymmetry needs a symmetry_cfg (use PPO otherwise).")
        # 親は再帰方策 + symmetry_cfg で例外にするので、対称性の設定は自分で持つ。
        super().__init__(policy, symmetry_cfg=None, **kwargs)
        if not policy.is_recurrent:
            raise ValueError("PPORecurrentSymmetry is for recurrent policies; rsl_rl's PPO handles the others.")
        if getattr(policy, "state_dependent_std", False):
            raise ValueError("state_dependent_std is not supported.")
        if self.rnd is not None:
            raise ValueError("RND is not supported together with recurrent symmetry.")
        if self.is_multi_gpu:
            raise ValueError("Multi-GPU training is not supported by PPORecurrentSymmetry.")
        func = symmetry_cfg["data_augmentation_func"]
        if isinstance(func, str):
            func = string_to_callable(func)
        if not callable(func):
            raise ValueError(f"symmetry_cfg.data_augmentation_func is not callable: {func}")
        symmetry_cfg["data_augmentation_func"] = func
        self.symmetry = symmetry_cfg
        if not (symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]):
            print("[PPORecurrentSymmetry] Symmetry not used for learning, only logged.")

        # 反転側の隠れ状態 (収集中に進める) と、rollout の最初のステップの保存先。
        self._mirror_hidden: list = [None, None]  # (actor, critic)、None はゼロ
        self._saved_mirror_hidden: list[list[torch.Tensor]] | None = None
        # リセット直後の隠れ状態が 0 でなかったか (収集中に積み、更新の前に確かめる)。
        self._nonzero_after_reset: torch.Tensor | None = None
        self._last_dones: torch.Tensor | None = None

    # ------------------------------------------------------------------ 反転
    def _mirror(self, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
        """反転した分だけを返す (先頭の次元は入力と同じ)。"""
        obs_aug, act_aug = self.symmetry["data_augmentation_func"](
            obs=obs, actions=actions, env=self.symmetry["_env"]
        )
        obs_m = None if obs_aug is None else obs_aug[obs.batch_size[0] :]
        act_m = None if act_aug is None else act_aug[actions.shape[0] :]
        return obs_m, act_m

    def _mirror_padded(self, obs: TensorDict) -> TensorDict:
        """[時間, 軌跡] の観測を反転する。"""
        t, b = obs.batch_size[:2]
        obs_m, _ = self._mirror(obs=obs.reshape(-1))
        return obs_m.reshape(t, b)

    def _mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = actions.shape
        _, act_m = self._mirror(actions=actions.reshape(-1, shape[-1]))
        return act_m.reshape(shape)

    # ------------------------------------------------------------------ 収集
    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)

        def zeros_like_hidden() -> list[list[torch.Tensor]]:
            out = []
            for memory in (self.policy.memory_a, self.policy.memory_c):
                rnn = memory.rnn
                n_states = 2 if isinstance(rnn, nn.LSTM) else 1
                out.append(
                    [torch.zeros(rnn.num_layers, num_envs, rnn.hidden_size, device=self.device) for _ in range(n_states)]
                )
            return out

        # 元の隠れ状態も rollout の最初のステップだけ保存する (rsl_rl の _save_hidden_states を差し替える)。
        self._saved_hidden = zeros_like_hidden()  # [actor, critic]、各 [層, 環境, H]
        self._saved_mirror_hidden = zeros_like_hidden()
        self._nonzero_after_reset = torch.zeros((), dtype=torch.bool, device=self.device)
        self.storage._save_hidden_states = self._save_first_hidden_states

    def _save_first_hidden_states(self, hidden_states) -> None:
        """rsl_rl の ``RolloutStorage._save_hidden_states`` の代わり。最初のステップだけ保存する。"""
        step = self.storage.step
        for saved, hidden in zip(self._saved_hidden, hidden_states):
            if step == 0:
                for buf, h in zip(saved, _as_tuple(hidden) if hidden is not None else (None,) * len(saved)):
                    buf.zero_() if h is None else buf.copy_(h)
            elif hidden is not None and self._last_dones is not None:
                # 前のステップで終わった環境 = このステップが軌跡の始まり。隠れ状態は 0 のはず。
                done = self._last_dones
                for h in _as_tuple(hidden):
                    self._nonzero_after_reset |= (h[..., done, :] != 0).any()

    def act(self, obs: TensorDict) -> torch.Tensor:
        # このステップの観測を読む前の反転側の隠れ状態を、rollout の最初のステップだけ保存する。
        if self.storage.step == 0:
            for saved, hidden in zip(self._saved_mirror_hidden, self._mirror_hidden):
                if hidden is None:
                    for buf in saved:
                        buf.zero_()
                else:
                    for buf, h in zip(saved, _as_tuple(hidden)):
                        buf.copy_(h)
        elif self._last_dones is not None:
            done = self._last_dones
            for hidden in self._mirror_hidden:
                if hidden is not None:
                    for h in _as_tuple(hidden):
                        self._nonzero_after_reset |= (h[..., done, :] != 0).any()

        actions = super().act(obs)

        # 反転した観測で actor / critic の RNN を 1 ステップ進める。
        obs_m, _ = self._mirror(obs=obs)
        policy = self.policy
        x_a = self._memory_input(obs_m, actor=True).unsqueeze(0)
        x_c = self._memory_input(obs_m, actor=False).unsqueeze(0)
        _, h_a = policy.memory_a.rnn(x_a, self._mirror_hidden[0])
        _, h_c = policy.memory_c.rnn(x_c, self._mirror_hidden[1])
        self._mirror_hidden = [h_a, h_c]
        return actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        super().process_env_step(obs, rewards, dones, extras)
        done = dones == 1
        self._last_dones = done.clone()
        for hidden in self._mirror_hidden:
            if hidden is not None:
                for h in _as_tuple(hidden):
                    h[..., done, :] = 0.0

    def compute_returns(self, obs: TensorDict) -> None:
        # rsl_rl #206: 最後の価値の評価で critic の隠れ状態を進めない。
        memory_c = self.policy.memory_c
        hidden = memory_c.hidden_state
        last_values = self.policy.evaluate(obs).detach()
        memory_c.hidden_state = hidden
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    # ------------------------------------------------------------------ 更新
    def _mini_batches(self):
        """``RolloutStorage.recurrent_mini_batch_generator`` と同じ切り方に、反転側の隠れ状態を足したもの。"""
        st = self.storage
        from rsl_rl.utils import split_and_pad_trajectories

        padded_obs, traj_masks = split_and_pad_trajectories(st.observations, st.dones)
        dones = st.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        starts = last_was_done.permute(1, 0)  # [環境, 時間]: 軌跡の始まり
        # 軌跡は [環境, 時間] の行優先の順に並ぶ (split_and_pad_trajectories と同じ)。
        traj_env, traj_t = starts.nonzero(as_tuple=True)
        from_first_step = traj_t == 0

        if bool(self._nonzero_after_reset):
            raise RuntimeError(
                "PPORecurrentSymmetry: a hidden state right after a reset was not zero, so it cannot be rebuilt "
                "from the first step only. Did the policy's reset() change?"
            )

        def traj_hidden(saved: list[torch.Tensor], first: int, last: int):
            # 軌跡の始まりの隠れ状態 [層, 軌跡, H]: rollout の最初のステップなら保存した値、それ以外は 0。
            out = []
            sel = slice(first, last)
            env, first_step = traj_env[sel], from_first_step[sel]
            for h in saved:  # [層, 環境, H]
                o = h.new_zeros(h.shape[0], env.shape[0], h.shape[2])
                o[:, first_step] = h[:, env[first_step]]
                out.append(o)
            return _from_tuple(tuple(out))

        mini_batch_size = st.num_envs // self.num_mini_batches
        for _ in range(self.num_learning_epochs):
            first = 0
            for i in range(self.num_mini_batches):
                start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
                last = first + int(torch.sum(last_was_done[:, start:stop]))
                yield {
                    "obs": padded_obs[:, first:last],
                    "masks": traj_masks[:, first:last],
                    "actions": st.actions[:, start:stop],
                    "values": st.values[:, start:stop],
                    "returns": st.returns[:, start:stop],
                    "advantages": st.advantages[:, start:stop],
                    "old_log_prob": st.actions_log_prob[:, start:stop],
                    "old_mu": st.mu[:, start:stop],
                    "old_sigma": st.sigma[:, start:stop],
                    "h_a": traj_hidden(self._saved_hidden[0], first, last),
                    "h_c": traj_hidden(self._saved_hidden[1], first, last),
                    "h_a_m": traj_hidden(self._saved_mirror_hidden[0], first, last),
                    "h_c_m": traj_hidden(self._saved_mirror_hidden[1], first, last),
                    "traj_per_env": last_was_done[:, start:stop].sum(0),
                }
                first = last

    def _memory_input(self, obs: TensorDict, actor: bool) -> torch.Tensor:
        """RNN に入る観測 (正規化後)。

        地図入りの方策 (:class:`~tsubame_isaac.models.actor_critic_perceptive (tsubame_isaac_env).ActorCriticPerceptiveRecurrent`)
        では、RNN に入るのは固有感覚のグループだけ (``extero_to_memory`` のときは地図の潜在も後ろに付く)。
        ``get_actor_obs`` は地図まで連結するので、そのまま RNN に入れると幅が合わない。
        """
        p = self.policy
        if not hasattr(p, "actor_proprio_groups"):
            if actor:
                return p.actor_obs_normalizer(p.get_actor_obs(obs))
            return p.critic_obs_normalizer(p.get_critic_obs(obs))
        side = "actor" if actor else "critic"
        x = getattr(p, f"{side}_obs_normalizer")(p._cat(obs, getattr(p, f"{side}_proprio_groups")))
        encoder = getattr(p, f"{side}_extero_encoder")
        if encoder is not None and p.extero_to_memory:
            extero = getattr(p, f"{side}_extero_normalizer")(p._cat(obs, getattr(p, f"{side}_extero_groups")))
            x = torch.cat([x, encoder(extero)], dim=-1)
        return x

    def _actor_mean(self, obs: TensorDict, masks: torch.Tensor, hidden) -> torch.Tensor:
        policy = self.policy
        if hasattr(policy, "_actor_features"):
            # 地図入りの方策: 地図は RNN の横から actor に入るので、方策自身の経路で計算する。
            return policy.actor(policy._actor_features(obs, masks, hidden))
        x = self._memory_input(obs, actor=True)
        return policy.actor(policy.memory_a(x, masks, hidden))

    def _chunks(self, b: dict, advantages: torch.Tensor):
        """ミニバッチを環境の範囲で ``update_chunks`` 個に分ける。

        軌跡は環境の順に並んでいるので、環境の範囲 [e0, e1) は軌跡の連続した範囲に対応する。
        各片の重み = 片の環境数 / ミニバッチの環境数 (損失はどれも環境 x 時間の平均なので、
        重み付きの和が分けないときの値と一致する)。
        """
        k = max(1, int(getattr(self.policy, "update_chunks", 1)))
        n = b["actions"].shape[1]
        if k == 1:
            yield b, advantages, 1.0
            return
        traj_cum = torch.cat([b["traj_per_env"].new_zeros(1), torch.cumsum(b["traj_per_env"], 0)]).tolist()
        bounds = [round(i * n / k) for i in range(k + 1)]
        for e0, e1 in zip(bounds[:-1], bounds[1:]):
            if e1 <= e0:
                continue
            t0, t1 = traj_cum[e0], traj_cum[e1]
            c = {}
            for key, v in b.items():
                if key in ("obs", "masks"):
                    c[key] = v[:, t0:t1]
                elif key in ("h_a", "h_c", "h_a_m", "h_c_m"):
                    c[key] = _from_tuple(tuple(h[:, t0:t1] for h in _as_tuple(v)))
                elif key == "traj_per_env":
                    c[key] = v[e0:e1]
                else:
                    c[key] = v[:, e0:e1]
            yield c, advantages[:, e0:e1], (e1 - e0) / n

    def update(self) -> dict[str, float]:  # noqa: C901
        """rsl_rl の PPO の更新と同じ。ただしミニバッチを ``update_chunks`` 個に分けて勾配を足し合わせ、
        1 回だけ step する (方策の ``update_chunks``、既定 1 = 分けない)。分けても数学的には同じ更新で、
        TCN の窓の組み立てと畳み込みの中間値がミニバッチ全体では GPU に載らないためのもの。"""
        sym = self.symmetry
        use_aug = sym["use_data_augmentation"]
        mean_value_loss = mean_surrogate_loss = mean_entropy = mean_symmetry_loss = 0.0
        skipped = 0

        for b in self._mini_batches():
            advantages_all = b["advantages"]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

            self.optimizer.zero_grad()
            kl_mean = 0.0
            for c, advantages, w in self._chunks(b, advantages_all):
                losses, kl_c = self._chunk_loss(c, advantages, use_aug, sym)
                (losses["loss"] * w).backward()
                kl_mean += kl_c * w
                mean_value_loss += losses["value"].item() * w
                mean_surrogate_loss += losses["surrogate"].item() * w
                mean_entropy += losses["entropy"].item() * w
                mean_symmetry_loss += losses["symmetry"].item() * w

            if self.desired_kl is not None and self.schedule == "adaptive":
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif self.desired_kl / 2.0 > kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # A non-finite gradient would write NaN into every weight (std included) and kill the run
            # at the next sample (Blind-TCN Phase 5 with the wall penalties off: twice, ~800 iterations
            # apart). Drop that mini-batch's step instead; Adam's moments stay untouched.
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad()
                skipped += 1
                print(f"[WARN] non-finite gradient (norm={grad_norm.item()}), mini-batch step skipped")
                continue
            self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "symmetry": mean_symmetry_loss / num_updates,
            "nonfinite_skips": float(skipped),
        }

    def _chunk_loss(self, b: dict, advantages: torch.Tensor, use_aug: bool, sym: dict):
        """1 片の損失 (片の中の平均) と KL の平均。"""
        n = b["actions"].shape[1]  # 元の環境数
        obs, masks, actions = b["obs"], b["masks"], b["actions"]
        h_a, h_c = b["h_a"], b["h_c"]
        old_log_prob, values, returns = b["old_log_prob"], b["values"], b["returns"]
        obs_m = self._mirror_padded(obs)
        if use_aug:
            obs = torch.cat([obs, obs_m], dim=1)
            masks = torch.cat([masks, masks], dim=1)
            actions = torch.cat([actions, self._mirror_actions(actions)], dim=1)
            h_a = _from_tuple(tuple(torch.cat(p, dim=1) for p in zip(_as_tuple(h_a), _as_tuple(b["h_a_m"]))))
            h_c = _from_tuple(tuple(torch.cat(p, dim=1) for p in zip(_as_tuple(h_c), _as_tuple(b["h_c_m"]))))
            # 式 6: 反転したサンプルにも元の確率・アドバンテージを使う。
            old_log_prob = torch.cat([old_log_prob, old_log_prob], dim=1)
            advantages = torch.cat([advantages, advantages], dim=1)
            values = torch.cat([values, values], dim=1)
            returns = torch.cat([returns, returns], dim=1)

        self.policy.act(obs, masks=masks, hidden_state=h_a)
        log_prob = self.policy.get_actions_log_prob(actions)
        value = self.policy.evaluate(obs, masks=masks, hidden_state=h_c)
        action_mean = self.policy.action_mean
        # KL とエントロピーは元のサンプルだけで見る (rsl_rl と同じ)。
        mu = action_mean[:, :n]
        sigma = self.policy.action_std[:, :n]
        entropy = self.policy.entropy[:, :n]

        kl_mean = 0.0
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma / b["old_sigma"] + 1.0e-5)
                    + (torch.square(b["old_sigma"]) + torch.square(b["old_mu"] - mu)) / (2.0 * torch.square(sigma))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = torch.mean(kl).item()

        # 代理損失 (rsl_rl #249: squeeze(-1))
        ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
        adv = advantages.squeeze(-1)
        surrogate = -adv * ratio
        surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = values + (value - values).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.max((value - returns).pow(2), (value_clipped - returns).pow(2)).mean()
        else:
            value_loss = (returns - value).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        # ミラー損失: 反転した観測での行動の平均 vs 元の行動の平均を反転したもの (目標側は勾配を止める)。
        with torch.set_grad_enabled(sym["use_mirror_loss"]):
            if use_aug:
                mean_mirrored_obs = action_mean[:, n:]
            else:
                mean_mirrored_obs = self._actor_mean(obs_m, b["masks"], b["h_a_m"])
            target = self._mirror_actions(mu.detach())
            symmetry_loss = nn.functional.mse_loss(mean_mirrored_obs, target)
        if sym["use_mirror_loss"]:
            loss = loss + sym["mirror_loss_coeff"] * symmetry_loss

        return {
            "loss": loss,
            "value": value_loss.detach(),
            "surrogate": surrogate_loss.detach(),
            "entropy": entropy.mean().detach(),
            "symmetry": symmetry_loss.detach(),
        }, kl_mean
