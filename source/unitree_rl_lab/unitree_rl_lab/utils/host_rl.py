"""HoST's multi-critic PPO, ported to Isaac Lab's RslRlVecEnvWrapper.

Port of OpenRobotLab/HoST ``rsl_rl`` (MIT): ``modules/actor_critic.py``,
``storage/rollout_storage.py``, ``algorithms/ppo.py``, ``runners/on_policy_runner.py``.
What HoST changes relative to plain rsl_rl, and so what this file exists for:

* one critic per reward group; each group's advantage is normalized separately and the
  groups are mixed with ``reward_group_weights`` (e.g. task / regu / style / target =
  2.5 / 0.1 / 1 / 1), so small regularization rewards are not drowned by the task reward;
* a Tanh on the actor output (action means stay in [-1, 1]);
* an L2C2-style smoothness loss: policy and value outputs should change little between an
  observation and a random interpolation towards the next one.

Not in HoST: an optional left/right symmetry loss (``symmetry_fn`` in the algorithm cfg, a
``"module:function"`` returning ``(mirror_obs, mirror_actions)`` for the env), pulling the
policy's mean action for the mirrored observation towards the mirrored mean action.

Group rewards come from the env's reward manager: every term whose name starts with
``<group>_`` belongs to that group. The ``task`` group uses the raw per-step value (HoST
does not scale it by dt); the other groups are scaled by dt, as HoST does for them.

Used by ``scripts/rsl_rl/train.py`` / ``play.py`` when the agent cfg's ``class_name`` is
``"HoSTRunner"``. The policy is exported through ``alg.policy.actor`` like rsl_rl's.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation, store_code_state, string_to_callable


class MultiCriticActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_critics: int,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256),
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__()

        def mlp(in_dim, dims, out_dim, out_act=None):
            layers, d = [], in_dim
            for h in dims:
                layers += [nn.Linear(d, h), resolve_nn_activation(activation)]
                d = h
            layers.append(nn.Linear(d, out_dim))
            if out_act is not None:
                layers.append(out_act)
            return nn.Sequential(*layers)

        self.actor = mlp(num_actor_obs, actor_hidden_dims, num_actions, nn.Tanh())
        self.critics = nn.ModuleList([mlp(num_critic_obs, critic_hidden_dims, 1) for _ in range(num_critics)])
        self.num_critics = num_critics
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP (x{num_critics}): {self.critics[0]}")

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        return self.actor(obs)

    def evaluate(self, critic_obs, **kwargs):
        return torch.cat([critic(critic_obs) for critic in self.critics], dim=-1)


class MultiCriticRolloutStorage:
    def __init__(self, num_envs, num_steps, obs_dim, critic_obs_dim, action_dim, num_critics, group_weights, device):
        self.device = device
        self.num_envs, self.num_steps, self.num_critics = num_envs, num_steps, num_critics
        self.group_weights = torch.tensor(group_weights, device=device).view(1, 1, -1)
        z = lambda *shape: torch.zeros(num_steps, num_envs, *shape, device=device)  # noqa: E731
        self.observations = z(obs_dim)
        self.critic_observations = z(critic_obs_dim)
        self.actions = z(action_dim)
        self.rewards = z(num_critics)
        self.dones = z(1)
        self.actions_log_prob = z(1)
        self.values = z(num_critics)
        self.returns = z(num_critics)
        self.advantages = z(num_critics)
        self.mu = z(action_dim)
        self.sigma = z(action_dim)
        self.step = 0

    def add(self, obs, critic_obs, actions, rewards, dones, values, log_prob, mu, sigma):
        if self.step >= self.num_steps:
            raise AssertionError("Rollout buffer overflow")
        i = self.step
        self.observations[i].copy_(obs)
        self.critic_observations[i].copy_(critic_obs)
        self.actions[i].copy_(actions)
        self.rewards[i].copy_(rewards)
        self.dones[i].copy_(dones.view(-1, 1))
        self.values[i].copy_(values)
        self.actions_log_prob[i].copy_(log_prob.view(-1, 1))
        self.mu[i].copy_(mu)
        self.sigma[i].copy_(sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step]
            delta = self.rewards[step] + not_done * gamma * next_values - self.values[step]
            advantage = delta + not_done * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        # normalize each critic's advantage separately, then mix with the group weights
        adv = self.returns - self.values
        adv = (adv - adv.mean(dim=(0, 1), keepdim=True)) / (adv.std(dim=(0, 1), keepdim=True) + 1e-8)
        self.advantages = adv
        self.mixed_advantages = torch.sum(adv * self.group_weights, dim=-1)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        # the last transition has no next observation, so it is dropped (as in HoST)
        batch_size = self.num_envs * (self.num_steps - 1)
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        flat = lambda x: x[:-1].flatten(0, 1)  # noqa: E731
        obs, critic_obs = flat(self.observations), flat(self.critic_observations)
        next_obs = self.observations[1:].flatten(0, 1)
        actions, values, returns = flat(self.actions), flat(self.values), flat(self.returns)
        log_prob, advantages = flat(self.actions_log_prob), flat(self.mixed_advantages)
        mu, sigma = flat(self.mu), flat(self.sigma)
        not_dones = 1.0 - flat(self.dones)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                b = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    obs[b], critic_obs[b], next_obs[b], not_dones[b], actions[b], values[b],
                    advantages[b], returns[b], log_prob[b], mu[b], sigma[b],
                )  # fmt: skip


class MultiCriticPPO:
    def __init__(
        self,
        policy: MultiCriticActorCritic,
        reward_group_weights,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        value_smoothness_coef=0.1,
        smoothness_upper_bound=1.0,
        smoothness_lower_bound=0.1,
        mirror_fns=None,
        symmetry_coef=1.0,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.policy = policy.to(device)
        self.actor_critic = self.policy
        self.reward_group_weights = reward_group_weights
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
        self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.max_grad_norm, self.use_clipped_value_loss = max_grad_norm, use_clipped_value_loss
        self.schedule, self.desired_kl = schedule, desired_kl
        # HoST: epsilon = lower / (upper - lower); policy coef = upper * epsilon
        epsilon = smoothness_lower_bound / (smoothness_upper_bound - smoothness_lower_bound)
        self.policy_smooth_coef = smoothness_upper_bound * epsilon
        self.value_smooth_coef = value_smoothness_coef * self.policy_smooth_coef
        self.mirror_fns = mirror_fns
        self.symmetry_coef = symmetry_coef
        self.storage: MultiCriticRolloutStorage | None = None
        self._step = {}

    def init_storage(self, num_envs, num_steps, obs_dim, critic_obs_dim, action_dim):
        self.storage = MultiCriticRolloutStorage(
            num_envs, num_steps, obs_dim, critic_obs_dim, action_dim, self.policy.num_critics,
            self.reward_group_weights, self.device,
        )  # fmt: skip

    def act(self, obs, critic_obs):
        actions = self.policy.act(obs).detach()
        self._step = dict(
            obs=obs,
            critic_obs=critic_obs,
            actions=actions,
            values=self.policy.evaluate(critic_obs).detach(),
            log_prob=self.policy.get_actions_log_prob(actions).detach(),
            mu=self.policy.action_mean.detach(),
            sigma=self.policy.action_std.detach(),
        )
        return actions

    def process_env_step(self, rewards, dones, time_outs):
        s = self._step
        rewards = rewards.clone()
        if time_outs is not None:
            # bootstrap on time-outs (per critic)
            rewards += self.gamma * s["values"] * time_outs.unsqueeze(1).to(self.device)
        self.storage.add(
            s["obs"], s["critic_obs"], s["actions"], rewards, dones.float(), s["values"], s["log_prob"],
            s["mu"], s["sigma"],
        )  # fmt: skip

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        stats = {"value_function": 0.0, "surrogate": 0.0, "smoothness": 0.0, "entropy": 0.0}
        if self.mirror_fns is not None:
            stats["symmetry"] = 0.0
        for (
            obs_b, critic_obs_b, next_obs_b, cont_b, actions_b, target_values_b, advantages_b, returns_b,
            old_log_prob_b, old_mu_b, old_sigma_b,
        ) in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):  # fmt: skip
            self.policy.act(obs_b)
            log_prob_b = self.policy.get_actions_log_prob(actions_b)
            value_b = self.policy.evaluate(critic_obs_b)
            mu_b, sigma_b, entropy_b = self.policy.action_mean, self.policy.action_std, self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_b / old_sigma_b + 1.0e-5)
                        + (torch.square(old_sigma_b) + torch.square(old_mu_b - mu_b)) / (2.0 * torch.square(sigma_b))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate

            ratio = torch.exp(log_prob_b - old_log_prob_b.squeeze(-1))
            adv = advantages_b.squeeze()
            surrogate = torch.max(-adv * ratio, -adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_b + (value_b - target_values_b).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((value_b - returns_b).pow(2), (value_clipped - returns_b).pow(2)).mean()
            else:
                value_loss = (returns_b - value_b).pow(2).mean()

            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy_b.mean()

            # smoothness: random interpolation towards the next observation (not across resets)
            mix = cont_b * (torch.rand_like(cont_b) - 0.5) * 2.0
            mix_obs = obs_b + mix * (next_obs_b - obs_b)
            policy_smooth = torch.square(torch.norm(mu_b - self.policy.act_inference(mix_obs), dim=-1)).mean()
            value_smooth = torch.square(torch.norm(value_b - self.policy.evaluate(mix_obs), dim=-1)).mean()
            smooth_loss = self.policy_smooth_coef * policy_smooth + self.value_smooth_coef * value_smooth
            loss = loss + smooth_loss

            if self.mirror_fns is not None:
                mirror_obs, mirror_actions = self.mirror_fns
                sym_loss = torch.square(
                    self.policy.act_inference(mirror_obs(obs_b)) - mirror_actions(mu_b).detach()
                ).sum(dim=-1).mean()
                loss = loss + self.symmetry_coef * sym_loss
                stats["symmetry"] += sym_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["value_function"] += value_loss.item()
            stats["surrogate"] += surrogate.item()
            stats["smoothness"] += smooth_loss.item()
            stats["entropy"] += entropy_b.mean().item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {k: v / n for k, v in stats.items()}


class HoSTRunner:
    """Drop-in for rsl_rl's ``OnPolicyRunner`` (train/play scripts) running HoST's PPO."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        alg_cfg = dict(train_cfg["algorithm"])
        policy_cfg = dict(train_cfg["policy"])

        self.reward_groups: list[str] = list(alg_cfg.pop("reward_groups"))
        group_weights = list(alg_cfg.pop("reward_group_weights"))
        assert len(group_weights) == len(self.reward_groups)
        self.critic_obs_group = alg_cfg.pop("critic_obs_group", "policy")
        symmetry_fn = alg_cfg.pop("symmetry_fn", None)
        mirror_fns = string_to_callable(symmetry_fn)(self.env) if symmetry_fn else None
        if mirror_fns is not None:
            print(f"[HoSTRunner] symmetry loss on ({symmetry_fn}), coef {alg_cfg.get('symmetry_coef', 1.0)}")
        self._resolve_reward_groups()

        obs = self.env.get_observations()
        num_obs = obs["policy"].shape[1]
        num_critic_obs = obs[self.critic_obs_group].shape[1]
        policy = MultiCriticActorCritic(
            num_obs, num_critic_obs, self.env.num_actions, len(self.reward_groups), **policy_cfg
        )
        for key in ("class_name",):
            alg_cfg.pop(key, None)
        self.alg = MultiCriticPPO(policy, group_weights, mirror_fns=mirror_fns, device=device, **alg_cfg)
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, num_obs, num_critic_obs, self.env.num_actions)

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.git_status_repos = []

    # -- rewards

    def _resolve_reward_groups(self):
        rm = self.env.unwrapped.reward_manager
        names = rm.active_terms
        self._group_masks = torch.zeros(len(self.reward_groups), len(names), device=self.device)
        unassigned = []
        for j, name in enumerate(names):
            group = name.split("_")[0]
            if group not in self.reward_groups:
                unassigned.append(name)
                continue
            self._group_masks[self.reward_groups.index(group), j] = 1.0
        if unassigned:
            raise ValueError(f"Reward terms without a group prefix {self.reward_groups}: {unassigned}")
        dt = self.env.unwrapped.step_dt
        # task: raw per-step value; other groups: scaled by dt (HoST)
        self._group_scale = torch.tensor(
            [1.0 if g == "task" else dt for g in self.reward_groups], device=self.device
        )
        print("[HoSTRunner] reward groups:")
        for g in self.reward_groups:
            print(f"  {g}: {[n for n in names if n.split('_')[0] == g]}")

    def _group_rewards(self) -> torch.Tensor:
        step_reward = self.env.unwrapped.reward_manager._step_reward  # (N, terms), weight * value
        return (step_reward @ self._group_masks.T) * self._group_scale

    # -- training

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            store_code_state(self.log_dir, self.git_status_repos)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_td = self.env.get_observations().to(self.device)
        obs, critic_obs = obs_td["policy"], obs_td[self.critic_obs_group]
        self.alg.policy.train()

        ep_infos = []
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs_td, _, dones, infos = self.env.step(actions.to(self.env.device))
                    rewards = self._group_rewards()
                    obs_td, dones = obs_td.to(self.device), dones.to(self.device)
                    obs, critic_obs = obs_td["policy"], obs_td[self.critic_obs_group]
                    self.alg.process_env_step(rewards, dones, infos.get("time_outs"))
                    if "log" in infos:
                        ep_infos.append(infos["log"])
                    cur_reward_sum += rewards.sum(dim=-1)
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self._log(it, tot_iter, collection_time, learn_time, loss_dict, ep_infos, rewbuffer, lenbuffer)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(self, it, tot_iter, collection_time, learn_time, loss_dict, ep_infos, rewbuffer, lenbuffer):
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += collection_time + learn_time
        fps = int(collection_size / (collection_time + learn_time))
        w = self.writer
        lines = []
        if ep_infos:
            for key in ep_infos[0]:
                vals = [torch.as_tensor(ep[key], device=self.device).float().view(-1) for ep in ep_infos if key in ep]
                value = torch.cat(vals).mean().item()
                tag = key if "/" in key else "Episode/" + key
                w.add_scalar(tag, value, it)
                lines.append(f"{key:>35}: {value:.4f}")
        for k, v in loss_dict.items():
            w.add_scalar(f"Loss/{k}", v, it)
        w.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        mean_std = self.alg.policy.std.mean().item()
        w.add_scalar("Policy/mean_noise_std", mean_std, it)
        w.add_scalar("Perf/total_fps", fps, it)
        if rewbuffer:
            w.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            w.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
        self._iters_done = getattr(self, "_iters_done", 0) + 1
        avg_it_time = self.tot_time / self._iters_done
        msg = [
            "#" * 80,
            f"{'Learning iteration ' + str(it) + '/' + str(tot_iter):^80}",
            "",
            f"{'Computation:':>35} {fps} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)",
            f"{'Mean action noise std:':>35} {mean_std:.2f}",
        ]
        msg += [f"{'Mean ' + k + ' loss:':>35} {v:.4f}" for k, v in loss_dict.items()]
        if rewbuffer:
            msg += [
                f"{'Mean reward (all groups):':>35} {statistics.mean(rewbuffer):.2f}",
                f"{'Mean episode length:':>35} {statistics.mean(lenbuffer):.2f}",
            ]
        msg += lines
        msg += [
            "-" * 80,
            f"{'Total timesteps:':>35} {self.tot_timesteps}",
            f"{'Iteration time:':>35} {collection_time + learn_time:.2f}s",
            f"{'Time elapsed:':>35} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
            f"{'ETA:':>35} {time.strftime('%H:%M:%S', time.gmtime(avg_it_time * (tot_iter - it - 1)))}",
        ]
        print("\n".join(msg))

    # -- io / inference (same surface as rsl_rl's OnPolicyRunner)

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.policy.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = loaded.get("iter", 0)
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.policy.eval()
        if device is not None:
            self.alg.policy.to(device)

        def policy(obs_td):
            return self.alg.policy.act_inference(obs_td["policy"])

        return policy

    def eval_mode(self):
        self.alg.policy.eval()

    def train_mode(self):
        self.alg.policy.train()

    def add_git_repo_to_log(self, repo_file_path: str):
        self.git_status_repos.append(repo_file_path)
