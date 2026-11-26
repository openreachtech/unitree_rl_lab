"""Multi-critic PPO for rsl-rl.

"Multi-critic Learning for Whole-body End-effector Twist Tracking" (3.3): one actor, one critic per
reward group. Each critic learns the value of its own group's reward; GAE runs per group, each
group's advantage is normalized on its own, and the policy is updated with their sum. The relative
scale of the groups' rewards therefore does not matter, only the scale within a group.

The rewards come from :class:`unitree_rl_lab.envs.multi_critic_env.MultiCriticRLEnv`, in
``extras["reward_groups"]``; the scalar reward the runner receives is only used for logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from rsl_rl.storage import RolloutStorage


class MultiCriticActorCritic(ActorCritic):
    """``ActorCritic`` with one critic MLP per reward group; ``evaluate`` returns (N, num_critics)."""

    def __init__(self, obs, obs_groups, num_actions, critic_names: list[str] | None = None, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if not critic_names:
            raise ValueError("MultiCriticActorCritic needs critic_names (set by the runner from the env).")
        self.critic_names = list(critic_names)
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        dims = kwargs.get("critic_hidden_dims", [256, 256, 256])
        activation = kwargs.get("activation", "elu")
        del self.critic
        self.critics = nn.ModuleList(MLP(num_critic_obs, 1, dims, activation) for _ in self.critic_names)
        print(f"Critics {self.critic_names}: {self.critics[0]}")

    @property
    def num_critics(self) -> int:
        return len(self.critics)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return torch.cat([critic(obs) for critic in self.critics], dim=-1)


class MultiCriticRolloutStorage(RolloutStorage):
    """Rewards, values and returns per critic (T, N, K); advantages summed over critics (T, N, 1)."""

    def __init__(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device, num_critics):
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.num_critics = num_critics
        shape = (num_transitions_per_env, num_envs, num_critics)
        self.rewards = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.returns = torch.zeros(*shape, device=device)

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        advantages = self.returns - self.values
        # per critic: the share of the return variance its values explain (1 good, <= 0 no better than the mean)
        self.explained_variance = 1.0 - advantages.var(dim=(0, 1)) / (self.returns.var(dim=(0, 1)) + 1e-8)
        # normalize each critic's advantage, then sum (no weights); the sum is normalized again so the
        # policy step does not grow with the number of critics
        mean = advantages.mean(dim=(0, 1), keepdim=True)
        std = advantages.std(dim=(0, 1), keepdim=True)
        summed = ((advantages - mean) / (std + 1e-8)).sum(dim=-1, keepdim=True)
        if normalize_advantage:
            summed = (summed - summed.mean()) / (summed.std() + 1e-8)
        self.advantages = summed


class MultiCriticPPO(PPO):
    """PPO on :class:`MultiCriticActorCritic`, trained on ``extras["reward_groups"]``.

    The value loss is averaged over all critics. Per-mini-batch advantage normalization would
    re-normalize the summed advantage, so it is not supported.
    """

    policy: MultiCriticActorCritic

    def __init__(self, policy: MultiCriticActorCritic, **kwargs):
        if kwargs.get("normalize_advantage_per_mini_batch"):
            raise ValueError("MultiCriticPPO normalizes the advantage per critic over the whole rollout.")
        if kwargs.get("rnd_cfg") is not None:
            raise ValueError("MultiCriticPPO does not support RND.")
        super().__init__(policy, **kwargs)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        self.storage = MultiCriticRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device, self.policy.num_critics
        )

    def update(self) -> dict[str, float]:
        explained_variance = self.storage.explained_variance.tolist()
        loss_dict = super().update()
        for name, ev in zip(self.policy.critic_names, explained_variance):
            loss_dict[f"explained_variance_{name}"] = ev
        return loss_dict

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        # time-out bootstrapping per critic, here: PPO's squeezes the (N, 1) values of a single critic
        rewards = extras["reward_groups"].to(self.device)
        if "time_outs" in extras:
            rewards = rewards + self.gamma * self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device)
            extras = {k: v for k, v in extras.items() if k != "time_outs"}
        super().process_env_step(obs, rewards, dones, extras)
