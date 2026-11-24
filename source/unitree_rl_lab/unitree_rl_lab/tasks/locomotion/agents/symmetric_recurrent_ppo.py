"""PPO with left-right symmetry data augmentation for a recurrent (GRU) policy.

rsl_rl 3.1.2 refuses symmetry augmentation for ``ActorCriticRecurrent``: the stored
GRU hidden states belong to the trajectory as it happened, and the mirrored
trajectory's hidden states are something else entirely -- they only exist if the
GRU has also been run on the mirrored observations from the start of the episode.

So this module runs that second, mirrored GRU stream during the rollout:

* ``ActorCriticRecurrentSymmetric`` keeps a mirrored hidden state for the actor and
  the critic next to the real ones, advanced every step on the mirrored
  observations (same weights, no extra parameters) and reset on the same dones.
* ``SymmetricRecurrentRolloutStorage`` stores those mirrored hidden states per step
  and hands them out with the real ones in ``recurrent_mini_batch_generator``.
* ``SymmetricRecurrentPPO.update`` appends the mirrored trajectories to each
  mini-batch along the trajectory axis (dim 1 of the padded [time, traj, ...]
  layout), each starting from its own mirrored hidden state -- the recurrent
  counterpart of what rsl_rl does along dim 0 for an MLP.

The mirrored hidden states are recorded with the policy weights of the rollout, like
the real ones, so both go equally stale within an update. GRU only (no LSTM tuple).
The mirror itself is the task's ``data_augmentation_func`` (mdp.symmetry).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories, string_to_callable

__all__ = ["ActorCriticRecurrentSymmetric", "SymmetricRecurrentPPO", "SymmetricRecurrentRolloutStorage"]


class ActorCriticRecurrentSymmetric(ActorCriticRecurrent):
    """``ActorCriticRecurrent`` that also carries the hidden states of the mirrored rollout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.memory_a.rnn, nn.GRU):
            raise ValueError("ActorCriticRecurrentSymmetric supports rnn_type='gru' only")
        self.mirror_hidden_a: torch.Tensor | None = None
        self.mirror_hidden_c: torch.Tensor | None = None

    @torch.no_grad()
    def step_mirror(self, obs_mirrored: TensorDict) -> None:
        """Advance the mirrored hidden states by one step on the mirrored observations."""
        a = self.actor_obs_normalizer(self.get_actor_obs(obs_mirrored))
        _, self.mirror_hidden_a = self.memory_a.rnn(a.unsqueeze(0), self.mirror_hidden_a)
        c = self.critic_obs_normalizer(self.get_critic_obs(obs_mirrored))
        _, self.mirror_hidden_c = self.memory_c.rnn(c.unsqueeze(0), self.mirror_hidden_c)

    def get_mirror_hidden_states(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.mirror_hidden_a, self.mirror_hidden_c

    def reset(self, dones: torch.Tensor | None = None) -> None:
        super().reset(dones)
        if dones is None:
            self.mirror_hidden_a = None
            self.mirror_hidden_c = None
        else:
            for h in (self.mirror_hidden_a, self.mirror_hidden_c):
                if h is not None:
                    h[..., dones == 1, :] = 0.0


class SymmetricRecurrentRolloutStorage(RolloutStorage):
    """``RolloutStorage`` that also stores the mirrored GRU hidden states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_mirror_hidden_a: torch.Tensor | None = None
        self.saved_mirror_hidden_c: torch.Tensor | None = None

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        mirror = getattr(transition, "mirror_hidden_states", (None, None))
        if mirror[0] is not None:
            if self.saved_mirror_hidden_a is None:
                n = self.num_transitions_per_env
                self.saved_mirror_hidden_a = torch.zeros(n, *mirror[0].shape, device=self.device)
                self.saved_mirror_hidden_c = torch.zeros(n, *mirror[1].shape, device=self.device)
            self.saved_mirror_hidden_a[self.step].copy_(mirror[0])
            self.saved_mirror_hidden_c[self.step].copy_(mirror[1])
        elif self.saved_mirror_hidden_a is not None:
            self.saved_mirror_hidden_a[self.step].zero_()
            self.saved_mirror_hidden_c[self.step].zero_()
        super().add_transitions(transition)

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Same batches as ``RolloutStorage.recurrent_mini_batch_generator``; the hidden-state
        entry is ``(actor, critic, mirrored actor, mirrored critic)``."""
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches

        dones = self.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        last_was_done_t = last_was_done.permute(1, 0)

        def select(saved: torch.Tensor | None, first: int, last: int) -> torch.Tensor:
            # [time, layers, envs, hidden] -> trajectory starts -> [layers, trajs, hidden]
            if saved is None:
                layers, hidden = self.saved_hidden_state_a[0].shape[1], self.saved_hidden_state_a[0].shape[3]
                return torch.zeros(layers, last - first, hidden, device=self.device)
            return saved.permute(2, 0, 1, 3)[last_was_done_t][first:last].transpose(1, 0).contiguous()

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
                last_traj = first_traj + int(torch.sum(last_was_done[:, start:stop]))
                hidden = (
                    select(self.saved_hidden_state_a[0], first_traj, last_traj),
                    select(self.saved_hidden_state_c[0], first_traj, last_traj),
                    select(self.saved_mirror_hidden_a, first_traj, last_traj),
                    select(self.saved_mirror_hidden_c, first_traj, last_traj),
                )
                yield (
                    padded_obs_trajectories[:, first_traj:last_traj],
                    self.actions[:, start:stop],
                    self.values[:, start:stop],
                    self.advantages[:, start:stop],
                    self.returns[:, start:stop],
                    self.actions_log_prob[:, start:stop],
                    self.mu[:, start:stop],
                    self.sigma[:, start:stop],
                    hidden,
                    trajectory_masks[:, first_traj:last_traj],
                )
                first_traj = last_traj


class SymmetricRecurrentPPO(PPO):
    """PPO + symmetry data augmentation for ``ActorCriticRecurrentSymmetric``."""

    policy: ActorCriticRecurrentSymmetric

    def __init__(self, policy, *args, symmetry_cfg: dict | None = None, **kwargs):
        # rsl_rl's PPO rejects symmetry for recurrent policies; set it up here instead
        super().__init__(policy, *args, symmetry_cfg=None, **kwargs)
        if not isinstance(policy, ActorCriticRecurrentSymmetric):
            raise TypeError("SymmetricRecurrentPPO needs an ActorCriticRecurrentSymmetric policy")
        if symmetry_cfg is None or not symmetry_cfg.get("use_data_augmentation", False):
            raise ValueError("SymmetricRecurrentPPO needs symmetry_cfg with use_data_augmentation=True")
        if symmetry_cfg.get("use_mirror_loss", False):
            raise NotImplementedError("mirror loss is not implemented for the recurrent policy")
        if isinstance(symmetry_cfg["data_augmentation_func"], str):
            symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
        if self.rnd is not None or self.is_multi_gpu:
            raise NotImplementedError("SymmetricRecurrentPPO does not support RND or multi-GPU")
        self.symmetry = symmetry_cfg

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        self.storage = SymmetricRecurrentRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    # -- rollout -------------------------------------------------------------------

    def _mirror_obs(self, obs: TensorDict) -> TensorDict:
        n = obs.batch_size[0]
        aug, _ = self.symmetry["data_augmentation_func"](obs=obs, actions=None, env=self.symmetry["_env"])
        return aug[n:]

    def _mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = actions.shape
        flat = actions.reshape(-1, shape[-1])
        _, aug = self.symmetry["data_augmentation_func"](obs=None, actions=flat, env=self.symmetry["_env"])
        return aug[flat.shape[0] :].reshape(shape)

    def act(self, obs: TensorDict) -> torch.Tensor:
        mirror_hidden = self.policy.get_mirror_hidden_states()
        actions = super().act(obs)  # records the real hidden states, then steps the real GRUs
        self.transition.mirror_hidden_states = mirror_hidden
        self.policy.step_mirror(self._mirror_obs(obs))
        return actions

    # -- update --------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_symmetry_loss = 0.0

        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            n_env = actions_batch.shape[1]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # append the mirrored trajectories along the trajectory axis
            with torch.no_grad():
                t, n_traj = obs_batch.batch_size
                obs_m = self._mirror_obs(obs_batch.reshape(-1)).reshape(t, n_traj)
                obs_batch = torch.cat([obs_batch, obs_m], dim=1)
                masks_batch = torch.cat([masks_batch, masks_batch], dim=1)
                actions_batch = torch.cat([actions_batch, self._mirror_actions(actions_batch)], dim=1)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(1, 2, 1)
                target_values_batch = target_values_batch.repeat(1, 2, 1)
                advantages_batch = advantages_batch.repeat(1, 2, 1)
                returns_batch = returns_batch.repeat(1, 2, 1)
            h_a = torch.cat([hidden_states_batch[0], hidden_states_batch[2]], dim=1)
            h_c = torch.cat([hidden_states_batch[1], hidden_states_batch[3]], dim=1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=h_a)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=h_c)
            # KL and entropy from the original half only, as rsl_rl does
            mu_all = self.policy.action_mean
            mu_batch = mu_all[:, :n_env]
            sigma_batch = self.policy.action_std[:, :n_env]
            entropy_batch = self.policy.entropy[:, :n_env]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch, -1))
            adv = torch.squeeze(advantages_batch, -1)
            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # logged only: how far the policy is from mirror-equivariant on this batch
            with torch.no_grad():
                symmetry_loss = torch.mean(
                    (mu_all[:, n_env:] - self._mirror_actions(mu_all[:, :n_env])).pow(2)
                )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "symmetry": mean_symmetry_loss / num_updates,
        }


def register_with_rsl_rl() -> None:
    """rsl_rl's runner resolves ``class_name`` with ``eval`` in its own module scope."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.ActorCriticRecurrentSymmetric = ActorCriticRecurrentSymmetric
    runner_module.SymmetricRecurrentPPO = SymmetricRecurrentPPO
