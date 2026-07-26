"""PPO with a jointly-trained TumblerNet-style auxiliary state estimator.

Extends ``rsl_rl.algorithms.PPO`` (rsl-rl 3.x). Parent handles rollout collection,
storage, GAE, and the surrogate/value/entropy PPO losses. This subclass adds one
extra term computed from the same minibatches, so the estimator is optimized by
the very same gradient steps as the policy/value functions (matching "the
estimators are trained in parallel with the deep reinforcement learning controller
using PPO instead of being trained separately", TumblerNet / Xiao et al. 2025).

Loss is the paper's *convex combination*, not a plain sum (Eq. 16):

    L = beta * loss_reg + (1 - beta) * loss_policy
    loss_reg    = MSE(vel_estimate, true_vel) + MSE(com_cop_estimate, true_com_cop)
    loss_policy = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy

with beta = ``aux_loss_coef`` = 0.5, i.e. "the regression loss weight is the same
as the weight of policy loss" (paper, directly below Eq. 16). Note this also
halves the *effective* weight of ``value_loss_coef`` / ``entropy_coef`` relative
to vanilla PPO defaults tuned assuming beta=0 -- that is what the paper specifies,
not an oversight.

``true_vel`` / ``true_com_cop`` come from the env's privileged ``estimator_target``
observation group (ground truth, never seen by the actor); ``vel_estimate`` /
``com_cop_estimate`` are ``BipedPolicy``'s own estimator output, cached on the
policy by ``BipedActorCritic.update_distribution()`` every time ``act()`` runs
(which this method calls once per minibatch, same as vanilla PPO).

This intentionally does not support the RND / symmetry extensions available in
upstream ``PPO`` -- neither is used by the biped task -- to keep this override
focused on the one addition that matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms import PPO as RslPPO


class BipedPPO(RslPPO):
    """PPO + auxiliary estimator loss for ``BipedActorCritic``."""

    def __init__(self, policy, *args, aux_loss_coef: float = 0.5, **kwargs):
        super().__init__(policy, *args, **kwargs)
        # Paper's `beta` (Eq. 16): convex-combination weight, not an additive scale.
        self.aux_loss_coef = aux_loss_coef

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_vel_loss = 0.0
        mean_com_cop_loss = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Recompute actions log prob, entropy, and the estimator's latest prediction
            # for the current batch of transitions (policy has been updated since collection).
            # NOTE: kwarg is `hidden_state` (singular) to match rsl_rl.modules.ActorCritic's
            # signature -- BipedActorCritic.act()/evaluate() ignore it via **kwargs anyway
            # (this policy is non-recurrent), but keep it faithful to upstream.
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL-adaptive learning rate.
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

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Auxiliary estimator loss: supervise BipedPolicy's estimator against the
            # privileged ground truth carried in the "estimator_target" obs group.
            estimator_target = obs_batch["estimator_target"]
            true_vel = estimator_target[..., 0:3]
            true_com_cop = estimator_target[..., 3:6]
            vel_loss = F.mse_loss(self.policy.last_vel_estimate, true_vel)
            com_cop_loss = F.mse_loss(self.policy.last_com_cop_estimate, true_com_cop)
            loss_reg = vel_loss + com_cop_loss
            loss_policy = (
                surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            )

            # Eq. 16: convex combination, not `loss_policy + aux_loss_coef * loss_reg`.
            loss = self.aux_loss_coef * loss_reg + (1.0 - self.aux_loss_coef) * loss_policy

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_vel_loss += vel_loss.item()
            mean_com_cop_loss += com_cop_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_vel_loss /= num_updates
        mean_com_cop_loss /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_vel": mean_vel_loss,
            "estimator_com_cop": mean_com_cop_loss,
        }
