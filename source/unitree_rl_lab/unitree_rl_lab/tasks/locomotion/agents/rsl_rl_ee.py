"""Explicit Estimator extension for rsl-rl-lib 3.1.2 (Ji et al. 2022,
"Concurrent Training of a Control Policy and a State Estimator",
arXiv:2202.05481 -- the state-estimator technique
[[reference_paper_impedance_matching_running_jump]] itself builds on).

This is a from-scratch reimplementation against the CURRENT rsl-rl-lib
(TensorDict + named ``obs_groups``), not a port of genesis_lr's
``actor_critic_ee.py``/``ppo_ee.py`` -- those target an older, vendored
rsl_rl whose class shapes differ substantially (flat-tensor observations,
no ``obs_groups``). See リファレンス_ExplicitEstimator実装仕様.md for the full
design writeup and the genesis_lr source this borrows the *design* (not
the code) from.

Injection mechanism: rsl_rl's ``OnPolicyRunner`` resolves
``policy_cfg["class_name"]``/``alg_cfg["class_name"]`` via
``eval(class_name)`` in ``rsl_rl.runners.on_policy_runner``'s own module
namespace. ``agents/rsl_rl_ppo_cfg.py`` imports this module and monkeypatches
``ActorCriticEE``/``PPOEE`` into it at import time (before any runner is
constructed) so ``eval("ActorCriticEE")``/``eval("PPOEE")`` resolve without
needing to fork the rsl-rl-lib pip package itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class ActorCriticEE(ActorCritic):
    """ActorCritic whose actor input is ``[actor_obs, estimator(actor_obs)]``.

    The estimator predicts the "estimator_target" observation group (ground
    truth: base velocity, per-link contact, foot height -- see
    longjump_env_cfg.py's ``EstimatorTargetCfg``) from the actor's own
    observation. Its output is concatenated onto that observation before
    reaching the actor MLP, so at deployment (no privileged sensors
    available) the actor still gets an estimate of these quantities.

    The critic is unaffected -- it already sees the true privileged values
    directly via its own ``critic`` observation group, so feeding it the
    (noisier) estimator prediction would only hurt the value function.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        estimator_hidden_dims: list[int] = [256, 128],
        actor_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            **kwargs,
        )
        if "estimator_target" not in obs_groups:
            raise ValueError(
                "ActorCriticEE requires an 'estimator_target' entry in obs_groups (see"
                " LongJumpPPORunnerCfg.obs_groups in rsl_rl_ppo_cfg.py)."
            )

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_estimator_labels = sum(obs[group].shape[-1] for group in obs_groups["estimator_target"])
        self.num_actor_obs = num_actor_obs
        self.num_estimator_labels = num_estimator_labels

        self.estimator = MLP(num_actor_obs, num_estimator_labels, estimator_hidden_dims, activation)
        print(f"Estimator MLP: {self.estimator}")

        # ActorCritic.__init__ (the super().__init__ call above) already
        # built self.actor sized for num_actor_obs alone. Rebuild it here
        # with room for the estimator's output appended -- this is the
        # only architectural difference from the stock class, which is what
        # makes a Net2Net-style transplant from a Stage-A (non-EE)
        # checkpoint straightforward: copy the old actor's weights into the
        # first num_actor_obs input columns of the new first layer, and
        # zero-init the num_estimator_labels columns added for the
        # estimator's output slice (see the Net2Net transplant script).
        actor_input_dim = num_actor_obs + num_estimator_labels
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP (rebuilt for Explicit Estimator, input={actor_input_dim}): {self.actor}")

        # Cache of the most recent estimator prediction, set by
        # _actor_input() and read by PPOEE.update() for the auxiliary loss
        # -- avoids recomputing the estimator forward pass a second time.
        self.last_estimator_prediction: torch.Tensor | None = None

    def _actor_input(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        estimator_output = self.estimator(actor_obs)
        # Keep the non-detached tensor for the estimator's own MSE loss
        # (PPOEE.update() backward()s through this one), but feed the actor
        # a *detached* copy. This is what makes the two networks' training
        # truly independent (matches the paper's separate-optimizers
        # design, confirmed 2026-08-27): without detaching here, the PPO
        # loss's backward() would also traverse the estimator's graph (its
        # gradient there is simply never applied, since estimator params
        # aren't in self.optimizer) and free intermediate buffers that
        # PPOEE.update() then needs for the estimator's own backward() --
        # causing a "backward through the graph a second time" RuntimeError.
        self.last_estimator_prediction = estimator_output
        return torch.cat((actor_obs, estimator_output.detach()), dim=-1)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        combined = self._actor_input(obs)
        self._update_distribution(combined)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        combined = self._actor_input(obs)
        if self.state_dependent_std:
            return self.actor(combined)[..., 0, :]
        return self.actor(combined)


class _ActorCriticEEOnnxExporter(nn.Module):
    """Deploy-time inference module for ActorCriticEE: normalizer -> estimator -> concat -> actor.

    isaaclab_rl.rsl_rl.exporter's generic export_policy_as_onnx()/export_policy_as_jit()
    only know how to wrap a plain ``policy.actor`` submodule behind a normalizer
    (``forward(x) = actor(normalizer(x))``). For ActorCriticEE, ``policy.actor``
    was Net2Net-widened to expect ``[actor_obs, estimator(actor_obs)]`` (see
    ``_actor_input`` above) -- exporting it directly via the generic path bakes
    in a graph whose "obs" input is num_actor_obs + num_estimator_labels wide
    (e.g. 87), silently mismatched against the num_actor_obs-wide (e.g. 46)
    observation vector the deploy C++ side actually has available (there is no
    privileged sensor to hand it the other half). This was only caught during
    2026-08-31 mujoco sim2sim testing -- the deploy OrtRunner (algorithms.h)
    builds its Ort::Value input tensor using the *model's* declared shape
    together with a data pointer sized to the *actual* observation vector, so
    the shape mismatch does not throw; it silently over-reads adjacent heap
    memory for the missing tail, feeding the actor garbage and producing
    nonsense actions (observed as an immediate crouch on entering the policy
    state). Exporting *this* wrapper instead makes num_actor_obs alone the
    ONNX graph's "obs" input -- matching deploy.yaml's actual observation
    vector -- with the estimator forward pass baked into the graph so the
    deploy side still benefits from it exactly as during training.
    """

    def __init__(self, policy: ActorCriticEE) -> None:
        super().__init__()
        import copy

        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.estimator = copy.deepcopy(policy.estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.num_actor_obs = policy.num_actor_obs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        estimator_output = self.estimator(x)
        combined = torch.cat((x, estimator_output), dim=-1)
        return self.actor(combined)

    def export_jit(self, path: str, filename: str = "policy.pt") -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        traced = torch.jit.script(self)
        traced.save(os.path.join(path, filename))

    def export_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.num_actor_obs)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_actor_critic_ee(policy: ActorCriticEE, path: str) -> None:
    """Export an ActorCriticEE policy to both policy.pt and policy.onnx, estimator included.

    Use this instead of isaaclab_rl.rsl_rl's export_policy_as_jit/export_policy_as_onnx
    for any ActorCriticEE policy -- see _ActorCriticEEOnnxExporter's docstring for why
    the generic exporters produce a silently-broken deploy graph for this class.
    """
    exporter = _ActorCriticEEOnnxExporter(policy)
    exporter.export_jit(path, filename="policy.pt")
    exporter.export_onnx(path, filename="policy.onnx")


class PPOEE(PPO):
    """PPO + Explicit Estimator, trained with two fully independent optimizers.

    Matches [[reference_explicit_estimator_impl_spec]]'s confirmed reading of
    genesis_lr's ppo_ee.py: the estimator's MSE loss is optimized by its own
    Adam optimizer, in a separate backward()/step() from the PPO loss --
    no gradient coupling, no shared parameters between the two optimizers.

    Known simplification vs. genesis_lr (flagged, not yet matched): that
    implementation masks out the terminal transition of each trajectory
    from the estimator loss (its ``terminated_batch`` is actually a
    *not-done* mask, confirmed 2026-08-27 by reading its rollout storage --
    the name is misleading). This class does not yet reproduce that masking
    since rsl-rl-lib 3.1.2's non-recurrent ``mini_batch_generator`` shuffles
    flat transitions and does not surface a per-sample done/not-done mask
    to the caller in an obviously equivalent way. In practice this means a
    small fraction of estimator-loss samples may be computed on
    already-reset (post-termination) observations; left as a known
    simplification for v1 rather than a blocker.
    """

    policy: ActorCriticEE

    def __init__(self, policy: ActorCriticEE, estimator_lr: float = 1.0e-3, **kwargs: dict[str, Any]) -> None:
        super().__init__(policy, **kwargs)

        if not isinstance(policy, ActorCriticEE):
            raise TypeError("PPOEE requires an ActorCriticEE policy.")

        # Exclude the estimator's parameters from the main PPO optimizer --
        # it is trained exclusively by estimator_optimizer below. The base
        # PPO.__init__ already built self.optimizer over ALL of
        # policy.parameters() (including the estimator's), so it is
        # rebuilt here without them.
        estimator_param_ids = {id(p) for p in policy.estimator.parameters()}
        policy_only_params = [p for p in policy.parameters() if id(p) not in estimator_param_ids]
        self.optimizer = optim.Adam(policy_only_params, lr=self.learning_rate)
        self._policy_only_params = policy_only_params

        self.estimator_optimizer = optim.Adam(policy.estimator.parameters(), lr=estimator_lr)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0

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
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob, entropy, and (as a side effect of
            # ActorCriticEE.act -> _actor_input) the estimator's prediction
            # for this batch's observations, under the *current* policy
            # parameters.
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

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

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
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

            # --- PPO step (estimator excluded, see __init__) ---
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._policy_only_params, self.max_grad_norm)
            self.optimizer.step()

            # --- Estimator step: fully independent optimizer/backward ---
            estimator_target = obs_batch["estimator_target"]
            estimator_loss = torch.nn.functional.mse_loss(self.policy.last_estimator_prediction, estimator_target)
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            self.estimator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
        }
