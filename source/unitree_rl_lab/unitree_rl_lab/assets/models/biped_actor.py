import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml

from rsl_rl.modules import ActorCritic

from unitree_rl_lab.assets.models.modules.base_nn import BaseNet


class BipedEstimator(BaseNet):
    """History-of-proprioception -> [lin_vel(3), com_cop(3)] (TumblerNet "Estimator Net").

    Trained jointly with PPO via an auxiliary supervised loss against the ground-truth
    privileged values (see ``unitree_rl_lab.assets.models.modules.biped_ppo.BipedPPO``),
    not via a separate pre-training stage. Its output -- not the privileged ground
    truth -- is what the policy actually conditions on, so the exact same network runs
    unchanged at deployment time.
    """

    def __init__(self, model_cfg):
        super().__init__(model_config=model_cfg)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """:param history: [(*batch, history_dim)] stacked past proprioception.
        :return: [(*batch, 6)] = [lin_vel_estimate(3), com_cop_estimate(3)]
        """
        return self.estimator_mlp(history)


class BipedPolicy(BaseNet):
    """
    Actor for the biped (2-leg stance) locomotion policy.

    Architecture from: "Learning stable bipedal locomotion skills for quadrupedal
    robots on challenging terrains with automatic fall recovery" (TumblerNet,
    Xiao et al., 2025), adapted to this repo's config-driven ``BaseNet`` MLPs.

    Inputs:
        - proprio_state  : current-step command + proprioception (o_t)
        - history_state  : stacked past proprioception, fed only to the estimator

    Outputs:
        - action_mean       : mean of the Gaussian action distribution
        - action_std        : std of the Gaussian action distribution (learnable, state-independent)
        - vel_estimate       : estimator's predicted base linear velocity (3,)
        - com_cop_estimate   : estimator's predicted CoM-CoP balance vector (3,)
    """

    def __init__(self, args, model_cfg):
        self.proprio_dim = args.proprio_obs_dim
        self.history_dim = args.history_obs_dim
        self.action_dim = args.action_dim
        self.model_cfg = model_cfg
        self._adapt(args)
        super().__init__(model_config=model_cfg["policy"])
        self.estimator = BipedEstimator(model_cfg["estimator"])
        # state-independent learnable log std (standard in PPO locomotion)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def _adapt(self, args):
        """Inject input/output dimensions into the config before BaseNet builds the layers."""
        estimator_cfg = self.model_cfg["estimator"]["MLP"]["estimator_mlp"]
        estimator_cfg["input"] = self.history_dim
        estimator_cfg["output"] = 6  # [lin_vel(3), com_cop(3)]

        base_cfg = self.model_cfg["policy"]["MLP"]["base_net"]
        base_cfg["input"] = self.proprio_dim + 6
        base_cfg["output"] = self.action_dim

    def forward(self, observations):
        """
        :param observations: [(*batch, proprio_dim + history_dim)] -- current-step proprio
            followed by stacked history. The policy MLP never sees privileged ground truth:
            it only ever sees the estimator's own prediction, matching real deployment.
        :return dict:
            action_mean      [(*batch, action_dim)]
            action_std       [(*batch, action_dim)]
            vel_estimate     [(*batch, 3)]
            com_cop_estimate [(*batch, 3)]
        """
        proprio_state = observations[..., : self.proprio_dim]
        history_state = observations[..., self.proprio_dim :]

        estimate = self.estimator(history_state)  # [*batch, 6]
        vel_estimate = estimate[..., 0:3]
        com_cop_estimate = estimate[..., 3:6]

        fused = torch.cat((vel_estimate, proprio_state, com_cop_estimate), dim=-1)
        action_mean = self.base_net(fused)
        action_std = self.log_std.exp().expand_as(action_mean)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "vel_estimate": vel_estimate,
            "com_cop_estimate": com_cop_estimate,
        }

    def get_distribution(self, observations):
        out = self.forward(observations)
        return Normal(out["action_mean"], out["action_std"]), out

    def act(self, observations):
        """
        Sample an action for rollout collection (no grad).
        :return: action, log_prob, action_mean
        """
        dist, out = self.get_distribution(observations)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)  # sum over action dimensions
        return action, log_prob, out["action_mean"]

    def evaluate_actions(self, observations, actions):
        """
        Re-evaluate sampled actions for the PPO update step.
        :return: log_prob [(*batch,)], entropy [(*batch,)]
        """
        dist, _ = self.get_distribution(observations)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class BipedCritic(BaseNet):
    """
    Critic for the biped policy (state-value function V(s)).

    Sees the full privileged observation (ground-truth base linear velocity and
    CoM-CoP, contact/joint effort, etc.) -- unlike the actor, which only ever sees
    the estimator's prediction of a subset of that information.
    """

    def __init__(self, args, model_cfg):
        self.critic_obs_dim = args.critic_obs_dim
        self.model_cfg = model_cfg
        self._adapt(args)
        super().__init__(model_config=model_cfg["critic"])

    def _adapt(self, args):
        cfg = self.model_cfg["critic"]["MLP"]["base_net"]
        cfg["input"] = self.critic_obs_dim
        cfg["output"] = 1  # scalar value

    def forward(self, observations):
        """
        :param observations: [(*batch, critic_obs_dim)]
        :return: value [(*batch, 1)]
        """
        return self.base_net(observations)


class BipedActorCritic(ActorCritic):
    """
    Combined Actor-Critic wrapper used by the PPO trainer.

    Actor  -> BipedPolicy  (estimator + stochastic Gaussian policy)
    Critic -> BipedCritic  (deterministic, outputs scalar value from privileged obs)

    Trained end-to-end by ``BipedPPO``, which adds an auxiliary supervised loss on
    ``BipedPolicy``'s estimator (see ``unitree_rl_lab.assets.models.modules.biped_ppo``).
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        proprio_obs_dim,
        history_obs_dim,
        critic_obs_dim,
        model_cfg_path=None,
        model_cfg_key="biped_model",
        **kwargs,
    ):
        """
        :param obs        : TensorDict of live observations from the env (rsl-rl 3.x API).
        :param obs_groups : e.g. {"policy": ["policy"], "critic": ["critic"]} (rsl-rl 3.x API).
        :param proprio_obs_dim / history_obs_dim : widths of the "policy" obs group, laid out
            as [current-step proprio | stacked history] (see ``tasks/biped`` env cfg).
        :param critic_obs_dim : width of the "critic" obs group (privileged).
        """
        # Do not call ActorCritic.__init__ because it expects the default MLP architecture args.
        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        # No observation normalization; rsl-rl's PPO calls update_normalization() unconditionally.
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False

        self.proprio_dim = int(proprio_obs_dim)
        self.history_dim = int(history_obs_dim)
        self.critic_dim = int(critic_obs_dim)
        self.action_dim = int(num_actions)
        assert self.proprio_dim > 0, f"proprio_obs_dim must be > 0, got {self.proprio_dim}."
        assert self.history_dim > 0, f"history_obs_dim must be > 0, got {self.history_dim}."
        assert self.critic_dim > 0, f"critic_obs_dim must be > 0, got {self.critic_dim}."

        # Sanity-check cfg widths against live observation tensors.
        policy_obs_dim = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        critic_obs_dim_live = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        expected_policy = self.proprio_dim + self.history_dim
        assert policy_obs_dim == expected_policy, (
            f"Policy obs width ({policy_obs_dim}) != proprio+history "
            f"({self.proprio_dim}+{self.history_dim}={expected_policy})."
        )
        assert critic_obs_dim_live == self.critic_dim, (
            f"Critic obs width ({critic_obs_dim_live}) != configured critic_obs_dim ({self.critic_dim})."
        )

        if model_cfg_path is None:
            model_cfg_path = Path(__file__).resolve().parent / "config.yaml"
        model_cfg_data = self._load_model_cfg(model_cfg_path, model_cfg_key)
        model_cfg_actor = deepcopy(model_cfg_data)
        model_cfg_critic = deepcopy(model_cfg_data)

        args = SimpleNamespace(
            proprio_obs_dim=self.proprio_dim,
            history_obs_dim=self.history_dim,
            critic_obs_dim=self.critic_dim,
            action_dim=self.action_dim,
        )

        self.actor = BipedPolicy(args, model_cfg_actor)
        self.critic = BipedCritic(args, model_cfg_critic)
        self.distribution: Optional[Normal] = None
        # Cached for BipedPPO's auxiliary estimator loss (populated in update_distribution()).
        self.last_vel_estimate: Optional[torch.Tensor] = None
        self.last_com_cop_estimate: Optional[torch.Tensor] = None

    @staticmethod
    def _load_model_cfg(model_cfg_path, model_cfg_key):
        cfg_path = Path(model_cfg_path).expanduser().resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if model_cfg_key not in raw:
            raise KeyError(f"Config key '{model_cfg_key}' not found in {cfg_path}.")
        return raw[model_cfg_key]

    def reset(self, dones=None):
        """rsl-rl API compatibility. This biped model is non-recurrent."""
        return None

    @property
    def action_mean(self):
        assert self.distribution is not None, "Call update_distribution() before reading action_mean."
        return self.distribution.mean

    @property
    def action_std(self):
        assert self.distribution is not None, "Call update_distribution() before reading action_std."
        return self.distribution.stddev

    @property
    def entropy(self):
        assert self.distribution is not None, "Call update_distribution() before reading entropy."
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        """:param obs: TensorDict of observation groups (rsl-rl 3.x API)."""
        out = self.actor(self.get_actor_obs(obs))
        self.distribution = Normal(out["action_mean"], out["action_std"])
        # Stashed here (rather than threaded through act()/return values) so BipedPPO's
        # update() -- which only calls act()/evaluate() through the plain rsl-rl PPO
        # interface -- can still read the freshly recomputed estimator output for the
        # auxiliary loss after every re-forward pass during the PPO epochs.
        self.last_vel_estimate = out["vel_estimate"]
        self.last_com_cop_estimate = out["com_cop_estimate"]

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        assert self.distribution is not None, "Call act()/update_distribution() before log-prob."
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        out = self.actor(self.get_actor_obs(obs))
        return out["action_mean"]

    def evaluate(self, obs, **kwargs):
        return self.critic(self.get_critic_obs(obs))
