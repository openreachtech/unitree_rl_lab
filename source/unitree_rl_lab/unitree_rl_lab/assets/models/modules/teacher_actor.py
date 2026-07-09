import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml

from base_nn import BaseNet
from rsl_rl.modules import ActorCritic

class TeacherPolicy(BaseNet):
    """
    Actor for the teacher policy.

    Architecture from: "Learning robust perceptive locomotion for quadrupedal robots in the wild"
    (Zhuang et al., 2023)

    Inputs:
        - proprio_state  : command + proprioception  (o_t^p)
        - extero_state   : height-scan                (o_t^e)  [exteroceptive encoder g_e]
        - priv_state     : privileged info (friction, contact forces, ...)  (s_t)  [encoder g_p]

    Outputs:
        - action_mean    : mean of the Gaussian action distribution
        - action_std     : std  of the Gaussian action distribution  (learnable, state-independent)
        - extero_latent  : l_t^e  (per-leg exteroceptive latent, concatenated)
        - priv_latent    : l_t^p  (privileged latent)
    """

    def __init__(self, args, model_cfg):
        self.proprio_dim = args.proprio_obs_dim
        self.extero_dim = args.extero_obs_dim
        self.priv_dim = args.priv_obs_dim
        self.action_dim = args.action_dim
        self.model_cfg = model_cfg
        self._adapt(args)
        super().__init__(model_config=model_cfg["policy"])
        # state-independent learnable log std (standard in PPO locomotion)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def _adapt(self, args):
        """Inject input/output dimensions into the config before BaseNet builds the layers."""
        cfg = self.model_cfg["policy"]["MLP"]
        cfg["extero_encoder"]["input"] = self.extero_dim
        cfg["privileged_encoder"]["input"] = self.priv_dim
        cfg["base_net"]["input"] = (
            self.proprio_dim
            + cfg["extero_encoder"]["output"]
            + cfg["privileged_encoder"]["output"]
        )
        cfg["base_net"]["output"] = self.action_dim

    def forward(self, proprio_state, extero_state, priv_state):
        """
        :param proprio_state : [(*batch, proprio_dim)]
        :param extero_state  : [(*batch, extero_dim)]
        :param priv_state    : [(*batch, priv_dim)]
        :return dict:
            action_mean   [(*batch, action_dim)]
            action_std    [(*batch, action_dim)]
            extero_latent [(*batch, extero_encoder_output)]
            priv_latent   [(*batch, privileged_encoder_output)]
        """
        extero_latent = self.extero_encoder(extero_state)      # [*batch, extero_output]
        priv_latent = self.privileged_encoder(priv_state)      # [*batch, priv_output]

        fused = torch.cat((proprio_state, extero_latent, priv_latent), dim=-1)
        action_mean = self.base_net(fused)
        action_std = self.log_std.exp().expand_as(action_mean)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "extero_latent": extero_latent,
            "priv_latent": priv_latent,
        }

    def get_distribution(self, proprio_state, extero_state, priv_state):
        out = self.forward(proprio_state, extero_state, priv_state)
        return Normal(out["action_mean"], out["action_std"]), out

    def act(self, proprio_state, extero_state, priv_state):
        """
        Sample an action for rollout collection (no grad).
        :return: action, log_prob, action_mean
        """
        dist, out = self.get_distribution(proprio_state, extero_state, priv_state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)    # sum over action dimensions
        return action, log_prob, out["action_mean"]

    def evaluate_actions(self, proprio_state, extero_state, priv_state, actions):
        """
        Re-evaluate sampled actions for the PPO update step.
        :return: log_prob [(*batch,)], entropy [(*batch,)]
        """
        dist, _ = self.get_distribution(proprio_state, extero_state, priv_state)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class TeacherCritic(BaseNet):
    """
    Critic for the teacher policy (state-value function V(s)).

    Same encoder architecture as the actor but outputs a scalar value estimate.
    """

    def __init__(self, args, model_cfg):
        self.proprio_dim = args.proprio_obs_dim
        self.extero_dim = args.extero_obs_dim
        self.priv_dim = args.priv_obs_dim
        self.model_cfg = model_cfg
        self._adapt(args)
        super().__init__(model_config=model_cfg["critic"])

    def _adapt(self, args):
        cfg = self.model_cfg["critic"]["MLP"]
        cfg["extero_encoder"]["input"] = self.extero_dim
        cfg["privileged_encoder"]["input"] = self.priv_dim
        cfg["base_net"]["input"] = (
            self.proprio_dim
            + cfg["extero_encoder"]["output"]
            + cfg["privileged_encoder"]["output"]
        )
        cfg["base_net"]["output"] = 1   # scalar value

    def forward(self, proprio_state, extero_state, priv_state):
        """
        :return: value [(*batch, 1)]
        """
        extero_latent = self.extero_encoder(extero_state)
        priv_latent = self.privileged_encoder(priv_state)
        fused = torch.cat((proprio_state, extero_latent, priv_latent), dim=-1)
        return self.base_net(fused)


class TeacherActorCritic(ActorCritic):
    """
    Combined Actor-Critic wrapper used by the PPO trainer.

    Actor  → TeacherPolicy  (stochastic, outputs Gaussian distribution)
    Critic → TeacherCritic  (deterministic, outputs scalar value)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        proprio_obs_dim,
        extero_obs_dim,
        priv_obs_dim,
        model_cfg_path=None,
        model_cfg_key="teacher_model",
        **kwargs,
    ):
        # Do not call ActorCritic.__init__ because it expects the default MLP architecture args.
        nn.Module.__init__(self)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.proprio_dim = int(proprio_obs_dim)
        self.extero_dim = int(extero_obs_dim)
        self.priv_dim = int(priv_obs_dim)
        self.action_dim = int(num_actions)

        if model_cfg_path is None:
            model_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        model_cfg_data = self._load_model_cfg(model_cfg_path, model_cfg_key)
        model_cfg_actor = deepcopy(model_cfg_data)
        model_cfg_critic = deepcopy(model_cfg_data)

        args = SimpleNamespace(
            proprio_obs_dim=self.proprio_dim,
            extero_obs_dim=self.extero_dim,
            priv_obs_dim=self.priv_dim,
            action_dim=self.action_dim,
        )

        self.actor = TeacherPolicy(args, model_cfg_actor)
        self.critic = TeacherCritic(args, model_cfg_critic)
        self.distribution: Optional[Normal] = None

    @staticmethod
    def _load_model_cfg(model_cfg_path, model_cfg_key):
        cfg_path = Path(model_cfg_path).expanduser().resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if model_cfg_key not in raw:
            raise KeyError(f"Config key '{model_cfg_key}' not found in {cfg_path}.")
        return raw[model_cfg_key]

    def reset(self, dones=None):
        """rsl-rl API compatibility. This teacher model is non-recurrent."""
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

    def _split_actor_obs(self, observations):
        if isinstance(observations, (tuple, list)):
            if len(observations) != 3:
                raise ValueError("Tuple/list observations must be (proprio, extero, privileged).")
            return observations[0], observations[1], observations[2]

        expected_with_priv = self.proprio_dim + self.extero_dim + self.priv_dim
        expected_without_priv = self.proprio_dim + self.extero_dim
        if observations.shape[-1] == expected_with_priv:
            proprio = observations[..., : self.proprio_dim]
            extero = observations[..., self.proprio_dim : self.proprio_dim + self.extero_dim]
            privileged = observations[..., self.proprio_dim + self.extero_dim :]
            return proprio, extero, privileged

        if observations.shape[-1] == expected_without_priv:
            # Actor observations may omit privileged features. Feed zeros to the privileged encoder.
            proprio = observations[..., : self.proprio_dim]
            extero = observations[..., self.proprio_dim :]
            privileged = torch.zeros(
                *observations.shape[:-1],
                self.priv_dim,
                device=observations.device,
                dtype=observations.dtype,
            )
            return proprio, extero, privileged

        raise ValueError(
            "Unexpected actor observation dim. "
            f"Expected {expected_without_priv} (no priv) or {expected_with_priv} (with priv), "
            f"got {observations.shape[-1]}."
        )

    def _split_critic_obs(self, observations):
        if isinstance(observations, (tuple, list)):
            if len(observations) != 3:
                raise ValueError("Tuple/list observations must be (proprio, extero, privileged).")
            return observations[0], observations[1], observations[2]

        expected = self.proprio_dim + self.extero_dim + self.priv_dim
        if observations.shape[-1] != expected:
            raise ValueError(
                "Unexpected critic observation dim. "
                f"Expected {expected} (proprio+extero+priv), got {observations.shape[-1]}."
            )
        proprio = observations[..., : self.proprio_dim]
        extero = observations[..., self.proprio_dim : self.proprio_dim + self.extero_dim]
        privileged = observations[..., self.proprio_dim + self.extero_dim :]
        return proprio, extero, privileged

    def update_distribution(self, observations):
        proprio, extero, privileged = self._split_actor_obs(observations)
        out = self.actor(proprio, extero, privileged)
        self.distribution = Normal(out["action_mean"], out["action_std"])

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        assert self.distribution is not None, "Call act()/update_distribution() before log-prob."
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        proprio, extero, privileged = self._split_actor_obs(observations)
        out = self.actor(proprio, extero, privileged)
        return out["action_mean"]

    def evaluate(self, critic_observations, **kwargs):
        proprio, extero, privileged = self._split_critic_obs(critic_observations)
        return self.critic(proprio, extero, privileged)

    def get_value(self, critic_observations):
        proprio, extero, privileged = self._split_critic_obs(critic_observations)
        return self.critic(proprio, extero, privileged)
