# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GruPPORunnerCfg(BasePPORunnerCfg):
    """BasePPORunnerCfg with the MLP actor-critic (``ActorCritic``) swapped for RSL-RL's
    recurrent ``ActorCriticRecurrent``, using a GRU. In that class the RNN sits in front of
    the same actor/critic MLP (obs -> GRU -> MLP -> output), so ``actor_hidden_dims``/
    ``critic_hidden_dims`` are left exactly as ``BasePPORunnerCfg``'s -- this only inserts
    the recurrence, it does not resize the network around it. Algorithm hyperparameters,
    ``num_steps_per_env``, etc. are all inherited unchanged, so a run using this differs
    from one using ``BasePPORunnerCfg`` in exactly one place: the network."""

    policy: RslRlPpoActorCriticRecurrentCfg = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


# ---------------------------------------------------------------------------
# Perceptive: the GRU policy with a height map beside it. See
# ``assets/models/actor_critic_perceptive.py`` for the architecture and why the map
# does not go through the recurrence by default.
# ---------------------------------------------------------------------------
@configclass
class PerceptiveActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    """``RslRlPpoActorCriticRecurrentCfg`` plus the exteroceptive branch's settings."""

    class_name: str = "ActorCriticPerceptiveRecurrent"

    extero_obs_groups: list[str] = MISSING
    """Observation groups carrying the height map, named rather than inferred.

    Any group listed here is routed through the exteroceptive encoder instead of the
    GRU; everything else in an observation set is proprioceptive. A group named here
    that a given set does not contain simply does not apply to that side, which is how
    the actor reads the sensor's grid while the critic reads the noise-free one.
    """

    extero_encoder_dims: list[int] = [160]
    """Hidden widths of the height-map encoder ``g_e``."""

    extero_latent_dim: int = 96
    """Width of ``l_e``. Miki et al. use 96 (24 per foot x 4); a body-centred grid has no
    per-foot structure, so this is one MLP over the whole grid at the same width."""

    extero_to_memory: bool = False
    """Feed ``l_e`` to the GRU as well. Off keeps the GRU's input at the blind policy's
    45 dimensions, so a blind checkpoint's recurrent weights load unchanged."""

    extero_skip: bool = True
    """Hand ``l_e`` to the actor MLP directly, around the GRU."""


@configclass
class PerceptiveGruPPORunnerCfg(BasePPORunnerCfg):
    """``GruPPORunnerCfg``'s network with a height map added as a second input group.

    ``obs_groups`` is spelled out because the whole point is that the two sides read
    different grids: the actor gets ``height_map``, built by the MID-360 with its noise
    model, and the critic gets ``height_map_clean``, the top-down raycast. Both are the
    full 29 x 21 with no body exclusion and the same sign convention, so they line up
    cell for cell.
    """

    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "height_map"],
        "critic": ["critic", "height_map_clean"],
    }

    policy: PerceptiveActorCriticCfg = PerceptiveActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        extero_obs_groups=["height_map", "height_map_clean"],
    )
