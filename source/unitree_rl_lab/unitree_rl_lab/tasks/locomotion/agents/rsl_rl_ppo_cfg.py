# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
