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

from unitree_rl_lab.tasks.locomotion.agents.actor_critic_tcn import ActorCriticTcn  # noqa: F401


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
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class RslRlPpoActorCriticTcnCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic whose actor is Lee et al. 2020's TCN student encoder."""

    class_name: str = "ActorCriticTcn"
    tcn_history_length: int = 100
    tcn_channels: int = 34
    tcn_kernel_size: int = 5
    tcn_latent_dim: int = 64
    tcn_concat_current_obs: bool = True
    use_privileged_encoder: bool = False
    privileged_encoder_hidden_dims: list[int] = [72]
    privileged_latent_dim: int = 64


@configclass
class TcnPPORunnerCfg(BasePPORunnerCfg):
    """BasePPORunnerCfg with the MLP actor swapped for the TCN student encoder from
    Lee et al. 2020 (Science Robotics, Table S5 TCN-100). That paper distills a
    privileged teacher into this TCN by imitation; here the same encoder is trained
    with PPO. The critic stays a feedforward MLP on the critic observation.

    TCN-100 covers 2 s of proprioception at the 50 Hz policy rate. Checkpoints cannot
    be shared with ``BasePPORunnerCfg`` or ``GruPPORunnerCfg``.
    """

    policy: RslRlPpoActorCriticTcnCfg = RslRlPpoActorCriticTcnCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        tcn_history_length=100,
        tcn_channels=34,
        tcn_kernel_size=5,
        tcn_latent_dim=64,
        tcn_concat_current_obs=True,
        use_privileged_encoder=False,
    )


@configclass
class TcnTeacherPPORunnerCfg(TcnPPORunnerCfg):
    """``TcnPPORunnerCfg`` with the paper's teacher-style critic.

    Actor is unchanged (TCN-100 student). Critic encodes privileged ``xt`` (ELU 72 → 64),
    concatenates current proprioception ``ot``, then the critic MLP. No distillation loss.
    Checkpoints cannot be shared with ``TcnPPORunnerCfg``.
    """

    policy: RslRlPpoActorCriticTcnCfg = RslRlPpoActorCriticTcnCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        tcn_history_length=100,
        tcn_channels=34,
        tcn_kernel_size=5,
        tcn_latent_dim=64,
        tcn_concat_current_obs=True,
        use_privileged_encoder=True,
        privileged_encoder_hidden_dims=[72],
        privileged_latent_dim=64,
    )
