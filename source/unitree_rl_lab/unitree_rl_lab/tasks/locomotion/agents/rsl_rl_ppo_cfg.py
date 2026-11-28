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
    RslRlSymmetryCfg,
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
class SymmetricPPORunnerCfg(BasePPORunnerCfg):
    """BasePPORunnerCfg + left-right symmetry data augmentation (mdp.symmetry)."""

    def __post_init__(self):
        from unitree_rl_lab.tasks.locomotion.mdp.symmetry import compute_symmetric_states

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=compute_symmetric_states
        )


@configclass
class SymmetricGruPPORunnerCfg(BasePPORunnerCfg):
    """GRU actor-critic (as feat/mid360's GruPPORunnerCfg) + left-right symmetry augmentation.

    rsl_rl does not support symmetry for recurrent policies, so this uses the
    mirrored-hidden-state PPO in ``agents/symmetric_recurrent_ppo.py``.
    """

    policy: RslRlPpoActorCriticRecurrentCfg = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrentSymmetric",
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

    def __post_init__(self):
        from unitree_rl_lab.tasks.locomotion.agents.symmetric_recurrent_ppo import register_with_rsl_rl
        from unitree_rl_lab.tasks.locomotion.mdp.symmetry import compute_symmetric_states

        register_with_rsl_rl()
        self.algorithm.class_name = "SymmetricRecurrentPPO"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=compute_symmetric_states
        )


@configclass
class TcnActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    """``agents/actor_critic_tcn.py:ActorCriticTcn`` (Lee et al. 2020 TCN-N, ported from tsubame_isaac_env)."""

    class_name: str = "ActorCriticTcn"
    tcn_history_length: int = 100
    """Actor window [steps]; 100 = 2 s at 50 Hz."""
    tcn_channels: int = 34
    tcn_kernel_size: int = 5
    tcn_latent_dim: int = 64
    tcn_concat_current_obs: bool = True
    tcn_bf16: bool = True
    critic_history_length: int = 3
    actor_history_groups: list[str] | None = None
    """Leading actor groups that go through the TCN window; later groups are current-only."""
    critic_history_groups: list[str] | None = None
    """Leading critic groups that are stacked over ``critic_history_length``; later groups are current-only."""
    update_chunks: int = 8
    """Mini-batch split for gradient accumulation (``PPORecurrentSymmetry``); GPU memory only."""


@configclass
class G1FpmPPORunnerCfg(BasePPORunnerCfg):
    """G1-FPM: TCN actor over proprioception + foothold height patches beside it,
    MLP critic over 3 steps of privileged obs + the clean map + the actor's patches.
    Left-right symmetry augmentation through ``PPORecurrentSymmetry``."""

    experiment_name = "g1_fpm"
    obs_groups = {
        "policy": ["policy", "foothold_patches"],
        "critic": ["critic", "height_map_clean", "foothold_patches"],
    }
    policy: TcnActorCriticCfg = TcnActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_history_groups=["policy"],
        critic_history_groups=["critic"],
    )

    def __post_init__(self):
        from unitree_rl_lab.tasks.locomotion.agents.actor_critic_tcn import ActorCriticTcn
        from unitree_rl_lab.tasks.locomotion.agents.ppo_recurrent_symmetry import PPORecurrentSymmetry
        from unitree_rl_lab.tasks.locomotion.mdp.symmetry import compute_symmetric_states
        import rsl_rl.runners.on_policy_runner as runner_module

        # rsl_rl resolves class_name with eval() in its runner module
        runner_module.ActorCriticTcn = ActorCriticTcn
        runner_module.PPORecurrentSymmetry = PPORecurrentSymmetry
        self.algorithm.class_name = "PPORecurrentSymmetry"
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=compute_symmetric_states
        )
