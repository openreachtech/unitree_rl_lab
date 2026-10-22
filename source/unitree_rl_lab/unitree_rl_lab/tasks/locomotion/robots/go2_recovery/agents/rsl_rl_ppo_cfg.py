"""RL agent configuration for the walk / fall / get-up mixture of experts."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.assets import models  # noqa: F401  (registers the classes with rsl_rl)
from unitree_rl_lab.tasks.locomotion.robots.go2_recovery.obs_spec import CRITIC_GRAVITY_Z, POLICY_GRAVITY_Z


@configclass
class RecoveryMoeActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "RecoveryMoEActorCritic"

    num_experts: int = 2
    """Expert 0 is the blind locomotion policy, expert 1 the get-up policy. The falling phase in
    between gets no expert of its own: the multi-task mixture's randomly initialised third expert
    measured a routing weight of 0.000-0.001 everywhere, and a fall is short enough for the gate to
    blend the two that exist."""

    gating_hidden_dims: tuple[int, ...] = (128,)
    gating_activation: str = "elu"
    gating_prior_scale: float = 5.0
    """With two experts, 5.0 puts 99.3% of the weight on the prior's expert away from the boundary,
    so each expert starts out seeing the states it was trained on."""

    actor_gravity_z_index: int = POLICY_GRAVITY_Z
    critic_gravity_z_index: int = CRITIC_GRAVITY_Z
    upright_threshold: float = 0.8
    upright_width: float = 0.08


@configclass
class RecoveryMoePpoCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "RecoveryMoEPPO"

    lr_scales: dict = {
        "actor_pretrained": 0.1,
        "actor_gating": 1.0,
        "critic_new": 1.0,
        "critic_gating": 1.0,
        "other": 1.0,
    }
    gravity_z_index: int = POLICY_GRAVITY_Z
    action_clip: float | None = 10.0

    actor_warmup_iterations: int = 100
    """The critic starts from random weights, so the actor is held still until it has a value
    function worth taking advantages from."""


@configclass
class RecoveryMoEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "go2_recovery_moe"
    empirical_normalization = False
    clip_actions = 10.0
    """See ``MoEPPORunnerCfg.clip_actions``. 10 is a 2.5 rad joint offset at scale 0.25; the get-up
    expert's actions, doubled by the scale change, stay well inside it."""
    policy: RecoveryMoeActorCriticCfg = RecoveryMoeActorCriticCfg(
        init_noise_std=1.0,  # overwritten by the experts' own noise in the starting checkpoint
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # Low entropy bonus for the same reason as the multi-task mixture: the experts already know good
    # actions, and a 0.01 bonus inflates the noise until it breaks them.
    algorithm: RecoveryMoePpoCfg = RecoveryMoePpoCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GetUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """A plain actor-critic for the get-up policy on the shared interface. Fine-tuning settings
    follow the mixture's: the converted actor already knows the motion."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "go2_getup"
    empirical_normalization = False
    clip_actions = 10.0
    """Same bound as the mixture, and built into the exported network by export_policy.py. The
    original get-up policy was trained unclipped and never needed it in Isaac Lab, but in MuJoCo,
    once inverted, its output fed back through ``last_action`` until it sat at the deploy side's +-100
    and the robot froze upside down (Go2: 4 of 25 falls). With the bound: 30 of 30. It does not
    bind during an ordinary get-up."""
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # overwritten by the converted checkpoint
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
