from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class HoSTAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """HoST's PPO extras (see ``unitree_rl_lab.utils.host_rl``)."""

    reward_groups: list[str] = MISSING
    """Reward-term name prefixes, one critic each."""
    reward_group_weights: list[float] = MISSING
    """Weights mixing the per-group normalized advantages."""
    critic_obs_group: str = "policy"
    """HoST's critic sees the actor observations."""
    value_smoothness_coef: float = 0.1
    smoothness_upper_bound: float = 1.0
    smoothness_lower_bound: float = 0.1
    symmetry_fn: str | None = None
    """Not in HoST: ``"module:function"`` returning ``(mirror_obs, mirror_actions)`` for a
    left/right symmetry loss on the policy mean."""
    symmetry_coef: float = 1.0


@configclass
class GetUpPPORunnerCfg(BasePPORunnerCfg):
    """HoST G1CfgPPO (g1_ground): multi-critic PPO with smoothness loss, 50 steps/env,
    12000 iterations, init std 0.8, actor [512, 256, 128] + Tanh, critics [512, 256]."""

    class_name = "HoSTRunner"
    experiment_name = "g1_getup"
    num_steps_per_env = 50
    max_iterations = 12000
    save_interval = 500
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256],
        activation="elu",
    )
    algorithm = HoSTAlgorithmCfg(
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
        reward_groups=["task", "regu", "style", "target"],
        # HoST: [2.5, 0.1, 1, 1]; target raised so the standing posture counts as much as getting up
        reward_group_weights=[2.5, 0.1, 1.0, 2.5],
        symmetry_fn="unitree_rl_lab.tasks.locomotion.robots.g1_getup.symmetry:build_mirror",
        symmetry_coef=1.0,
    )


@configclass
class StandUpPPORunnerCfg(GetUpPPORunnerCfg):
    """Stand up from crouch, with the same HoST learner."""

    experiment_name = "g1_standup"
    max_iterations = 5000


@configclass
class GetUp29PPORunnerCfg(GetUpPPORunnerCfg):
    """Same learner for the G1 29dof get-up."""

    experiment_name = "g1_29dof_getup"


@configclass
class GetUp29HoSTPPORunnerCfg(GetUp29PPORunnerCfg):
    """HoST's reward group weights [2.5, 0.1, 1, 1] for training from scratch."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.reward_group_weights = [2.5, 0.1, 1.0, 1.0]


@configclass
class GetUpStagedPPORunnerCfg(GetUpPPORunnerCfg):
    """G1 23dof from scratch with the staged rewards: HoST's group weights."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.reward_group_weights = [2.5, 0.1, 1.0, 1.0]
