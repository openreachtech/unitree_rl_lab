"""RL agent configuration for the mixture-of-experts multi-task policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.assets import models  # noqa: F401  (registers the classes with rsl_rl)
from unitree_rl_lab.tasks.multitask.obs_spec import CRITIC_UNIFIED, POLICY_UNIFIED, block_offsets


@configclass
class RslRlMoeActorCriticCfg(RslRlPpoActorCriticCfg):
    """Mixture-of-experts actor-critic.

    ``class_name`` is resolved by ``OnPolicyRunner`` with a bare ``eval`` in its own module
    namespace, which is why importing :mod:`unitree_rl_lab.assets.models` above matters: that import binds the class
    there. Without it the name would simply not resolve.
    """

    class_name: str = "MoEActorCritic"

    num_experts: int = 3
    """Expert 0 is the locomotion policy, expert 1 the acrobatics policy, expert 2 starts random and
    exists to absorb the run/take-off and landing/run transitions neither pre-trained policy has
    visited."""

    gating_hidden_dims: tuple[int, ...] = (128,)
    """Matches MoE-Loco's gating network (Table VI). The gate only has to decide which regime the
    robot is in; it does not need an expert-sized network to do it."""

    gating_activation: str = "elu"

    gating_prior_scale: float = 5.0
    """Additive logit prior steering the gate toward the acrobatics expert while a move is
    commanded, written into the network rather than learned.

    Enabled after a 2000-iteration run with it off produced ``max_height`` of exactly zero
    throughout: at a uniform ``[1/3, 1/3, 1/3]`` the blend of a walking action, an acrobatic action
    and noise cannot leave the ground, so no flip ever succeeds, so nothing rewards the gate for
    routing to the acrobatics expert. The prior exists to break that circle. Locomotion was
    unaffected in that run -- it ran, and faster than at iteration 200 -- so only the acrobatic
    half needed the help.

    Known limitation: the prior keys off the jump command's ``enabled`` flag, held for
    ``command_duration_s`` (0.5 s), while the move runs ~1.2 s. It therefore covers the take-off --
    the part that needs the acrobatics expert's explosive extension -- and hands back to the
    locomotion expert mid-flight. If take-offs start happening but landings do not, that handoff is
    the next thing to fix, by giving the observation an explicit "motion in progress" flag that
    spans the whole window.
    """

    actor_prior_index: int = block_offsets(POLICY_UNIFIED)["jump_command"]
    critic_prior_index: int = block_offsets(CRITIC_UNIFIED)["jump_command"]
    """Column of the jump command's ``enabled`` flag in each observation. Derived from the layout
    rather than written out, so a change to obs_spec cannot leave these pointing at the wrong
    column."""


@configclass
class RslRlMoePpoCfg(RslRlPpoAlgorithmCfg):
    """PPO with per-parameter-group learning-rate scales."""

    class_name: str = "MoEPPO"

    lr_scales: dict = {
        # Pre-trained experts: fine-tune gently instead of overwriting what they know.
        "actor_pretrained": 0.1,
        "critic_pretrained": 0.1,
        # Randomly initialised transition expert and the gates: nothing to preserve.
        "actor_new": 1.0,
        "critic_new": 1.0,
        "actor_gating": 1.0,
        "critic_gating": 1.0,
        "other": 1.0,
    }
    """Applied inside the optimizer's ``step``. rsl_rl's adaptive KL schedule overwrites every
    parameter group's ``lr`` on each update, so a scale written into the group up front would be
    erased before use. Scaling gradients instead would not work either: Adam normalises by gradient
    magnitude, so a uniformly scaled gradient produces the same step."""

    gravity_z_index: int = block_offsets(POLICY_UNIFIED)["projected_gravity"] + 2
    """Column of projected gravity's z component, used to split the gate statistics by whether the
    trunk is upright, tilted or inverted. Derived from the layout so it cannot drift out of step."""

    action_clip: float | None = 10.0
    """Mirror of the runner's ``clip_actions``, for reporting only.

    Lets the logs show how often that clip actually binds. It matters because ``clip_actions`` is
    applied by the vec-env wrapper, so it is not part of the environment and not exported to
    deploy -- if it binds during training, the deployed policy is a different one.
    """

    actor_warmup_iterations: int = 50
    """Iterations during which every ``actor_*`` group is held at scale 0.

    The value function is initialised from critics trained against different reward compositions and
    has never seen a transition state at all, so its first advantage estimates are noise -- and
    applying that noise to a good actor at the adaptive schedule's starting rate is how a good
    initialisation gets undone. Letting the critic catch up first costs 50 iterations and reuses the
    scale mechanism already here.
    """


@configclass
class MoEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    clip_actions = 10.0
    """Bound on the policy's raw action, applied before the environment sees it.

    A mixture has nothing holding its output small the way a single trained policy does. In the
    first full run the raw action diverged to a magnitude of roughly 50 while the *motion* stayed
    fine -- PhysX clamps a joint target to the joint's range, so a nonsensical target is harmless --
    but ``action_rate_l2`` reads the raw action and reached -9.9e4 against every other term below 1.
    It dominated the return, KL went with it, and the adaptive schedule pinned the learning rate to
    its 1e-5 floor from iteration 270 onward: 1700 iterations that could not learn.

    ``RslRlVecEnvWrapper`` clamps before ``env.step``, so the physics and the penalty see the same
    bounded value. 10 corresponds to a 2.5 rad joint offset, orders of magnitude beyond the trained
    experts' operating range (their per-step action deltas measure ~0.02), so it constrains nothing
    either of them does.
    """
    policy: RslRlMoeActorCriticCfg = RslRlMoeActorCriticCfg(
        init_noise_std=1.0,  # overwritten by the experts' own noise, see modules.initialize_experts
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # entropy_coef is 0.001 rather than the source tasks' 0.01. Those tasks learned from scratch,
    # where a strong reward gradient held the action noise down and the entropy bonus bought useful
    # exploration. Here the experts already know good actions, so the reward gradient is weak and the
    # bonus is the dominant force on the noise parameter: it inflated std from the experts' 0.195 to
    # 0.65 -- roughly a +-0.14 rad disturbance on every joint target at an action scale of 0.25.
    # A backflip does not survive that, and the measured flip success rate fell from 1.00 standalone
    # to 0.11 merged. Loss/entropy tells the same story: -3.2 for the standalone expert against
    # +10.3 here, a distribution pushed wide open. Carrying the experts' noise level in at
    # initialisation was necessary but not sufficient -- something also had to stop it drifting back.
    algorithm: RslRlMoePpoCfg = RslRlMoePpoCfg(
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
