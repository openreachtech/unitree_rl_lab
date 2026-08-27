"""Policy networks for Unitree RL Lab.

Networks live here; the training pieces they need -- algorithm variants, checkpoint surgery --
live in :mod:`.modules`, following the layout the other branches use (``biped_actor.py`` beside
``modules/biped_ppo.py``).
"""

from .modules.expert_init import initialize_experts  # noqa: F401
from .modules.moe_ppo import DEFAULT_LR_SCALES, MoEPPO  # noqa: F401
from .modules.weight_surgery import expand_state_dict, scatter_observation  # noqa: F401
from .moe_actor import (  # noqa: F401
    EXPERT_ACROBATICS,
    EXPERT_LOCOMOTION,
    NUM_EXPERTS,
    Gating,
    MixtureOfExperts,
    MoEActorCritic,
)


def register_with_rsl_rl() -> None:
    """Make ``MoEActorCritic`` and ``MoEPPO`` resolvable by ``OnPolicyRunner``.

    The runner turns the configured class names into classes with a bare ``eval`` evaluated in its
    own module namespace::

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))   # rsl_rl/runners/...
        alg_class = eval(self.alg_cfg.pop("class_name"))

    so only names already imported *there* can be selected. Binding ours into that namespace is what
    makes ``class_name="MoEActorCritic"`` work without patching rsl_rl. Idempotent.
    """
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.MoEActorCritic = MoEActorCritic
    on_policy_runner.MoEPPO = MoEPPO


register_with_rsl_rl()
