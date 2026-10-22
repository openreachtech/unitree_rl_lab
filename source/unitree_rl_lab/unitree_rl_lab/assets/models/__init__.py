"""Policy networks for Unitree RL Lab.

Networks live here; the training pieces they need -- algorithm variants, checkpoint surgery -- live
in :mod:`.modules`. The mixture-of-experts base classes (``moe_actor.py``, ``modules/moe_ppo.py``,
``modules/weight_surgery.py``) are taken unchanged from ``feat/multi-task``.
"""

from .modules.moe_ppo import DEFAULT_LR_SCALES, MoEPPO  # noqa: F401
from .modules.recovery_ppo import RecoveryMoEPPO  # noqa: F401
from .modules.weight_surgery import expand_state_dict, scatter_observation  # noqa: F401
from .moe_actor import Gating, MixtureOfExperts, MoEActorCritic  # noqa: F401
from .recovery_moe import AttitudeGating, RecoveryMoEActorCritic  # noqa: F401


def register_with_rsl_rl() -> None:
    """Make the classes above resolvable by ``OnPolicyRunner``.

    The runner turns the configured class names into classes with a bare ``eval`` in its own module
    namespace, so only names already bound *there* can be selected. Idempotent.
    """
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.MoEActorCritic = MoEActorCritic
    on_policy_runner.MoEPPO = MoEPPO
    on_policy_runner.RecoveryMoEActorCritic = RecoveryMoEActorCritic
    on_policy_runner.RecoveryMoEPPO = RecoveryMoEPPO


register_with_rsl_rl()
