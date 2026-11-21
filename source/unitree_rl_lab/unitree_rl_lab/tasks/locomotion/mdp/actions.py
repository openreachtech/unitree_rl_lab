"""Action terms with sim-to-real randomization.

Ported from humanoid-gym (``XBotLFreeEnv.step``): every policy step, each env
samples a delay coefficient ``d ~ U(0, max_delay)`` and the applied action is a
blend of the new action and the previously applied one, plus multiplicative
Gaussian noise::

    a_applied = (1 - d) * a_new + d * a_prev_applied
    a_applied += noise_std * randn * a_applied

This only lives in simulation. The policy output and the ``last_action``
observation stay the raw network output, so the exported deploy.yaml and the
C++ deploy code are unchanged (extra cfg keys are ignored by the deploy side).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["DelayedNoisyJointPositionAction", "DelayedNoisyJointPositionActionCfg"]


class DelayedNoisyJointPositionAction(JointPositionAction):
    cfg: DelayedNoisyJointPositionActionCfg

    def __init__(self, cfg: DelayedNoisyJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # last action actually applied (after blending / noise), in raw action units
        self._applied_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # keep the raw policy output for observations (last_action)
        self._raw_actions[:] = actions

        applied = actions
        if self.cfg.max_delay > 0.0:
            d = torch.rand(self.num_envs, 1, device=self.device) * self.cfg.max_delay
            applied = (1.0 - d) * applied + d * self._applied_actions
        if self.cfg.noise_std > 0.0:
            applied = applied + self.cfg.noise_std * torch.randn_like(applied) * applied
        self._applied_actions[:] = applied

        self._processed_actions = applied * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._applied_actions[env_ids] = 0.0


@configclass
class DelayedNoisyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedNoisyJointPositionAction

    max_delay: float = 0.5
    """Upper bound of the per-step blend coefficient with the previous applied action (humanoid-gym: 0.5)."""

    noise_std: float = 0.02
    """Std of the multiplicative Gaussian action noise (humanoid-gym: 0.02)."""
