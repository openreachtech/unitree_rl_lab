"""Discrete gait-mode command: lets a single policy switch between quadruped walking
and the two biped stances (hind-leg / front-leg) within an episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Mode ids shared by the mode-aware observation/reward/termination functions in
# ``rewards.py`` / ``terminations.py`` -- 0 must stay "quadruped" (the default/reset
# mode; see ``GaitModeCommand``).
MODE_QUAD = 0
MODE_HIND_BIPED = 1
MODE_FRONT_BIPED = 2


class GaitModeCommand(CommandTerm):
    """Discrete stance-mode command (quad / hind-biped / front-biped).

    Exposes the currently-commanded mode as a one-hot ``(num_envs, 3)`` tensor via the
    ``command`` property (same "commanded input" convention as a velocity command, so
    the policy conditions on it directly), and the raw integer mode id via
    ``mode_ids`` for the mode-aware reward/termination functions that need to branch
    on it (see ``unitree_rl_lab.tasks.biped.mdp.rewards`` /
    ``unitree_rl_lab.tasks.biped.mdp.terminations``).

    Every episode's *first* mode is forced to quadruped (id 0) -- the "default mode is
    normal walking" behavior -- regardless of ``cfg.mode_probs``; later in-episode
    resamples (mode switches, at ``cfg.resampling_time_range`` cadence) draw from the
    full distribution, so a single training run also practices the quad<->biped
    transitions themselves, not just each mode in isolation.
    """

    cfg: GaitModeCommandCfg

    def __init__(self, cfg: GaitModeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.mode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_one_hot = torch.zeros(self.num_envs, 3, device=self.device)
        self._mode_one_hot[:, MODE_QUAD] = 1.0
        self._mode_probs = torch.tensor(cfg.mode_probs, device=self.device)
        self._is_first_resample = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "GaitModeCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tMode probabilities [quad, hind_biped, front_biped]: {self.cfg.mode_probs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """One-hot ``[quad, hind_biped, front_biped]``. Shape ``(num_envs, 3)``."""
        return self._mode_one_hot

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        new_ids = torch.multinomial(self._mode_probs, len(env_ids), replacement=True).to(
            device=self.device, dtype=torch.long
        )
        # Force every episode's first mode to quad; later in-episode resamples
        # (mode switches) draw from the full cfg.mode_probs distribution.
        first = self._is_first_resample[env_ids]
        new_ids = torch.where(first, torch.zeros_like(new_ids), new_ids)

        self.mode_ids[env_ids] = new_ids
        self._is_first_resample[env_ids] = False
        self._mode_one_hot[env_ids] = 0.0
        self._mode_one_hot[env_ids, new_ids] = 1.0

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            self._is_first_resample[:] = True
        else:
            self._is_first_resample[env_ids] = True
        return super().reset(env_ids)


@configclass
class GaitModeCommandCfg(CommandTermCfg):
    """Configuration for :class:`GaitModeCommand`."""

    class_type: type = GaitModeCommand

    asset_name: str = MISSING

    mode_probs: tuple[float, float, float] = (0.5, 0.25, 0.25)
    """Sampling probability for ``[quad, hind_biped, front_biped]`` at each in-episode
    resample (not the episode's first mode, which is always quad -- see
    :class:`GaitModeCommand`)."""
