"""Foot swing height schedule ("Multi-critic Learning for Whole-body End-effector Twist Tracking", 3.2).

A gait phase ``phi`` advances at a fixed frequency; each foot ``f`` has a fixed phase offset
``theta_f`` (a static gait pattern). The desired foot heights are (eq. 5)

    h_f = h_max * sin(2 pi (phi + theta_f))

A foot is scheduled in swing where ``h_f > 0`` and in stance elsewhere (``contact_schedule``). The
schedule runs whatever the base velocity command, so standing still does not satisfy it.
``command`` is ``h_f`` (num_envs, 4), observed by the policy. The phase is random at reset.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitScheduleCommand(CommandTerm):
    cfg: GaitScheduleCommandCfg

    def __init__(self, cfg: GaitScheduleCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.offsets = torch.tensor(cfg.phase_offsets, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.heights = torch.zeros(self.num_envs, len(cfg.phase_offsets), device=self.device)

    def __str__(self) -> str:
        return (
            f"GaitScheduleCommand: {self.cfg.frequency} Hz, swing height {self.cfg.swing_height} m,"
            f" offsets {self.cfg.phase_offsets}"
        )

    @property
    def command(self) -> torch.Tensor:
        """Desired foot heights above the ground. Shape (num_envs, 4)."""
        return self.heights

    @property
    def contact_schedule(self) -> torch.Tensor:
        """1 where a foot is scheduled in stance, 0 in swing. Shape (num_envs, 4)."""
        return (self.heights <= 0.0).float()

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # the phase only restarts with the episode (resampling_time_range is the episode length)
        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        self._update_heights()

    def _update_command(self):
        self.phase = (self.phase + self.cfg.frequency * self._env.step_dt) % 1.0
        self._update_heights()

    def _update_heights(self):
        self.heights[:] = self.cfg.swing_height * torch.sin(2 * math.pi * (self.phase.unsqueeze(-1) + self.offsets))


@configclass
class GaitScheduleCommandCfg(CommandTermCfg):
    class_type: type = GaitScheduleCommand

    frequency: float = MISSING
    """Gait frequency [Hz]."""

    swing_height: float = MISSING
    """``h_max``: the peak of the swing [m]."""

    phase_offsets: tuple[float, ...] = MISSING
    """Per-foot phase offset ``theta_f``, in the order of the feet the rewards use."""
