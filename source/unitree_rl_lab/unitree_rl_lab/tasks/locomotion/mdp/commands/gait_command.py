from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitCommand(CommandTerm):
    """Samples a gait *style* -- frequency, per-foot phase offset, duty cycle -- as a
    policy-visible command, following Margolis & Agrawal, "Walk These Ways: Tuning Robot
    Control for Generalization with Multiplicity of Behavior" (2022).

    Feet are ordered ``[FL, FR, RL, RR]``; FL is the phase-0 reference leg, the other three
    carry a commanded offset relative to it. ``leg_phases()`` returns each foot's phase in
    ``[0, 1)`` for use by observation/reward terms (``gait_clock_obs``, ``gait_tracking_reward``).
    """

    cfg: GaitCommandCfg

    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        # [frequency, theta_FR, theta_RL, theta_RR, duty_cycle]
        self.gait_command = torch.zeros(self.num_envs, 5, device=self.device)
        self.metrics["frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["duty_cycle"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """``[frequency, theta_FR, theta_RL, theta_RR, duty_cycle]``. Shape is (num_envs, 5)."""
        return self.gait_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["frequency"] = self.gait_command[:, 0]
        self.metrics["duty_cycle"] = self.gait_command[:, 4]

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequency)
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.theta_fr)
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.theta_rl)
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.theta_rr)
        self.gait_command[env_ids, 4] = r.uniform_(*self.cfg.ranges.duty_cycle)
        # Restart the clock whenever the gait style is (re)sampled, so a freshly sampled
        # style doesn't inherit a stance/swing phase left over from a different one.
        self.phase[env_ids] = 0.0

    def _update_command(self):
        self.phase = (self.phase + self.gait_command[:, 0] * self._env.step_dt) % 1.0

    """
    Helpers used by observation/reward terms.
    """

    def leg_phases(self) -> torch.Tensor:
        """Per-leg phase in ``[0, 1)``, ordered ``[FL, FR, RL, RR]``."""
        theta = torch.cat([torch.zeros(self.num_envs, 1, device=self.device), self.gait_command[:, 1:4]], dim=1)
        return (self.phase.unsqueeze(1) + theta) % 1.0

    def duty_cycle(self) -> torch.Tensor:
        """Commanded stance fraction of the cycle. Shape is (num_envs,)."""
        return self.gait_command[:, 4]


@configclass
class GaitCommandCfg(CommandTermCfg):
    """Configuration for :class:`GaitCommand`."""

    class_type: type = GaitCommand

    @configclass
    class Ranges:
        frequency: tuple[float, float] = MISSING
        """Stride frequency range, in Hz."""
        theta_fr: tuple[float, float] = MISSING
        """FR phase offset relative to FL, in cycle fractions [0, 1)."""
        theta_rl: tuple[float, float] = MISSING
        """RL phase offset relative to FL, in cycle fractions [0, 1)."""
        theta_rr: tuple[float, float] = MISSING
        """RR phase offset relative to FL, in cycle fractions [0, 1)."""
        duty_cycle: tuple[float, float] = MISSING
        """Fraction of the cycle each foot spends in stance, in (0, 1)."""

    ranges: Ranges = MISSING
