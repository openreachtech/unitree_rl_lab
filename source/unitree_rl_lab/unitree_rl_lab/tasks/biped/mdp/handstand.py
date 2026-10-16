"""The bipedal-stance command: which end the robot is standing on.

Ported from ``feat/multi-task``'s ``HandstandCommand``, cut down to the single-skill case. There the
stance was also scheduled, released and descended from inside a mixture-of-experts episode; here it
is pinned on for the whole episode, which is also how that branch trained its biped expert --
``feat/biped`` found that learning the stance and learning *when* to switch into it at the same time
did not work, and pinning the mode was what did.

The single-stance tasks hold one stance and do not observe the term (a pinned ``(enabled, stance)``
is a constant); it supplies the posture quantities the rewards share -- pitch alignment, roll error,
the upright ramp that gates the walking rewards, lifted-end contact -- and the ``handstand/*``
metrics. The two-stance task sets ``stance_front_probability``: the stance is drawn per episode and
the policy reads it from the ``stance`` column.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STANCE_FRONT = 1.0
"""Standing on the front legs, hind legs lifted: the body pitches nose-down."""

STANCE_HIND = -1.0
"""Standing on the hind legs, front legs lifted: the body pitches nose-up."""


class HandstandCommand(CommandTerm):
    """Commands a two-legged stance, held for the whole episode: ``[enabled, stance]``."""

    cfg: HandstandCommandCfg

    def __init__(self, cfg: HandstandCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.stance = torch.full((self.num_envs,), cfg.stance, device=self.device)
        self.front_foot_ids, _ = self.robot.find_bodies(cfg.front_foot_names, preserve_order=True)
        self.hind_foot_ids, _ = self.robot.find_bodies(cfg.hind_foot_names, preserve_order=True)

        for name in ("pitch_alignment", "roll_error", "base_height", "upright", "airborne", "success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # Split by stance: the population mean cannot tell "both stances work" from "one works and
        # the other never leaves the floor", and collapsing onto one stance is how the two-stance
        # task is expected to fail if it fails.
        for name in ("success_front", "success_hind", "share_front"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    # -- state ------------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        enabled = self.enabled.float()
        return torch.stack((enabled, self.stance * enabled), dim=-1)

    @property
    def pitch_alignment(self) -> torch.Tensor:
        """``cos`` of the angle between world-down and the direction the stance points, in ``[-1, 1]``.

        Flat is 0 and +1 is the body pitched a full 90 degrees into the commanded stance, read
        straight off projected gravity: its body-x component is positive nose-down (front stance)
        and negative nose-up, and the stance sign folds both into one expression.
        """
        return self.stance * self.robot.data.projected_gravity_b[:, 0]

    @property
    def roll_error(self) -> torch.Tensor:
        """Gravity's body-y component: 0 upright, +-1 lying on a side.

        Separate from the pitch so that toppling sideways is not paid like rising.
        """
        return self.robot.data.projected_gravity_b[:, 1]

    @property
    def is_upright(self) -> torch.Tensor:
        """Arrived in the commanded stance: pitch alignment above ``success_alignment``."""
        return self.enabled & (self.pitch_alignment > self.cfg.success_alignment)

    @property
    def upright_ramp(self) -> torch.Tensor:
        """How far into the stance the robot is, as a weight in ``[0, 1]``.

        Gates the walking rewards. Paying velocity tracking from a four-legged stance rewards never
        attempting the rise; a hard step at the success threshold puts a cliff in the middle of it.
        The ramp from ``gate_alignment_low`` to ``success_alignment`` avoids both.
        """
        low, high = self.cfg.gate_alignment_low, self.cfg.success_alignment
        ramp = (self.pitch_alignment - low) / max(high - low, 1.0e-6)
        return ramp.clamp(0.0, 1.0) * self.enabled.float()

    @property
    def lifted_contact(self) -> torch.Tensor:
        """Fraction of the lifted end's feet touching the ground, in ``[0, 1]``."""
        sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        threshold = self.cfg.contact_threshold

        def touching(ids: list[int]) -> torch.Tensor:
            forces = sensor.data.net_forces_w[:, ids, :]
            return (torch.linalg.norm(forces, dim=-1) > threshold).float().mean(dim=-1)

        return torch.where(self.stance > 0, touching(self.hind_foot_ids), touching(self.front_foot_ids))

    @property
    def success(self) -> torch.Tensor:
        """In the commanded stance with the lifted end fully off the ground."""
        return self.is_upright & (self.lifted_contact == 0.0)

    # -- command term interface ------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        self.enabled[env_ids] = True
        if self.cfg.stance_front_probability is None:
            self.stance[env_ids] = self.cfg.stance
            return
        # Per episode, not per step: reversing a 90-degree pitch on no notice is a different skill.
        probability = self.cfg.stance_front_probability
        if self.cfg.balance_stance_population:
            # Steer toward the target share of *running* environments, not of draws: a stance whose
            # episodes end quickly is under-represented in every batch for as long as it fails.
            keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            keep[env_ids] = False
            if bool(keep.any()):
                share = (self.stance[keep] > 0).float().mean().item()
                probability = min(max(probability + (probability - share), 0.0), 1.0)
        draw = torch.rand(len(env_ids), device=self.device)
        self.stance[env_ids] = torch.where(
            draw < probability,
            torch.full_like(draw, STANCE_FRONT),
            torch.full_like(draw, STANCE_HIND),
        )

    def _update_command(self):
        pass

    def _update_metrics(self):
        enabled = self.enabled.float()
        self.metrics["pitch_alignment"][:] = self.pitch_alignment * enabled
        self.metrics["roll_error"][:] = self.roll_error.abs() * enabled
        self.metrics["base_height"][:] = self.robot.data.root_pos_w[:, 2] * enabled
        self.metrics["upright"][:] = self.is_upright.float()
        self.metrics["airborne"][:] = (1.0 - self.lifted_contact) * enabled
        self.metrics["success"][:] = self.success.float()
        # Conditional means written into every slot, so the manager's average reproduces them.
        front = self.stance > 0
        success = self.success.float()
        self.metrics["share_front"][:] = front.float().mean()
        for name, mask in (("success_front", front), ("success_hind", ~front)):
            self.metrics[name][:] = success[mask].mean() if bool(mask.any()) else 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class HandstandCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandstandCommand`."""

    class_type: type = HandstandCommand

    asset_name: str = MISSING

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Required by ``CommandTermCfg`` and meaningless here: the stance is held for the episode."""

    stance: float = STANCE_FRONT
    """Which end stands, when ``stance_front_probability`` is None."""

    stance_front_probability: float | None = None
    """Probability of drawing the front stance, per episode. ``None`` pins :attr:`stance`."""

    balance_stance_population: bool = False
    """Hold the share of *running* environments in each stance at ``stance_front_probability``
    instead of the share of draws. With plain draws, a stance the policy cannot hold yet ends its
    episodes in a few steps and so fills only a sliver of each batch -- starting from the hind
    policy, front-commanded environments were 2% of the population and the front stance did not
    start to learn."""

    success_alignment: float = 0.93
    """``pitch_alignment`` at which the stance counts as achieved -- about 68 degrees of pitch, just
    under the 75 degrees ``feat/biped``'s front stance settles at. ``feat/multi-task`` first set it at
    0.6 (37 degrees) and the policy held a crouch it was already being paid in full for."""

    gate_alignment_low: float = 0.6
    """Where the walking-reward gate starts to open, ramping to full weight at ``success_alignment``."""

    front_foot_names: tuple[str, ...] = ("FR_foot", "FL_foot")
    hind_foot_names: tuple[str, ...] = ("RR_foot", "RL_foot")

    contact_sensor_name: str = "contact_forces"
    contact_threshold: float = 1.0
