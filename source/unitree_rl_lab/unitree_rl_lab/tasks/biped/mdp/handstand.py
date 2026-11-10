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

The switching task sets ``scheduled``: the episode starts on four wheels and the stance is switched
on and off on a timetable -- on at a drawn time, held for a drawn duration, released, and on again
after a drawn rest -- so the policy learns to rise on command, come back down when the command
drops, and drive on four wheels in between. Its gates read :attr:`HandstandCommand.stance_weight`
(which reward set is in force) and :attr:`HandstandCommand.descending` (the grace after a release
in which the robot may still be tilted). The scheduling follows ``feat/multi-task``'s
``HandstandCommand`` (d6510da), cycled instead of once per episode.
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
        # Schedule, in policy steps of the episode. Unused (trigger at step 0, never released) unless
        # `scheduled`.
        self.trigger_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.release_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.next_trigger_time = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.hold_duration = torch.full((self.num_envs,), float("inf"), device=self.device)
        # Whether the robot got back down after the last release, latched over the grace: the
        # descent is a transient and reading it at an arbitrary instant would miss most of them.
        self._recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Indices into the *contact sensor's* bodies, which are ordered differently from the
        # articulation's (depth-first per leg vs. breadth-first): on Go2w the articulation's
        # FR_foot is 16, the sensor's 16 is RR_thigh. Resolved on first use, once the sensor
        # has its body list.
        self._foot_ids: tuple[list[int], list[int]] | None = None

        for name in ("pitch_alignment", "roll_error", "base_height", "upright", "airborne", "success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # Split by stance: the population mean cannot tell "both stances work" from "one works and
        # the other never leaves the floor", and collapsing onto one stance is how the two-stance
        # task is expected to fail if it fails.
        for name in ("success_front", "success_hind", "share_front", "stance_on", "descent_success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    # -- state ------------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        enabled = self.enabled.float()
        return torch.stack((enabled, self.stance * enabled), dim=-1)

    def _seconds_since(self, step: torch.Tensor) -> torch.Tensor:
        """Seconds since ``step`` of the episode; ``-inf`` where ``step`` is unset (< 0)."""
        elapsed = (self._env.episode_length_buf - step).float() * self._env.step_dt
        return torch.where(step >= 0, elapsed, torch.full_like(elapsed, float("-inf")))

    @property
    def elapsed_since_trigger(self) -> torch.Tensor:
        """Seconds since the stance was last switched on; ``-inf`` before the first trigger."""
        return self._seconds_since(self.trigger_step)

    @property
    def since_release(self) -> torch.Tensor:
        """Seconds since the stance was last released; ``-inf`` while held or never released."""
        return torch.where(
            self.enabled, torch.full_like(self.next_trigger_time, float("-inf")), self._seconds_since(self.release_step)
        )

    @property
    def stance_weight(self) -> torch.Tensor:
        """Which reward set is in force, in ``[0, 1]``: 1 the stance's, 0 four-wheel driving.

        Switched on at once at the trigger (the stance rewards are what pay for rising, so fading
        them in would only delay it) and faded out over ``crossfade_s`` after the release, so the
        four-wheel rewards return as a pull back down rather than as a step while the robot is
        still pitched 75 degrees.
        """
        since = self.since_release
        fade = (1.0 - since / max(self.cfg.crossfade_s, 1.0e-6)).clamp(0.0, 1.0)
        # Before the first trigger `since` is -inf: four wheels, weight 0.
        fade = torch.where(since >= 0.0, fade, torch.zeros_like(fade))
        return torch.where(self.enabled, torch.ones_like(fade), fade)

    @property
    def descending(self) -> torch.Tensor:
        """Inside the grace after a release, in which the robot may still be tilted.

        Without it the four-wheel orientation limit returns at the release, with the robot still
        75 degrees from level, and ends the episode on the next step -- on ``feat/multi-task`` every
        released environment terminated at exactly release + 0.02 s, so the descent was never
        reached, let alone learned. Contact of the trunk with the floor still ends the episode.
        """
        since = self.since_release
        return (since >= 0.0) & (since < self.cfg.release_grace_s)

    @property
    def is_quadruped(self) -> torch.Tensor:
        """Back down: level, carrying itself, and standing on the end it had lifted.

        The tilt is the angle ``bad_orientation`` measures (a planar gravity norm cannot tell level
        from upside down); the height floor separates standing from lying on the floor; the lifted
        end's contact is what says the stance ended rather than the robot merely being level.
        """
        tilt = torch.acos(-self.robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0))
        return (
            (tilt < self.cfg.recovered_tilt_angle)
            & (self.lifted_contact > 0.0)
            & (self.robot.data.root_pos_w[:, 2] > self.cfg.recovered_height)
        )

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
        if self._foot_ids is None:
            self._foot_ids = (
                sensor.find_bodies(list(self.cfg.front_foot_names), preserve_order=True)[0],
                sensor.find_bodies(list(self.cfg.hind_foot_names), preserve_order=True)[0],
            )
        front_ids, hind_ids = self._foot_ids
        threshold = self.cfg.contact_threshold

        def touching(ids: list[int]) -> torch.Tensor:
            forces = sensor.data.net_forces_w[:, ids, :]
            return (torch.linalg.norm(forces, dim=-1) > threshold).float().mean(dim=-1)

        return torch.where(self.stance > 0, touching(hind_ids), touching(front_ids))

    @property
    def success(self) -> torch.Tensor:
        """In the commanded stance with the lifted end fully off the ground."""
        return self.is_upright & (self.lifted_contact == 0.0)

    # -- command term interface ------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        if not self.cfg.scheduled:
            self.enabled[env_ids] = True
            self.trigger_step[env_ids] = 0
            self._draw_stance(env_ids)
            return
        # Every episode starts on four wheels; a share of them never stands at all.
        n = len(env_ids)
        self.enabled[env_ids] = False
        self.trigger_step[env_ids] = -1
        self.release_step[env_ids] = -1
        self._recovered[env_ids] = False
        first = torch.empty(n, device=self.device).uniform_(*self.cfg.trigger_time_range)
        never = torch.rand(n, device=self.device) < self.cfg.never_stand_probability
        self.next_trigger_time[env_ids] = torch.where(never, torch.full_like(first, float("inf")), first)

    def _draw_stance(self, env_ids: Sequence[int] | torch.Tensor):
        if self.cfg.stance_front_probability is None:
            self.stance[env_ids] = self.cfg.stance
            return
        # Per stance, not per step: reversing a 90-degree pitch on no notice is a different skill.
        probability = self.cfg.stance_front_probability
        if self.cfg.balance_stance_population:
            # Steer toward the target share of *running* stances, not of draws: a stance whose
            # episodes end quickly is under-represented in every batch for as long as it fails.
            # Counted over the environments standing (all of them, when the stance is pinned on).
            keep = self.enabled.clone()
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
        if not self.cfg.scheduled:
            return
        step = self._env.episode_length_buf
        time = step.float() * self._env.step_dt

        starting = (~self.enabled & (time >= self.next_trigger_time)).nonzero(as_tuple=False).squeeze(-1)
        if len(starting) > 0:
            self._draw_stance(starting)
            self.enabled[starting] = True
            self.trigger_step[starting] = step[starting]
            self.release_step[starting] = -1
            self._recovered[starting] = False
            self.hold_duration[starting] = torch.empty(len(starting), device=self.device).uniform_(
                *self.cfg.hold_duration_range
            )

        ending = (self.enabled & (self.elapsed_since_trigger >= self.hold_duration)).nonzero(as_tuple=False).squeeze(-1)
        if len(ending) > 0:
            self.enabled[ending] = False
            self.release_step[ending] = step[ending]
            rest = torch.empty(len(ending), device=self.device).uniform_(*self.cfg.rest_duration_range)
            self.next_trigger_time[ending] = time[ending] + rest

        descending = self.descending
        if bool(descending.any()):
            self._recovered |= descending & self.is_quadruped

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
        # Success is conditioned on the stance being commanded: with the stance scheduled, a
        # population mean would mostly count environments driving on four wheels.
        for name, mask in (("success_front", front & self.enabled), ("success_hind", ~front & self.enabled)):
            self.metrics[name][:] = success[mask].mean() if bool(mask.any()) else 0.0
        self.metrics["stance_on"][:] = self.enabled.float().mean()
        # The share of releases that reached a level four-wheel stance inside the grace, over the
        # environments whose last release is at least a grace old (the verdict is in).
        decided = self.since_release >= self.cfg.release_grace_s
        self.metrics["descent_success"][:] = (
            self._recovered[decided].float().mean() if bool(decided.any()) else 0.0
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class HandstandCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandstandCommand`."""

    class_type: type = HandstandCommand

    asset_name: str = MISSING

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Required by ``CommandTermCfg`` and meaningless here: resampled at reset only; the schedule,
    if any, runs in ``_update_command``."""

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

    # -- the switching task (``scheduled``) ------------------------------------------------------

    scheduled: bool = False
    """Switch the stance on and off on a timetable instead of holding it for the whole episode."""

    trigger_time_range: tuple[float, float] = (0.5, 4.0)
    """When the first stance is called for, in seconds from the start of the episode."""

    hold_duration_range: tuple[float, float] = (3.0, 6.0)
    """How long each stance is held before it is released."""

    rest_duration_range: tuple[float, float] = (2.0, 5.0)
    """Four-wheel driving between a release and the next stance."""

    never_stand_probability: float = 0.2
    """Share of episodes in which no stance is called for at all -- pure four-wheel driving."""

    crossfade_s: float = 0.25
    """Fade from the stance rewards to the four-wheel ones after a release (go2's value)."""

    release_grace_s: float = 1.5
    """Time after a release in which the robot may still be tilted (go2's value, measured there as
    what pitching down, catching the fall on the lifted pair and settling takes)."""

    recovered_tilt_angle: float = 0.4
    """Tilt under which the robot counts as back down, in radians: half the 0.8 rad limit."""

    recovered_height: float = 0.25
    """Root height above which the robot counts as carrying itself (standing is ~0.40 m)."""
