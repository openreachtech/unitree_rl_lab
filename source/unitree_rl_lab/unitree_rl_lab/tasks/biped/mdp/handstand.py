"""The bipedal-stance command: which end the robot is standing on, and when.

The quadruped rises onto two legs and walks there. Two things about this task shape the command
term, and both come from the recorded history of ``feat/biped``'s own attempts:

*The stance is a mode, not an event.* An acrobatic move is a one-shot: it fires, runs for about a
second, and hands the robot back. A handstand is entered and then *held* while the robot tracks a
velocity command on two legs. So this command has no ``rearm``, no motion code, and no per-attempt
scoring; it has an on-state with a duration, and the rewards that describe bipedal walking are
switched on for exactly that span.

*Learning the stance and learning when to switch are separate problems, and doing both at once did
not work.* ``feat/biped``'s sandbox spent a whole round on it -- probabilistic mode resampling, a
scripted quad/biped schedule with an assist force, a single transition per episode, and a direct
multimodal edit -- and every variant produced a policy that provably received the mode observation
(verified against the printed observation table) and never changed behaviour with it, up to 9900
iterations. What worked was pinning the mode on and training the stance alone, leaving the
observation slot occupied by a real if constant signal, so a later stage could vary it without
changing the network's shape.

``pinned`` is that first stage and is the default. The merged policy needs no second stage of the
same kind, because there the mixture's *gate* is what decides which expert drives -- the same
division of labour the acrobatics expert already relies on.
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
    """Commands a two-legged stance: ``[enabled, stance]``.

    ``stance`` is :data:`STANCE_FRONT` or :data:`STANCE_HIND` while enabled and 0 otherwise, so the
    two columns are never both meaningless and the mirror stance can be trained later without
    moving any other column of the observation.

    The term exposes ``enabled``, ``trigger_step`` and ``elapsed_since_trigger`` under the same
    names the jump command uses, because the reward gate reads those names and there is no reason
    for two commands in one environment to describe "a mode is running" differently.
    """

    cfg: HandstandCommandCfg

    def __init__(self, cfg: HandstandCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stance = torch.full((self.num_envs,), cfg.stance, device=self.device)
        self.trigger_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.scheduled_trigger_time = torch.zeros(self.num_envs, device=self.device)
        self.hold_duration = torch.zeros(self.num_envs, device=self.device)

        # Both pairs, always. Which is the stance and which is lifted follows `stance` per
        # environment -- see `lifted_contact`.
        self.front_foot_ids, _ = self.robot.find_bodies(cfg.front_foot_names, preserve_order=True)
        self.hind_foot_ids, _ = self.robot.find_bodies(cfg.hind_foot_names, preserve_order=True)

        for name in ("pitch_alignment", "roll_error", "base_height", "upright", "airborne", "success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # Split by stance, because the population mean cannot distinguish "both stances work" from
        # "one works and the other never leaves the floor" -- and collapsing onto a single stance is
        # the specific way the unified task is expected to fail if it fails.
        for name in ("success_front", "success_hind", "share_front"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["takeoff_speed_limit"] = torch.zeros(self.num_envs, device=self.device)

        # Whether the stance was reached at any point in its window, latched, plus the running
        # tallies the take-off curriculum decides on. Latched rather than sampled at the end
        # because a window that ends counts as a success if the robot ever got up in it -- reading
        # `success` at the closing step would only ever catch the ones still holding it.
        self._achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.attempts = 0
        self.successes = 0
        self.takeoff_speed_limit = float(cfg.initial_takeoff_speed_limit)


    # -- state ------------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        enabled = self.enabled.float()
        return torch.stack((enabled, self.stance * enabled), dim=-1)

    @property
    def commanded_speed(self) -> torch.Tensor:
        """Magnitude of the commanded planar velocity, which the take-off gate reads."""
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        return torch.linalg.norm(command[:, :2], dim=-1)

    @property
    def elapsed_since_trigger(self) -> torch.Tensor:
        """Seconds since the stance was commanded; 0 before the first trigger.

        Retained after the command turns off, matching ``JumpCommand``, so a gate can keep a term
        alive through the settle-back into a quadruped stance.
        """
        elapsed_steps = self._env.episode_length_buf - self.trigger_step
        return torch.where(
            self.trigger_step >= 0,
            elapsed_steps.float() * self._env.step_dt,
            torch.zeros(self.num_envs, device=self.device),
        )

    @property
    def pitch_alignment(self) -> torch.Tensor:
        """``cos`` of the angle between world-down and the direction the stance points, in ``[-1, 1]``.

        Reads straight off projected gravity: flat is 0, and +1 is the body pitched a full 90
        degrees into the commanded stance. Equivalent to the paper's *Base Pitch* term
        (``-cos(p^c - p)`` with ``p^c = +-90`` degrees) but written in the quantity the observation
        already carries, so there is no angle extraction and no wrap-around to get wrong.

        The stance sign is what makes one expression serve both ends of the robot: gravity's body-x
        component is positive when the nose is down (front stance) and negative when it is up.
        """
        return self.stance * self.robot.data.projected_gravity_b[:, 0]

    @property
    def roll_error(self) -> torch.Tensor:
        """Gravity's body-y component: 0 upright, +-1 lying on a side.

        Tracked separately from the pitch because a tilt reward that does not distinguish them is
        indifferent between rising and toppling sideways. ``feat/biped``'s hardware trace is the
        cautionary case: the robot reached 43 degrees of sagittal lean with 33 degrees of lateral,
        the lateral rise leading the sagittal one.
        """
        return self.robot.data.projected_gravity_b[:, 1]

    @property
    def is_upright(self) -> torch.Tensor:
        """Whether the robot has actually arrived in the commanded stance.

        A hard, high threshold, because this is what :attr:`success` and the completion reward are
        built on. The first run of this task set it at 0.6 -- 37 degrees of pitch -- and shared it
        with the reward gate, which turned out to be the whole problem: the completion term banked
        +1.0 at a shallow crouch and stayed banked, while everything asking the robot to stand
        taller totalled about 0.03 per step. The measured stance settled at 60 degrees and 0.187 m
        of stance-hip height, 17 cm short of the reference, and crept upward for 2000 iterations
        without arriving. See ``sandbox/SUMMARY.md``.
        """
        return self.enabled & (self.pitch_alignment > self.cfg.success_alignment)

    @property
    def upright_ramp(self) -> torch.Tensor:
        """How far into the stance the robot is, as a weight in ``[0, 1]``.

        What the reward gate uses, instead of :attr:`is_upright`. The paper gates its tracking and
        balance rewards on being upright ("if is upright, else 0") and the reason holds -- velocity
        tracking on two legs is only meaningful once the robot is on two legs, and paying it from a
        quadruped stance rewards not attempting the task. But a step at the success threshold puts
        a cliff in the middle of the rise: everything the robot is working toward arrives at once,
        and nothing grades the approach.

        Ramping from ``gate_alignment_low`` to ``success_alignment`` keeps the gradient continuous
        across the whole rise and still puts the full value at the top, where the posture is
        actually good.
        """
        low, high = self.cfg.gate_alignment_low, self.cfg.success_alignment
        ramp = (self.pitch_alignment - low) / max(high - low, 1.0e-6)
        return ramp.clamp(0.0, 1.0) * self.enabled.float()

    @property
    def lifted_contact(self) -> torch.Tensor:
        """Fraction of the *lifted* end's feet touching the ground, in ``[0, 1]``, per environment.

        Which pair is lifted follows the commanded stance, so this reads correctly in the unified
        task where neighbouring environments are standing on opposite ends of the robot.

        It was a fixed list once, and the unified task set that list to all four feet on the
        reasoning that both pairs are candidates there. That silently broke :attr:`success`, which
        requires this to be exactly zero: with four feet in the list, "nothing lifted is touching"
        becomes "no foot is touching at all", which a robot standing on two legs never satisfies.
        The metric then counted the moments all four feet happened to leave the ground rather than
        the stance being held -- so it still separated a real two-legged gait from a tripod, and the
        comparisons drawn from it were not meaningless, but the number it reported was not the
        success rate it was labelled as.
        """
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

    # -- external control -------------------------------------------------------------------

    def set_command(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        enabled: bool,
        stance: float | None = None,
        hold_duration: float | None = None,
    ) -> None:
        """Command a stance from outside, the counterpart of ``JumpCommand.set_command``.

        Enabling does not flip ``enabled`` directly. It clears the per-episode latch and makes the
        term's own schedule due at the next update, so the stance starts exactly the way a scheduled
        one does -- ``trigger_step`` stamped inside the update, the speed gate honoured, the hold
        ended by ``hold_duration`` and the attempt tallied. Disabling collapses the hold so the same
        ending path runs at the next update. Requires ``pinned=False``: a pinned term never updates.

        ``hold_duration`` of ``None`` holds until released.
        """
        if self.cfg.pinned:
            raise RuntimeError("HandstandCommand.set_command needs pinned=False")
        if enabled:
            if stance is not None:
                self.stance[env_ids] = stance
            self.hold_duration[env_ids] = float("inf") if hold_duration is None else hold_duration
            self.enabled[env_ids] = False
            self.trigger_step[env_ids] = -1
            self.scheduled_trigger_time[env_ids] = 0.0
            self._achieved[env_ids] = False
        else:
            self.hold_duration[env_ids] = 0.0
            self.scheduled_trigger_time[env_ids] = 1.0e9

    # -- schedule ---------------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.direction_matched:
            # The commanded heading picks the stance that goes with it: moving forward, planting the
            # front feet lets the robot's own momentum carry it over them into a nose-down rise;
            # moving backward does the same about the hind feet. The same principle the acrobatics
            # command uses to match a flip to the direction it is travelling in.
            #
            # Below `direction_conflict_speed` there is no direction to match, so the stance is
            # drawn. That covers the standstill, which is the case the expert knows best -- every
            # one of its training episodes begins at rest and rises from there.
            velocity = self._env.command_manager.get_command(self.cfg.velocity_command_name)
            forward, lateral = velocity[env_ids, 0], velocity[env_ids, 1]
            limit = self.cfg.direction_conflict_speed
            directed = torch.where(
                forward > limit,
                torch.full_like(forward, STANCE_FRONT),
                torch.full_like(forward, STANCE_HIND),
            )
            drawn = torch.where(
                torch.rand(len(env_ids), device=self.device) < 0.5,
                torch.full_like(forward, STANCE_FRONT),
                torch.full_like(forward, STANCE_HIND),
            )
            # A lateral command matches neither stance -- sideways momentum is symmetric and helps
            # neither rotation -- so those environments fall back to the draw and are then filtered
            # out by the take-off gate below.
            fore_aft = forward.abs() > lateral.abs()
            self.stance[env_ids] = torch.where(fore_aft & (forward.abs() > limit), directed, drawn)
        elif self.cfg.stance_front_probability is None:
            self.stance[env_ids] = self.cfg.stance
        else:
            # Drawn per episode, not per step: a stance is a mode the robot commits to and holds,
            # and the policy has to be able to plan a rise around it. Resampling mid-episode would
            # ask it to reverse a 90-degree pitch on no notice, which is a different skill.
            draw = torch.rand(len(env_ids), device=self.device)
            self.stance[env_ids] = torch.where(
                draw < self.cfg.stance_front_probability,
                torch.full_like(draw, STANCE_FRONT),
                torch.full_like(draw, STANCE_HIND),
            )
        if self.cfg.pinned:
            # On from the first step of the episode, for the whole episode. `trigger_step` is set
            # to the current step rather than 0 so `elapsed_since_trigger` is measured from the
            # reset, not from an episode boundary the environment has already passed.
            self.enabled[env_ids] = True
            self.trigger_step[env_ids] = self._env.episode_length_buf[env_ids]
            self.scheduled_trigger_time[env_ids] = 0.0
            self.hold_duration[env_ids] = float("inf")
            return
        self.enabled[env_ids] = False
        self.trigger_step[env_ids] = -1
        self.scheduled_trigger_time[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.trigger_time_range
        )
        self.hold_duration[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.hold_duration_range
        )
        # Only a share of episodes carry a stance at all. Holding one for ten seconds of a twenty
        # second episode halves what is left for the acrobatic moves, which interleave at 1.5-3 s
        # and were already the weaker half of the merged policy. Splitting per episode keeps the
        # stance long enough to include a rise, a hold and a descent, while leaving the other half
        # of the episodes at the acrobatics' original exposure.
        if self.cfg.episode_probability < 1.0:
            skipped = torch.rand(len(env_ids), device=self.device) >= self.cfg.episode_probability
            self.scheduled_trigger_time[env_ids] = torch.where(
                skipped, torch.full_like(self.scheduled_trigger_time[env_ids], 1.0e9),
                self.scheduled_trigger_time[env_ids],
            )
        self._achieved[env_ids] = False

    def _update_command(self):
        if self.cfg.pinned:
            self._achieved |= self.success
            return
        episode_time = self._env.episode_length_buf.float() * self._env.step_dt
        due = (~self.enabled) & (self.trigger_step < 0) & (episode_time >= self.scheduled_trigger_time)
        # Deferred rather than cancelled while the commanded speed is above the curriculum's limit,
        # so the environment stays eligible: as soon as the velocity command resamples to something
        # slower, or the limit rises, the stance happens. Same mechanism as the acrobatics
        # take-off gate, and the reason a lateral command never produces a stance -- its planar
        # speed is above the limit whenever it is worth calling lateral.
        too_fast = due & (self.commanded_speed > self.takeoff_speed_limit)
        if torch.any(too_fast):
            self.scheduled_trigger_time[too_fast] = (
                episode_time[too_fast] + self.cfg.takeoff_retry_interval_s
            )
        starting = due & ~too_fast
        self._achieved |= self.success
        if torch.any(starting):
            self.enabled[starting] = True
            self.trigger_step[starting] = self._env.episode_length_buf[starting]
        ending = self.enabled & (self.elapsed_since_trigger >= self.hold_duration)
        if torch.any(ending):
            self.enabled[ending] = False
            self.attempts += int(ending.sum().item())
            self.successes += int((self._achieved & ending).sum().item())

    def _update_metrics(self):
        enabled = self.enabled.float()
        self.metrics["pitch_alignment"][:] = self.pitch_alignment * enabled
        self.metrics["roll_error"][:] = self.roll_error.abs() * enabled
        self.metrics["base_height"][:] = self.robot.data.root_pos_w[:, 2] * enabled
        self.metrics["upright"][:] = self.is_upright.float()
        self.metrics["airborne"][:] = (1.0 - self.lifted_contact) * enabled
        self.metrics["success"][:] = self.success.float()
        self.metrics["takeoff_speed_limit"][:] = self.takeoff_speed_limit

        # Conditional means, written into every slot so the manager's own average over environments
        # reproduces them. A stance with no environments in it reports 0 rather than a stale value.
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
    """Required by ``CommandTermCfg`` and meaningless here: the stance is scheduled per episode by
    this term, not resampled on a timer, so it is pushed beyond any episode. Same convention the
    jump command uses."""

    direction_matched: bool = False
    """Choose the stance from the commanded heading rather than by drawing it. Forward takes the
    front stance and backward the hind one, because the robot's own momentum carries it over the
    planted feet in exactly that direction. Off for the single-stance and unified biped tasks,
    which pin or draw instead."""

    episode_probability: float = 1.0
    """Share of episodes that carry a stance at all. Below 1.0 the rest of the episode is ordinary
    locomotion and whatever else the environment schedules."""

    velocity_command_name: str = "base_velocity"

    direction_conflict_speed: float = 0.3
    """Commanded speed below which no heading is worth matching, and the stance is drawn."""

    initial_takeoff_speed_limit: float = 0.3
    takeoff_retry_interval_s: float = 0.5
    """The rise from a moving start is unlearned -- the expert's episodes all begin at rest -- so
    the stance is only offered below a commanded speed a curriculum raises as rises keep landing.
    Mirrors the acrobatics take-off gate, including deferring rather than cancelling."""

    stance_front_probability: float | None = None
    """Probability of drawing the front stance, resampled per episode. ``None`` pins every
    environment to :attr:`stance` instead.

    Set for the unified task, where one network serves both stances and the observation's stance
    column is what tells it which. ``feat/biped`` spent a whole round failing to make one policy
    respond to a mode observation, so this is not free -- but its modes were quadruped against
    biped, where one is far easier and collects most of the reward whatever the command says.
    Front against hind are siblings of comparable difficulty with no easy option to collapse onto,
    and unlike that round this one starts from two policies that already work.
    """

    stance: float = STANCE_FRONT
    """Which end stands, when ``stance_front_probability`` is None. The front stance is the one to train first: ``feat/biped`` got both
    working, and the hind-leg one needed seven sandbox rounds to stop shuffling on a tripod while
    the front stance did not."""

    pinned: bool = True
    """Hold the stance on for the whole episode. See this module's docstring for why the first
    stage does not ask the policy to learn the switch as well."""

    trigger_time_range: tuple[float, float] = (2.0, 6.0)
    hold_duration_range: tuple[float, float] = (6.0, 12.0)
    """When ``pinned`` is off: when the stance starts, and how long it is held. Unused otherwise."""

    success_alignment: float = 0.93
    """``pitch_alignment`` at which the stance counts as achieved -- about 68 degrees of pitch, just
    under the 75 degrees ``feat/biped``'s front stance settles at.

    This is the bar the completion reward has to clear, so it has to describe the posture actually
    wanted rather than the point where the rise becomes recognisable. Set at 0.6 (37 degrees) in the
    first run, and shared with the gate below; that combination is what left the policy holding a
    crouch it was already being paid in full for."""

    gate_alignment_low: float = 0.6
    """Where the reward gate starts to open, ramping to full weight at ``success_alignment``. Kept
    low so the terms that describe bipedal walking begin to apply while the robot is still on its
    way up, rather than switching on all at once at the top."""

    front_foot_names: tuple[str, ...] = ("FR_foot", "FL_foot")
    hind_foot_names: tuple[str, ...] = ("RR_foot", "RL_foot")
    """The two pairs, named by where they are on the robot rather than by their role. The role is
    decided per environment by :attr:`stance`, which is what lets one command term serve a task
    where both stances are running at once."""

    contact_sensor_name: str = "contact_forces"
    contact_threshold: float = 1.0
