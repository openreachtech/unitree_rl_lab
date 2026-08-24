from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JumpCommand(CommandTerm):
    """One-shot jump command with command-edge-triggered physical assistance.

    The policy command is ``[enabled, target_height, target_pitch_turns,
    target_roll_turns]``. Rotation targets are expressed in turns rather than
    radians to keep their scale close to one. During training, ``enabled``
    rises at a sampled time. Assistance starts on that rising edge and lasts
    for ``assist_duration_s``; keeping the command high does not retrigger it.
    """

    cfg: JumpCommandCfg
    MOTION_JUMP = 1
    MOTION_BACKFLIP = 2
    MOTION_SIDEFLIP = 3

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_ids, self.assist_body_names = self.robot.find_bodies(
            cfg.assist_body_names, preserve_order=True
        )
        if len(self.body_ids) == 0:
            raise ValueError(f"No assist bodies matched: {cfg.assist_body_names}")
        body_index = {name: index for index, name in enumerate(self.assist_body_names)}

        def resolve_profile(names: tuple[str, ...]) -> list[int]:
            profile_names = names or tuple(self.assist_body_names)
            missing = set(profile_names) - set(body_index)
            if missing:
                raise ValueError(f"Assist profile bodies are not in assist_body_names: {sorted(missing)}")
            return [body_index[name] for name in profile_names]

        self.jump_force_indices = resolve_profile(cfg.jump_assist_body_names)
        self.backflip_force_indices = resolve_profile(cfg.backflip_assist_body_names)
        self.sideflip_force_indices = resolve_profile(cfg.sideflip_assist_body_names)

        self.jump_assist_mass = (
            cfg.jump_assist_mass
            if cfg.jump_assist_mass is not None
            else float(self.robot.data.default_mass[0].sum().item())
        )

        self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_enabled = torch.zeros_like(self.enabled)
        self.command_issued = torch.zeros_like(self.enabled)
        self.motion_code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_height = torch.zeros(self.num_envs, device=self.device)
        # What the launch force is sized for, as opposed to what the reward asks for. These
        # are the same number for a jump and deliberately different for a flip -- see
        # `flip_launch_height`.
        self.launch_height = torch.zeros(self.num_envs, device=self.device)
        self.target_pitch_turns = torch.zeros(self.num_envs, device=self.device)
        self.target_roll_turns = torch.zeros(self.num_envs, device=self.device)
        self.accumulated_pitch = torch.zeros(self.num_envs, device=self.device)
        self.accumulated_roll = torch.zeros(self.num_envs, device=self.device)
        self.scheduled_trigger_time = torch.zeros(self.num_envs, device=self.device)
        self.trigger_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.standing_height = torch.full(
            (self.num_envs,), cfg.nominal_standing_height, dtype=torch.float, device=self.device
        )
        self.max_height = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.assist_scale = cfg.initial_assist_scale
        self.curriculum_success_rate = 0.0
        self.curriculum_episode_count = 0
        self.curriculum_success_count = 0
        self.curriculum_episode_count_by_motion = torch.zeros(4, dtype=torch.long, device=self.device)
        self.curriculum_success_count_by_motion = torch.zeros(4, dtype=torch.long, device=self.device)

        if cfg.state_file is not None and os.path.isfile(cfg.state_file):
            with open(cfg.state_file) as f:
                saved_state = json.load(f)
            self.assist_scale = saved_state["assist_scale"]
            self.curriculum_success_rate = saved_state["curriculum_success_rate"]
            self.curriculum_episode_count_by_motion = torch.tensor(
                saved_state["curriculum_episode_count_by_motion"], dtype=torch.long, device=self.device
            )
            self.curriculum_success_count_by_motion = torch.tensor(
                saved_state["curriculum_success_count_by_motion"], dtype=torch.long, device=self.device
            )

        self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assist_scale"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        enabled = self.enabled.float()
        return torch.stack(
            (
                enabled,
                self.target_height * enabled,
                self.target_pitch_turns * enabled,
                self.target_roll_turns * enabled,
            ),
            dim=-1,
        )

    @property
    def height_delta(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.standing_height

    @property
    def time_since_trigger(self) -> torch.Tensor:
        return torch.where(
            self.enabled,
            self.elapsed_since_trigger,
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        )

    @property
    def elapsed_since_trigger(self) -> torch.Tensor:
        """Elapsed time retained after the command turns off."""
        elapsed_steps = self._env.episode_length_buf - self.trigger_step
        return torch.where(
            self.trigger_step >= 0,
            elapsed_steps.float() * self._env.step_dt,
            torch.zeros_like(elapsed_steps, dtype=torch.float),
        )

    def set_command(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        enabled: bool,
        target_height: float | None = None,
        target_pitch_turns: float | None = None,
        target_roll_turns: float | None = None,
    ) -> None:
        """Set the command externally; a false-to-true edge triggers assistance."""
        self.enabled[env_ids] = enabled
        if target_height is not None:
            self.target_height[env_ids] = target_height
        if target_pitch_turns is not None:
            self.target_pitch_turns[env_ids] = target_pitch_turns
        if target_roll_turns is not None:
            self.target_roll_turns[env_ids] = target_roll_turns

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self.enabled[env_ids] = False
        self.previous_enabled[env_ids] = False
        self.command_issued[env_ids] = False
        self.motion_code[env_ids] = 0
        self.trigger_step[env_ids] = -1
        self.max_height[env_ids] = 0.0
        self.accumulated_pitch[env_ids] = 0.0
        self.accumulated_roll[env_ids] = 0.0
        self.success[env_ids] = False
        self.standing_height[env_ids] = self.cfg.nominal_standing_height
        self.target_height[env_ids] = 0.0
        self.launch_height[env_ids] = 0.0
        self.target_pitch_turns[env_ids] = 0.0
        self.target_roll_turns[env_ids] = 0.0

        zero_forces = torch.zeros(
            len(env_ids), len(self.body_ids), 3, dtype=torch.float, device=self.device
        )
        self.robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=torch.zeros_like(zero_forces),
            body_ids=self.body_ids,
            env_ids=env_ids,
            is_global=True,
        )

        if self.cfg.auto_trigger:
            enabled_motions = []
            if self.cfg.enable_jump:
                enabled_motions.append(self.MOTION_JUMP)
            if self.cfg.enable_backflip:
                enabled_motions.append(self.MOTION_BACKFLIP)
            if self.cfg.enable_sideflip:
                enabled_motions.append(self.MOTION_SIDEFLIP)
            if not enabled_motions:
                raise ValueError("At least one motion type must be enabled when auto_trigger=True")

            sampled_motion_indices = torch.randint(
                len(enabled_motions), (len(env_ids),), device=self.device
            )
            sampled_motion_codes = torch.tensor(enabled_motions, device=self.device)[
                sampled_motion_indices
            ]
            self.motion_code[env_ids] = sampled_motion_codes

            sampled_height = sample_uniform(
                self.cfg.target_height_range[0],
                self.cfg.target_height_range[1],
                (len(env_ids),),
                device=self.device,
            )
            sampled_pitch = sample_uniform(
                self.cfg.target_pitch_turns_range[0],
                self.cfg.target_pitch_turns_range[1],
                (len(env_ids),),
                device=self.device,
            )
            sampled_roll = sample_uniform(
                self.cfg.target_roll_turns_range[0],
                self.cfg.target_roll_turns_range[1],
                (len(env_ids),),
                device=self.device,
            )
            # Flips get `flip_target_height` rather than a flat zero. A sideflip needs air
            # time to rotate in, and with no height it stays on the ground: the one-turn
            # policy reaches max_height 0.076 m and rotates just clear of the floor, which
            # works for one turn and cannot work for two (0.36 s of flight demands
            # ~35 rad/s, against ~12.6 rad/s measured). Defaults to 0.0, so Phase 2 and
            # every existing flip task keep the previous behaviour exactly.
            self.target_height[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_JUMP,
                sampled_height,
                torch.full_like(sampled_height, self.cfg.flip_target_height),
            )
            # The launch force follows `flip_launch_height` when one is given, so the force
            # can be sized for what it takes to get the robot airborne without dragging the
            # reward target along with it. They were one number, and raising it to buy
            # flight time silently destroyed `height_progress`: that reward is
            # exp(-(max_height - target_height)^2 / 0.16), so a 1.50 m target against the
            # 0.66 m actually reached reads 0.012 instead of 0.975 against 0.60 m. Zero
            # means "same as the target", i.e. the previous behaviour exactly.
            launch_flip = (
                self.cfg.flip_launch_height
                if self.cfg.flip_launch_height > 0.0
                else self.cfg.flip_target_height
            )
            self.launch_height[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_JUMP,
                sampled_height,
                torch.full_like(sampled_height, launch_flip),
            )
            self.target_pitch_turns[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_BACKFLIP, sampled_pitch, 0.0
            )
            self.target_roll_turns[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_SIDEFLIP, sampled_roll, 0.0
            )
            self.scheduled_trigger_time[env_ids] = sample_uniform(
                self.cfg.trigger_time_range[0],
                self.cfg.trigger_time_range[1],
                (len(env_ids),),
                device=self.device,
            )
        else:
            self.scheduled_trigger_time[env_ids] = 0.0

    def _update_metrics(self):
        self.metrics["max_height"][:] = self.max_height
        self.metrics["success"][:] = self.success.float()
        self.metrics["assist_scale"][:] = self.assist_scale

    def _update_command(self):
        if self.cfg.auto_trigger:
            scheduled_on = (
                self._env.episode_length_buf.float() * self._env.step_dt
                >= self.scheduled_trigger_time
            )
            self.enabled |= scheduled_on & ~self.command_issued

        rising_edge = self.enabled & ~self.previous_enabled
        if torch.any(rising_edge):
            self.command_issued[rising_edge] = True
            self.success[rising_edge] = False
            self.trigger_step[rising_edge] = self._env.episode_length_buf[rising_edge]
            self.max_height[rising_edge] = 0.0
            self.accumulated_pitch[rising_edge] = 0.0
            self.accumulated_roll[rising_edge] = 0.0

        active = self.trigger_step >= 0
        self.max_height[active] = torch.maximum(self.max_height[active], self.height_delta[active])
        self.accumulated_roll[active] += self.robot.data.root_ang_vel_b[active, 0] * self._env.step_dt
        self.accumulated_pitch[active] += self.robot.data.root_ang_vel_b[active, 1] * self._env.step_dt

        upright = self.robot.data.projected_gravity_b[:, 2] < self.cfg.landing_upright_threshold
        landed = (
            active
            & (self.elapsed_since_trigger >= self.cfg.minimum_landing_time_s)
            & (torch.abs(self.height_delta) < self.cfg.landing_height_tolerance)
            & (
                torch.abs(self.robot.data.root_lin_vel_w[:, 2])
                < self.cfg.landing_vertical_speed_tolerance
            )
            & upright
        )
        jump_target_reached = (
            torch.abs(self.max_height - self.target_height) < self.cfg.height_tolerance
        )
        pitch_target = self.target_pitch_turns * (2.0 * math.pi)
        roll_target = self.target_roll_turns * (2.0 * math.pi)
        backflip_target_reached = (
            torch.abs(self.accumulated_pitch - pitch_target)
            < self.cfg.rotation_tolerance_rad
        )
        sideflip_target_reached = (
            torch.abs(self.accumulated_roll - roll_target)
            < self.cfg.rotation_tolerance_rad
        )
        reached_target = (
            ((self.motion_code == self.MOTION_JUMP) & jump_target_reached)
            | ((self.motion_code == self.MOTION_BACKFLIP) & backflip_target_reached)
            | ((self.motion_code == self.MOTION_SIDEFLIP) & sideflip_target_reached)
        )
        self.success |= landed & reached_target

        self._apply_assistance()
        command_expired = self.enabled & (
            self.elapsed_since_trigger >= self.cfg.command_duration_s
        )
        self.enabled[command_expired] = False
        self.previous_enabled.copy_(self.enabled)

    def _apply_assistance(self):
        elapsed = self.elapsed_since_trigger
        delay = self.cfg.assist_delay_s
        ramp = self.cfg.assist_ramp_s
        assist_active = (
            self.enabled
            & (self.trigger_step >= 0)
            & (elapsed >= delay)
            & (elapsed < delay + ramp + self.cfg.assist_duration_s)
            & (self.assist_scale > 0.0)
        )
        # Smooth 0->1 ramp over `assist_ramp_s` (measured from the end of the delay), instead
        # of a hard step onset. With ramp == 0.0 (default) this is identically 1.0 whenever
        # active, matching prior step-function behavior.
        if ramp > 0.0:
            ramp_progress = ((elapsed - delay) / ramp).clamp(0.0, 1.0)
        else:
            ramp_progress = torch.ones_like(elapsed)

        forces = torch.zeros(
            self.num_envs, len(self.body_ids), 3, dtype=torch.float, device=self.device
        )

        # Crouch-assist: a brief downward pulse on all assist bodies, right at trigger,
        # before the launch force -- physically teaches a genuine crouch-load instead of
        # relying on reward shaping alone to elicit correct timing. Shaped as a linear
        # triangular envelope (0 -> peak -> 0) so it starts and ends at zero force, same
        # as the launch ramp's continuity, and `assist_delay_s` is expected to be set to
        # `crouch_assist_duration_s` so the launch ramp begins exactly as this ends --
        # both sides of that handoff are at ~0 force, so there's no discontinuity there
        # either. Disabled by default (crouch_assist_duration_s == 0.0).
        crouch_duration = self.cfg.crouch_assist_duration_s
        if crouch_duration > 0.0 and self.cfg.crouch_assist_force > 0.0:
            crouch_active = (
                (self.trigger_step >= 0)
                & (elapsed >= 0.0)
                & (elapsed < crouch_duration)
                & (self.assist_scale > 0.0)
            )
            if torch.any(crouch_active):
                half = crouch_duration / 2.0
                envelope = torch.minimum(elapsed / half, (crouch_duration - elapsed) / half).clamp(0.0, 1.0)
                crouch_force_per_body = (
                    self.cfg.crouch_assist_force * self.assist_scale * envelope / len(self.body_ids)
                )
                for body_index in range(len(self.body_ids)):
                    forces[crouch_active, body_index, 2] = -crouch_force_per_body[crouch_active]

        def apply_profile(
            motion_code: int,
            force_indices: list[int],
            total_force: float,
        ) -> None:
            motion_mask = assist_active & (self.motion_code == motion_code)
            if torch.any(motion_mask):
                force_per_body = total_force * self.assist_scale / len(force_indices)
                for force_index in force_indices:
                    forces[motion_mask, force_index, 2] = force_per_body * ramp_progress[motion_mask]

        # Jump assist force is derived per-env from projectile motion, following the
        # paper's f_jump(h_target): the average force needed to reach the initial
        # vertical velocity v0 = sqrt(2*g*h_target) over the assist window. By design this
        # is strong enough alone to fully launch the robot at assist_scale=1.0 -- the paper's
        # intent is for the robot to physically experience the successful trajectory early on,
        # not to require the policy's own contribution from the start.
        # Keyed on the height itself rather than on MOTION_JUMP, so a flip that has been
        # given a height gets the lift that goes with it. The force is derived from
        # target_height, so it delivers that height and no more -- unlike raising the
        # one-sided sideflip force, which lifts as much as it spins and reached
        # max_height 2.351 m when doubled for a second turn.
        jump_mask = assist_active & (self.launch_height > 0.0)
        if torch.any(jump_mask):
            initial_velocity = torch.sqrt(
                2.0 * self.cfg.gravity * self.launch_height[jump_mask].clamp(min=0.0)
            )
            total_force = self.jump_assist_mass * initial_velocity / self.cfg.assist_duration_s
            force_per_body = total_force * self.assist_scale / len(self.jump_force_indices)
            for force_index in self.jump_force_indices:
                forces[jump_mask, force_index, 2] = force_per_body * ramp_progress[jump_mask]

        apply_profile(
            self.MOTION_BACKFLIP,
            self.backflip_force_indices,
            self.cfg.backflip_assist_force,
        )
        # Sideflip assist scales with the number of turns asked for, the way the jump force
        # scales with target_height. As a fixed constant it delivered one rotation whatever
        # the target said, so a two-turn command could never reach `reached_target`,
        # `success` stayed 0, the 60%-success gate never opened and assist_scale sat at 1.0
        # -- the same deadlock seen when asking for 0.70 m.
        #
        # Being an upward force on one side, it supplies lift as well as roll torque, so
        # scaling it buys the extra flight time the extra rotation needs at the same time.
        # At |turns| = 1.0 this is exactly the previous behaviour, so Phase 2 is unchanged.
        # The couple runs on its own schedule, because sharing the launch window wasted it.
        # A 350 N couple is worth 0.284*350 = 99.4 N*m against a roll inertia of about
        # 0.166 kg*m^2, i.e. 599 rad/s^2 -- 60 rad/s over 0.1 s, more than three times the
        # ~18 rad/s two turns need. Yet the robot managed half a turn, because the window
        # (0.12 s to 0.34 s) is spent almost entirely with the feet still planted, and the
        # ground simply absorbs the torque. A standing policy under full assist showed the
        # same half turn, which is what confirmed it: the force was never the problem.
        #
        # Delaying it past take-off lets the same mechanism act on a free body, where it is
        # so effective that the magnitude has to come DOWN rather than up -- 350 N in the
        # air would be roughly nine rotations.
        sideflip_mask = assist_active & (self.motion_code == self.MOTION_SIDEFLIP)
        couple_delay = self.cfg.sideflip_couple_delay_s
        if couple_delay > 0.0:
            couple_mask = (
                (self.motion_code == self.MOTION_SIDEFLIP)
                & (self.trigger_step >= 0)
                & (elapsed >= couple_delay)
                & (elapsed < couple_delay + self.cfg.sideflip_couple_duration_s)
                & (self.assist_scale > 0.0)
            )
        else:
            couple_mask = sideflip_mask
        if torch.any(sideflip_mask):
            force_per_body = (
                self.cfg.sideflip_assist_force * self.assist_scale / len(self.sideflip_force_indices)
            )
            for force_index in self.sideflip_force_indices:
                forces[sideflip_mask, force_index, 2] = force_per_body * ramp_progress[sideflip_mask]

            # Extra roll, added as a COUPLE: up on the sideflip bodies, down on the others.
            # The one-sided force above cannot be scaled up to buy more rotation, because it
            # lifts as much as it spins -- doubling it for a second turn threw the robot to
            # max_height 2.351 m with base_contact on every episode. Geometrically the
            # one-sided force gives torque 0.142*F and translation F, while a couple gives
            # torque 0.284*F and translation 0: twice the spin per newton, and no lift at
            # all. So this knob adds rotation without touching the launch behaviour the
            # single-turn sideflip already gets right.
            #
            # Sizing, from the measured single rotation (omega ~ 12.6 rad/s off 350 N
            # one-sided, implying I_roll ~ 0.166 kg*m^2): a second turn needs roughly the
            # same angular impulse again, which a couple supplies at about half the force.
        if self.cfg.sideflip_couple_force > 0.0:
            couple_per_body = (
                self.cfg.sideflip_couple_force * self.assist_scale / len(self.sideflip_force_indices)
            )
            opposite = [i for i in range(len(self.body_ids)) if i not in self.sideflip_force_indices]
            for force_index in self.sideflip_force_indices:
                forces[couple_mask, force_index, 2] += couple_per_body
            for force_index in opposite:
                forces[couple_mask, force_index, 2] -= (
                    self.cfg.sideflip_couple_force * self.assist_scale / max(len(opposite), 1)
                )


        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=self.body_ids,
            is_global=True,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class JumpCommandCfg(CommandTermCfg):
    """Configuration for :class:`JumpCommand`."""

    class_type: type = JumpCommand
    asset_name: str = MISSING  # type: ignore[assignment]
    assist_body_names: list[str] = MISSING  # type: ignore[assignment]
    jump_assist_body_names: tuple[str, ...] = ()
    backflip_assist_body_names: tuple[str, ...] = ()
    sideflip_assist_body_names: tuple[str, ...] = ()

    state_file: str | None = None
    """Path used to persist/restore the EFGCL assist-force curriculum (assist_scale and the
    per-motion episode/success counters) across process restarts. rsl_rl checkpoints only save
    network weights, so without this, every ``--resume`` silently restarts the curriculum's
    assist-force decay from ``initial_assist_scale``."""

    auto_trigger: bool = False
    enable_jump: bool = True
    enable_backflip: bool = False
    enable_sideflip: bool = False
    trigger_time_range: tuple[float, float] = (0.8, 1.2)
    target_height_range: tuple[float, float] = (0.20, 0.20)
    target_pitch_turns_range: tuple[float, float] = (0.0, 0.0)
    target_roll_turns_range: tuple[float, float] = (0.0, 0.0)
    nominal_standing_height: float = 0.40
    flip_target_height: float = 0.0
    flip_launch_height: float = 0.0
    """Height the flip's launch force is sized for, when it should differ from
    ``flip_target_height``. The launch force is derived from a height, and that same height
    is the reward's target, so buying flight time by raising it moves the reward target out
    of reach at the same time. Zero (the default) keeps them identical, matching all prior
    behaviour."""
    """Height commanded alongside a backflip or sideflip, giving the rotation air time to
    happen in. The launch assist keys off ``target_height``, so a non-zero value here also
    turns that lift on for the flip. Defaults to 0.0, matching the previous behaviour where
    only ``MOTION_JUMP`` carried a height."""

    command_duration_s: float = 0.50
    assist_duration_s: float = 0.10
    assist_delay_s: float = 0.0
    """Delay between the trigger rising edge and the start of assist force. Gives the
    policy a windup window -- already exempt from idle-phase pose penalties since those
    gate on ``~enabled``, which flips true at the same instant as the trigger -- to crouch
    and load its legs before the shove lands, mirroring how a real quadruped briefly
    lowers its body just before push-off. Defaults to 0.0 (assist starts immediately),
    matching all prior behavior."""
    assist_ramp_s: float = 0.0
    """Duration over which assist force ramps linearly from 0 to full, starting once
    ``assist_delay_s`` has elapsed, instead of turning on as a hard step. Defaults to 0.0
    (instant full force), matching all prior behavior."""
    crouch_assist_force: float = 0.0
    """Downward force (summed across all assist bodies) applied as a brief triangular
    pulse right at trigger, before the launch assist force. Physically teaches a genuine
    crouch-load instead of relying on reward shaping alone to elicit correct timing.
    Defaults to 0.0 (disabled), matching all prior behavior. Set ``assist_delay_s`` to
    ``crouch_assist_duration_s`` so the launch force begins ramping in exactly as this
    pulse ends, avoiding a force discontinuity at the handoff."""
    crouch_assist_duration_s: float = 0.0
    """Duration of the crouch-assist pulse. Ramps 0 -> peak -> 0 linearly (continuous at
    both endpoints, same as the launch ramp). Defaults to 0.0 (disabled)."""
    gravity: float = 9.81
    jump_assist_mass: float | None = None
    """Mass (kg) used to derive the jump assist force via projectile motion. If ``None``,
    it is auto-detected from the robot's simulated total mass at init time."""
    backflip_assist_force: float = 350.0
    sideflip_assist_force: float = 600.0
    sideflip_couple_delay_s: float = 0.0
    """Delay from the trigger before the roll couple starts, measured separately from
    ``assist_delay_s``. Zero keeps the couple inside the launch window, which is where it
    was wasted: the feet are still on the ground for most of that window and the ground
    cancels the torque. Set this past take-off so the couple acts on a body in free
    flight."""
    sideflip_couple_duration_s: float = 0.10
    """How long the couple is applied once ``sideflip_couple_delay_s`` has elapsed. Only
    used when that delay is non-zero."""
    sideflip_couple_force: float = 0.0
    """Additional roll applied as a couple: +this on the sideflip bodies, -this on the rest,
    so it contributes torque without any net lift. Defaults to 0.0, leaving every existing
    task unchanged. Use this rather than raising ``sideflip_assist_force`` when more
    rotation is wanted -- that force is one-sided, so it adds translation faster than spin
    and at 700 N launched the robot to 2.351 m with base_contact on every episode."""
    initial_assist_scale: float = 1.0

    minimum_landing_time_s: float = 0.40
    height_tolerance: float = 0.10
    rotation_tolerance_rad: float = 0.30
    landing_height_tolerance: float = 0.10
    landing_vertical_speed_tolerance: float = 0.30
    landing_upright_threshold: float = -0.8
    """How upright the robot must be for ``landed`` to count, as a bound on
    ``projected_gravity_b[2]``. The -0.8 default admits up to 37 degrees of tilt, which is
    loose enough to hide a systematically crooked jump: a Go2-Jump-60 policy scoring
    success 0.997 was measured leaving the ground at -0.671 rad/s pitch and +0.366 rad/s
    roll on all 64 environments -- same sign every time, std under 0.11 -- and peaking at
    33.7 deg mean tilt, i.e. sitting 0.9 deg under the gate rather than landing cleanly.
    Tighten it to require a genuinely upright landing; -0.90 is 26 deg, -0.95 is 18 deg."""
