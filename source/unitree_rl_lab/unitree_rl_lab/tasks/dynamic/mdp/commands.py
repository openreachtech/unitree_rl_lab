from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_apply, sample_uniform

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

        self._uses_whole_body_spin = cfg.sideflip_spin_torque > 0.0
        if self._uses_whole_body_spin:
            # Static per-body mass properties for the whole-body spin; the COM-frame inertia
            # is constant, only its orientation changes.
            self._body_mass = self.robot.data.default_mass.to(self.device)
            self._body_inertia = self.robot.data.default_inertia.to(self.device).reshape(
                self.num_envs, -1, 3, 3
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
        self.target_pitch_turns[env_ids] = 0.0
        self.target_roll_turns[env_ids] = 0.0

        # The whole-body spin writes every body, so it has to be cleared on every body too.
        reset_body_ids = None if self._uses_whole_body_spin else self.body_ids
        num_reset_bodies = self.robot.num_bodies if self._uses_whole_body_spin else len(self.body_ids)
        zero_forces = torch.zeros(
            len(env_ids), num_reset_bodies, 3, dtype=torch.float, device=self.device
        )
        self.robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=torch.zeros_like(zero_forces),
            body_ids=reset_body_ids,
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
            # Flips get a height of their own, so the jump's projectile launch force (keyed on
            # target_height) also buys them air time to rotate in. Without it go2's
            # sideflip rotated just clear of the floor (max_height 0.076 m), and go2w's
            # backflip tipped onto its back.
            flip_height = torch.where(
                sampled_motion_codes == self.MOTION_BACKFLIP,
                torch.full_like(sampled_height, self.cfg.backflip_target_height),
                torch.full_like(sampled_height, self.cfg.sideflip_target_height),
            )
            self.target_height[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_JUMP, sampled_height, flip_height
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
        if self.cfg.success_allow_overshoot:
            jump_target_reached = self.max_height > self.target_height - self.cfg.height_tolerance
        else:
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

        # Jump assist force is derived per-env from projectile motion, following the
        # paper's f_jump(h_target): the average force needed to reach the initial
        # vertical velocity v0 = sqrt(2*g*h_target) over the assist window. By design this
        # is strong enough alone to fully launch the robot at assist_scale=1.0 -- the paper's
        # intent is for the robot to physically experience the successful trajectory early on,
        # not to require the policy's own contribution from the start. Keyed on the height
        # rather than on MOTION_JUMP, so a flip given a height gets the lift that goes with it.
        jump_mask = assist_active & (self.target_height > 0.0)
        if torch.any(jump_mask):
            initial_velocity = torch.sqrt(2.0 * self.cfg.gravity * self.target_height[jump_mask])
            total_force = self.jump_assist_mass * initial_velocity / self.cfg.assist_duration_s
            force_per_body = total_force * self.assist_scale / len(self.jump_force_indices)
            for force_index in self.jump_force_indices:
                forces[jump_mask, force_index, 2] = force_per_body * ramp_progress[jump_mask]

        # One-sided backflip lift on the front hips. Assigned, so it REPLACES the launch force
        # there rather than adding to it -- the go2w backflip assist was swept and trained
        # with exactly this, so it is kept as is.
        backflip_mask = assist_active & (self.motion_code == self.MOTION_BACKFLIP)
        if torch.any(backflip_mask):
            force_per_body = (
                self.cfg.backflip_assist_force * self.assist_scale / len(self.backflip_force_indices)
            )
            for force_index in self.backflip_force_indices:
                forces[backflip_mask, force_index, 2] = force_per_body * ramp_progress[backflip_mask]

        # One-sided sideflip lift on the right hips, added to the launch force. (Assigning it
        # overwrote the launch there: with sideflip_assist_force = 0 the robot lifted on one
        # side only, rolled the wrong way at take-off and reached 0.13 m instead of ~0.6 m.)
        sideflip_mask = assist_active & (self.motion_code == self.MOTION_SIDEFLIP)
        if torch.any(sideflip_mask):
            force_per_body = (
                self.cfg.sideflip_assist_force * self.assist_scale / len(self.sideflip_force_indices)
            )
            for force_index in self.sideflip_force_indices:
                forces[sideflip_mask, force_index, 2] += force_per_body * ramp_progress[sideflip_mask]

        # The same post-take-off couple for the backflip: up on the backflip bodies (front
        # hips), down on the rest (rear hips), so it adds pitch without adding lift. On go2w
        # the one-sided front force alone left the robot short of a turn and landing on its
        # back; pushing it harder would mostly roll the rear wheels backwards while the feet
        # are still down.
        if self.cfg.backflip_couple_force > 0.0:
            backflip_couple_mask = (
                (self.motion_code == self.MOTION_BACKFLIP)
                & (self.trigger_step >= 0)
                & (elapsed >= self.cfg.backflip_couple_delay_s)
                & (elapsed < self.cfg.backflip_couple_delay_s + self.cfg.backflip_couple_duration_s)
                & (self.assist_scale > 0.0)
            )
            opposite = [i for i in range(len(self.body_ids)) if i not in self.backflip_force_indices]
            couple = self.cfg.backflip_couple_force * self.assist_scale
            for force_index in self.backflip_force_indices:
                forces[backflip_couple_mask, force_index, 2] += couple / len(self.backflip_force_indices)
            for force_index in opposite:
                forces[backflip_couple_mask, force_index, 2] -= couple / max(len(opposite), 1)

        self.applied_forces = forces
        if not self._uses_whole_body_spin:
            self.robot.set_external_force_and_torque(
                forces=forces,
                torques=torch.zeros_like(forces),
                body_ids=self.body_ids,
                is_global=True,
            )
            return

        all_forces = torch.zeros(self.num_envs, self.robot.num_bodies, 3, device=self.device)
        all_torques = torch.zeros_like(all_forces)
        all_forces[:, self.body_ids] = forces
        spin_mask = (
            (self.motion_code == self.MOTION_SIDEFLIP)
            & (self.trigger_step >= 0)
            & (elapsed >= self.cfg.sideflip_spin_delay_s)
            & (elapsed < self.cfg.sideflip_spin_delay_s + self.cfg.sideflip_spin_duration_s)
            & (self.assist_scale > 0.0)
        )
        if torch.any(spin_mask):
            torque = self.cfg.sideflip_spin_torque * self.assist_scale * torch.sign(self.target_roll_turns)
            spin_forces, spin_torques = self._whole_body_spin(axis_b=(1.0, 0.0, 0.0), torque=torque)
            all_forces[spin_mask] += spin_forces[spin_mask]
            all_torques[spin_mask] += spin_torques[spin_mask]
        self.robot.set_external_force_and_torque(
            forces=all_forces,
            torques=all_torques,
            body_ids=None,
            is_global=True,
        )

    def _whole_body_spin(self, axis_b: tuple[float, float, float], torque: torch.Tensor):
        """Per-body wrenches that spin the whole robot rigidly about ``axis_b`` (base frame).

        A torque on one body (or a force couple on the hips) only reaches the legs through
        the hip joints, and on go2w most of the roll inertia is in the legs and wheels
        (whole-body Ixx 0.956 vs go2's 0.236): pushing hard enough to turn the robot
        just flexes the hips and the base springs back. Instead every body gets exactly
        the force and torque it would need to follow one shared angular acceleration
        alpha -- F_i = m_i * (alpha x d_i) at its COM, tau_i = I_i * alpha -- so the
        joints carry nothing and the total torque about the whole-body COM is ``torque``.
        """
        mass = self._body_mass
        com = self.robot.data.body_com_pos_w
        center = (mass.unsqueeze(-1) * com).sum(dim=1, keepdim=True) / mass.sum(dim=1).view(-1, 1, 1)
        offset = com - center
        axis = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor(axis_b, device=self.device).expand(self.num_envs, 3),
        )
        rotation = matrix_from_quat(self.robot.data.body_com_quat_w)
        inertia_w = rotation @ self._body_inertia @ rotation.transpose(-1, -2)
        axis_per_body = axis.unsqueeze(1).expand_as(offset)
        own = torch.einsum("nbi,nbij,nbj->nb", axis_per_body, inertia_w, axis_per_body)
        along = (offset * axis_per_body).sum(dim=-1)
        parallel_axis = mass * ((offset * offset).sum(dim=-1) - along**2)
        axis_inertia = (own + parallel_axis).sum(dim=1)
        alpha = (torque / axis_inertia).unsqueeze(-1) * axis
        alpha_per_body = alpha.unsqueeze(1).expand_as(offset)
        forces = mass.unsqueeze(-1) * torch.cross(alpha_per_body, offset, dim=-1)
        torques = (inertia_w @ alpha_per_body.unsqueeze(-1)).squeeze(-1)
        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "force_visualizer"):
                self.force_visualizer = VisualizationMarkers(self.cfg.force_visualizer_cfg)
            self.force_visualizer.set_visibility(True)
        elif hasattr(self, "force_visualizer"):
            self.force_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw the assist force on each assist body: red = up, blue = down (crouch pulse)."""
        if not self.robot.is_initialized or not hasattr(self, "applied_forces"):
            return
        force_z = self.applied_forces[..., 2].reshape(-1)
        positions = self.robot.data.body_pos_w[:, self.body_ids].reshape(-1, 3)
        # The arrow marker points along +x; pitch it by -90 deg (up) or +90 deg (down).
        half = math.sqrt(0.5)
        orientations = torch.zeros(force_z.shape[0], 4, device=self.device)
        orientations[:, 0] = half
        orientations[:, 2] = torch.where(force_z >= 0.0, -half, half)
        scales = torch.tensor(
            self.cfg.force_visualizer_cfg.markers["up"].scale, device=self.device
        ).repeat(force_z.shape[0], 1)
        scales[:, 0] = force_z.abs() / self.cfg.force_vis_newtons_per_meter
        marker_indices = (force_z < 0.0).long()
        self.force_visualizer.visualize(positions, orientations, scales, marker_indices)


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
    backflip_target_height: float = 0.0
    """Height commanded alongside a backflip, giving the rotation air time. The jump's
    launch force keys off it, so a non-zero value also turns that lift on for the flip."""
    sideflip_target_height: float = 0.0
    """As ``backflip_target_height``, for the sideflip."""

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
    sideflip_spin_torque: float = 0.0
    """Roll torque (N*m, about the whole-body COM) applied as a rigid whole-body spin --
    every body gets the wrench of one shared angular acceleration, so nothing has to pass
    through the joints (see ``JumpCommand._whole_body_spin``). For robots whose roll
    inertia sits in heavy legs, where a hip couple only flexes the hips. Defaults to 0.0
    (off), which also keeps the hip-only wrench path."""
    sideflip_spin_delay_s: float = 0.0
    """Delay from the trigger before the whole-body spin starts; set it past take-off."""
    sideflip_spin_duration_s: float = 0.10
    """How long the whole-body spin is applied."""
    backflip_couple_force: float = 0.0
    """Pitch couple for the backflip: +this on the backflip bodies, -this on the rest, so it
    adds rotation with no net lift. Defaults to 0.0 (off)."""
    backflip_couple_delay_s: float = 0.0
    """Delay from the trigger before the backflip couple starts; set it past take-off so the
    ground does not absorb the torque."""
    backflip_couple_duration_s: float = 0.10
    """How long the backflip couple is applied."""
    initial_assist_scale: float = 1.0

    force_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/assist_force",
        markers={
            "up": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "down": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """Assist-force arrows drawn when ``debug_vis`` is on (red up, blue down)."""
    force_vis_newtons_per_meter: float = 200.0
    """Arrow length scale: this many newtons draw a 1 m arrow."""

    minimum_landing_time_s: float = 0.40
    height_tolerance: float = 0.10
    success_allow_overshoot: bool = False
    """Count a jump as reaching its target when ``max_height`` clears
    ``target_height - height_tolerance``, with no upper bound. Only ``success`` (and so the
    assist curriculum) changes; the height reward stays two-sided. Under full assist the
    jump overshoots, and a two-sided check then holds ``success`` at 0 and the assist at
    1.0 until the policy learns to jump *lower*. Defaults to False (two-sided)."""
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
