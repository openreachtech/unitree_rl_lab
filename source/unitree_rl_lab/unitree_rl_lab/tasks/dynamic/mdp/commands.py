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
            self.target_height[env_ids] = torch.where(
                sampled_motion_codes == self.MOTION_JUMP, sampled_height, 0.0
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
            self.standing_height[rising_edge] = self.robot.data.root_pos_w[rising_edge, 2]
            self.max_height[rising_edge] = 0.0
            self.accumulated_pitch[rising_edge] = 0.0
            self.accumulated_roll[rising_edge] = 0.0

        active = self.trigger_step >= 0
        self.max_height[active] = torch.maximum(self.max_height[active], self.height_delta[active])
        self.accumulated_roll[active] += self.robot.data.root_ang_vel_b[active, 0] * self._env.step_dt
        self.accumulated_pitch[active] += self.robot.data.root_ang_vel_b[active, 1] * self._env.step_dt

        upright = self.robot.data.projected_gravity_b[:, 2] < -0.8
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
        assist_active = (
            self.enabled
            & (self.trigger_step >= 0)
            & (elapsed >= 0.0)
            & (elapsed < self.cfg.assist_duration_s)
            & (self.assist_scale > 0.0)
        )

        forces = torch.zeros(
            self.num_envs, len(self.body_ids), 3, dtype=torch.float, device=self.device
        )

        def apply_profile(
            motion_code: int,
            force_indices: list[int],
            total_force: float,
        ) -> None:
            motion_mask = assist_active & (self.motion_code == motion_code)
            if torch.any(motion_mask):
                force_per_body = total_force * self.assist_scale / len(force_indices)
                for force_index in force_indices:
                    forces[motion_mask, force_index, 2] = force_per_body

        # Jump assist force is derived per-env from projectile motion, following the
        # paper's f_jump(h_target): the average force needed to reach the initial
        # vertical velocity v0 = sqrt(2*g*h_target) over the assist window. By design this
        # is strong enough alone to fully launch the robot at assist_scale=1.0 -- the paper's
        # intent is for the robot to physically experience the successful trajectory early on,
        # not to require the policy's own contribution from the start.
        jump_mask = assist_active & (self.motion_code == self.MOTION_JUMP)
        if torch.any(jump_mask):
            initial_velocity = torch.sqrt(
                2.0 * self.cfg.gravity * self.target_height[jump_mask].clamp(min=0.0)
            )
            total_force = self.jump_assist_mass * initial_velocity / self.cfg.assist_duration_s
            force_per_body = total_force * self.assist_scale / len(self.jump_force_indices)
            for force_index in self.jump_force_indices:
                forces[jump_mask, force_index, 2] = force_per_body

        apply_profile(
            self.MOTION_BACKFLIP,
            self.backflip_force_indices,
            self.cfg.backflip_assist_force,
        )
        apply_profile(
            self.MOTION_SIDEFLIP,
            self.sideflip_force_indices,
            self.cfg.sideflip_assist_force,
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

    command_duration_s: float = 0.50
    assist_duration_s: float = 0.10
    gravity: float = 9.81
    jump_assist_mass: float | None = None
    """Mass (kg) used to derive the jump assist force via projectile motion. If ``None``,
    it is auto-detected from the robot's simulated total mass at init time."""
    backflip_assist_force: float = 350.0
    sideflip_assist_force: float = 600.0
    initial_assist_scale: float = 1.0

    minimum_landing_time_s: float = 0.40
    height_tolerance: float = 0.10
    rotation_tolerance_rad: float = 0.30
    landing_height_tolerance: float = 0.10
    landing_vertical_speed_tolerance: float = 0.30
