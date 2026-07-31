from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TowAssistCommand(CommandTerm):
    """Continuous forward "tow" assist force (EFGCL-style physical guidance -- Yoneda et al.,
    "Spotting-Inspired External Force Guided Curriculum Learning", 2026 -- adapted from a
    one-shot launch impulse to a continuous velocity-tracking assist).

    Unlike the jump/backflip/sideflip case, running has no single trigger moment to assist:
    the robot is expected to hold a commanded forward speed indefinitely. So instead of a
    timed force pulse, this applies a force proportional to how far actual forward speed
    trails the commanded speed -- a "leash" pulling the robot up to speed -- every step, only
    ever helping (never braking), scaled by a global ``assist_scale`` that decays via
    :func:`~unitree_rl_lab.tasks.locomotion.mdp.tow_assist_decay` once velocity-tracking
    success is consistently good. ``assist_scale`` is the same for every env at any point in
    training, so unlike the jump command's per-trigger time encoding, no observation of it is
    needed -- the policy never has to distinguish assisted vs. unassisted steps.
    """

    cfg: TowAssistCommandCfg

    def __init__(self, cfg: TowAssistCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_ids, _ = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        if len(self.body_ids) == 0:
            raise ValueError(f"No tow-assist bodies matched: {cfg.body_names}")

        self.assist_scale = cfg.initial_assist_scale
        self.curriculum_success_rate = 0.0
        self.curriculum_episode_count = 0
        self.curriculum_success_count = 0

        if cfg.state_file is not None and os.path.isfile(cfg.state_file):
            with open(cfg.state_file) as f:
                saved_state = json.load(f)
            self.assist_scale = saved_state["assist_scale"]
            self.curriculum_success_rate = saved_state["curriculum_success_rate"]
            self.curriculum_episode_count = saved_state["curriculum_episode_count"]
            self.curriculum_success_count = saved_state["curriculum_success_count"]

        self._last_force_x = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assist_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_x"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TowAssistCommand:\n"
        msg += f"\tAssist scale: {self.assist_scale}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Not policy-visible -- just the currently applied per-env forward force, for logging."""
        return self._last_force_x.unsqueeze(-1)

    def _update_metrics(self):
        self.metrics["assist_scale"][:] = self.assist_scale
        self.metrics["force_x"][:] = self._last_force_x

    def _resample_command(self, env_ids: Sequence[int]):
        # Nothing per-episode to resample: assist_scale is a single global value that decays
        # across training (see tow_assist_decay), not something sampled per env/episode.
        pass

    def _update_command(self):
        velocity_command = self._env.command_manager.get_term(self.cfg.velocity_command_name).command
        actual_vel_xy = self.robot.data.root_lin_vel_b[:, :2]

        vx_error = (velocity_command[:, 0] - actual_vel_xy[:, 0]).clamp(min=0.0)
        cmd_norm = torch.norm(velocity_command[:, :2], dim=1)

        force_x = (self.cfg.gain * vx_error * self.assist_scale).clamp(max=self.cfg.max_force)
        force_x = torch.where(cmd_norm > 0.1, force_x, torch.zeros_like(force_x))
        self._last_force_x = force_x

        forces = torch.zeros(self.num_envs, len(self.body_ids), 3, dtype=torch.float, device=self.device)
        forces[:, 0, 0] = force_x
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=self.body_ids,
            is_global=False,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class TowAssistCommandCfg(CommandTermCfg):
    """Configuration for :class:`TowAssistCommand`."""

    class_type: type = TowAssistCommand
    asset_name: str = MISSING  # type: ignore[assignment]
    body_names: list[str] = MISSING  # type: ignore[assignment]
    """Bodies to apply the tow force to (in the body's own local frame, so it always pulls
    along the robot's current forward heading, not a fixed world direction). A single body
    (e.g. ``["base"]``) is the natural choice for "being towed", unlike the jump command's
    per-leg assist which needs to distribute a launch force across the legs doing the pushing.
    """

    velocity_command_name: str = "base_velocity"
    gain: float = MISSING  # type: ignore[assignment]
    """Forward force per m/s of tracking error (N per m/s)."""
    max_force: float = MISSING  # type: ignore[assignment]
    """Clamp on the (pre-``assist_scale``) forward force, in N."""
    initial_assist_scale: float = 1.0

    state_file: str | None = None
    """Path used to persist/restore the EFGCL assist-force curriculum across process
    restarts, same rationale as ``JumpCommandCfg.state_file`` -- rsl_rl checkpoints only save
    network weights, so without this, every ``--resume`` silently restarts the assist-force
    decay from ``initial_assist_scale``."""
