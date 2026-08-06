"""Discrete gait-mode command: lets a single policy switch between quadruped walking
and the two biped stances (hind-leg / front-leg) within an episode.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, yaw_quat

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


class ForwardAssistVelocityCommand(UniformVelocityCommand):
    """`UniformVelocityCommand` that also applies a decaying, forward-and-up external
    force to help the robot practice forward weight-shift early in training
    (External Force Guided Curriculum Learning, EFGCL -- see
    ``doc/papers/EFGCL_Learning_Dynamic_Motion_through_Spotting-Inspired_External_Force_Guided_Curriculum_Learning.md``,
    and this codebase's own real-hardware-deployed use of the same technique for
    backflip on the ``feat/jump`` branch).

    Promoted from ``robots/go2/sandbox/try15.py``/``try16.py`` (forward-only 2-leg
    walking without leaning on a front foot). Direction is diagonally forward-and-up
    rather than purely horizontal: horizontal alone only helps translation, not the
    other half of what makes forward hardest on this reared-up hind-leg stance --
    momentarily carrying weight forward over the support legs without toppling onto
    the front legs. A modest upward component takes some of that weight off while
    the robot practices the timing, similar to a gymnastics spotter, then hands it
    back as ``assist_scale`` decays to 0.

    Also tracks, per episode, the fraction of steps any front foot spent in ground
    contact -- fed to :func:`unitree_rl_lab.tasks.biped.mdp.curriculums.assist_force_decay`
    as part of the success criterion. "Survived the episode" alone is not a
    sufficient success signal: the robot can trivially satisfy that by leaning on a
    front foot (the exact crutch behavior the up-component of this force exists to
    wean it off of), which let the assist decay to 0 well before any of the
    front-leg-avoidance behavior it was meant to teach had a chance to take hold.
    """

    cfg: ForwardAssistVelocityCommandCfg

    def __init__(self, cfg: ForwardAssistVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.body_ids, _ = self.robot.find_bodies(cfg.assist_body_name)

        self.contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.front_foot_body_ids, _ = self.contact_sensor.find_bodies(cfg.front_foot_body_names)

        self.front_contact_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device)

        self.assist_scale = cfg.initial_assist_scale
        self.curriculum_episode_count = 0
        self.curriculum_success_count = 0
        self.curriculum_success_rate = 0.0

        if cfg.state_file is not None and os.path.isfile(cfg.state_file):
            with open(cfg.state_file) as f:
                saved_state = json.load(f)
            self.assist_scale = saved_state["assist_scale"]
            self.curriculum_success_rate = saved_state["curriculum_success_rate"]

        self.metrics["assist_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["front_contact_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def save_curriculum_state(self) -> None:
        if self.cfg.state_file is None:
            return
        state = {
            "assist_scale": self.assist_scale,
            "curriculum_success_rate": self.curriculum_success_rate,
        }
        os.makedirs(os.path.dirname(self.cfg.state_file), exist_ok=True)
        tmp_path = self.cfg.state_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, self.cfg.state_file)

    @property
    def front_contact_fraction(self) -> torch.Tensor:
        """Fraction of the current episode (so far) any front foot spent in contact."""
        return self.front_contact_steps / self.episode_steps.clamp(min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.front_contact_steps[env_ids] = 0.0
        self.episode_steps[env_ids] = 0.0

        if self.cfg.lin_vel_x_min_magnitude > 0.0:
            n = len(env_ids)
            max_magnitude = max(abs(self.cfg.ranges.lin_vel_x[0]), abs(self.cfg.ranges.lin_vel_x[1]))
            magnitude = torch.empty(n, device=self.device).uniform_(
                self.cfg.lin_vel_x_min_magnitude, max_magnitude
            )
            sign = torch.where(
                torch.rand(n, device=self.device) < 0.5,
                torch.full((n,), -1.0, device=self.device),
                torch.full((n,), 1.0, device=self.device),
            )
            self.vel_command_b[env_ids, 0] = magnitude * sign

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["assist_scale"][:] = self.assist_scale
        self.metrics["front_contact_fraction"][:] = self.front_contact_fraction

    def _update_command(self):
        super()._update_command()
        self._track_front_contact()
        self._apply_assist_force()

    def _track_front_contact(self):
        forces = self.contact_sensor.data.net_forces_w[:, self.front_foot_body_ids, :]
        any_front_in_contact = (torch.norm(forces, dim=-1) > 1.0).any(dim=-1)
        self.front_contact_steps += any_front_in_contact.float()
        self.episode_steps += 1.0

    def _apply_assist_force(self):
        # Symmetric: pushes in whichever direction (forward or backward) is
        # currently commanded, via sign(vel_command_b[:, 0]) -- not forward-only,
        # since Phase2 now also trains backward walking and the same
        # weight-shift-over-the-support-legs challenge applies in reverse.
        forces = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        active = (~self.is_standing_env) & (self.vel_command_b[:, 0] != 0.0) & (self.assist_scale > 0.0)
        if torch.any(active):
            direction_local = torch.zeros(int(active.sum()), 3, device=self.device)
            direction_local[:, 0] = torch.sign(self.vel_command_b[active, 0])
            direction_w = quat_apply(yaw_quat(self.robot.data.root_quat_w[active]), direction_local)
            forces[active, 0, :2] = direction_w[:, :2] * (self.cfg.assist_force_forward * self.assist_scale)
            forces[active, 0, 2] = self.cfg.assist_force_up * self.assist_scale
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=self.body_ids,
            is_global=True,
        )


@configclass
class ForwardAssistVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`ForwardAssistVelocityCommand`."""

    class_type: type = ForwardAssistVelocityCommand

    contact_sensor_name: str = "contact_forces"
    front_foot_body_names: list[str] = MISSING

    assist_body_name: str = "base"
    assist_force_forward: float = 20.0
    assist_force_up: float = 10.0
    initial_assist_scale: float = 1.0
    state_file: str | None = None
    """Path used to persist/restore the EFGCL assist-scale + success-rate across process
    restarts -- rsl_rl checkpoints only save network weights, so without this, a
    ``--resume`` would silently restart the decay from ``initial_assist_scale``."""

    lin_vel_x_min_magnitude: float = 0.0
    """If > 0, ``lin_vel_x`` is resampled as ``sign * magnitude`` with ``magnitude``
    drawn uniformly from ``[lin_vel_x_min_magnitude, max(abs(ranges.lin_vel_x))]``
    and ``sign`` a coin flip -- excludes the near-zero band from both directions,
    instead of ``ranges.lin_vel_x`` alone (a single continuous interval, which
    can't express "no near-zero speed" without also excluding one whole sign).
    Matches this task's own established lesson (try16): a demanded pace slow
    enough that a tripod can satisfy it removes the pressure to commit to a
    genuine 2-leg gait. Default 0.0 leaves ``ranges.lin_vel_x`` sampling as-is."""


class PinnedGaitModeCommand(CommandTerm):
    """Discrete gait-mode command permanently pinned to a single mode (never
    resampled, never switches) -- exposes the same one-hot ``command``/``mode_ids``
    interface as :class:`GaitModeCommand`, so any mode-aware reward/termination
    function written against that interface keeps working unmodified, but the
    mode itself never varies within or across episodes.

    Built as a foundation stage: warm up the network's ``gait_mode`` input
    dimension (and let a checkpoint form around it) on top of an *already-proven*
    single-stance skill -- concretely, pinning to hind-biped on top of the
    original ``Go2-Biped-Phase1`` reward/termination set (which reliably produces
    genuine hind-leg standing at 7000 iterations on its own, no gait_mode at all)
    -- before ever asking the policy to condition its behavior on an actually
    *varying* mode signal. Every prior attempt at that in one step (Try-1, a
    direct multimodal Phase1 edit, Try-8) converged to a policy that technically
    received the gait_mode observation (confirmed correct via the printed
    observation-term table) but never produced any visible mode-dependent
    behavior change -- motivating decoupling "learn to use the gait_mode input
    slot at all" from "learn when to switch" into two separate stages.

    A later stage can swap this command out for an actually-varying one (e.g.
    :class:`GaitModeCommand`) via a config change alone, resuming from this
    stage's checkpoint -- no observation-shape change, since the
    ``command``/``mode_ids`` interface is identical.
    """

    cfg: PinnedGaitModeCommandCfg

    def __init__(self, cfg: PinnedGaitModeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.mode_ids = torch.full((self.num_envs,), cfg.pinned_mode, dtype=torch.long, device=self.device)
        self._mode_one_hot = torch.zeros(self.num_envs, 3, device=self.device)
        self._mode_one_hot[:, cfg.pinned_mode] = 1.0

    def __str__(self) -> str:
        return f"PinnedGaitModeCommand: pinned_mode={self.cfg.pinned_mode}"

    @property
    def command(self) -> torch.Tensor:
        """One-hot ``[quad, hind_biped, front_biped]``. Shape ``(num_envs, 3)``."""
        return self._mode_one_hot

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass  # pinned -- nothing to (re)sample, mode_ids/one-hot never change.

    def _update_command(self):
        pass


@configclass
class PinnedGaitModeCommandCfg(CommandTermCfg):
    """Configuration for :class:`PinnedGaitModeCommand`."""

    class_type: type = PinnedGaitModeCommand

    asset_name: str = MISSING

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Unused -- the mode is pinned and never resampled. Set far beyond any
    episode length only because :class:`CommandTermCfg` declares this field
    ``MISSING``."""

    pinned_mode: int = MODE_HIND_BIPED
    """Which mode id (``MODE_QUAD``/``MODE_HIND_BIPED``/``MODE_FRONT_BIPED``) the
    command permanently reports."""



    gait_mode_command_name: str = "gait_mode"
