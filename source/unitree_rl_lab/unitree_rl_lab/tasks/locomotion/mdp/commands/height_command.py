from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class UniformHeightCommand(CommandTerm):
    """Samples a per-env target base height uniformly from ``cfg.ranges.base_height``.

    Same shape as ``isaaclab.envs.mdp.commands.UniformVelocityCommand`` -- a buffer
    resampled every ``resampling_time_range`` seconds -- but for a single scalar (world-frame
    target base height) instead of a 3-vector body-frame velocity. Exists so a reward like
    ``track_base_height_depth_scaled_exp`` can read the target from ``env.command_manager`` instead of a
    constant float baked into its params, which is what lets the target differ per env or be
    widened by a curriculum later (mirroring how ``base_velocity`` widens from
    ``UniformLevelVelocityCommandCfg.ranges`` toward ``limit_ranges``). For now, tasks that
    want a fixed target just set ``ranges.base_height`` to ``(h, h)``.
    """

    cfg: "UniformHeightCommandCfg"

    def __init__(self, cfg: "UniformHeightCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformHeightCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n\tResampling time range: {self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        """The desired base height command, world-frame meters. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_height"] += (
            torch.abs(self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.base_height)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for ``UniformHeightCommand``."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the asset in the scene whose base height is tracked."""

    @configclass
    class Ranges:
        base_height: tuple[float, float] = MISSING
        """World-frame target base height range (m). Set min == max for a fixed target."""

    ranges: Ranges = MISSING


@configclass
class UniformLevelHeightCommandCfg(UniformHeightCommandCfg):
    """``UniformHeightCommandCfg`` plus a ``limit_ranges`` field, exactly mirroring
    ``UniformLevelVelocityCommandCfg`` -- a curriculum term (e.g. ``crouch_depth_levels``)
    widens ``ranges`` toward this ceiling over training instead of it being fixed for the
    whole run."""

    limit_ranges: UniformHeightCommandCfg.Ranges = MISSING
