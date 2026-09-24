from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class MixedGoalVelocityCommand(UniformVelocityCommand):
    """Splits command synthesis by terrain column: a full omnidirectional command on
    "rough" columns, goal-directed steering (one random goal per episode, dropping to
    zero on arrival) on every other column.

    - Columns named in ``cfg.rough_terrain_names`` (default ``("rough",)``): sampled
      exactly like the stock ``UniformVelocityCommand`` -- independent lin_vel_x/lin_vel_y/
      ang_vel_z draws from ``cfg.ranges`` every resample, widened toward
      ``cfg.limit_ranges`` by the existing ``lin_vel_cmd_levels``/``ang_vel_cmd_levels``
      curriculum terms exactly as Phase1/Phase2 do. No goal, no arrival logic -- this
      terrain has no obstacle for a goal to be "beyond".
    - Every other column (the wall rings): one random goal per episode, steered toward
      every step (``lin_vel_x``/``ang_vel_z`` synthesized from the heading error to the
      goal -- no reverse or strafe component at all), command zeroed on arrival.

    Rationale (2026-08-16): an earlier version of this class applied the goal-directed
    branch to the *whole* scene, including "rough" columns -- but "rough" has no wall to
    place a goal beyond, and Phase1/2 already trained a full omnidirectional
    (forward/reverse/strafe/turn) command there that goal-directed steering can't
    reproduce. Splitting by column keeps "rough" doing what Phase1/2 already validated
    while still getting the goal-directed "arrived -> stop" exposure where it's actually
    needed -- the wall.

    An env's column is fixed for its lifetime: ``terrain_levels`` moves an env up and
    down *rows*, never across columns, so the sub-terrain type it was assigned at reset
    never changes. The mask below is therefore computed once in ``__init__`` rather than
    tracked per promotion.
    """

    cfg: "MixedGoalVelocityCommandCfg"

    def __init__(self, cfg: "MixedGoalVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.rough_env_mask = self._compute_rough_env_mask()
        self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

    def _compute_rough_env_mask(self) -> torch.Tensor:
        terrain = self._env.scene.terrain
        terrain_generator_cfg = terrain.cfg.terrain_generator
        if terrain_generator_cfg is None or not terrain_generator_cfg.curriculum:
            # No deterministic column -> sub-terrain mapping -- fall back to treating
            # every env as "rough" (stock UniformVelocityCommand behaviour everywhere)
            # rather than guess which envs might be on a wall.
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Replicates TerrainGenerator._generate_curriculum_terrains' own column ->
        # sub-terrain assignment exactly: column i takes the first sub-terrain whose
        # running proportion total exceeds i / num_cols. Getting this wrong would label
        # the wrong envs, so it mirrors that loop line for line rather than approximating.
        names = list(terrain_generator_cfg.sub_terrains.keys())
        proportions = torch.tensor(
            [terrain_generator_cfg.sub_terrains[n].proportion for n in names], dtype=torch.float32
        )
        proportions = proportions / proportions.sum()
        cumsum = torch.cumsum(proportions, dim=0)
        num_cols = terrain_generator_cfg.num_cols
        col_is_rough = torch.zeros(num_cols, dtype=torch.bool)
        for col in range(num_cols):
            sub_index = int(torch.nonzero(col / num_cols + 0.001 < cumsum, as_tuple=False)[0])
            col_is_rough[col] = names[sub_index] in self.cfg.rough_terrain_names

        return col_is_rough.to(self.device)[terrain.terrain_types]

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rough_ids = env_ids_t[self.rough_env_mask[env_ids_t]]
        wall_ids = env_ids_t[~self.rough_env_mask[env_ids_t]]

        if len(rough_ids) > 0:
            # Stock UniformVelocityCommand sampling -- draws vel_command_b and
            # is_standing_env from cfg.ranges/cfg.rel_standing_envs, unchanged.
            super()._resample_command(rough_ids)

        if len(wall_ids) > 0:
            r = torch.empty(len(wall_ids), device=self.device)
            radius = r.uniform_(*self.cfg.goal_radius_range).clone()
            theta = torch.empty(len(wall_ids), device=self.device).uniform_(-math.pi, math.pi)
            origin_xy = self._env.scene.env_origins[wall_ids, :2]
            self.goal_pos_w[wall_ids, 0] = origin_xy[:, 0] + radius * torch.cos(theta)
            self.goal_pos_w[wall_ids, 1] = origin_xy[:, 1] + radius * torch.sin(theta)
            self.is_standing_env[wall_ids] = (
                torch.empty(len(wall_ids), device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
            )

    def _update_command(self):
        # Rough envs: identical to the stock post-processing (standing-env zeroing;
        # heading_command is unused by this cfg so that branch never triggers). Runs over
        # the whole scene, same as the parent -- harmless for wall envs too, since the
        # wall branch below overwrites their vel_command_b unconditionally right after.
        super()._update_command()

        wall_ids = (~self.rough_env_mask).nonzero(as_tuple=False).flatten()
        if len(wall_ids) == 0:
            return

        goal_vec_w = self.goal_pos_w[wall_ids] - self.robot.data.root_pos_w[wall_ids, :2]
        distance = torch.norm(goal_vec_w, dim=-1)
        arrived = distance < self.cfg.arrival_radius

        desired_heading = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
        heading_error = math_utils.wrap_to_pi(desired_heading - self.robot.data.heading_w[wall_ids])
        ang_vel_z = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,
            min=-self.cfg.max_ang_vel,
            max=self.cfg.max_ang_vel,
        )
        lin_vel_x = self.cfg.max_lin_vel * torch.cos(heading_error).clamp(min=0.0)

        self.vel_command_b[wall_ids, 0] = torch.where(arrived, torch.zeros_like(lin_vel_x), lin_vel_x)
        self.vel_command_b[wall_ids, 1] = 0.0
        self.vel_command_b[wall_ids, 2] = torch.where(arrived, torch.zeros_like(ang_vel_z), ang_vel_z)

        standing_wall_ids = wall_ids[self.is_standing_env[wall_ids]]
        self.vel_command_b[standing_wall_ids, :] = 0.0

        self.metrics["goal_distance"][wall_ids] = distance


@configclass
class MixedGoalVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = MixedGoalVelocityCommand

    rough_terrain_names: tuple[str, ...] = ("rough",)
    """Sub-terrain names sampled with the stock UniformVelocityCommand behaviour, using
    ``ranges``/``limit_ranges`` below (widened by lin_vel_cmd_levels/ang_vel_cmd_levels,
    same as Phase1/Phase2). Every other column gets goal-directed steering instead."""

    goal_radius_range: tuple[float, float] = (1.75, 2.5)
    """Per-episode goal distance from the env's spawn origin (in m), for wall-column envs.
    Must clear the terrain's own curriculum-promotion rim (``tile_size * 0.35``) so
    reaching the goal actually requires crossing the obstacle, not just approaching it."""

    arrival_radius: float = 0.5
    """Distance (in m) within which a wall-column env's goal counts as reached and its
    command drops to zero for the remainder of the episode."""

    max_lin_vel: float = 1.0
    """Forward speed commanded to wall-column envs while not yet facing/at the goal (m/s)."""

    max_ang_vel: float = 1.0
    """Yaw-rate cap while a wall-column env steers toward its goal (rad/s)."""
