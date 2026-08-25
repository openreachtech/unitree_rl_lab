from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class UniformTerrainGatedVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand whose lin_vel_y is only ever nonzero on the sub-terrain names
    listed in ``cfg.lateral_terrain_names`` (every other column samples/holds it at exactly
    0), and whose lin_vel_x is clamped to >= ``cfg.restricted_lin_vel_x_min`` (forward-only,
    no reverse) on every column *outside* that set.

    Folded into Go2w-v1-Phase5 from the sandbox (Try10 introduced the lin_vel_y gate; Try13
    added the lin_vel_x floor) -- see sandbox/SUMMARY.md's Try-by-try table and Lesson 7 for
    why: Phase5's terrain mix is mostly stairs, where lateral motion and reverse have no
    task value and pinning them to a single global range either wastes training capacity
    (strafe/reverse learned on stairs where it's pointless) or forgets a skill trained
    earlier (Phase1-3's strafing) that a *different part of the same terrain* ("rough"
    columns) still has legitimate use for.

    An env's sub-terrain *type* is fixed for its whole lifetime: ``TerrainImporter``
    assigns ``terrain_types`` (one column index per env) once, in
    ``_compute_env_origins_curriculum``, and ``update_env_origins`` -- which the
    terrain_levels curriculum calls every episode -- only ever mutates ``terrain_levels``
    (the row/difficulty), never ``terrain_types`` (isaaclab terrain_importer.py:314-329). So
    the lateral-env mask below is computed once in ``__init__`` and stays valid; it does not
    need to track terrain curriculum promotions.
    """

    cfg: "UniformTerrainGatedVelocityCommandCfg"

    def __init__(self, cfg: "UniformTerrainGatedVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._lateral_env_mask = self._compute_lateral_env_mask()

    def _compute_lateral_env_mask(self) -> torch.Tensor:
        terrain = self._env.scene.terrain
        terrain_generator_cfg = terrain.cfg.terrain_generator
        if terrain_generator_cfg is None or not terrain_generator_cfg.curriculum:
            # No deterministic column -> sub-terrain mapping to gate on (curriculum=False
            # picks sub-terrains randomly per tile, not per column) -- fall back to
            # stock UniformVelocityCommand behaviour rather than guess.
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Replicate isaaclab terrain_generator.py::_generate_curriculum_terrains's column
        # assignment exactly (num_cols is deterministic per sub-terrain proportion; only the
        # row/difficulty is randomized within a column), so that "column index -> sub-terrain
        # name" here matches what TerrainGenerator actually built.
        names = list(terrain_generator_cfg.sub_terrains.keys())
        proportions = torch.tensor(
            [terrain_generator_cfg.sub_terrains[n].proportion for n in names], dtype=torch.float32
        )
        proportions = proportions / proportions.sum()
        cumsum = torch.cumsum(proportions, dim=0)
        num_cols = terrain_generator_cfg.num_cols
        col_is_lateral = torch.zeros(num_cols, dtype=torch.bool)
        for col in range(num_cols):
            sub_index = int(torch.nonzero(col / num_cols + 0.001 < cumsum, as_tuple=False)[0])
            col_is_lateral[col] = names[sub_index] in self.cfg.lateral_terrain_names

        return col_is_lateral.to(self.device)[terrain.terrain_types]

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        restricted_ids = env_ids_t[~self._lateral_env_mask[env_ids_t]]
        self.vel_command_b[restricted_ids, 1] = 0.0
        self.vel_command_b[restricted_ids, 0] = self.vel_command_b[restricted_ids, 0].clamp(
            min=self.cfg.restricted_lin_vel_x_min
        )


@configclass
class UniformTerrainGatedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = UniformTerrainGatedVelocityCommand

    lateral_terrain_names: tuple[str, ...] = ("rough",)
    """Sub-terrain names (``TerrainGeneratorCfg.sub_terrains`` keys) on which lin_vel_y is
    drawn from ``ranges``/``limit_ranges`` and lin_vel_x may go negative (reverse). Every
    other name holds lin_vel_y at exactly 0 and clamps lin_vel_x to
    ``restricted_lin_vel_x_min``, regardless of what ``ranges``/``limit_ranges`` say."""

    restricted_lin_vel_x_min: float = 0.4
    """lin_vel_x floor enforced on every env outside ``lateral_terrain_names``."""


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

    An env's column (sub-terrain type) is fixed for its lifetime -- see
    ``UniformTerrainGatedVelocityCommand``'s docstring for why the mask below only needs
    computing once, in ``__init__``, rather than tracking terrain_levels promotions.
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

        # Same column-assignment replication as UniformTerrainGatedVelocityCommand's
        # _compute_lateral_env_mask -- see its docstring for why this matches
        # TerrainGenerator's actual column -> sub-terrain mapping exactly.
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
