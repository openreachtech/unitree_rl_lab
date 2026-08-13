from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
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

    Folded into Go2W-v1-Phase5 from the sandbox (Try10 introduced the lin_vel_y gate; Try13
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
