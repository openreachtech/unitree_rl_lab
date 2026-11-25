"""Velocity command sampled by category (van Marum et al.): stand, sagittal only,
lateral only, turn in place, or omnidirectional.

Uniform sampling of (vx, vy, wz) almost never produces a pure single-axis command,
so the policy rarely practices walking straight, side-stepping or turning in place.
This term samples the usual uniform command, then zeroes the axes its category does
not use. Ranges (and the level curriculum acting on them) are unchanged.

``forward_only_types`` lists terrain columns (``TerrainImporter.terrain_types``) whose envs
always get a straight-forward command, vx ~ U(``forward_speed_range``), never stand / turn
(after tsubame_isaac_env's ``TerrainTypeVelocityCommand``). On a wall or a stair column a
stand or turn command would let the policy score its tracking reward without ever going
near the obstacle. An env's column never changes, so its command distribution is fixed.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from .velocity_command import UniformLevelVelocityCommandCfg

# rows: stand, sagittal, lateral, turn, omni ; columns: vx, vy, wz
_CATEGORY_MASKS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
]


class CategoricalVelocityCommand(UniformVelocityCommand):
    cfg: CategoricalLevelVelocityCommandCfg

    def __init__(self, cfg: CategoricalLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self._probs = torch.tensor(cfg.category_probs, dtype=torch.float, device=self.device)
        self._masks = torch.tensor(_CATEGORY_MASKS, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        n = self.vel_command_b[env_ids].shape[0]
        category = torch.multinomial(self._probs, n, replacement=True)
        self.vel_command_b[env_ids] *= self._masks[category]
        # standing envs are held at zero by _update_command
        self.is_standing_env[env_ids] = category == 0

        if self.cfg.forward_only_types:
            terrain = self._env.scene.terrain
            if getattr(terrain, "terrain_types", None) is None:
                return
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            cols = torch.tensor(self.cfg.forward_only_types, device=self.device)
            ids = ids[torch.isin(terrain.terrain_types[ids], cols)]
            self.vel_command_b[ids] = 0.0
            self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(
                *self.cfg.forward_speed_range
            )
            self.is_standing_env[ids] = False


@configclass
class CategoricalLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = CategoricalVelocityCommand

    category_probs: tuple[float, float, float, float, float] = (0.1, 0.225, 0.225, 0.225, 0.225)
    """Probabilities of (stand, sagittal, lateral, turn, omni). ``rel_standing_envs`` is ignored."""

    forward_only_types: tuple[int, ...] = ()
    """Terrain columns whose envs only get a forward command (see the module docstring)."""

    forward_speed_range: tuple[float, float] = (0.3, 1.0)
    """vx (m/s) for ``forward_only_types``."""
