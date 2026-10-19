"""Terminations for the hind-stance wall crossing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["illegal_contact_off_wall", "illegal_contact_off_wall_after_jump", "wall_success_done", "wall_timeout"]


class illegal_contact_off_wall(ManagerTermBase):
    """``illegal_contact``, except for contact that happens on the wall.

    A body touching something counts as a fall unless it is within ``margin`` of the wall (inner face
    to outer face, horizontally) *and* at least ``min_height`` above the floor. The height condition
    is what keeps the exemption from covering a robot lying on the floor at the foot of the wall;
    the distance is read from the wall table, not the force direction, because a trunk resting on
    the floor pushes up exactly as one resting on the wall's top does.

    A class so the sensor-to-articulation body mapping is resolved once: ``ContactSensor.body_names``
    rebuilds its list from every environment's prim paths on each call, and doing that every step
    doubled the step time (278 vs 148 ms at 4096 environments, measured)."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        sensor = env.scene.sensors[sensor_cfg.name]
        names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.robot_ids, _ = robot.find_bodies(names, preserve_order=True)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        margin: float = 0.15,
        min_height: float = 0.15,
        command_name: str = "wallcross",
        asset_name: str = "robot",
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        touching = torch.linalg.norm(forces, dim=-1).amax(dim=1) > threshold

        wall = env.command_manager.get_term(command_name)
        pos = wall.robot.data.body_pos_w[:, self.robot_ids]
        half, _, thick = wall.wall()
        cheb = wall.chebyshev(pos)
        near_wall = (cheb > (half - margin)[:, None]) & (cheb < (half + thick + margin)[:, None])
        raised = (pos[..., 2] - env.scene.env_origins[:, None, 2]) >= min_height
        return torch.any(touching & ~(near_wall & raised), dim=1)


class illegal_contact_off_wall_after_jump(illegal_contact_off_wall):
    """``illegal_contact_off_wall`` from the jump on (zero before it). For the thighs, calves and head:
    in the hind stance and the approach the lifted legs are handled by their own terms."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        margin: float = 0.15,
        min_height: float = 0.15,
        command_name: str = "wallcross",
        asset_name: str = "robot",
    ) -> torch.Tensor:
        hit = super().__call__(env, sensor_cfg, threshold, margin, min_height, command_name, asset_name)
        return hit & env.command_manager.get_term(command_name).triggered


def wall_success_done(env: ManagerBasedRLEnv, command_name: str = "wallcross") -> torch.Tensor:
    """End the episode ``success_end_s`` after the crossing succeeded."""
    wall = env.command_manager.get_term(command_name)
    return wall.success & (wall.elapsed - wall.success_time >= wall.cfg.success_end_s)


def wall_timeout(env: ManagerBasedRLEnv, command_name: str = "wallcross") -> torch.Tensor:
    """End the episode if the crossing has not succeeded ``wall_timeout_s`` after the jump."""
    wall = env.command_manager.get_term(command_name)
    return wall.triggered & ~wall.success & (wall.elapsed - wall.trigger_time >= wall.cfg.wall_timeout_s)
