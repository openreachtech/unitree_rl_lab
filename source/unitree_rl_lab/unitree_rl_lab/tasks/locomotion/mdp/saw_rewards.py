"""Clock-free gait rewards from van Marum et al., "Revisiting Reward Design and
Evaluation for Robust Humanoid Standing and Walking" (single-contact SaW).

Neither term needs a gait clock or reference: the policy picks its own cadence.
"Standing" means the commanded (vx, vy, wz) norm is below ``stand_threshold``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["single_foot_contact", "feet_air_time_cmd"]


def _standing(env: ManagerBasedRLEnv, command_name: str, stand_threshold: float) -> torch.Tensor:
    return env.command_manager.get_command(command_name).norm(dim=1) < stand_threshold


class single_foot_contact(ManagerTermBase):
    """1 if exactly one foot touched the ground at some point in the last ``grace_s`` seconds, else 0.

    Discourages hopping (both feet together) without prescribing timing. The grace
    period allows a double-support phase between steps. Constant 1 when standing, so
    recovery steps under disturbances are never penalized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.since_single = torch.full((env.num_envs,), 1.0e3, device=env.device)

    def reset(self, env_ids=None):
        self.since_single[slice(None) if env_ids is None else env_ids] = 1.0e3

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        grace_s: float = 0.2,
        stand_threshold: float = 0.1,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        single = in_contact.sum(dim=1) == 1
        self.since_single = torch.where(single, torch.zeros_like(self.since_single), self.since_single + env.step_dt)
        reward = (self.since_single <= grace_s).float()
        return torch.where(_standing(env, command_name, stand_threshold), torch.ones_like(reward), reward)


def feet_air_time_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.4,
    stand_threshold: float = 0.1,
) -> torch.Tensor:
    """At each touchdown, ``air_time - threshold`` (seconds) summed over feet; 0 when standing.

    Same as IsaacLab's ``feet_air_time``, except the standing gate uses the full command
    including yaw, so turning in place still regularizes stepping frequency.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * ~_standing(env, command_name, stand_threshold)
