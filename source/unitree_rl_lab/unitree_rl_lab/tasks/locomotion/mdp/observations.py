from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def gait_clock_obs(env: ManagerBasedRLEnv, command_name: str = "gait_command") -> torch.Tensor:
    """Per-leg clock input for a :class:`~unitree_rl_lab.tasks.locomotion.mdp.GaitCommand`.

    Returns ``sin``/``cos`` of each foot's commanded phase, ordered ``[FL, FR, RL, RR]``
    (8 dims total). Unlike :func:`gait_phase`, which encodes one fixed-period clock shared by
    all feet, this reflects the per-foot phase offsets sampled by ``GaitCommand`` -- i.e. it
    changes shape with the commanded gait style (trot/pace/bound/gallop/...), not just time.
    """
    gait_term = env.command_manager.get_term(command_name)
    phases = gait_term.leg_phases() * (2 * torch.pi)
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
