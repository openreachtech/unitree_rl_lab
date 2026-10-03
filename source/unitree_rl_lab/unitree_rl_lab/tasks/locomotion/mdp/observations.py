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
def gait_reference_phase(env, command_name: str = "gait_ref") -> torch.Tensor:
    """基準軌道の位相 (N, 2) = [sin, cos]。

    ref_contact などが採点している位相そのものを返す。

    upstream の gait_phase は period 0.8s 固定の時計で、GaitReferenceCommand が
    積分している位相（速度依存・env ごとにランダム初期化）とは無関係。
    観測できない位相で採点すると、ポリシーにとっては学習不能なノイズになり、
    報酬の分散だけが増えて track_lin_vel_xy の信号が埋もれる。
    """
    term = env.command_manager.get_term(command_name)
    angle = term.phase * 2.0 * torch.pi
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)    
def gait_onehot(env) -> torch.Tensor:
    """現在の歩容を one-hot で返す。"""
    if not hasattr(env, "current_gait_id"):
        env.current_gait_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return torch.nn.functional.one_hot(env.current_gait_id, num_classes=5).float()
def gait_reference_foothold(env, command_name: str = "gait_ref") -> torch.Tensor:
    """基準着地点を胴体ヨー座標系で返す (N, 6) = [xL, xR, yL, yR, zL, zR]。"""
    term = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    rel = term.foot_ref_w - asset.data.root_pos_w[:, None, :]      # (N, 2, 3)

    q = asset.data.root_quat_w
    siny = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
    cosy = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    yaw = torch.atan2(siny, cosy)
    cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]

    x = cy * rel[..., 0] + sy * rel[..., 1]
    y = -sy * rel[..., 0] + cy * rel[..., 1]
    return torch.cat([x, y, rel[..., 2]], dim=-1)


def gait_reference_slope(env, command_name: str = "gait_ref") -> torch.Tensor:
    """進行方向の地形勾配 (N, 1)。"""
    term = env.command_manager.get_term(command_name)
    return term.terrain_slope.unsqueeze(-1)