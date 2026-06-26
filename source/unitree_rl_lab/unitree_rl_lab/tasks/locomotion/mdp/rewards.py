from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body_stairs(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    step_height_range: tuple[float, float],
    tanh_mult: float,
    stance_foot_height_body: float = -0.32,
) -> torch.Tensor:
    """Penalize foot height error with a per-env target tied to terrain stair difficulty.
    ``step_height_range`` is extra lift above the nominal stance foot height in body frame,
    not an absolute world/body z coordinate.
    """
    step_min, step_max = step_height_range
    terrain = env.scene.terrain
    terrain_gen = terrain.cfg.terrain_generator
    num_rows = terrain_gen.num_rows if terrain_gen is not None else 1
    difficulty = terrain.terrain_levels.float() / max(num_rows - 1, 1)
    step_height = step_min + difficulty * (step_max - step_min)
    target_height = stance_foot_height_body + step_height

    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height.unsqueeze(1))
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def _cpg_leg_phases_rad(env: ManagerBasedRLEnv, period: float, offset: list[float]) -> torch.Tensor:
    """Per-leg open-loop CPG phase in [0, 2π). Swing when phase < π."""
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = [(global_phase + offset_) % 1.0 for offset_ in offset]
    return torch.cat(phases, dim=-1) * (2 * torch.pi)

def wild_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    radius: float = 0.1,
    target_clearance: float = 0.05,
) -> torch.Tensor:
    """Reward swinging feet for clearance above local max terrain height under each foot.

    Uses continuous clearance (foot z minus local h_max) scaled by ``target_clearance``,
    gated by open-loop CPG swing phase.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    foot_ids = asset_cfg.body_ids
    # 足のworld位置 [num_envs, num_feet, 3]
    foot_pos_w = robot.data.body_pos_w[:, foot_ids, :]

    # レイヒット点 [num_envs, num_rays, 3]
    ray_hits_w = sensor.data.ray_hits_w

    # 各足 vs 各レイ点のxy距離 [num_envs, num_feet, num_rays]
    diff_xy = foot_pos_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
    dist_xy = torch.norm(diff_xy, dim=-1)

    # 半径内マスク
    in_range = dist_xy <= radius # [num_envs, num_feet, num_rays]

    # 半径内のz値だけ取り出してmax（範囲外は -inf で無視）
    ray_z = ray_hits_w[:, None, :, 2].expand(-1, len(foot_ids), -1)
    valid_rays = torch.isfinite(ray_z)
    in_range = in_range & valid_rays
    masked_z = torch.where(in_range, ray_z, torch.full_like(ray_z, -1e9))
    h_max, _ = torch.max(masked_z, dim=-1)

    # 半径内に1点もない場合のフォールバック（全体max）
    no_hit = ~in_range.any(dim=-1)
    global_max = torch.where(valid_rays, ray_z, torch.full_like(ray_z, -1e9)).amax(dim=-1)
    h_max = torch.where(no_hit, global_max, h_max)

    clearance = foot_pos_w[:, :, 2] - h_max
    reward = torch.clamp(clearance / target_clearance, 0.0, 1.0)

    # --- スイング判定 ---
    phases = _cpg_leg_phases_rad(env, period, offset)
    is_swing = phases < torch.pi

    reward = reward * is_swing.float()
    return torch.sum(reward, dim=-1)


def foot_clearance_terrain_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    target_clearance: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """POSITIVE reward for swing feet that clear the local terrain by ``target_clearance``.
    This replaces an ``exp(-penalty)`` formulation whose maximum (1.0) was reached by
    **not lifting at all**: a stationary, fully-planted robot scored full marks because
    every foot was masked out as "stance", so the term gave zero gradient toward
    stepping and actively reinforced standing still (the dominant local optimum that
    stalled the terrain curriculum). Both the old foot-speed gate and the later
    swing-only ``exp(-penalty)`` shared this flaw.
    The new term is strictly positive and only pays out for a foot that is
    simultaneously:
      * in swing phase (no ground contact), **and**
      * earning clearance above the terrain directly below it, **and**
      * while a non-zero base velocity is commanded.
    The per-foot reward is the achieved clearance normalised to ``target_clearance``
    and **capped at 1.0**, so it cannot be farmed by kicking a leg absurdly high and a
    held static leg is worth at most one unit. A planted/standing robot earns 0.
    Terrain height under each foot is read from the height scanner (nearest ray),
    so it works for both ascending ``pyramid_stairs`` and descending
    ``pyramid_stairs_inv`` (clearance is measured against whatever is directly below).
    """
    from isaaclab.sensors import RayCaster

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]   # (N, F, 3)
    foot_z_w   = foot_pos_w[:, :, 2]                                # (N, F)
    foot_xy_w  = foot_pos_w[:, :, :2]                               # (N, F, 2)

    ray_hits_w = sensor.data.ray_hits_w                             # (N, R, 3)
    ray_xy_w   = ray_hits_w[:, :, :2]                               # (N, R, 2)
    ray_z_w    = ray_hits_w[:, :, 2]                                # (N, R)

    # Nearest height-scan point to each foot: (N, F, R) -> argmin -> (N, F)
    dist = torch.norm(foot_xy_w.unsqueeze(2) - ray_xy_w.unsqueeze(1), dim=-1)
    nearest = dist.argmin(dim=-1)                                    # (N, F)

    terrain_z = torch.gather(
        ray_z_w.unsqueeze(1).expand(-1, foot_z_w.shape[1], -1),
        dim=2,
        index=nearest.unsqueeze(-1),
    ).squeeze(-1)                                                    # (N, F)

    clearance = foot_z_w - terrain_z                                 # (N, F)

    # Positive, capped: fraction of the target clearance the swing foot achieves.
    achieved = torch.clamp(clearance, min=0.0, max=target_clearance) / target_clearance  # (N, F) in [0, 1]

    # Swing mask: only reward feet that are off the ground (no current contact).
    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0  # (N, F)
    swing = (~in_contact).float()                                   # (N, F)

    reward = torch.sum(achieved * swing, dim=1)                     # (N,)

    # Only while motion is commanded: standing envs must not be pushed to step.
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)  # (N,)
    return reward * (cmd_norm > 0.1)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""

def forward_command_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward actual base-frame velocity projected onto the commanded direction.
    ``track_lin_vel_xy_exp`` uses an exponential kernel that saturates near the
    target speed and gives almost no gradient pushing a stalled robot to *start*
    moving across hard terrain. This term adds a small linear incentive for net
    progress toward the commanded heading, which is what the terrain curriculum
    actually measures (displacement from spawn). It is the positive signal that
    lets the robot bootstrap onto, and climb across, the stair terrain.
    The reward is clamped to the commanded speed (no bonus for overshooting) and
    is zero when no motion is commanded (standing envs).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]      # (N, 2)
    cmd_norm = torch.norm(cmd_xy, dim=1)                               # (N,)
    cmd_dir = cmd_xy / cmd_norm.clamp(min=1e-3).unsqueeze(1)           # (N, 2)
    vel_b = asset.data.root_lin_vel_b[:, :2]                           # (N, 2)
    progress = torch.sum(vel_b * cmd_dir, dim=1)                       # (N,)
    # no reward for overshooting the command (per-env Tensor cap) and none for
    # moving backwards; torch.clamp cannot mix a float min with a Tensor max, so
    # apply the lower and upper bounds separately.
    progress = torch.minimum(progress.clamp(min=0.0), cmd_norm)
    return progress * (cmd_norm > 0.1)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
