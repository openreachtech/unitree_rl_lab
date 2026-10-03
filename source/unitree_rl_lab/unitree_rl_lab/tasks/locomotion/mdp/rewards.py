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
import math                       # gait_phase_contact が使う
from isaaclab.envs import mdp     # gait_joint_deviation_l1 が mdp.joint_deviation_l1 を呼ぶ
from . import reference          # モデル由来の基準軌道（ref_* 報酬が使う）

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


""" def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward """

def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ratio: float = 4.0,
    min_force: float = 10.0,
) -> torch.Tensor:

    """垂直面への衝突（段鼻へのつま先/足側面のヒット）を検出する。

    水平力が鉛直力の `ratio` 倍を超える接触を衝突とみなす。
    戻り値は該当した足の本数 (0.0 〜 2.0) なので、weight は負で与える。

    元実装からの変更点:
    - `min_force` を追加。比だけで判定すると、足がほとんど浮いていて
      鉛直力がノイズ程度のときにも条件が成立し、遊脚中ずっと罰が入る。
      これは feet_clearance と正面から競合する。
    - `net_forces_w` から `net_forces_w_history` の最大へ変更。
      ポリシーは 50 Hz、物理は 200 Hz なので、単フレーム読みだと
      段鼻への一瞬の衝突を取りこぼす。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    forces_z = torch.abs(forces[..., 2])
    forces_xy = torch.linalg.norm(forces[..., :2], dim=-1)
    hit = (forces_xy > ratio * forces_z) & (forces_xy > min_force)
    return hit.any(dim=1).float().sum(dim=-1)


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


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


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
def gait_joint_deviation_l1(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Penalize joint deviation only for a specific gait phase."""
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return mdp.joint_deviation_l1(env, asset_cfg) * gait_mask


# Phase-aligned contact pattern
def gait_phase_contact(
    env,
    left_foot: str,
    right_foot: str,
    cycle_time: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Reward phase-aligned left/right contacts with smooth cyclic targets."""
    if cycle_time <= 0.0:
        raise ValueError(f"`cycle_time` must be positive, got: {cycle_time}")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_foot_id = contact_sensor.find_bodies([left_foot])[0][0]
    right_foot_id = contact_sensor.find_bodies([right_foot])[0][0]

    contact_force_hist = contact_sensor.data.net_forces_w_history
    # Use the latest contact-force sample and a sigmoid gate to reduce contact jitter.
    left_force = torch.norm(contact_force_hist[:, -1, left_foot_id, :], dim=-1)
    right_force = torch.norm(contact_force_hist[:, -1, right_foot_id, :], dim=-1)
    left_contact = torch.sigmoid((left_force - 1.0) * 5.0)
    right_contact = torch.sigmoid((right_force - 1.0) * 5.0)

    speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    adaptive_cycle_time = torch.clamp(cycle_time * 0.6 + 0.4 / (speed + 1.0e-3), 0.5, 1.2)
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, adaptive_cycle_time) / adaptive_cycle_time
    phase_angle = 2.0 * math.pi * phase

    left_target = 0.5 * (1.0 + torch.sin(phase_angle))
    right_target = 0.5 * (1.0 + torch.sin(phase_angle + math.pi))

    left_reward = 1.0 - torch.square(left_contact - left_target)
    right_reward = 1.0 - torch.square(right_contact - right_target)

    support = left_contact + right_contact
    support_reward = torch.exp(-5.0 * torch.square(support - 1.0))

    reward = 0.7 * 0.5 * (left_reward + right_reward) + 0.3 * support_reward
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * (speed > 0.1) * gait_mask


# Straight knee during stance
def stance_knee_extension(
    env,
    knee_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    target: float = 0.1,
    scale: float = 5.0,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Simple, robust reward for straight knees during stance."""
    asset = env.scene[knee_cfg.name]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- knee positions (N, K)
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]

    # --- contact forces (use norm, not just z)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)
    # --- soft stance (simple and smooth)
    stance = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- squared error
    knee_error = torch.square(knee_pos - target)
    if knee_error.shape[1] != stance.shape[1]:
        raise ValueError(
            f"Mismatch between knees ({knee_error.shape[1]}) and feet ({stance.shape[1]}). "
            "Set `knee_cfg.joint_names` and `foot_sensor_cfg.body_names` to matching left/right pairs."
        )

    # --- apply stance weighting
    weighted_error = knee_error * stance
    # --- normalize so flight phase doesn't give free reward
    stance_sum = torch.sum(stance, dim=1) + 1.0e-6
    avg_error = torch.sum(weighted_error, dim=1) / stance_sum
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * avg_error) * gait_mask


# Contact Pattern
# Standing: Encourage consistent double-foot support for static balance.
# Walking: Encourage alternating single-leg contact and flight phases.
# Running: Encourage flight phases.
def contact_pattern_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Unified contact-pattern reward selected by gait id.

    Pattern by `target_gait_id`:
    - 0 (stand): prefer double support
    - 1 (walk): prefer single support
    - 3 (run): prefer flight
    """
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- (N, 2, 3) -> (N, 2)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- soft contact (0~1)
    contact = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- continuous number of contacts (0~2)
    num_contacts = torch.sum(contact, dim=1)

    # --- smooth reward
    single_support = torch.exp(-5.0 * (num_contacts - 1.0) ** 2)
    flight = torch.exp(-5.0 * (num_contacts - 0.0) ** 2)
    double_support = torch.exp(-5.0 * (num_contacts - 2.0) ** 2)

    gait_mask = (env.current_gait_id == target_gait_id).float()

    if target_gait_id == 0:
        # Stand: strong double-support, discourage shuffle/flight.
        reward = double_support - 0.5 * single_support - 1.0 * flight
    elif target_gait_id == 3:
        # Run: prioritize flight, weakly discourage support phases.
        reward = flight - 0.5 * single_support - 1.0 * double_support
    else:
        # Walk (and fallback): prioritize alternating single support.
        reward = single_support + 0.5 * flight - 1.0 * double_support

    return reward * gait_mask

# Base Stability:
# Penalize base and joint motion to maintain upright posture.
def base_stability_standing(
    env,
    std_base: float = 0.25,
    std_joint: float = 0.1,
    target_gait_id: int = 0,
) -> torch.Tensor:
    """Penalize base/joint motion and tilt while in standing gait mode."""
    asset = env.scene["robot"]

    # ---- base motion ----
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    base_error = (
        torch.sum(lin_vel**2, dim=1) +
        0.5 * torch.sum(ang_vel**2, dim=1)   # reduce angular dominance
    )

    # ---- joint motion ----
    joint_vel = asset.data.joint_vel
    joint_error = torch.mean(joint_vel**2, dim=1)  # mean is more stable than sum

    # ---- upright ----
    # only x,y tilt matters
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    upright_error = torch.sum(gravity_xy**2, dim=1)

    # ---- combine (still simple) ----
    reward = torch.exp(-base_error / std_base) \
           * torch.exp(-joint_error / std_joint) \
           * torch.exp(-upright_error / 0.1)

    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * gait_mask


# Push-Off Dynamics
# Reward strong vertical and forward velocity during push-off.
def push_off_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    command_name: str,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Reward velocity along commanded direction during push-off."""
    asset = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- commanded velocity (N, 2)
    v_cmd = env.command_manager.get_command(command_name)[:, :2]

    # --- actual velocity in world frame (N, 2)
    v = asset.data.root_lin_vel_w[:, :2]

    # --- projection (how well we move in commanded direction)
    proj_vel = torch.sum(v * v_cmd, dim=1)

    # --- contact (N, 2)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)
    contacts = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- push-off phase (partial contact)
    push_off = contacts * (1.0 - contacts)
    push_off_strength = torch.sum(push_off, dim=1)
    gait_mask = (env.current_gait_id == target_gait_id).float()

    return proj_vel * push_off_strength * scale * gait_mask


# Short Contact
# Penalize prolonged stance to promote dynamic running
def short_contact_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    max_steps: int = 15,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Penalize long foot contact duration to encourage dynamic running."""
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- (N, F, 3) -> (N, F)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- soft contact (0~1)
    contact = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # contact duration buffer (per-env, per-foot) (N, F)
    duration = getattr(env, "_contact_duration", None)
    if duration is None or duration.shape != contact.shape:
        duration = torch.zeros_like(contact)

    # Reset durations for envs marked for reset in this step.
    # `reset_buf` is computed before reward terms, so this catches episode boundaries reliably.
    if hasattr(env, "reset_buf"):
        duration = torch.where(env.reset_buf.unsqueeze(1), torch.zeros_like(duration), duration)

    # update: accumulate only when in contact
    duration = (duration + 1.0) * contact
    env._contact_duration = duration

    # penalty only when duration exceeds threshold
    excess = torch.relu(duration - max_steps)
    penalty = torch.sum(excess, dim=1)
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * penalty) * gait_mask


# Feet Swing Height Penalty
# Ensure the robot lifts its feet sufficiently during the swing phase to prevent tripping and dragging
def feet_swing_height_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    target_height: float = 0.1,
    scale: float = 20.0,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Encourage sufficient foot clearance during swing."""
    asset = env.scene["robot"]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- contact force (N, F)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- smooth swing (1 = swing, 0 = stance)
    swing = 1.0 - torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- foot height (world)
    # Contact sensors provide forces but not body poses. Resolve matching bodies on the robot asset.
    foot_body_ids, _ = asset.find_bodies(foot_sensor_cfg.body_names, preserve_order=foot_sensor_cfg.preserve_order)
    foot_pos = asset.data.body_pos_w[:, foot_body_ids, :]
    foot_height = foot_pos[..., 2]

    # --- terrain-relative foot height
    # `TerrainImporter` does not expose `get_heights` in this setup.
    # For flat-plane tasks, ground is z=0; keep a guarded path for terrains that implement height queries.
    foot_xy = foot_pos[..., :2].reshape(-1, 2)
    get_heights_fn = getattr(env.scene.terrain, "get_heights", None)
    if callable(get_heights_fn):
        terrain_h = get_heights_fn(foot_xy).reshape(foot_height.shape)
    else:
        terrain_h = torch.zeros_like(foot_height)
    rel_height = foot_height - terrain_h

    # --- penalty only when below target (smooth)
    height_error = torch.relu(target_height - rel_height)
    penalty = torch.square(height_error) * swing

    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * torch.sum(penalty, dim=1)) * gait_mask


# Arm–leg momentum balance
# Penalizes residual whole-body angular momentum, particularly in the yaw (vertical) direction.
# Encourages anti-phase yaw swing, where the arms move in opposition to the legs to cancel out leg-induced rotation
# - Whole-body Momentum Minimization
# - Arm Symmetry and Coordination
def arm_leg_momentum_penalty(
    env,
    robot_cfg: SceneEntityCfg,
    left_arm_cfg: SceneEntityCfg,
    right_arm_cfg: SceneEntityCfg,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Simplified angular momentum reward (yaw only)."""
    asset = env.scene[robot_cfg.name]

    body_pos = asset.data.body_pos_w  # (N, B, 3)
    body_vel = asset.data.body_lin_vel_w  # (N, B, 3)

    # Cache body masses for efficiency and rebuild when shape changes.
    masses = getattr(env, "_body_masses", None)
    expected_shape = (body_pos.shape[0], body_pos.shape[1], 1)
    if masses is None or tuple(masses.shape) != expected_shape:
        raw_masses = asset.root_physx_view.get_masses()
        if raw_masses.ndim == 1:
            raw_masses = raw_masses.unsqueeze(0).expand(body_pos.shape[0], -1)
        masses = raw_masses.unsqueeze(-1).to(body_pos.device)
        env._body_masses = masses

    # --- CoM
    com = torch.sum(body_pos * masses, dim=1) / torch.sum(masses, dim=1)

    def momentum_z(body_ids) -> torch.Tensor:
        r = body_pos[:, body_ids, :] - com.unsqueeze(1)
        v = body_vel[:, body_ids, :]
        m = masses[:, body_ids, 0]
        # r x (m v) -> z = x*vy - y*vx
        return torch.sum(m * (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]), dim=1)

    l_total = momentum_z(slice(None))
    l_la = momentum_z(left_arm_cfg.body_ids)
    l_ra = momentum_z(right_arm_cfg.body_ids)

    reward = -(l_total**2) - 0.4 * (l_la - l_ra) ** 2
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * scale * gait_mask


def gait_orientation_asym_l2(
    env,
    free_pitch_forward=0.35,
    roll_scale=1.0,
    pitch_back_scale=1.0,
    target_gait_id=None,
    asset_cfg=SceneEntityCfg("robot"),
):
    """前後非対称な姿勢コスト。前傾は許容範囲を持たせ、後傾とロールは罰する。"""
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b

    pitch_fwd = torch.clamp(g[:, 0] - math.sin(free_pitch_forward), min=0.0)
    pitch_back = torch.clamp(-g[:, 0], min=0.0)
    roll = g[:, 1]

    cost = (
        roll_scale * roll.square()
        + pitch_back_scale * pitch_back.square()
        + pitch_fwd.square()
    )
    if target_gait_id is None:
        return cost
    return cost * (env.current_gait_id == target_gait_id).float()


def gait_joint_pos_l2(env, asset_cfg, target_gait_id=None):
    """指定した関節の、デフォルト姿勢からのずれ（二乗和）。gait で絞れる。"""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cost = torch.sum(torch.square(q - q0), dim=1)
    if target_gait_id is None:
        return cost
    return cost * (env.current_gait_id == target_gait_id).float()
def foot_clearance_reward_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """foot_clearance_reward の地形非依存版。接地足の高さを基準にする。

    平地（接地足 z≈0）では元の実装とほぼ一致するため、
    平地で学習済みの報酬ランドスケープを壊さない。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                      # (N, F)

    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(forces, dim=-1) > 1.0                                    # (N, F)

    # 基準高さ = 接地している足の最低高さ。両脚浮遊中は低い方の足。
    ref = torch.where(contact, foot_z, torch.full_like(foot_z, float("inf")))
    ref = ref.min(dim=1).values
    ref = torch.where(torch.isinf(ref), foot_z.min(dim=1).values, ref)

    rel_z = foot_z - ref.unsqueeze(-1)

    foot_z_target_error = torch.square(rel_z - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
def root_height_below_minimum_relative(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """足を基準にした胴体高さ判定。地形の絶対高さに依存しない。"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1).values
    return (asset.data.root_pos_w[:, 2] - foot_z) < minimum_height
"""
Model-generated reference trajectory rewards.

mdp/commands/gait_reference.py の GaitReferenceCommand が計算した
粗い基準軌道を読むだけの報酬項。基準の中身（周期・デューティ比・
地形へスナップした着地点など）は mdp/reference.py にある。

役割分担:
    ref_foot       どこへ着地するか      （地形の踏面へスナップした目標）
    ref_clearance  そこまでどう運ぶか    （段との衝突を未然に防ぐ）
    feet_stumble   実際にぶつかったとき  （事後の罰則）

区間の指定は term.phase_mask を使う。走行は滞空期、歩行・階段は単脚支持期。
階段には飛翔期がないので in_flight で絞ると基準が常にゼロになる。

両足・全関節にわたる量は和ではなく平均にしてあるので、関節数や足の本数に
依存せず weight を調整できる。

注意: ManagerBasedRLEnv.step() は reward_manager.compute() を
command_manager.compute() より先に呼ぶので、ここで読む基準は
1 ステップ（20 ms）前の値になる。1 周期 0.6 s に対して 3% 程度。
"""

FEET_SENSOR = SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
FEET_BODY = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
SCANNER = SceneEntityCfg("foothold_scanner")


def _gait_reference(env, command_name: str = "gait_ref"):
    return env.command_manager.get_term(command_name)


def _touchdown_mask(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """接触が始まった瞬間のマスク (N, F)。

        m_j = 1( c_j(t) = 1 かつ c_j(t - dt) = 0 )

    `current_contact_time` は接触開始からの経過時間なので、1 ステップ分
    以下であることが立ち上がりエッジと同値になる。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    t = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return ((t > 0.0) & (t <= env.step_dt * 1.5)).float()


def ref_contact_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = FEET_SENSOR,
    command_name: str = "gait_ref",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """基準の接地系列と実接地の一致率 (0.0 〜 1.0、両足の平均)。

    feet_gait の置き換え。feet_gait は threshold=0.55 固定なので
    「常にどちらか片足が接地」が満点で、滞空も両脚支持も減点になる。
    こちらはデューティ比が歩容ごとに変わるので、走行 (delta=0.35) では
    「両足とも離れている」区間が満点になる。
    """
    term = _gait_reference(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    actual = (torch.norm(forces, dim=-1).amax(dim=1) > force_threshold).float()
    return (actual == term.contact_ref).float().mean(dim=-1)


def ref_com_vz(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    command_name: str = "gait_ref",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """重心の上下移動が基準に乗っているか。

    走行（滞空期）: 接触力ゼロなので r_ddot = g、鉛直速度は離地速度から
        一意に決まる。近似ではなく厳密。
    階段・歩行（単脚支持期）: 指令水平速度 x 地形勾配。階段では
        dz/dx = 蹴上/踏面 なので、前進指令が鉛直上昇の要求を含意する。
    """
    term = _gait_reference(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    masses = reference.cached_body_masses(env, asset)
    _, com_vel = reference.com_position_velocity(
        asset.data.body_pos_w, asset.data.body_lin_vel_w, masses
    )
    err = com_vel[:, 2] - term.com_vz_ref
    return torch.exp(-err.square() / (std**2)) * term.com_vz_mask


def ref_angular_momentum(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    yaw_std: float = 0.30,
    command_name: str = "gait_ref",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """全身の角運動量が基準の範囲に収まっているか。

    走行（滞空期）: 外力がないので h は全軸で保存されるはず。h_ref = 離地時の値。

    歩行・階段（単脚支持期）: **ヨー軸だけ**を見る。
        ピッチ軸の角運動量は遊脚の前後振りそのもので、歩行に必要な量なので
        縛ってはいけない（実測 angmom_pitch ≈ 0.72 が常時出る）。
        一方ヨーは、脚の振り出しが作る反動を腕・上体が打ち消して初めて
        ゼロに近づく量なので、ここを締めるのが腕振りの動機になる。

    yaw_std は実測 angmom_yaw と同じオーダーに取ること。std を制御したい量より
    大きく取ると exp がほぼ 1 に張り付き、無条件報酬になって何も効かない。
    """
    term = _gait_reference(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    masses = reference.cached_body_masses(env, asset)
    h = reference.angular_momentum_com(
        asset.data.body_pos_w, asset.data.body_lin_vel_w, masses
    )

    err_flight = torch.norm(h - term.h_ref, dim=-1)
    r_flight = torch.exp(-err_flight.square() / (std**2))

    err_yaw = torch.clamp(h[:, 2].abs() - term.h_tol, min=0.0)
    r_stance = torch.exp(-err_yaw.square() / (yaw_std**2))

    value = torch.where(term.in_flight > 0.5, r_flight, r_stance)
    return value * term.phase_mask


def ref_foot_placement(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    command_name: str = "gait_ref",
    asset_cfg: SceneEntityCfg = FEET_BODY,
    sensor_cfg: SceneEntityCfg = FEET_SENSOR,
) -> torch.Tensor:
    """接触が始まった瞬間の足位置が、地形から求めた基準着地点に近いか。

        r = (1/2) sum_j exp(-||p_j - p_j_ref||^2 / sigma^2) * m_j_touch

    p_j も p_j_ref もワールド座標。基準は Raibert の名目位置を足裏矩形の
    平坦さで選び直して踏面へ寄せたもの（reference.snap_foothold）なので、
    階段では段鼻を避けた点になる。

    立脚中ずっと評価しないのは、立脚中は足が地面に固定されたまま胴体が
    前へ出るため。常時評価すると足が滑って基準位置に留まることを報酬する。
    """
    term = _gait_reference(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    err = torch.norm(foot_w - term.foot_ref_w, dim=-1)
    return (torch.exp(-err.square() / (std**2)) * _touchdown_mask(env, sensor_cfg)).mean(dim=-1)


def ref_swing_clearance(
    env: ManagerBasedRLEnv,
    margin: float = 0.03,
    command_name: str = "gait_ref",
    asset_cfg: SceneEntityCfg = FEET_BODY,
    sensor_cfg: SceneEntityCfg = FEET_SENSOR,
    scan_cfg: SceneEntityCfg = SCANNER,
) -> torch.Tensor:
    """遊脚が地形に対して必要な高さを確保しているか。weight は負で与える。

        z_safe = max_{足のやや前方の矩形} z_terrain + margin
        r      = (1/2) sum_j m_j_swing * [max(0, z_safe - z_j)]^2

    足そのものの真下ではなく、進行方向にやや広げた矩形の最大を取るのが要点。
    足が段鼻に届く手前で蹴上が矩形に入るので、ぶつかる前に持ち上げを要求できる。
    feet_clearance が「地面から何 cm 上げろ」という下限しか言わないのに対し、
    こちらは段の垂直面そのものを避けさせる。

    戻り値は正の違反量なので、weight は負。
    """
   
    term = _gait_reference(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 地形スキャンが無い構成（平地タスクなど）では z_terrain = 0 として扱う。
    # 平地ではこれが厳密に正しいので、項を外さずに済む。
    scanner = env.scene.sensors.get(scan_cfg.name, None)
    if scanner is None:
        z_safe = torch.full_like(foot[..., 2], margin)
    else:
        z_safe = reference.clearance_query(
            scanner.data.ray_hits_w, foot[..., :2], term.forward_xy
        ) + margin
    swing = (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0).float()
    violation = torch.clamp(z_safe - foot[..., 2], min=0.0)
    return (violation.square() * swing).mean(dim=-1)


def load_torque_normalized(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """定格に対する正規化トルク負荷の平均 (1/N) sum (tau_i / tau_i_max)^2。

    dof_torques (joint_torques_l2) と違い関節ごとの定格で割るので、足首や腕の
    小さいモーターも公平に評価される。29 関節の和だと他項より値が大きく
    なりやすいので平均にしてある。weight は負で与える。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limits = None
    for attr in ("joint_effort_limits", "joint_effort_limits_sim", "default_joint_effort_limits"):
        limits = getattr(asset.data, attr, None)
        if limits is not None:
            break
    if limits is None:
        raise AttributeError("joint effort limits not found on ArticulationData")
    limits = limits[:, asset_cfg.joint_ids].clamp(min=1e-3)
    return torch.mean((tau / limits).square(), dim=-1)
def arm_swing_antisymmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """左右の肩ピッチが逆位相になることだけを促す。

    q_L + q_R = 0 という対称性のみを課すので振幅は自由。両腕が
    逆位相なら振幅がいくつでも 0 になり、腕振り自体は抑制しない。
    片手だけ振るフォームは q_L + q_R != 0 になるので罰が立つ。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.square(q.sum(dim=-1))