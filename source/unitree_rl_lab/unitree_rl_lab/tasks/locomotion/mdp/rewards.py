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


def base_height_climb_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_clearance: float = 0.35,
    std: float = 0.15,
    command_name: str | None = None,
    min_cmd_norm: float = 0.1,
) -> torch.Tensor:
    """Reward the base for sitting ``nominal_clearance`` above whatever
    terrain is directly beneath it right now.

    None of the existing terrain rewards (``forward_command_progress``,
    ``adaptive_foot_clearance_reward``) reward the vertical body-height gain
    needed to actually haul the torso up onto a step -- they reward
    horizontal progress and foot lift. This targets a specific failure mode:
    the front feet reach a step but the body/hind legs don't follow, so the
    body stays low even though the terrain right under it has risen.

    This is a potential-based tracking signal,
    ``exp(-(base_z - target_z)^2 / std^2)``, where ``target_z`` is the local
    terrain height (nearest height-scan point to the base) plus
    ``nominal_clearance``. It collapses to "maintain ordinary standing
    height" on flat ground (no conflict with flat-ground gait naturalness),
    and as the base's own xy position advances over a riser, ``target_z``
    rises with it -- pulling the body upward instead of letting it lag
    behind the front legs.

    ``command_name``: if set, the reward is masked to 0 whenever
    ``|command| <= min_cmd_norm`` (mirrors ``stall_penalty``'s gating). This
    term is otherwise unconditional on command, so on tasks with a
    meaningful standing-env fraction, a zero-command env parked on uneven
    curriculum terrain still gets a tight (std-scale) gradient to chase the
    local terrain height with rapid leg motion -- reinforcing a twitchy
    "correct height even at rest" habit that has nothing to do with climbing
    and transfers badly to genuinely flat ground. Gating by command restricts
    the term to when there's an actual climb/traverse attempt underway.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    base_xy = asset.data.root_pos_w[:, :2]
    base_z = asset.data.root_pos_w[:, 2]

    ray_hits_w = sensor.data.ray_hits_w  # (N, R, 3)
    ray_xy = ray_hits_w[:, :, :2]
    ray_z = ray_hits_w[:, :, 2]
    # RayCaster reports +-inf for rays that miss the terrain mesh entirely.
    # Masking those out of the nearest-ray search below avoids picking an
    # invalid ray when a valid one exists, but if a whole scan misses the
    # mesh (e.g. robot toppled off a terrain edge) the gather would still
    # return a raw +-inf. Sanitize the source so local_terrain_h is always
    # finite, and zero the reward outright when there is no valid ray at all.
    valid_rays = torch.isfinite(ray_z)
    ray_z = torch.nan_to_num(ray_z, nan=-1e3, posinf=-1e3, neginf=-1e3)

    dist = torch.norm(base_xy.unsqueeze(1) - ray_xy, dim=-1)  # (N, R)
    dist = torch.where(valid_rays, dist, torch.full_like(dist, 1e9))
    nearest = dist.argmin(dim=-1)  # (N,)
    local_terrain_h = torch.gather(ray_z, 1, nearest.unsqueeze(-1)).squeeze(-1)  # (N,)

    target_z = local_terrain_h + nominal_clearance
    height_error = base_z - target_z
    reward = torch.exp(-torch.square(height_error) / std**2)
    reward = reward * valid_rays.any(dim=1).float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward = reward * (cmd_norm > min_cmd_norm)

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
    # RayCaster reports +-inf for rays that miss the terrain mesh. This
    # function previously didn't exclude invalid rays at all (it could pick
    # one as "nearest" even with a valid ray close by). Mask them out of the
    # search and sanitize the source so terrain_z is always finite; the reward
    # is zeroed below when a scan has no valid ray at all.
    valid_rays = torch.isfinite(ray_z_w)                            # (N, R)
    ray_z_w = torch.nan_to_num(ray_z_w, nan=-1e3, posinf=-1e3, neginf=-1e3)

    # Nearest *valid* height-scan point to each foot: (N, F, R) -> argmin -> (N, F)
    dist = torch.norm(foot_xy_w.unsqueeze(2) - ray_xy_w.unsqueeze(1), dim=-1)
    dist = torch.where(valid_rays.unsqueeze(1), dist, torch.full_like(dist, 1e9))
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
    reward = reward * valid_rays.any(dim=1).float()

    # Only while motion is commanded: standing envs must not be pushed to step.
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)  # (N,)
    return reward * (cmd_norm > 0.1)


def adaptive_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    lookahead_distance: float = 0.15,
    natural_clearance: float = 0.03,
    max_clearance: float = 0.20,
    roughness_ref: float = 0.05,
    command_name: str | None = None,
    min_cmd_norm: float = 0.1,
) -> torch.Tensor:
    """Reward swing feet for clearing the terrain ahead, with the clearance
    *target* itself scaled by how much clearance is actually needed.

    ``wild_foot_clearance_reward`` and ``foot_clearance_terrain_adaptive`` both
    reward hitting a single fixed ``target_clearance`` everywhere. That value
    has to be set high enough to clear the tallest stair riser, so the
    reward-maximizing gait is "lift every foot to stair height," including on
    flat ground where doing so is unnecessary and looks unnatural. This term
    replaces the fixed target with a per-foot value derived from two signals:

      1. Obstacle lookahead: the height-scan point nearest to
         ``lookahead_distance`` ahead of the foot (along the base's current
         world-frame travel direction) minus the height directly under the
         foot. This is ~0 on flat ground and grows toward the real riser
         height as a step approaches.
      2. Terrain roughness gate: the height range across the whole scanned
         patch under the robot, normalized by ``roughness_ref``. This zeroes
         out the lookahead term (rather than trusting a single possibly noisy
         ray) whenever the immediate surroundings are essentially flat.

    ``target = natural_clearance + roughness_gate * obstacle_height``, so a
    flat-ground target collapses to an ordinary walking lift
    (``natural_clearance``) while an approaching stair riser raises it toward
    ``max_clearance``. Swing detection reuses the same open-loop CPG gate as
    ``wild_foot_clearance_reward`` to keep gait timing/coordination unchanged.

    ``command_name``: if set, the reward is masked to 0 whenever
    ``|command| <= min_cmd_norm`` (mirrors ``stall_penalty``/
    ``base_height_climb_reward``'s gating). The CPG swing phase
    (:func:`_cpg_leg_phases_rad`) is a pure open-loop clock driven by elapsed
    episode time, not by command or actual motion -- so without this gate,
    a standing, zero-command robot still cycles through "swing" phases every
    ``period`` seconds and is still paid to lift each foot by
    ``natural_clearance``, i.e. march in place. Gating restricts that payout
    to when there's an actual command to walk.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    foot_ids = asset_cfg.body_ids
    foot_pos_w = robot.data.body_pos_w[:, foot_ids, :]  # (N, F, 3)
    foot_xy_w = foot_pos_w[:, :, :2]
    foot_z_w = foot_pos_w[:, :, 2]

    ray_hits_w = sensor.data.ray_hits_w  # (N, R, 3)
    ray_xy_w = ray_hits_w[:, :, :2]
    ray_z_w = ray_hits_w[:, :, 2]
    valid_rays = torch.isfinite(ray_z_w)  # (N, R)
    # local_h/ahead_h below are two *independently* nearest-ray-selected
    # heights that get differenced in obstacle_height. The argmin masking
    # avoids invalid rays when a valid one exists, but if a whole scan misses
    # the terrain mesh (e.g. robot toppled off a stair edge) the gather would
    # return raw +inf and inf - inf = NaN, which then propagates into PPO's
    # returns/advantages and corrupts the value function. Sanitize the source
    # so the gather is always finite, and zero the reward for all-invalid scans.
    ray_z_w = torch.nan_to_num(ray_z_w, nan=-1e3, posinf=-1e3, neginf=-1e3)

    def _nearest_height(query_xy: torch.Tensor) -> torch.Tensor:
        # query_xy: (N, F, 2) -> nearest scanned height per foot: (N, F)
        dist = torch.norm(query_xy.unsqueeze(2) - ray_xy_w.unsqueeze(1), dim=-1)  # (N, F, R)
        dist = torch.where(valid_rays.unsqueeze(1), dist, torch.full_like(dist, 1e9))
        nearest = dist.argmin(dim=-1)  # (N, F)
        return torch.gather(
            ray_z_w.unsqueeze(1).expand(-1, query_xy.shape[1], -1), dim=2, index=nearest.unsqueeze(-1)
        ).squeeze(-1)

    local_h = _nearest_height(foot_xy_w)  # (N, F)

    # Direction of travel: base world-frame xy velocity, zeroed when ~stationary
    # so the lookahead point collapses onto the foot itself (no obstacle to see).
    base_vel_xy = robot.data.root_lin_vel_w[:, :2]  # (N, 2)
    base_speed = torch.norm(base_vel_xy, dim=1)  # (N,)
    moving = (base_speed > 0.05).float().view(-1, 1, 1)  # (N, 1, 1)
    unit_dir = (base_vel_xy / base_speed.clamp(min=1e-3).unsqueeze(1)).unsqueeze(1)  # (N, 1, 2)
    ahead_xy = foot_xy_w + unit_dir * lookahead_distance * moving  # (N, F, 2)

    ahead_h = _nearest_height(ahead_xy)  # (N, F)
    obstacle_height = torch.clamp(ahead_h - local_h, min=0.0, max=max_clearance - natural_clearance)  # (N, F)

    # Roughness gate: full height range of the scanned patch under the robot.
    z_for_max = torch.where(valid_rays, ray_z_w, torch.full_like(ray_z_w, -1e9))
    z_for_min = torch.where(valid_rays, ray_z_w, torch.full_like(ray_z_w, 1e9))
    roughness = (z_for_max.amax(dim=1) - z_for_min.amin(dim=1)).clamp(min=0.0)  # (N,)
    roughness_gate = torch.clamp(roughness / roughness_ref, 0.0, 1.0).unsqueeze(1)  # (N, 1)

    target_clearance = natural_clearance + roughness_gate * obstacle_height  # (N, F)

    clearance = foot_z_w - local_h
    reward = torch.clamp(clearance / target_clearance.clamp(min=1e-3), 0.0, 1.0)

    phases = _cpg_leg_phases_rad(env, period, offset)
    is_swing = phases < torch.pi
    reward = reward * is_swing.float()
    reward = torch.sum(reward, dim=-1) * valid_rays.any(dim=1).float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward = reward * (cmd_norm > min_cmd_norm)

    return reward


def quiet_standing_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str = "base_velocity",
    max_cmd_norm: float = 0.1,
    joint_vel_std: float = 1.0,
    roughness_ref: float = 0.05,
) -> torch.Tensor:
    """Reward literal stillness (all feet planted, low joint velocity) while the
    commanded velocity is ~0 *and* the terrain underneath is flat.

    base_height_climb_reward, wild_foot_clearance, and stall_penalty are all
    gated *off* near zero command -- that removes pressure toward motion, but
    nothing pushes back toward hard stillness being the unambiguous optimum.
    Generic penalties left active there (action_rate, joint_torques, energy,
    joint_vel) are small and uniform, and don't specifically cost a small
    asymmetric single-leg lift more than doing nothing, so residual noise
    (policy stochasticity, height-scan noise still feeding the observation
    even though it no longer drives reward at rest) can leave that lift as a
    locally-costless habit. This adds the missing positive pressure: reward is
    ``all_feet_planted * exp(-mean(joint_vel^2) / joint_vel_std^2)``, masked to
    0 whenever ``|command| > max_cmd_norm`` -- the inverse of the existing
    motion-reward gates, active only in the regime they turn off.

    ``sensor_cfg``: if set, also masks the reward by terrain flatness under
    the robot (same height-range-over-roughness_ref gate
    ``adaptive_foot_clearance_reward`` uses). Standing envs are spawned across
    the *whole* curriculum terrain, not just flat ground -- a zero-command env
    can be sitting on or next to a stair. Without this gate, "freeze here" gets
    rewarded there too, which can bias the shared policy toward hesitating
    right when a climb needs a committed push. With it, the reward only ever
    fires on genuinely flat terrain, so it can't compete with
    stall_penalty/stair_commit's "don't freeze on the stairs" pressure.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    all_planted = in_contact.all(dim=1).float()

    joint_vel = asset.data.joint_vel
    stillness = torch.exp(-torch.mean(torch.square(joint_vel), dim=1) / joint_vel_std**2)

    reward = all_planted * stillness

    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_z = sensor.data.ray_hits_w[:, :, 2]
        valid_rays = torch.isfinite(ray_z)
        z_for_max = torch.where(valid_rays, ray_z, torch.full_like(ray_z, -1e9))
        z_for_min = torch.where(valid_rays, ray_z, torch.full_like(ray_z, 1e9))
        roughness = (z_for_max.amax(dim=1) - z_for_min.amin(dim=1)).clamp(min=0.0)
        flatness_gate = torch.clamp(1.0 - roughness / roughness_ref, 0.0, 1.0)
        reward = reward * flatness_gate

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = reward * (cmd_norm <= max_cmd_norm)

    return reward


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


def stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_scale: float = 0.1,
) -> torch.Tensor:
    """Penalize near-zero body speed while a non-trivial command is active.

    ``forward_command_progress`` rewards positive progress but gives zero
    gradient once a robot has already stopped -- "attempted and failed" and
    "never tried" score identically. On terrain hard enough that most
    climbing attempts fail, that makes freezing at the base of an obstacle a
    locally safer strategy than attempting to climb: it avoids
    ``bad_orientation``/``base_contact`` termination while forfeiting reward
    that's already near zero either way. This term makes standing still
    itself costly instead of merely non-rewarding: ``exp(-body_speed /
    speed_scale)`` is ~1 when the body is essentially stationary and decays
    quickly as soon as there is *any* real forward motion, so it targets true
    stalling without penalizing slow, deliberate climbing steps.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    body_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    stall = torch.exp(-body_speed / speed_scale)
    return stall * (cmd_norm > 0.1)


def stair_commit_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    height_gap_threshold: float = 0.08,
    max_forward_speed: float = 1.0,
    max_climb_speed: float = 0.5,
) -> torch.Tensor:
    """Strongly reward forward+upward body progress at the exact moment the
    front feet are planted on a step higher than the hind feet.

    MuJoCo testing found a specific, repeated failure: the front feet reach a
    step but the hind legs stay on the ground below, and the robot often
    freezes or gives up right there rather than driving the hind legs to
    follow. ``base_height_climb_reward`` only rewards *having arrived* at the
    right body height, and ``forward_command_progress``/``stall_penalty``
    apply the same incentive everywhere -- neither concentrates gradient on
    this exact, narrow "straddling" window where a decisive push matters most.

    This term detects that state directly (front feet in contact with terrain
    meaningfully higher than the terrain under the hind feet) and rewards
    forward+upward body velocity only while it holds, so the incentive is
    sharply localized to the moment the hind legs need to drive.

    ``asset_cfg``/``contact_sensor_cfg`` must list feet in
    ``[FR, FL, RR, RL]`` order (front pair first, hind pair second) -- the
    same convention used by ``adaptive_foot_clearance_reward``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (N, 4, 2)
    front_xy = foot_xy[:, :2, :]
    hind_xy = foot_xy[:, 2:, :]

    ray_hits_w = sensor.data.ray_hits_w  # (N, R, 3)
    ray_xy = ray_hits_w[:, :, :2]
    ray_z = ray_hits_w[:, :, 2]
    # front_terrain_h - hind_terrain_h below differences two independent
    # nearest-ray lookups; inf - inf = NaN if a scan misses the terrain mesh
    # entirely. Sanitize the source so the gather is always finite, and zero
    # the reward for all-invalid scans (see adaptive_foot_clearance_reward).
    valid_rays = torch.isfinite(ray_z)
    ray_z = torch.nan_to_num(ray_z, nan=-1e3, posinf=-1e3, neginf=-1e3)

    def _nearest_height(query_xy: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(query_xy.unsqueeze(2) - ray_xy.unsqueeze(1), dim=-1)  # (N, 2, R)
        dist = torch.where(valid_rays.unsqueeze(1), dist, torch.full_like(dist, 1e9))
        nearest = dist.argmin(dim=-1)  # (N, 2)
        return torch.gather(
            ray_z.unsqueeze(1).expand(-1, query_xy.shape[1], -1), dim=2, index=nearest.unsqueeze(-1)
        ).squeeze(-1)

    front_terrain_h = _nearest_height(front_xy).mean(dim=1)  # (N,)
    hind_terrain_h = _nearest_height(hind_xy).mean(dim=1)  # (N,)

    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0  # (N, 4)
    front_planted = in_contact[:, :2].any(dim=1)
    hind_down = in_contact[:, 2:].any(dim=1)
    straddling = front_planted & hind_down & ((front_terrain_h - hind_terrain_h) > height_gap_threshold)

    # A scan that misses the terrain mesh entirely can't establish a valid
    # front/hind height gap, so it must not count as straddling.
    straddling = straddling & valid_rays.any(dim=1)

    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=max_forward_speed)
    climb_speed = torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0, max=max_climb_speed)
    progress = forward_speed + climb_speed
    return progress * straddling.float()


def landing_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    landing_window: float = 0.1,
    ang_vel_std: float = 2.0,
) -> torch.Tensor:
    """Reward quickly re-stabilizing body orientation right after a foot lands.

    ``stair_commit_reward`` pushes the robot to lunge onto risers, but a
    forceful landing that isn't quickly stabilized can leave small
    orientation/angular-velocity errors that compound across several steps
    until the robot topples -- a "climbs several steps then falls" failure
    mode. This detects a recent landing event (any foot's current contact
    time is still within ``landing_window`` of having started) and rewards
    low body angular velocity specifically in that window, so settling
    quickly after impact is worth something on its own instead of only being
    implicitly required to avoid termination later.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    current_contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
    just_landed = ((current_contact_time > 0.0) & (current_contact_time <= landing_window)).any(dim=1)

    ang_vel = torch.norm(asset.data.root_ang_vel_b, dim=1)
    stability = torch.exp(-torch.square(ang_vel) / ang_vel_std**2)
    return stability * just_landed.float()


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
