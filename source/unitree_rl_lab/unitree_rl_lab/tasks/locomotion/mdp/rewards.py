from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import wrap_to_pi
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


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_terrain_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    target_clearance: float,
    command_name: str = "base_velocity",
    radius: float = 0.1,
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

    Terrain height under each foot is the max height-scan hit within ``radius`` of the
    foot's xy position (matches go2-blind's ``wild_foot_clearance_reward``), so a step
    edge under or beside the foot is captured rather than only the single nearest ray.
    Falls back to the global max hit if no ray lands within ``radius``. Works for both
    ascending ``pyramid_stairs`` and descending ``pyramid_stairs_inv``.
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

    # Max height-scan hit within `radius` of each foot: (N, F, R) -> max -> (N, F)
    dist_xy = torch.norm(foot_xy_w.unsqueeze(2) - ray_xy_w.unsqueeze(1), dim=-1)  # (N, F, R)
    in_range = dist_xy <= radius                                    # (N, F, R)

    ray_z_expanded = ray_z_w.unsqueeze(1).expand(-1, foot_z_w.shape[1], -1)  # (N, F, R)
    valid_rays = torch.isfinite(ray_z_expanded)
    in_range = in_range & valid_rays

    masked_z = torch.where(in_range, ray_z_expanded, torch.full_like(ray_z_expanded, -1e9))
    terrain_z, _ = torch.max(masked_z, dim=-1)                      # (N, F)

    # Fallback to the global max hit if no ray landed within `radius`.
    no_hit = ~in_range.any(dim=-1)                                  # (N, F)
    global_max = torch.where(valid_rays, ray_z_expanded, torch.full_like(ray_z_expanded, -1e9)).amax(dim=-1)
    terrain_z = torch.where(no_hit, global_max, terrain_z)          # (N, F)

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


def heading_drift_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drifting away from what the commanded ang_vel_z, integrated
    since spawn, predicts.

    ``forward_command_progress``/``track_lin_vel_xy_exp`` only ever check
    BODY-frame velocity, which is invariant to the robot's absolute yaw:
    nothing else stops it from spinning around and climbing stairs
    hindquarters-first (discovered empirically on 2026-07-06 -- it climbs
    20cm/20cm steps far more easily backward). ``heading_command=True`` was
    tried as a fix but reverted (2026-07-07): forcing ang_vel_z to a
    continuous heading-error correction measurably degraded general
    locomotion (broken lateral movement, straight-line drift) in ways
    terrain_levels/bad_orientation never surfaced.

    This is a reward-only alternative that does not touch command generation
    at all. It tracks a per-env "expected yaw", reset to the actual yaw at
    spawn and integrated forward every step using the COMMANDED ang_vel_z
    (not the actual one), and penalises the squared error between actual and
    expected yaw. A robot that faithfully tracks whatever turn rate is
    commanded -- including large, legitimate ones -- keeps this error near
    zero no matter how much it turns; only an UNREQUESTED rotation (e.g. a
    spontaneous 180-degree flip to climb backward) opens a large, sustained
    gap, since wrap_to_pi keeps the wrapped error pinned near its max
    (~pi**2) for as long as the wrong heading is held, rather than only
    charging the one-off transient cost of the turn itself.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    yaw = asset.data.heading_w
    cmd_ang_vel_z = env.command_manager.get_command(command_name)[:, 2]

    if not hasattr(env, "_heading_drift_expected_yaw") or env._heading_drift_expected_yaw is None:
        env._heading_drift_expected_yaw = yaw.clone()

    # `episode_length_buf` is incremented BEFORE reward computation each step
    # (see ManagerBasedRLEnv.step()), and reset to 0 AFTER reward computation
    # for envs terminating this step. So it is never 0 at this point -- the
    # first reward call following a reset sees it at 1. Checking `== 0` here
    # (an earlier version of this function did) never fires, so the expected-
    # yaw tracker never resets across episode boundaries and compares each
    # fresh episode's actual (randomly-spawned) yaw against a stale value
    # from the previous episode -- a large, meaningless, essentially random
    # "error" almost always, not a real drift signal.
    reset_mask = env.episode_length_buf == 1
    env._heading_drift_expected_yaw[reset_mask] = yaw[reset_mask]
    env._heading_drift_expected_yaw += cmd_ang_vel_z * env.step_dt

    error = wrap_to_pi(yaw - env._heading_drift_expected_yaw)
    return torch.square(error)


def body_height_gain(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_climb_speed: float = 1.0,
) -> torch.Tensor:
    """Reward the base for gaining absolute world height (Try-8 / hypothesis A).

    Direct mujoco testing (2026-07-09) found the robot gets physically stuck
    when its front feet are staggered across two step heights: it cannot
    generate enough lift through the higher-placed front leg to raise the
    body, so the rear feet never reach the step behind it. Neither
    ``forward_command_progress`` (body-frame XY velocity only) nor
    ``feet_gait`` (contact-timing only) give any reward during that stuck
    moment -- straining to lift with zero horizontal progress currently earns
    nothing. This term fills that gap directly: it rewards positive dz/dt of
    the base regardless of horizontal motion, so the exact moment the robot
    needs to learn (an effortful vertical push with no forward progress yet)
    finally has a gradient pointing toward "keep pushing" instead of being
    reward-invisible. Scaled as an implied vertical speed (m/s, via
    ``env.step_dt``) so its magnitude is directly comparable to
    ``forward_command_progress``'s body-frame velocity reward.

    Only positive gains count (falling back down after a failed attempt is
    already penalised elsewhere, e.g. ``flat_orientation_l2``); the result is
    clamped to ``max_climb_speed`` as a safety cap against any single-step
    physics anomaly, and gated on a nonzero command like the other locomotion
    terms so standing envs cannot farm it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]

    if not hasattr(env, "_body_height_gain_prev_z") or env._body_height_gain_prev_z is None:
        env._body_height_gain_prev_z = z.clone()

    # Same reset-detection fix as heading_drift_penalty: episode_length_buf is
    # never 0 at reward-computation time, only ever 1 at the earliest.
    reset_mask = env.episode_length_buf == 1
    env._body_height_gain_prev_z[reset_mask] = z[reset_mask]

    gain = torch.clamp((z - env._body_height_gain_prev_z) / env.step_dt, min=0.0, max=max_climb_speed)
    env._body_height_gain_prev_z = z.clone()

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return gain * (cmd_norm > 0.1)


def foot_touchdown_height_gain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_gain: float = 0.15,
) -> torch.Tensor:
    """Reward each foot for touching down higher than its best height so far
    this episode (Try-9 / hypothesis B).

    Same motivating observation as ``body_height_gain`` (front feet staggered
    across two step heights -> the robot gets stuck, the rear feet never
    reach the step behind), but targets it per-foot and event-based instead
    of via the base's continuous height: the instant a swing foot lands
    higher than any previous touchdown of that same foot this episode -- most
    critically, the rear foot finally hooking the step the front feet are
    already standing on -- pays out immediately, rather than only being
    picked up indirectly through overall base motion.

    Tracks a per-foot running best touchdown height (not just the last one),
    so a foot cannot farm this by bouncing down and back up to the same
    height repeatedly -- only genuine net progress for that foot pays out.
    Per-event gain is normalised by ``target_gain`` (matches the 0.15m
    ``target_clearance`` used elsewhere) and capped at 1.0 per foot per step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                        # (N, F)
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # (N, F)

    if not hasattr(env, "_foot_td_prev_contact") or env._foot_td_prev_contact is None:
        env._foot_td_prev_contact = is_contact.clone()
        env._foot_td_best_z = foot_z.clone()

    # Same reset-detection fix as heading_drift_penalty (episode_length_buf==1,
    # not ==0). Without it, a fresh episode's first touchdowns would be
    # compared against stale heights from a completely unrelated previous
    # episode/spawn point.
    reset_mask = env.episode_length_buf == 1
    env._foot_td_prev_contact[reset_mask] = is_contact[reset_mask]
    env._foot_td_best_z[reset_mask] = foot_z[reset_mask]

    rising_edge = is_contact & (~env._foot_td_prev_contact)                         # (N, F) new touchdown this step

    gain = torch.clamp(foot_z - env._foot_td_best_z, min=0.0, max=target_gain) / target_gain
    reward = torch.sum(gain * rising_edge.float(), dim=1)

    env._foot_td_best_z = torch.maximum(env._foot_td_best_z, foot_z)
    env._foot_td_prev_contact = is_contact.clone()

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm > 0.1)


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


"""
Jump rewards (Unitree-Go2-LongJump-v1, Stage B).

Design ported from [[reference_paper_impedance_matching_running_jump]]
(Guan et al., arXiv:2404.15096) -- the paper's own code is not public
(confirmed 2026-08-27, see リファレンス_ExplicitEstimator実装仕様.md), so this
is a from-scratch reimplementation of what the paper describes, not a code
port. Both terms are gated on the ``jump_command`` (see
mdp.commands.jump_command.JumpCommand): 0 while running normally, 1 while
in a jump attempt.
"""


def jump_dense_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    normalize_by_weight: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """-std(foot contact force) across the four feet, active only while jumping.

    Encourages a symmetric, simultaneous push-off. Gated to the
    jump_command window so it does not fight the asymmetric trot gait used
    for the running approach.

    2026-09-14: ``normalize_by_weight`` divides by the robot's own weight m*g, making the
    term dimensionless and comparable across machines. It is off by default so Go2's
    tuning is untouched. **This is the only quantity in the jump reward set carried in
    Newtons**, and that is not a cosmetic problem: on the 24.31 kg Anaguma it billed
    -3.478 per episode at the moment the machine actually jumped, against -0.0064 for the
    15 kg Go2 under the same config -- 540x -- while every reward that pays for jumping is
    dimensionless and paid Anaguma +0.203 against Go2's +2.06. The push-off was a -3.4
    proposition, so not jumping was the optimum and the Anaguma port never left the
    ground in four runs (docs/anaguma_port.md §4-3). Lowering the curriculum weight 2.5 ->
    0.25 was not enough on its own (2026-09-14_19-56-07: still -1.46 against +0.30 of
    carrots at iter 48); the scale has to come out of the units, not the weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)  # (N, 4)
    jumping = env.command_manager.get_command(command_name)[:, 0] > 0.0
    # 2026-09-02: restricted to the push-off window (jump commanded, not yet airborne).
    # It previously ran for the whole jump window including the flight and, worse, the
    # whole time a fallen robot held jump_command==1 -- where std(F_foot) ~ 0, i.e. lying
    # on the ground *scored better on this term than pushing off did*. Combined with the
    # net-negative reward budget that was part of what made falling attractive; see
    # プロジェクト_StageB崩壊の原因分析.md.
    command_term = env.command_manager.get_term(command_name)
    # 2026-09-03: also require at least one foot actually loaded. -std(F_foot) is
    # maximised (0) when every foot reads zero force, i.e. **while airborne** -- so the
    # term paid full marks for being off the ground, and its cheapest optimum was
    # "retime the gait's existing suspension phase into the jump window" rather than
    # "push off symmetrically". That is the rhythm-change local optimum observed in
    # mujoco. Restricting it to the loaded push-off makes it shape what it was named
    # for. See プロジェクト_StageB跳躍が小さい原因分析.md.
    in_contact = (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).any(dim=-1)
    pushing_off = jumping & (~command_term.has_been_airborne) & in_contact
    reward = -torch.std(foot_force, dim=-1) * pushing_off.float()
    if normalize_by_weight:
        asset: Articulation = env.scene[asset_cfg.name]
        weight_n = float(asset.data.default_mass[0].sum().item()) * 9.81
        reward = reward / weight_n
    return reward


def jump_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_target: float,
    std: float,
    min_vertical_vel: float = 0.3,
    max_tilt_angle: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """exp(-(v_liftoff - v_target)^2 / (2*std^2)), evaluated once at the liftoff instant.

    v_liftoff is the horizontal (xy, body-frame) base speed at the single
    step where all four feet transition from "some in contact" to "none in
    contact" while jump_command==1. The paper does not spell out its
    liftoff-detection algorithm (code is not public); this is the most
    direct reading consistent with the paper's own use of contact sensing
    elsewhere (confirmed 2026-08-27). The paper also does not state whether
    its 2.5 m/s target is a specific velocity component or a resultant
    speed; xy-plane speed is used here as the closest match to "forward
    liftoff speed" for a *running* long jump.

    ``v_target`` is meant to be raised over training by a curriculum term
    (mdp.jump_vel_target_levels) mutating this reward term's ``params``
    dict in place -- the same mechanism mdp.lin_vel_cmd_levels uses on
    base_velocity's command range. This is a deliberate deviation from the
    paper (which trains toward one fixed 2.5 m/s target): this task's goal
    is to push toward Go2's theoretical limit
    ([[project_long_jump_theoretical_calc]]), not to hit one target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if not hasattr(env, "_ljb_prev_all_airborne") or env._ljb_prev_all_airborne is None:
        env._ljb_prev_all_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_ljb_liftoff_count") or env._ljb_liftoff_count is None:
        # Per-env count of liftoff events this episode. Read by
        # mdp.jump_vel_target_levels to normalize this reward's episode sum
        # by *attempts* instead of episode duration -- see that function's
        # docstring for why the naive per-second normalization never gates
        # (2026-08-31 fix, see フィードバック_resume時カリキュラムリセット.md
        # for the related but distinct curriculum-on-resume issue).
        env._ljb_liftoff_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Reset-detection idiom matching heading_drift_penalty/foot touchdown
    # rewards elsewhere in this file: episode_length_buf==1, not ==0.
    reset_mask = env.episode_length_buf == 1
    env._ljb_prev_all_airborne[reset_mask] = False
    env._ljb_liftoff_count[reset_mask] = 0

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # (N, 4)
    all_airborne = ~is_contact.any(dim=-1)
    jumping = env.command_manager.get_command(command_name)[:, 0] > 0.0
    # 2026-09-02: a "lift-off" now has to look like a jump, not like a topple.
    # The old condition was "all four feet lost contact while jump_command==1", which a
    # robot tipping over satisfies just as well as one pushing off -- and at weight 250
    # that paid ~15 reward per fall, more than a *full 18 s episode* of good running
    # earned from velocity tracking (21.6). That was the single largest positive term in
    # the collapsed policy's return. Requiring an upward base velocity and an upright
    # body at the moment of lift-off makes the term unreachable by falling over.
    # See プロジェクト_StageB崩壊の原因分析.md.
    rising = asset.data.root_lin_vel_w[:, 2] > min_vertical_vel
    upright = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() < max_tilt_angle
    # Only the FIRST airborne transition of each commanded jump scores. Foot contact
    # flickers many times per second while a robot tumbles, and every flicker used to
    # count as a fresh "lift-off": reconstructing the collapsed policy's numbers shows
    # the raw term summing to ~3.1 within a single 0.55 s episode, i.e. 3-4 payouts per
    # fall. JumpCommand.has_been_airborne is updated after the reward manager in the
    # step order, so on the genuine first transition it is still False here.
    command_term = env.command_manager.get_term(command_name)
    first_liftoff = ~command_term.has_been_airborne
    liftoff_event = all_airborne & (~env._ljb_prev_all_airborne) & jumping & rising & upright & first_liftoff

    v_liftoff = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    reward = torch.exp(-torch.square(v_liftoff - v_target) / (2.0 * std**2)) * liftoff_event.float()

    env._ljb_liftoff_count += liftoff_event.long()
    env._ljb_prev_all_airborne = all_airborne.clone()
    return reward


def jump_height_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.20,
    height_scale: float = 0.01,
    payout_window_s: float = 0.6,
) -> torch.Tensor:
    """exp(-(peak rise - target)^2 / scale) every step of the jump window. 0 otherwise.

    2026-09-03, and the single most important term added: **nothing in the previous
    reward set rewarded jump height, vertical velocity or air time at all**, while
    ``base_linear_velocity`` (lin_vel_z_l2, weight -2.0) actively penalised vertical
    motion. The arithmetic on the run that produced a barely-visible jump:

        cost of a 0.20 m jump (v0 = 1.98 m/s) from lin_vel_z_l2   ~ -0.8 per jump
        best case income from the old horizontal-speed sparse term ~ +0.5 per jump

    i.e. a *perfect* jump was a net loss, so not jumping was correct play. This term
    is dense (paid every step the jump window is open) rather than sparse, which is
    what gives the push-off an actual gradient to climb -- exp(-(h-0.2)^2/0.01) reads
    0.02 at h=0, 0.11 at 0.05 m, 0.37 at 0.10 m, 0.78 at 0.15 m, 1.0 at 0.20 m. Form
    and scale are taken from tak's motion_progress_reward (tag ``jump-demo``), which
    does produce real jumps on this robot.

    ``target_height`` is a FIXED point, not a curriculum: tak measured that giving the
    height target a range collapsed learning outright (max_height 0.233 -> 0.026,
    success 0.000 by iteration 700), and our own ratcheting v_target curriculum never
    moved off its initial value for three consecutive runs. 0.20 m matches the apex
    [[reference_paper_impedance_matching_running_jump]] measured for a forward running
    jump (Table III: 24.4 cm at a 2 m/s approach).
    """
    command_term = env.command_manager.get_term(command_name)
    # 2026-09-06 (Loop 17): same payout bound as mdp.jump_takeoff_apex, for the same
    # reason -- ``max_height_gain`` is a running maximum, so it too paid more the longer
    # the window stayed open.
    jumping = (command_term.command[:, 0] > 0.0) & (command_term.time_since_liftoff <= payout_window_s)
    progress = torch.exp(-torch.square(command_term.max_height_gain - target_height) / height_scale)
    return progress * jumping.float()


def jump_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_distance: float = 3.0,
) -> torch.Tensor:
    """Horizontal distance covered while airborne, paid once at a scoring landing.

    2026-09-03: this replaces ``jump_sparse_reward``'s exp(-(v_liftoff_xy - 1.5)^2/2).
    That term asked for a 1.5 m/s horizontal lift-off speed while lin_vel_cmd_levels
    was commanding up to 4.4 m/s and track_lin_vel_xy (weight 1.5, measured +1.155/s)
    was rewarding it -- the two objectives were in direct opposition and the jump term
    lost by a factor of ~4000, which is why it read ~0.0003/s all run and why
    jump_vel_target_levels never advanced.

    Distance is the task's actual objective and is what the theoretical estimate
    (2.02 m at a 2 m/s approach, 3.89 m at 5.3 m/s -- [[project_long_jump_theoretical_calc]])
    and the paper's 96 cm are quoted in, so it is directly comparable. Clamped only to
    keep a physics glitch from paying out unboundedly. JumpCommand only fills this in
    for attempts that clear its real-jump gate, so a gait suspension phase earns
    nothing.
    """
    command_term = env.command_manager.get_term(command_name)
    return command_term.jump_distance.clamp(0.0, max_distance)


def lin_vel_z_l2_grounded(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(isaaclab) mdp.lin_vel_z_l2, switched off while a jump is commanded.

    2026-09-03: at weight -2.0 this term charges the square of vertical velocity, so a
    1.98 m/s lift-off (a 0.20 m apex) cost about -0.8 reward per jump -- more than the
    entire jump-reward income. Penalising vertical velocity is right for a locomotion
    task, where it means bobbing, and exactly wrong during a jump, where it is the
    point. Gating it on the jump command keeps the flat-running behaviour it was added
    for. See プロジェクト_StageB跳躍が小さい原因分析.md.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jumping = env.command_manager.get_term(command_name).command[:, 0] > 0.0
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * (~jumping).float()


def joint_vel_l2_jump_gated(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(isaaclab) mdp.joint_vel_l2, switched off while a jump is commanded.

    2026-09-15 (Go2 サイクル1). ``joint_vel`` は完走時点の Go2 で **単独最大のペナルティ**
    (`2026-09-14_17-02-52/model_10899` で −0.2416、次点 action_rate −0.1773) でありながら、
    踏切はまさに関節を最速で伸ばす動作で、膝は 37〜58 rad/s 回る
    ([[feedback_mujoco_no_torque_speed_curve]])。**跳べば跳ぶほど課金される**構造で、
    飛距離が 0.799 m で頭打ちになった run の最も大きな逆風がこれだった。

    ``mdp.lin_vel_z_l2_grounded`` (v4) と ``mdp.track_lin_vel_xy_exp_grounded`` (Loop 13) が
    同じ理由で既に jump 窓の外に出されている。三つ目として同じ扱いをする。平地走行側の
    正則化は jump 窓の外で従来どおり効く。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jumping = env.command_manager.get_term(command_name).command[:, 0] > 0.0
    return torch.sum(torch.square(asset.data.joint_vel), dim=1) * (~jumping).float()


def upright_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-tilt / std^2) on the body's deviation from flat. Ported from tak's Go2-Jump.

    2026-09-02: added as one half of the fix for the reward budget that made *dying* the
    optimal policy. The velocity task's positive terms are capped by their weights
    (track_lin_vel_xy 1.5 + track_ang_vel_z 0.75 = 2.25 per second at best) while its
    penalties are L2 in joint velocity/acceleration/action rate and therefore grow with
    the square of commanded speed -- measured at -3.46/s once lin_vel_cmd_levels reached
    2.2 m/s, i.e. a *perfect* policy still scored negative and terminating early was
    worth ~+40 return per episode. This term plus mdp.is_alive restore a positive
    baseline for staying upright and alive. See プロジェクト_StageB崩壊の原因分析.md.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt / std**2)


def jump_trunk_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_drop_m: float = 0.08,
    max_shortfall_m: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the trunk dropping more than ``max_drop_m`` below its take-off height.

    2026-09-03 (v5), added from tanaka's mujoco observation: "the body does not float
    enough, so the landing fails -- it is like throwing the body at the ground". The v4
    reward set had nothing shaping the landing. The trunk reaching the floor was covered
    only by the ``base_contact`` termination, which is binary and terminal: it tells the
    policy "that episode is over" but gives no gradient for *how* to arrive, so nothing
    pushed it to get the legs underneath the body first.

    This is the graded version of the same idea -- a quadratic penalty that grows as the
    trunk sinks, active for the whole jump window (descent included), so the pressure to
    keep the body up and reach down with the legs is felt continuously and *before* any
    contact happens. Squared rather than linear so a near miss is cheap and an actual
    belly-scrape is expensive.

    The reference is ``trigger_height`` (the base height captured at the trigger), not an
    absolute world z: this task's terrain is a COBBLESTONE_ROAD generator, so world z is
    not height above ground and a fixed threshold would mean different things in
    different tiles. Same self-calibrating reference ``max_height_gain`` uses, and it
    sidesteps the class of bug tak documented where one wrong standing-height constant
    fed several quantities at once.

    Deliberately NOT gated on ``has_been_airborne``: the crouch that starts the jump also
    lowers the trunk, and making the crouch free while penalising only the descent would
    just teach a deeper crouch. ``max_drop_m`` = 0.08 leaves a normal crouch untaxed.

    2026-09-03 (v6) -- two corrections, both from watching the v5 policy in mujoco:

    1. **It now stops at the first touchdown.** Running for the whole window was a
       mistake with a very specific consequence: absorbing a landing means bending the
       legs, which lowers the trunk, which this term punished. The only way to avoid the
       penalty was to stay tall on contact -- i.e. to *not* absorb -- so the robot
       bounced. tanaka saw exactly that ("it lands cleanly, then bounces like a ball,
       about three vertical oscillations"), and airborne_time (0.72 s against the 0.277 s
       a single 0.094 m jump takes) confirmed ~2.6 flights per window. The term is meant
       to say "get the legs under you before you arrive", so it belongs strictly before
       arrival; after that the legs must be free to compress.
    2. **The shortfall is clipped.** Unbounded ``shortfall**2`` at weight -30 reached
       -0.42/s in v5 against a -0.001/s baseline (400x) and coincided with mean-reward
       spikes down to -14,867. On generated terrain a robot can descend well below its
       trigger height with nothing wrong, and the quadratic turns that into an arbitrary
       number. 0.15 m caps the term at -0.68/s at weight -30.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    active = (command_term.command[:, 0] > 0.0) & (~command_term.touched_down_once)
    floor = command_term.trigger_height - max_drop_m
    shortfall = (floor - asset.data.root_pos_w[:, 2]).clamp(min=0.0, max=max_shortfall_m)
    return torch.square(shortfall) * active.float()


def jump_repeat_liftoff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Charge per step for every flight phase after the first one in a jump window.

    2026-09-03 (v6). A long jump is one flight, not a series of hops. The v5 policy
    produced roughly three (see mdp.jump_trunk_clearance for the mechanism that caused
    it), and nothing in the reward set had an opinion about that: ``jump_height_progress``
    reads a running maximum, so once the peak is set the extra bounces are free.

    Returned as a per-step count of the surplus flights rather than a one-shot pulse at
    each liftoff, for the same reason mdp.jump_landing_feet_first is per-step: a single
    0.02 s pulse is worth ~2% of what the policy earns from one second of running, which
    is how v3's sparse jump reward ended up irrelevant.

    2026-09-05 (Loop 10): the surplus is counted from the TAKE-OFF's flight phase, not
    from the first flight phase of the window. ``(liftoff_count - 1)`` was the seventh
    instance of this project's recurring bug -- the term was named for bounces but was
    measuring "a running stride happened before the push-off". At a 3.3 m/s approach the
    approach itself has a flight phase per stride, and the take-off is the *strongest*
    liftoff of the window rather than the first, so the two disagree by construction.
    Measured on v8's model_7600: ``unaided_liftoff_index`` was 2.0000 in all five sampled
    iterations, i.e. the take-off was never the first flight. The penalty was therefore
    levied on every real jump, for every step from lift-off to landing, at -0.368/s
    in-window against jump_height's +0.494/s -- and because it is per-step it charged the
    *most* for the long flights the height reward is trying to buy.

    Counting from ``liftoff_index`` leaves the intent intact (the v5 ball-bouncing this
    was written for is still charged, since those flights come after the take-off) while
    costing an ordinary approach stride nothing.
    """
    command_term = env.command_manager.get_term(command_name)
    jumping = command_term.command[:, 0] > 0.0
    surplus = (command_term.liftoff_count - command_term.liftoff_index.long()).clamp(min=0).float()
    return surplus * jumping.float()


def jump_liftoff_vz_cap(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_z_cap: float = 2.0,
    payout_window_s: float = 0.6,
) -> torch.Tensor:
    """踏切の鉛直速度のうち、上限を超えた分だけを二乗で罰する（Go2 サイクル3、2026-09-15）。

    **外から機構を指定した報酬ではなく、実測された相関から来ている。**同じ日に mujoco で
    測った Go2 の5個体は、頂点が高いほど着地率が低いという関係を例外なく示した:

        頂点 0.243 -> 着地10% / 0.267 -> 28% / 0.303 -> 15% / 0.362 -> 2% / 0.386 -> 12%

    比較対象として、同じ報酬セットで学習した Anaguma の成功個体は頂点 0.199 で着地 90%。
    飛距離は ``2*v_x*v_z/g`` なので、**同じ距離なら v_z を下げて v_x を上げたほうが降りやすい**。
    ``jump_takeoff_distance`` は積しか見ないので配分には無関心で、その自由度をここで縛る。

    上限超過分だけを罰するのは、跳躍そのものを潰さないため。v_z <= cap の跳躍は無料で、
    ``real_jump`` ゲートの内側かつ離陸後 ``payout_window_s`` の間だけ課金する
    （Loop 16 の「窓が開いている限り払うと着地しないのが得になる」を繰り返さない）。

    着地を直接ねらう報酬は Go2 で3回外している（踏切対称性 Loop 30、着地スタッガ、
    着地衝撃 Loop 33）。4回目は着地そのものではなく**着地を難しくしている入力（v_z）**を縛る。
    """
    command_term = env.command_manager.get_term(command_name)
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & (command_term.time_since_liftoff <= payout_window_s)
    )
    # 符号の約束: この関数は **正の大きさ** を返し、罰であることは RewTerm の負の weight が
    # 担う（jump_flight_pitch / jump_landing_order と同じ側）。2026-09-15 のスモークで、
    # 関数側にも weight 側にも負を置いて **報酬が +0.163 になっていた**ミスを実測で検出した
    # ——上限を超えるほど得をする、意図と正反対の項だった。
    excess = (command_term.liftoff_vel_z - v_z_cap).clamp(min=0.0)
    return torch.square(excess) * active.float()


def jump_takeoff_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_distance: float = 2.06,
    payout_window_s: float = 0.6,
) -> torch.Tensor:
    """Reward the ballistic distance the take-off has bought, ``2*v_x*v_z/g``.

    2026-09-05 (Loop 11). This replaces mdp.jump_takeoff_speed, which by the end of
    Loop 10 was a constant: it is ``clamp(v_x / 2.0, 0, 1)`` and the policy took off at
    v_x = 2.158 m/s, i.e. saturated, contributing a flat +2.5 with **zero gradient**.

    Loop 10 raised the approach band to U(2.8, 3.3) and the split inverted: v_x went
    1.222 -> 2.158 while v_z fell 1.624 -> 1.301. Distance is the product, and at a
    fixed take-off energy C = v_x^2 + v_z^2 it is maximised at v_x = v_z, so the Loop 10
    end state leaves 30% on the table (ballistic 0.572 m against 0.741 m at C = 7.27).
    Loop 9's post-mortem measured the opposite imbalance and found only +4% there; the
    lever that was empty then is the one that has re-opened now.

    Rewarding the product rather than either factor is what makes the balance the
    physics' problem instead of a weight the next loop has to guess:

        dR/dv_x proportional to v_z,  dR/dv_z proportional to v_x

    which is exactly the objective's own sensitivity ratio, at every operating point.
    A larger jump_height weight was the alternative considered and rejected: apex height
    can also be bought by braking into the take-off, which is the behaviour Loop 10 just
    removed, whereas the product cannot be raised by giving up v_x.

    Dense inside the jump window and gated on ``real_jump``, matching the term it
    replaces -- so the gate against a running stride being scored as a take-off is the
    same one already in use, not a new exposure. ``target_distance`` is the task's own
    2.06 m goal, so the term is still climbing (0.278 of full at the Loop 10 end state)
    rather than saturating the way its predecessor did.
    """
    # 2026-09-13 (Loop 47): payout_window_s added, ported from jump_takeoff_apex.
    # As written in Loop 11 this paid for every step the jump window happened to stay
    # open, which makes NOT LANDING profitable -- Loop 16 regressed exactly that way on
    # the apex term (window 0.81 -> 2.09 s, income 0.75 -> 1.93) and Loop 17 fixed it by
    # bounding the payout to 0.6 s after lift-off. Re-enabling this term without the
    # same bound would reintroduce a failure this project has already paid for once.
    command_term = env.command_manager.get_term(command_name)
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & (command_term.time_since_liftoff <= payout_window_s)
    )
    gravity = 9.81
    ballistic = 2.0 * command_term.liftoff_vel_x * command_term.liftoff_vel_z.clamp(min=0.0) / gravity
    progress = (ballistic / target_distance).clamp(0.0, 1.0)
    return progress * active.float()


def jump_takeoff_energy(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_energy: float = 16.0,
    payout_window_s: float = 0.6,
) -> torch.Tensor:
    """踏切エネルギー ``v_x^2 + v_z^2`` を払う。

    2026-09-15 (サイクル7)。このプロジェクトが3ループかけて出した結論は
    「**v_x 対 v_z の配分比は報酬では動かず、伸びるのは常に踏切エネルギー C**」だった
    (プロジェクト_走り幅跳びLoop7-11と現在地)。にもかかわらず C を払う項が今まで無く、
    目的関数は積 ``2*v_x*v_z/g`` (mdp.jump_takeoff_distance) だけだった。
    積は**同じ C の中での配分**を最適化する勾配しか持たない (固定 C では v_x = v_z が最大)。

    実測でも距離は C で説明できる:

        C = 14.2 (v=2.91,2.40) -> 1.674 m   据え置きベスト model_29200
        C = 12.0 (v=2.76,2.09) -> 1.761 m   記録 model_12200
        C = 10.1 (v=2.49,1.98) -> 1.389 m
        C =  8.9 (v=2.55,1.56) -> 1.310 m   サイクル6

    ``unaided_takeoff_energy`` として既に計測されている量なので、測る側は増えない。
    ゲートは mdp.jump_takeoff_distance と同一 (``real_jump`` かつ離陸後 ``payout_window_s``)
    -- 走行の滞空相を踏切と誤認する経路を新しく作らないため。
    ``target_energy`` 16.0 は実測最大 14.2 の外側なので飽和しない
    (飽和した項が勾配を失い、エントロピー項に舵を渡して崩壊を招くのは
     サイクル2〜4で3回踏んでいる)。
    """
    command_term = env.command_manager.get_term(command_name)
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & (command_term.time_since_liftoff <= payout_window_s)
    )
    energy = command_term.liftoff_vel_x**2 + command_term.liftoff_vel_z.clamp(min=0.0) ** 2
    return (energy / target_energy).clamp(0.0, 1.0) * active.float()


def jump_landing_feet_first(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    body_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["base", "Head_.*", ".*_hip", ".*_thigh"]
    ),
    threshold: float = 1.0,
    landing_window_s: float = 0.3,
) -> torch.Tensor:
    """+1 per step of a touchdown carried entirely by the feet, 0 if anything else lands.

    2026-09-03 (v5). Companion to mdp.jump_trunk_clearance: that term says "do not let
    the trunk reach the floor", this one says "put the feet there instead". Active only
    at the touchdown of a jump that actually cleared the real-jump thresholds -- the
    interval tanaka described going wrong in mujoco, the body arriving before the legs
    were set.

    Paying it per step rather than once at the landing instant matters: a one-shot bonus
    is worth a single 0.02 s step and is swamped by the running income, the same
    magnitude mistake that made v3's sparse jump reward (+0.0003/s) irrelevant next to
    track_lin_vel_xy (+1.15/s).

    2026-09-03 (v6b) -- THE GATE WAS WRONG AND IT KILLED THE JUMP. As first written the
    condition was ``jumping & has_been_airborne & any_foot_down & ~any_body_down``, and
    at a 3.3 m/s approach the running gait's own suspension phase sets
    ``has_been_airborne``. So a robot that never jumped could set the flag by running,
    then collect the full +1.5/step for the remaining ~2 s of the window simply by
    having its feet on the ground -- while a *real* jump, which is airborne and therefore
    has no foot contact, earned nothing during its flight. The term paid for not jumping.

    The run data is unambiguous: over 80-iteration windows this reward climbed
    monotonically 0.0447 -> 0.0584 while jump height fell 0.242 -> 0.048 m and
    real_jump_fraction fell 0.72 -> 0.00, ending 15x larger than the height reward it was
    competing against. Same shape of mistake as v3's -std(F_foot), which was maximised
    while airborne: a term named for one thing that measures another.

    Two gates fix it. ``real_jump`` (0.20 s airborne AND 0.08 m of rise) is what the
    running suspension phase cannot satisfy, and ``landing_window_s`` confines the payout
    to the touchdown itself rather than the rest of the window.
    """
    foot_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    body_sensor: ContactSensor = env.scene.sensors[body_sensor_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    # NOTE: deliberately does NOT test ``jump_command`` or ``has_been_airborne``. The
    # touchdown this term exists to reward is the same event that satisfies ``landed``,
    # and ``landed`` clears both of those in the same _update_command call -- so gating
    # on either makes the term unable to ever fire (verified: 12/12 smoke-test
    # iterations at exactly 0.0000 despite real_jump_fraction reaching 0.25). The three
    # flags below all persist past the window's close and are reset only at the next
    # trigger, and ``time_since_touchdown`` bounds the payout on its own.
    landing_phase = (
        command_term.real_jump
        & command_term.touched_down_once
        & (command_term.time_since_touchdown <= landing_window_s)
    )

    foot_forces = foot_sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :].norm(dim=-1)
    body_forces = body_sensor.data.net_forces_w[:, body_sensor_cfg.body_ids, :].norm(dim=-1)
    any_foot_down = (foot_forces > threshold).any(dim=-1)
    any_body_down = (body_forces > threshold).any(dim=-1)

    return (landing_phase & any_foot_down & (~any_body_down)).float()


def jump_takeoff_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_speed: float = 2.0,
) -> torch.Tensor:
    """Reward horizontal speed carried through the take-off, as a fraction of target.

    2026-09-03 (v7). Measured cause of the short jumps: against a 3.3 m/s approach the
    take-off keeps only **0.70 m/s (v5's model_4300) / 0.84 m/s (v6b's model_5300)** --
    75-79% of the run's speed is thrown away braking into the push-off, so the robot is
    close to jumping from a standstill. The reference paper jumps 96 cm while holding a
    2 m/s approach.

    Distance is ``2*v_x*v_z/g`` and mdp.jump_height_progress already drives v_z, so this
    term supplies the other factor and nothing else. At v6b's measured v_z of 1.25 m/s,
    lifting v_x from 0.84 to 2.0 m/s takes the take-off distance from 0.21 m to 0.51 m.

    Clipped-linear rather than the exp-around-a-target form used for height: there is no
    value to hit, more is strictly better, and the approach speed caps it physically.

    Gated on ``real_jump``. Without that gate the term would be farmable by never
    jumping at all -- ``liftoff_vel_x`` latches on the strongest flight phase in the
    window, and at 3.3 m/s an ordinary running stride produces one. That is the same
    trap that has now cost this project six separate bugs; see
    プロジェクト_報酬項の名前と実測対象のズレ5例.md.
    """
    command_term = env.command_manager.get_term(command_name)
    active = (command_term.command[:, 0] > 0.0) & command_term.real_jump
    progress = (command_term.liftoff_vel_x / target_speed).clamp(0.0, 1.0)
    return progress * active.float()


def jump_takeoff_apex(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_apex: float = 0.28,
    payout_window_s: float = 0.6,
    approach_lo: float = 2.0,
    approach_hi: float = 2.5,
) -> torch.Tensor:
    """Reward the ballistic apex the take-off has bought, ``v_z^2 / (2g)``.

    2026-09-05 (Loop 12). The vertical counterpart of mdp.jump_takeoff_distance, and its
    replacement: the objective for this loop is apex height rather than distance, because
    height is what the jump has to show and it is the factor that has not moved in four
    loops (0.098 -> 0.119 -> 0.103 -> 0.116 m across v7..v10, against a 0.28 m goal).

    Why this rather than simply raising ``jump_height``'s weight -- the two are not the
    same lever, and the weight route has now been tried three times:

    * ``jump_height`` scores ``max_height_gain``, the trunk's measured rise. That can be
      bought by extending the legs, by a late pitch, or by braking into the push-off,
      none of which is a take-off. This term scores only what the lift-off velocity can
      pay for, exactly as its distance predecessor did for ``2*v_x*v_z/g``.
    * Its gradient does not fade near the target. ``jump_height`` is a Gaussian around
      0.28 m whose pull *weakens* as the peak approaches, whereas ``v_z^2`` keeps
      ``dR/dv_z`` proportional to v_z all the way up.

    The measured reason the distance term is being retired is the split, not the term:
    Loops 9, 10 and 11 all moved the reward weights on one factor and all three ended
    with the ratio v_x/v_z unchanged at 1.6-1.7 while the take-off *energy* rose. Paying
    for the product keeps rewarding whichever factor is cheapest to buy, and at a 2.1 m/s
    approach that is v_x every time. Removing v_x from the objective is what makes the
    height the only way to earn. The approach itself is not at risk -- it is held by the
    velocity-tracking reward against the U(2.8, 3.3) command, not by this term.

    Gated on ``real_jump`` and dense inside the jump window, the same exposure the two
    terms it replaces already had -- a running stride's suspension phase does not score.
    """
    command_term = env.command_manager.get_term(command_name)
    # 2026-09-06 (Loop 17): the payout is bounded to ``payout_window_s`` after the latched
    # take-off. ``liftoff_vel_z`` does not change after the take-off, so paying it for
    # every step the window happens to stay open made NOT LANDING profitable -- Loop 16
    # regressed exactly that way (window 0.81 -> 2.09 s, this term's income 0.75 -> 1.93,
    # every landing term to ~0, falls 0.006 -> 0.10). Bounding the payout keeps the dense
    # gradient that makes the take-off learnable while making the window's length worth
    # nothing.
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & (command_term.time_since_liftoff <= payout_window_s)
    )
    gravity = 9.81
    apex = torch.square(command_term.liftoff_vel_z.clamp(min=0.0)) / (2.0 * gravity)
    progress = (apex / target_apex).clamp(0.0, 1.0)
    # 2026-09-09 (Loop 31): scale by the approach speed at the moment the jump was asked
    # for. tanaka, jumping model_27000 in mujoco: "with a run-up it failed; standing jumps
    # succeeded several times". The trace agrees, and the cause is in the task, not the
    # policy -- **nothing in the reward set has ever required the jump to happen while
    # running**. jump_command fires irrespective of speed and every jump reward (this one
    # is the largest at +1.68) pays the same for a standing jump, which is the easier one.
    # The objective dropped v_x in Loop 12 for a good reason (a product buys the cheap
    # factor) and nineteen loops later the task called "running long jump" had drifted into
    # a standing one. Same shape as Loop 10's APPROACH_SPEED_MS mistake.
    #
    # A gate rather than a product or a separate speed reward: a product would let the
    # cheap factor be bought again, and paying for speed separately leaves the "sprint,
    # stop, then jump" loophole. Multiplying the jump's own payout by the approach makes a
    # slow jump worth less *as a jump*, with no way to collect the speed on its own.
    #
    # Ramp, not a step, and placed above the measured 2.256 m/s rather than below it: at
    # the operating point the gate reads 0.51, mid-range with a live gradient, and reaching
    # the 2.5 m/s target doubles the largest reward in the set. A gate that started
    # saturated would be a dead term (Loop 11 / 16 / 18), and a hard step would give zero
    # gradient on either side of it.
    #
    # The income cost is deliberate and NOT compensated by the weight: 1.68 -> 0.86 at
    # today's speed. Compensating would restore exactly what this change exists to remove.
    # Jumping still pays (jump_height + jump_foot_clearance + this = ~1.45 against 0 for
    # not jumping), so the cheapest response is to run faster, not to stop jumping.
    gate = ((command_term.trigger_speed - approach_lo) / (approach_hi - approach_lo)).clamp(0.0, 1.0)
    return progress * gate * active.float()


def track_lin_vel_xy_exp_grounded(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    jump_command_name: str = "jump_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(isaaclab) mdp.track_lin_vel_xy_exp, switched off while a jump is commanded.

    2026-09-06 (Loop 13). The same treatment mdp.lin_vel_z_l2_grounded got in v4, applied
    to the term on the other side of the same trade. Velocity tracking pays a measured
    ~1.15 per second against the U(2.8, 3.3) forward command, and it was never gated on
    the jump window -- so for the whole of the take-off the policy has been paid to hold
    its forward speed at the exact moment the task needs it converted into vertical
    speed. Loop 12 shows the size of the tax directly: v_z rose 1.378 -> 1.893 only by
    giving up v_x 2.27 -> 1.315, i.e. the height that was won was paid for out of this
    term's income.

    Gating is on the jump window, not on the flight phase, because the conversion happens
    in the take-off stance while the feet are still down. The approach itself is
    unaffected: the window opens at the trigger, so everything before it still tracks.

    The risk this creates is the opposite one -- with nothing paying for forward speed
    inside the window, the run could decay into a standing vertical jump, which is not
    this task and is already covered by another team. The approach speed at the trigger
    (2.0 m/s measured) and the ~0.2 s take-off stance bound how much can be shed, and
    ``unaided_liftoff_vel_x`` is the number to watch: below about 0.8 m/s the jump has
    stopped being a running jump.
    """
    from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp

    jumping = env.command_manager.get_term(jump_command_name).command[:, 0] > 0.0
    base = track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    return base * (~jumping).float()


def jump_pitch_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_rate: float = 6.0,
) -> torch.Tensor:
    """Bounded penalty on body pitch rate while a jump is commanded.

    2026-09-06 (Loop 13), from tanaka's mujoco description of the Loop 12 policy: "the
    front legs still think they are flying, the rear legs touch down at once, and the
    front legs get slammed into the ground". That is a nose-up flight attitude, produced
    by a push-off the rear legs dominate, and nothing in the reward set constrains it --
    ``flat_orientation_l2`` (-2.5) charges the tilt itself but not the rotation that
    creates it, and by the time the tilt exists in flight there is nothing left to push
    against to undo it.

    Attacking the rate rather than the attitude is deliberate: the rotation is set during
    the take-off stance, which is also where a symmetric four-leg push-off would put more
    of the same energy into pure vertical translation instead. So this term and the height
    objective are expected to pull the same way, not to trade off.

    Bounded on purpose -- ``(|w_y| / max_rate)^2`` clipped at 1, so the worst case costs
    exactly ``weight`` per second. An unbounded L2 on pitch rate is the shape that made
    ``action_rate`` 38% of the entire penalty budget in the v2 collapse, and it would
    charge the explosive part of the push-off hardest.
    """
    asset: Articulation = env.scene["robot"]
    jumping = env.command_manager.get_term(command_name).command[:, 0] > 0.0
    rate = asset.data.root_ang_vel_b[:, 1].abs()
    return torch.square(rate / max_rate).clamp(0.0, 1.0) * jumping.float()


def jump_roll_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_rate: float = 2.2,
) -> torch.Tensor:
    """Bounded penalty on body roll rate during the flight of a real jump.

    2026-09-09 (Loop 30). 43 unaided jumps traced in mujoco on model_26600 (23 landed,
    20 did not) put the mean flight roll rate at 0.50 rad/s for the landings and 1.07 for
    the failures -- a single threshold on it classifies 79% of the attempts, essentially
    as well as the pitch rate does (81%). Flight time and jump height classify nothing.

    The first design attacked the cause instead: the rear calves sat 1.017 rad apart at
    lift-off in mujoco and that gap correlated with the roll at r = -0.86, so a take-off
    symmetry term looked like the right lever. **Measuring the same quantity in Isaac
    before writing that reward killed it** -- Isaac's rear-calf asymmetry is 0.107 rad at
    the same operating point (approach 2.19 m/s, real_jump 0.744), nine times smaller,
    while the flight roll rate is 1.350 rad/s, HIGHER than mujoco's failure group. The
    roll is not produced by the asymmetry; the correlation was between two quantities that
    co-vary under mujoco's conditions only.

    What that measurement leaves is this: the roll exists in Isaac, at a level mujoco
    punishes and Isaac does not (0.4% falls at 1.35 rad/s), and **nothing in the reward
    set charges it**. ``jump_pitch_rate`` covers the other axis and has sat at weight 0.0
    since Loop 13; there has never been a roll term at all.

    ``max_rate`` is 2.2 rather than the measured 1.35 on purpose: at the measured value the
    term reads ``(1.35/2.2)^2 = 0.38``, in the middle of its range with a live gradient.
    Setting it at or below the operating point would start the term saturated, which is how
    ``jump_takeoff_speed`` (Loop 11), ``jump_takeoff_apex`` (Loop 16) and ``jump_height``
    (Loop 18) each died.

    Charged only while airborne inside a ``real_jump``. The airborne gate matters: at a
    2 m/s approach every running stride has a flight phase, and charging body roll during
    normal running would price the gait rather than the jump. Magnitude, not signed --
    unlike pitch, the failures roll both ways, so this is two-sided by construction and
    does not have the runaway shape of the ``relu()`` penalties in Loop 19.
    """
    asset: Articulation = env.scene["robot"]
    command_term = env.command_manager.get_term(command_name)
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & command_term.airborne_now
    )
    rate = asset.data.root_ang_vel_b[:, 0].abs()
    return torch.square(rate / max_rate).clamp(0.0, 1.0) * active.float()


def jump_flight_pitch(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_tilt: float = 0.35,
) -> torch.Tensor:
    """Bounded penalty on the nose-up attitude while a jump is commanded.

    2026-09-06 (Loop 13). tanaka's mujoco description of the Loop 12 policy -- "the front
    legs still think they are flying, the rear legs touch down at once, and the front legs
    get slammed into the ground" -- turned out to be exactly right, and universal:
    instrumenting the landing gives ``unaided_rear_first_fraction`` = **1.00** and a
    landing attitude of ``projected_gravity_b[:, 0]`` = **-0.173** (about 10 degrees
    nose-up). Every single real jump lands rear-first.

    It was also invisible until the measurement was gated correctly. Ungated, the same
    metric read 0.124 -- because at a 2 m/s approach the first contact inside the jump
    window is an ordinary running stride, not the jump's landing. That is the ninth
    instance of the failure mode collected in プロジェクト_報酬項の名前と実測対象のズレ7例.md,
    and the first one where the wrong number would have said "no problem here".

    The sign is taken from that measurement, not from a rotation convention: negative
    ``projected_gravity_b[:, 0]`` is the attitude that lands rear-first on this robot, so
    only ``relu(-g_x)`` is charged. A symmetric penalty would also charge nose-down, which
    is the direction a long jump actually wants to land in.

    Bounded and squared -- ``clamp(relu(-g_x)/max_tilt, 0, 1)^2`` -- so the worst case
    costs exactly ``weight`` per second in the window and the gradient is gentle near
    zero, where the attitude is already fine.

    ``flat_orientation_l2`` (weight -2.5, always on) does not cover this: it is symmetric,
    it charges roll and pitch together, and at -0.0035 measured it is three orders of
    magnitude below the jump terms -- it has never been large enough to shape a take-off.
    """
    asset: Articulation = env.scene["robot"]
    jumping = env.command_manager.get_term(command_name).command[:, 0] > 0.0
    nose_up = (-asset.data.projected_gravity_b[:, 0]).clamp(min=0.0)
    return torch.square((nose_up / max_tilt).clamp(0.0, 1.0)) * jumping.float()


def jump_landing_gear(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta: float = 0.12,
    deadband: float = 0.05,
) -> torch.Tensor:
    """Penalty for keeping the front feet above the rear ones while descending from a jump.

    2026-09-06 (Loop 14). Loop 13 removed the nose-up landing attitude entirely (landing
    pitch -0.173 -> -0.035) and ``unaided_rear_first_fraction`` did not move off **1.00**.
    So the landing order is not the body's attitude, it is where the legs are: the front
    pair is still tucked when the rear pair arrives, which is exactly tanaka's "the front
    legs still think they are flying, and then they get slammed into the ground".

    ``jump_flight_pitch`` cannot reach this -- a perfectly level body can still land on
    its rear feet if only the rear legs are extended -- and neither can any of the
    existing landing terms: ``jump_trunk_clearance`` only says the trunk must stay up, and
    ``jump_landing_feet_first`` pays for a touchdown carried by feet rather than by the
    body, without caring which feet.

    Charged only while airborne and descending. Ascending, a tuck is correct (it is what
    buys the 0.25 m foot clearance the jump is measured by), so charging the whole flight
    would fight the height objective this loop is supposed to protect.

    Bounded: ``clamp(relu(front_z - rear_z)/max_delta, 0, 1)^2``, worst case ``weight`` per
    second. 0.12 m is roughly the leg travel between a tucked and an extended foot.
    """
    asset: Articulation = env.scene["robot"]
    command_term = env.command_manager.get_term(command_name)
    jumping = command_term.command[:, 0] > 0.0
    descending = asset.data.root_lin_vel_w[:, 2] < 0.0
    foot_z = asset.data.body_pos_w[:, command_term._foot_link_ids, 2]
    front_z = foot_z[:, command_term._front_link_slots].mean(dim=1)
    rear_z = foot_z[:, command_term._rear_link_slots].mean(dim=1)
    airborne = ~command_term.contact_sensor.data.current_contact_time[
        :, command_term.cfg.contact_sensor_cfg.body_ids
    ].gt(0.0).any(dim=-1)
    active = jumping & descending & airborne
    # 2026-09-06 (Loop 19 -> 20): TWO-SIDED, with a deadband. The one-sided version --
    # relu(front_z - rear_z) -- did its job and then kept going: the gap went +0.279 m
    # (front tucked, lands rear-first) through 0.000 to **-0.10 to -0.15 m**, i.e. the
    # front feet now arrive well BELOW the rear ones, nose-down, and falls rose from 0.84%
    # to 2.2%. Nothing charged the far side, so the policy ran off it. Same shape of error
    # as jump_flight_pitch, which also only charges one direction.
    #
    # The deadband keeps what the term was for: a long jump SHOULD land slightly
    # front-first, so 0 to ``deadband`` below the rear feet is free. Only excursions past
    # that, in either direction, are charged.
    # 2026-09-06 (Loop 20 -> 21): the free band is ASYMMETRIC, [-deadband, 0]. The
    # symmetric ``|delta| - deadband`` form left delta in (0, +deadband] free -- but
    # positive delta means the front feet are ABOVE the rear ones, which is precisely the
    # rear-first geometry mdp.jump_landing_order charges. The two terms were disagreeing
    # about a 5 cm band. Here front-above-rear is charged from 0, front-below-rear is free
    # to ``deadband`` (a long jump should land slightly front-first) and charged past it.
    delta = front_z - rear_z
    excess = delta.clamp(min=0.0) + (-deadband - delta).clamp(min=0.0)
    return torch.square((excess / max_delta).clamp(0.0, 1.0)) * active.float()


def jump_landing_order(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalty held from touchdown to window close when the jump landed rear-feet-first.

    2026-09-06 (Loop 19). Loop 14's ``jump_landing_gear`` closed the GEOMETRY -- the front
    feet were 0.279 m above the rear ones at touchdown and are now level (0.000 m) -- and
    the order followed it down to 0.246 at Loop 17's best. It then drifted back to
    0.53-0.79 through Loop 18. With the geometry equalised, which end touches first is
    decided by swing timing, and no term rewards that; the order has become a coin flip
    that nothing is holding.

    Dense from the recorded touchdown to the end of the window (about 0.25 s) rather than
    a single-step charge at the touchdown itself, because IsaacLab computes rewards BEFORE
    the command term updates -- a reward reading a flag set in the same step's command
    update reads the previous step's value. That off-by-one is what made
    ``jump_landing_feet_first`` structurally unable to fire for the whole of v5 and v6.
    Reading a flag that persists until the next trigger sidesteps the ordering entirely.
    """
    command_term = env.command_manager.get_term(command_name)
    # Gated on time-since-touchdown rather than on the jump window still being open. With
    # min_feet_down_for_landing at 1 the window closes on essentially the same step as the
    # touchdown, so a ``command > 0`` gate left this term one step to charge in and it
    # measured -0.0003 in the smoke against a rear-first rate of 0.82. Charging for a fixed
    # 0.3 s after the landing makes the penalty independent of how fast the window closes.
    active = (
        command_term.touched_down_once
        & command_term.landed_rear_first
        & (command_term.time_since_touchdown <= 0.3)
    )
    return active.float()


def jump_leg_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalty for a thigh or calf link touching the ground during a jump.

    2026-09-09 (Loop 34), tanaka: "it sometimes lands on its knees and elbows -- add a
    penalty for that." Measured before changing anything: a knee or elbow touches the
    ground on **99.8% of jumps**. Not sometimes -- essentially always.

    A penalty already existed: ``undesired_contacts`` (weight -1, on Head/hip/thigh/calf,
    all the time). It pays **-0.040**, i.e. 3% of ``jump_takeoff_apex``'s +1.35. That is
    the same shape as every other landing term in this project -- present in the config,
    never large enough to change anything -- and 99.8% is what a policy does with a penalty
    it can afford to ignore. Nineteen loops have therefore trained landing on the knees as
    standard practice, and it took a person watching to notice.

    A separate jump-window term rather than raising ``undesired_contacts``: that one is
    always on, so raising it would also charge leg contacts during ordinary running and
    reprice the gait. This one only fires from the latched take-off onward.

    Charged per step while the contact is present, not latched per jump. Latched is a
    single payment however long the robot lies on its knee; per-step makes a brief scuff
    cheap and a sustained collapse expensive, which is the actual difference between the
    two behaviours.

    Cheapest ways to avoid it, listed before implementing: land on the feet (the intent);
    do not jump (blocked by ``real_jump`` and the take-off terms); stay airborne (blocked
    by the window timeout). No path here buys a lower jump, unlike the landing-impact
    penalty this replaces -- leg contact is a discrete event, not a magnitude that scales
    with jump height.
    """
    return env.command_manager.get_term(command_name).knee_contact_now.float()


def jump_landing_impact(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_impact_n: float = 1400.0,
    payout_window_s: float = 0.3,
) -> torch.Tensor:
    """Bounded penalty on the peak ground reaction force of the landing.

    2026-09-09 (Loop 33). tanaka, on model_27200: "even when it lands it is a case of just
    about staying up rather than landing lightly. You could not put this on the real
    robot." Measured, that is 846 N of peak summed vertical foot force -- **5.8x body
    weight** (Go2 is about 15 kg, 147 N). ``success_rate`` reads 97% at the same time,
    because it asks only for feet down and an upright trunk and has never had any term for
    how hard the robot arrived. This penalty is that missing term.

    It charges the RESULT, not a mechanism, and that is deliberate. The mechanism was tried
    first and was wrong: tanaka's model of how quadrupeds land a running jump -- front feet
    first, rear following, the gap spreading the impact -- was implemented as
    mdp.jump_landing_stagger and measured before being trusted. Per-env over ~700 landings,
    the correlation between the front/rear gap and the impact is **+0.20**, and landings
    with the front feet more than 10 ms early hit at **875 N** against **792 N** for
    rear-first. On this machine the stagger makes the landing HARDER, not softer -- the
    front legs are shorter, lighter and rigidly jointed to the trunk, with none of the
    scapular suspension that lets an animal absorb a front-first landing. The reward was
    already written at weight 30 and would have gone into a full run had the correlation
    not been checked.

    Two mechanisms have now been specified from outside and both were wrong on this robot
    (take-off symmetry in Loop 30, landing stagger here), while every measurement has paid.
    So this term names the objective and leaves the means to the policy: shorter, softer
    contacts, a deeper knee bend, a lower approach velocity into the touchdown -- whatever
    the optimiser finds.

    The cheapest ways to collect it, listed before implementing:
      - **Jump lower.** A smaller jump lands softer, and this is the real risk. It is held
        off by ``jump_takeoff_apex`` (+1.35) and ``jump_foot_clearance`` (+0.39) against
        this term's ~0.37 at the operating point, and by the ``real_jump`` gate underneath
        (0.20 s airborne and 0.08 m of rise). The height metrics are the thing to watch in
        the smoke; if they move, the weight is too high.
      - **Do not land at all** -- the payout is bounded to 0.3 s from the recorded
        touchdown, and not landing forfeits every landing term.
      - **Put fewer feet down** -- the quantity is the SUM over feet, i.e. the total ground
        reaction, so spreading the same impulse across fewer feet does not reduce it.

    ``max_impact_n`` is 1400 rather than the measured 846 for the usual reason: at 846 the
    term reads ``(846/1400)^2 = 0.37``, mid-range with a live gradient. Normalising by the
    operating point itself would start it saturated, which is how ``jump_takeoff_speed``
    (Loop 11), ``jump_takeoff_apex`` (Loop 16) and ``jump_height`` (Loop 18) each died.

    Dense from the recorded touchdown rather than a single-step charge, because IsaacLab
    evaluates rewards BEFORE the command term updates -- the off-by-one that made
    ``jump_landing_feet_first`` unable to fire at all through v5 and v6.
    """
    command_term = env.command_manager.get_term(command_name)
    active = (
        command_term.touched_down_once
        & command_term.real_jump
        & (command_term.time_since_touchdown <= payout_window_s)
    )
    return torch.square((command_term.landing_impact / max_impact_n).clamp(0.0, 1.0)) * active.float()


def jump_landing_stagger(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_lag_s: float = 0.030,
    tolerance_s: float = 0.030,
    payout_window_s: float = 0.3,
) -> torch.Tensor:
    """Reward landing front feet first with a TIME gap before the rear feet arrive.

    2026-09-09 (Loop 33), from tanaka after jumping model_27200: "the landing rate is very
    low, and even when it lands it is a case of just about staying up rather than landing
    lightly. Go2 pushes off with both rear legs at once and lands on both at once, and that
    is braking it. Most quadrupeds taking a running jump land front feet first with the
    rear following immediately -- staggering the two touchdowns spreads the impact."

    Measured on model_27200 before writing this: the gap between the first front contact
    and the first rear contact is **-6.9 ms**, i.e. the rear lands FIRST and within a
    single 20 ms control step -- effectively simultaneous, and in the wrong order. Peak
    summed vertical foot force at the landing is 870 N, 5.9x body weight.

    This is not the same quantity as ``jump_landing_order`` (Loop 19), which charges a
    boolean sampled at one instant, nor as ``jump_landing_gear`` (Loop 14/20/21), which
    charges the front/rear foot HEIGHT difference. Both are already doing their job:
    rear_first_fraction is down to 0.035 and the front feet are geometrically lower. **The
    feet are positioned correctly and still arrive at the same time.** That is Loop 13's
    finding again -- the ordering is set by swing timing, not by geometry -- and the time
    gap is what none of the existing terms can see.

    It also explains a gap between the logs and what a person watching sees.
    ``success_rate`` reads 97% because it asks only for feet down and an upright trunk; it
    has never had any term for how hard the robot arrived. 870 N is what "just about
    staying up" looks like as a number.

    Shape: a band around ``target_lag_s`` with a flat top of width ``tolerance_s``, falling
    off on BOTH sides. Two-sided by construction, because a one-sided ``relu`` is what sent
    ``jump_landing_gear`` from +0.279 m through zero to -0.15 m and ``jump_flight_pitch``
    from nose-up to nose-down in Loop 19 -- a one-sided penalty does not define a target,
    it defines a direction to run in. An unbounded reward for "front earlier" would buy a
    nose-dive onto the front legs with the rear crashing down afterwards, which is worse
    than what it replaces.

    The cheapest ways to collect this were listed before implementing:
      - Land front-first with an enormous gap -> priced out by the upper side of the band.
      - Never put the rear feet down -> ``landing_lag`` stays 0 and this pays nothing,
        because the metric requires BOTH sides to have touched.
      - Do not jump at all -> ``real_jump`` gates it, and the take-off terms dominate.
      - Stretch the window open -> the payout is bounded to 0.3 s from the touchdown, the
        fix Loop 17 had to make after the Loop 16 regression.

    Paid densely from the recorded touchdown for the same reason ``jump_landing_order`` is:
    IsaacLab evaluates rewards BEFORE the command term updates, so a single-step charge at
    the touchdown reads the previous step's value. That off-by-one made
    ``jump_landing_feet_first`` structurally unable to fire for the whole of v5 and v6.
    """
    command_term = env.command_manager.get_term(command_name)
    active = (
        command_term.touched_down_once
        & command_term.real_jump
        & (command_term.time_since_touchdown <= payout_window_s)
    )
    lag = command_term.landing_lag
    # Distance outside the flat top of the band, normalised by the band's own half-width.
    excess = (lag - target_lag_s).abs() - tolerance_s
    score = 1.0 - (excess.clamp(min=0.0) / tolerance_s).clamp(0.0, 1.0)
    # A lag of exactly zero means neither side has been recorded yet; pay nothing for it
    # rather than letting the band's tail reward an unmeasured landing.
    score = torch.where(lag == 0.0, torch.zeros_like(score), score)
    return score * active.float()


def jump_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_clearance: float = 0.65,
    payout_window_s: float = 0.6,
) -> torch.Tensor:
    """Reward the ground clearance of the lowest foot, ``peak_foot_clearance / target``.

    2026-09-06 (Loop 26). The height line has rewarded two quantities -- lift-off v_z and
    trunk rise -- and both are now done: v_z sits at 2.73 m/s with the knee actuators
    saturated at 45.43 N.m, and the trunk rise has passed its 0.28 m goal. Neither is what
    tanaka was looking at when he said the jump did not look very high. **What a person
    watching sees is daylight under the feet**, and that quantity has been measured since
    Loop 13 (``unaided_peak_foot_clearance``) without ever being rewarded.

    It is also the one height number that is not capped by the actuators. model_23600 and
    model_22100 have the same take-off -- v_z 2.731 vs 2.692 -- but clearances of 0.497 m
    and 0.603 m. That 0.11 m difference is entirely in how the legs are folded in flight,
    which costs the knees nothing.

    Bounded to ``payout_window_s`` after the take-off for the same reason
    mdp.jump_takeoff_apex is: ``peak_foot_clearance`` is a running maximum, so paying it for
    every step the window stays open would make not landing profitable -- the exploit that
    produced Loop 16's regression.

    The obvious risk is that a harder tuck arrives at the landing with the legs not extended
    -- which is exactly what mdp.jump_landing_gear now measures and charges on both sides,
    so the guard is already in place rather than being added alongside.
    """
    command_term = env.command_manager.get_term(command_name)
    active = (
        (command_term.command[:, 0] > 0.0)
        & command_term.real_jump
        & (command_term.time_since_liftoff <= payout_window_s)
    )
    progress = (command_term.peak_foot_clearance / target_clearance).clamp(0.0, 1.0)
    return progress * active.float()
