"""MDP terms for the walk / fall / get-up environment.

Walking and getting up want opposite things from the same reward terms -- ``flat_orientation_l2``,
``lin_vel_z_l2`` and ``undesired_contacts`` are exactly what a robot rolling onto its feet has to
violate. So each source task's terms are kept and switched by the robot's *state*: the walking set
applies while the robot is standing, the get-up set while it is not. The switch is a soft weight
rather than a step so the return does not jump at an arbitrary attitude.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STATE_STANDING = "standing"
STATE_DOWN = "down"


def standing_weight(
    env: ManagerBasedRLEnv,
    upright_range: tuple[float, float] = (0.8, 0.95),
    height_range: tuple[float, float] = (0.18, 0.25),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Weight in ``[0, 1]`` for "on its feet": trunk upright *and* base at walking height.

    Both, because each alone is satisfied by a fallen robot: lying prone with folded legs is upright
    by gravity, and a robot on its side can hold its base at 0.25 m. Ramps linearly across each
    range.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    neg_gz = -asset.data.projected_gravity_b[:, 2]
    upright = ((neg_gz - upright_range[0]) / (upright_range[1] - upright_range[0])).clamp(0.0, 1.0)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    tall = ((height - height_range[0]) / (height_range[1] - height_range[0])).clamp(0.0, 1.0)
    return upright * tall


def state_gated(
    env: ManagerBasedRLEnv,
    state: str,
    term: Callable[..., torch.Tensor],
    term_params: dict | None = None,
) -> torch.Tensor:
    """Scale an existing reward term by :func:`standing_weight` (``"standing"``) or its complement
    (``"down"``). The wrapped term's own arguments go in ``term_params``; see
    :func:`resolve_gated_term_params` for how the ``SceneEntityCfg`` inside them gets resolved."""
    weight = standing_weight(env)
    if state == STATE_DOWN:
        weight = 1.0 - weight
    elif state != STATE_STANDING:
        raise ValueError(f"Unknown state {state!r}; expected {STATE_STANDING!r} or {STATE_DOWN!r}.")
    return term(env, **(term_params or {})) * weight


def is_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 while the robot is not on its feet. As a small negative reward it both asks for a quick
    recovery and puts the fraction of time spent down in the logs."""
    return 1.0 - standing_weight(env)


def body_impact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Contact force above ``threshold`` on the given bodies, summed.

    Lying on the trunk is expected (the get-up starts there), so this does not charge for contact,
    only for the spike of hitting the ground -- the part of a fall the policy can soften by putting
    a leg out or tucking in.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0]
    return (forces - threshold).clamp(min=0.0).sum(dim=1)


def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base height above the environment origin, for the critic. Flat ground only."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def knockdown(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    angular_speed_range: tuple[float, float],
    linear_speed_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Shove the robot hard enough to topple it: a roll or pitch rate plus a matching sideways kick.

    ``push_by_setting_velocity`` samples each axis uniformly around zero, so most of its draws are
    small and the strong ones point in arbitrary combinations. Here the magnitude is drawn from a
    range bounded away from zero and the linear kick is aligned with the tipping direction, so the
    event reliably produces falls of every kind -- to either side, forward, backward -- and the top
    of the range still leaves some the locomotion expert can catch.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = asset.device

    # Tipping axis in the horizontal plane, uniformly in heading: covers roll, pitch and between.
    heading = torch.rand(n, device=device) * 2.0 * torch.pi
    axis = torch.stack([torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)], dim=-1)
    angular = math_utils.sample_uniform(*angular_speed_range, (n, 1), device=device)
    linear = math_utils.sample_uniform(*linear_speed_range, (n, 1), device=device)
    # A positive rate about `axis` tips the top of the trunk towards `axis x z`; kick the base that way.
    push_dir = torch.stack([axis[:, 1], -axis[:, 0], torch.zeros_like(heading)], dim=-1)

    vel_w = asset.data.root_vel_w[env_ids].clone()
    vel_w[:, :3] += linear * push_dir
    vel_w[:, 3:] += angular * axis
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def resolve_gated_term_params(env: ManagerBasedRLEnv, env_ids=None) -> None:
    """Resolve ``SceneEntityCfg`` entries nested inside a term's ``term_params``. Startup event.

    The wrapped term's arguments are nested because Isaac Lab rejects a ``**kwargs`` catch-all in a
    term signature, and it resolves scene entities only at the top level of ``params``. An
    unresolved one does not fail: ``body_ids`` stays ``slice(None)`` and silently selects every
    body. (Same as ``tasks.multitask.mdp.rewards.resolve_gated_term_params`` on ``feat/multi-task``.)
    """
    for manager in (env.reward_manager, env.termination_manager):
        for name in manager.active_terms:
            params = manager.get_term_cfg(name).params.get("term_params")
            if isinstance(params, dict):
                _resolve_nested(env, params, f"{name}:term_params")


def _resolve_nested(env: ManagerBasedRLEnv, params: dict, where: str) -> None:
    for key, value in params.items():
        if isinstance(value, SceneEntityCfg):
            try:
                value.resolve(env.scene)
            except ValueError as error:
                raise ValueError(f"Error while parsing '{where}:{key}'. {error}")
        elif isinstance(value, dict):
            _resolve_nested(env, value, f"{where}:{key}")


# -- get-up terms, from the original 42-dim get-up task this one replaced (iit-DLSLab/get-up-isaaclab)


def reset_fallen_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos_range: tuple[float, float] = (-3.14159, 3.14159),
    standing_prob: float = 0.1,
):
    """Reset to a random fallen pose (or, with ``standing_prob``, to standing)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # -- joints: default + wide uniform offset, clamped to the robot's own limits.
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*joint_pos_range)
    joint_limits = asset.data.default_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)

    # -- root: random full-SO(3) orientation, lifted slightly so the tumbled
    # pose doesn't spawn in self/ground collision.
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 2] += 0.15
    root_state[:, 7:] = 0.0

    keep_standing = torch.rand(len(env_ids), device=env.device) < standing_prob
    rand_quat = math_utils.random_orientation(len(env_ids), device=env.device)
    root_state[:, 3:7] = torch.where(keep_standing.unsqueeze(-1), root_state[:, 3:7], rand_quat)
    joint_pos[keep_standing] = asset.data.default_joint_pos[env_ids][keep_standing]

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-((z - target) / std)^2)``, gated to only pay out once the base
    is roughly upright -- otherwise the reward can be farmed by pushing a
    sideways- or upside-down torso up to the target height without actually
    standing up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    upright = asset.data.projected_gravity_b[:, 2] < -0.5
    return torch.exp(-height_error / std**2) * upright


def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint-position distance from the nominal standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
