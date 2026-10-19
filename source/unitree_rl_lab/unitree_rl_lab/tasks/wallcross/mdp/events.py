"""Reset events for the hind-stance wall crossing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_BANKS: dict[str, dict[str, torch.Tensor]] = {}


def reset_from_state_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start from a recorded walking state instead of standing still.

    The crossing is only switched on while the walking expert is walking, so the expert takes over
    mid-stride, moving at the approach speed. ``path`` is a bank written by
    ``scripts/rsl_rl/record_walker_states.py``: joint state, height, roll/pitch and body-frame
    velocities, heading removed. Runs after ``reset_base`` and ``reset_robot_joints``: the tile
    position and heading they drew are kept, everything else is replaced by a random bank entry.
    """
    if path not in _BANKS:
        _BANKS[path] = {k: v.to(env.device) for k, v in torch.load(path, map_location="cpu").items()}
    bank = _BANKS[path]
    asset: Articulation = env.scene[asset_cfg.name]
    k = len(env_ids)
    pick = torch.randint(0, bank["root_z"].shape[0], (k,), device=env.device)

    pos = asset.data.root_pos_w[env_ids].clone()
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + bank["root_z"][pick]
    yaw = math_utils.yaw_quat(asset.data.root_quat_w[env_ids])
    quat = math_utils.quat_mul(yaw, bank["root_quat_rp"][pick])
    lin = math_utils.quat_apply(quat, bank["root_lin_vel_b"][pick])
    ang = math_utils.quat_apply(quat, bank["root_ang_vel_b"][pick])
    asset.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat((lin, ang), dim=-1), env_ids=env_ids)
    asset.write_joint_state_to_sim(bank["joint_pos"][pick], bank["joint_vel"][pick], env_ids=env_ids)


def assert_observation_layout_with_heightmap(env: ManagerBasedEnv, env_ids=None) -> None:
    """The unified layout check, allowing the height-map variant's one trailing term.

    Replaces the shared ``assert_observation_layout`` startup event in the height-map tasks only: the
    first columns must still be exactly the unified layout (that is what lets a blind checkpoint be
    widened at the end), followed by ``wall_heightmap``."""
    import math

    from unitree_rl_lab.tasks.multitask.obs_spec import CRITIC_UNIFIED, POLICY_UNIFIED, assert_layout_matches

    from .observations import HEIGHTMAP_DIM

    manager = env.observation_manager
    for group_name, layout in (("policy", POLICY_UNIFIED), ("critic", CRITIC_UNIFIED)):
        names = list(manager.active_terms[group_name])
        dims = [math.prod(s) for s in manager.group_obs_term_dim[group_name]]
        if not names or names[-1] != "wall_heightmap" or dims[-1] != HEIGHTMAP_DIM:
            raise RuntimeError(f"{group_name}: expected the unified layout followed by wall_heightmap, got {names}")
        assert_layout_matches(layout, dict(zip(names[:-1], dims[:-1])), group_name)


def assert_observation_layout_with_mid360(env: ManagerBasedEnv, env_ids=None) -> None:
    """Layout check of the MID-360 tasks: the unified layout, then ``mid360_map`` (609) on the actor and
    ``wall_heightmap`` (155) on the critic."""
    import math

    from unitree_rl_lab.tasks.multitask.obs_spec import CRITIC_UNIFIED, POLICY_UNIFIED, assert_layout_matches

    from ..mid360 import MID360_MAP_DIM
    from .observations import HEIGHTMAP_DIM

    manager = env.observation_manager
    for group_name, layout, last, dim in (
        ("policy", POLICY_UNIFIED, "mid360_map", MID360_MAP_DIM),
        ("critic", CRITIC_UNIFIED, "wall_heightmap", HEIGHTMAP_DIM),
    ):
        names = list(manager.active_terms[group_name])
        dims = [math.prod(s) for s in manager.group_obs_term_dim[group_name]]
        if not names or names[-1] != last or dims[-1] != dim:
            raise RuntimeError(f"{group_name}: expected the unified layout followed by {last} ({dim}), got {names} {dims}")
        assert_layout_matches(layout, dict(zip(names[:-1], dims[:-1])), group_name)


def randomize_joint_velocity_limit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    calf_range: tuple[float, float] = (15.7, 35.0),
    hip_thigh_range: tuple[float, float] = (30.1, 45.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Draw PhysX's joint speed limits per episode.

    The MuJoCo check (2026-10-04) traced its 0.15-0.2 m higher jump to this limit: the URDF caps the
    calf at 15.7 rad/s (motor 30 rad/s / knee linkage ~1.94) and the hip/thigh at 30.1, PhysX enforces
    that inside its solver and takes energy out of the push-off below the cap, while MuJoCo has no
    cap. With the cap lifted Isaac's take-off matches MuJoCo's within 5%. Which one the real knee is
    closer to is unknown, so the policy is trained across both."""
    asset: Articulation = env.scene[asset_cfg.name]
    names = asset.joint_names
    k = len(env_ids)
    lim = torch.empty(k, len(names), device=env.device)
    for j, n in enumerate(names):
        lo, hi = calf_range if "calf" in n else hip_thigh_range
        lim[:, j] = torch.empty(k, device=env.device).uniform_(lo, hi)
    asset.write_joint_velocity_limit_to_sim(lim, env_ids=env_ids)


def takeoff_kick(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float = 0.75,
    dvz_range: tuple[float, float] = (0.0, 0.9),
    dvx_range: tuple[float, float] = (-0.4, 0.1),
    dang_range: tuple[float, float] = (-1.0, 1.0),
    max_after_trigger_s: float = 1.0,
    command_name: str = "wallcross",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Once per jump, at the first step with all four feet off the ground after the cue, add a random
    velocity to the base: up (dvz), along the heading (dvx) and a random angular velocity.

    For landing practice in the dynamics MuJoCo shows (2026-10-05): there the same policy takes off
    ~0.9 m/s faster upward (peak vz 2.68 vs 1.78 m/s in Isaac, traced to PhysX's joint speed cap),
    rises ~0.15 m higher, travels forward less and turns, and most failures were landings collapsing
    on the far side. Run every control step as an ``interval`` event; ``env_ids`` is ignored."""
    from .observations import heading_xy

    wall = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    n = env.num_envs
    if not hasattr(env, "_wallcross_kicked"):
        env._wallcross_kicked = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._wallcross_kick_trigger = torch.full((n,), -1.0, device=env.device)
    # a new jump (new trigger time) re-arms the kick
    new = wall.triggered & (wall.trigger_time != env._wallcross_kick_trigger)
    env._wallcross_kicked[new] = False
    env._wallcross_kick_trigger[new] = wall.trigger_time[new]
    sensor = env.scene.sensors[wall.cfg.contact_sensor_name]
    feet_off = (torch.linalg.norm(sensor.data.net_forces_w[:, wall.sensor_foot_ids], dim=-1) < 1.0).all(dim=1)
    fire = wall.triggered & ~env._wallcross_kicked & feet_off & ((wall.elapsed - wall.trigger_time) < max_after_trigger_s)
    if not torch.any(fire):
        return
    ids = fire.nonzero(as_tuple=False).squeeze(-1)
    env._wallcross_kicked[ids] = True
    ids = ids[torch.rand(len(ids), device=env.device) < probability]
    if len(ids) == 0:
        return
    k = len(ids)
    vel = asset.data.root_vel_w[ids].clone()
    f = heading_xy(asset.data.root_quat_w[ids])
    dvx = torch.empty(k, device=env.device).uniform_(*dvx_range)
    vel[:, 0:2] += dvx[:, None] * f
    vel[:, 2] += torch.empty(k, device=env.device).uniform_(*dvz_range)
    vel[:, 3:6] += torch.empty(k, 3, device=env.device).uniform_(*dang_range)
    asset.write_root_velocity_to_sim(vel, env_ids=ids)


def descent_kick(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float = 0.75,
    dvz_range: tuple[float, float] = (-1.2, 0.0),
    dvx_range: tuple[float, float] = (-0.5, 0.1),
    dang_range: tuple[float, float] = (-2.0, 2.0),
    min_rise_m: float = 0.15,
    descend_vz: float = -0.2,
    command_name: str = "wallcross",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Once per jump, when the base first comes down (vz < ``descend_vz``) after rising at least
    ``min_rise_m`` since the cue, add a random velocity: down (dvz <= 0), along the heading, angular.

    The take-off version (``takeoff_kick``, HM4) backfired: the policy learned to push off weaker,
    counting on the boost, and in MuJoCo fell short (rise 0.44 -> 0.40 m, 4/50 and 0/50 against HM3's
    19/50 and 25/50). A kick after the apex makes only the landing harder -- a weaker push-off cannot
    cancel it. Run every control step as an ``interval`` event; ``env_ids`` is ignored."""
    from .observations import heading_xy

    wall = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    n = env.num_envs
    if not hasattr(env, "_wallcross_dkicked"):
        env._wallcross_dkicked = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._wallcross_dkick_trigger = torch.full((n,), -1.0, device=env.device)
    new = wall.triggered & (wall.trigger_time != env._wallcross_dkick_trigger)
    env._wallcross_dkicked[new] = False
    env._wallcross_dkick_trigger[new] = wall.trigger_time[new]
    risen = (wall.z_max - wall.z_trigger) > min_rise_m
    falling = asset.data.root_lin_vel_w[:, 2] < descend_vz
    fire = wall.triggered & ~wall.success & ~env._wallcross_dkicked & risen & falling
    if not torch.any(fire):
        return
    ids = fire.nonzero(as_tuple=False).squeeze(-1)
    env._wallcross_dkicked[ids] = True
    ids = ids[torch.rand(len(ids), device=env.device) < probability]
    if len(ids) == 0:
        return
    k = len(ids)
    vel = asset.data.root_vel_w[ids].clone()
    f = heading_xy(asset.data.root_quat_w[ids])
    vel[:, 0:2] += torch.empty(k, device=env.device).uniform_(*dvx_range)[:, None] * f
    vel[:, 2] += torch.empty(k, device=env.device).uniform_(*dvz_range)
    vel[:, 3:6] += torch.empty(k, 3, device=env.device).uniform_(*dang_range)
    asset.write_root_velocity_to_sim(vel, env_ids=ids)
