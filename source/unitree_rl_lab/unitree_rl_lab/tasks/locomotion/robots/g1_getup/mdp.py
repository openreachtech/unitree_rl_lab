"""Get-up-specific MDP terms for the G1 (23dof), ported from HoST.

Reference: OpenRobotLab/HoST (RSS 2025), ``legged_gym/envs/base/host_ground.py`` and
``legged_gym/envs/g1/g1_config_ground.py``. Function names and constants below follow
HoST's ``_reward_<name>`` / cfg fields so the two can be compared side by side.

What lives here and why it can't come from ``tasks.locomotion.mdp``:

* ``DeltaJointPositionAction``: HoST drives the joints relative to the *current* joint
  position (``target = q + s * a``), with a per-env scale ``s`` that a curriculum lowers
  from 1.0 to 0.25 (the implicit speed bound). The same term also holds the robot limp
  for the first ``unactuated_steps`` so it settles into a natural lying pose, adds a
  0-4 physics-step actuation delay, and applies HoST's vertical pulling force to the
  torso. These all share the same per-env curriculum state, so they live together.
* ``update_getup_curriculum``: HoST's success-based curriculum (lower force and action
  scale for envs whose head ended the episode above a threshold).
* rewards: HoST's task / regu / style / target terms. The task term is the *product*
  of orientation and head height, so it is a single function here.
"""

from __future__ import annotations

import math
import os
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

# HoST attaches a fixed ``keyframe_head`` link 0.45 m above ``torso_link``. The unitree USD
# may or may not keep a head body, so compute the same point from the torso pose instead.
HEAD_OFFSET_IN_TORSO = (0.0, 0.0, 0.45)
# HoST's auxiliary foot points (fixed links on ``*_ankle_roll_link``) used to measure how
# flat each sole is. The ankle_roll origin itself is the 5th point.
FOOT_SOLE_POINTS = (
    (0.0, 0.0, 0.0),
    (0.035, 0.1, 0.0),
    (0.035, -0.1, 0.0),
    (0.2, 0.0, 0.0),
    (-0.15, 0.0, 0.0),
)

ACTION_NAME = "JointPositionDeltaAction"


##
# Helpers
##


def tolerance(x: torch.Tensor, lower: float, upper: float, margin: float, value_at_margin: float) -> torch.Tensor:
    """HoST ``g1_utils.tolerance`` (dm_control style, gaussian sigmoid): 1 inside
    ``[lower, upper]``, decaying to ``value_at_margin`` at ``margin`` outside."""
    in_bounds = (x >= lower) & (x <= upper)
    if margin == 0:
        return in_bounds.float()
    d = torch.where(x < lower, lower - x, x - upper) / margin
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    return torch.where(in_bounds, torch.ones_like(x), torch.exp(-0.5 * (d * scale) ** 2))


def _delta_action_term(env: ManagerBasedEnv) -> DeltaJointPositionAction:
    return env.action_manager.get_term(ACTION_NAME)


def _base_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _head_height(env: ManagerBasedEnv, torso_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """Head height above the mean foot height (HoST ``head_height - feet_height``)."""
    asset: Articulation = env.scene[torso_cfg.name]
    torso_pos = asset.data.body_pos_w[:, torso_cfg.body_ids[0]]
    torso_quat = asset.data.body_quat_w[:, torso_cfg.body_ids[0]]
    offset = torch.tensor(HEAD_OFFSET_IN_TORSO, device=env.device).expand(env.num_envs, 3)
    head_z = (torso_pos + math_utils.quat_apply(torso_quat, offset))[:, 2]
    feet_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2].mean(dim=1)
    return head_z - feet_z


##
# Action
##


class DeltaJointPositionAction(JointPositionAction):
    """``target = q_now + s * a`` with HoST's actuation extras (see module docstring).

    With ``recompute_every_substep`` (HoST) the target is rebuilt from the current joint
    position at every physics step, so the PD torque is ~``kp * s * a - kd * qd`` for the
    whole policy step. Otherwise it is computed once per policy step, as a motor PD tracking
    a 50 Hz target would see it.
    """

    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        # curriculum state (per env), lowered by ``update_getup_curriculum``
        self.action_scale = torch.full((n, 1), cfg.scale_init, device=self.device)
        self.assist_frac = torch.full((n,), cfg.assist_frac_init, device=self.device)
        # action history for smoothness terms (HoST last_actions / last_last_actions)
        self.prev_actions = torch.zeros_like(self._raw_actions)
        self.prev_prev_actions = torch.zeros_like(self._raw_actions)
        # actuation delay, in physics steps, resampled per env on reset
        self._delay = torch.zeros(n, 1, dtype=torch.long, device=self.device)
        self._prev_target = torch.zeros_like(self._raw_actions)
        self._substep = 0
        # pulling force
        body_ids, _ = self._asset.find_bodies(cfg.assist_body_name)
        self._assist_body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
        self._weight = self._asset.data.default_mass.to(self.device).sum(dim=1) * 9.81

    def _actuated(self) -> torch.Tensor:
        return self._env.episode_length_buf >= self.cfg.unactuated_steps

    def process_actions(self, actions: torch.Tensor):
        self.prev_prev_actions[:] = self.prev_actions
        self.prev_actions[:] = self._raw_actions
        # HoST: clip_actions = 100 (the Tanh actor keeps the mean in [-1, 1] anyway)
        actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        # HoST zeroes the action while unactuated, so the policy sees/gets penalized for zeros too
        self._raw_actions[:] = actions * self._actuated().unsqueeze(1)
        self._prev_target[:] = self._processed_actions
        q = self._asset.data.joint_pos[:, self._joint_ids]
        self._processed_actions = q + self.action_scale * self._raw_actions
        self._substep = 0

    def apply_actions(self):
        actuated = self._actuated()
        q = self._asset.data.joint_pos[:, self._joint_ids]
        delayed = self._substep < self._delay
        if self.cfg.recompute_every_substep:
            # delay: the previous policy step's action for the first ``delay`` physics steps
            action = torch.where(delayed, self.prev_actions, self._raw_actions)
            target = q + self.action_scale * action
        else:
            target = torch.where(delayed, self._prev_target, self._processed_actions)
        # unactuated: track the current position every physics step, i.e. go limp
        target = torch.where(actuated.unsqueeze(1), target, q)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._substep += 1

        if self.cfg.assist_frac_init > 0.0:
            on = actuated
            if self.cfg.assist_only_upright:
                on = on & (self._asset.data.projected_gravity_b[:, 2] < self.cfg.assist_upright_gravity_z)
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            forces[:, 0, 2] = self.assist_frac * self._weight * on
            # instantaneous: cleared after every physics step, and the world->body conversion
            # uses the current torso pose, so "up" stays world-up while the robot rotates
            self._asset.instantaneous_wrench_composer.add_forces_and_torques(
                forces=forces, torques=torch.zeros_like(forces), body_ids=self._assist_body_ids, is_global=True
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        n = self._delay[env_ids].shape[0]
        self._delay[env_ids] = torch.randint(0, self.cfg.max_delay_substeps + 1, (n, 1), device=self.device)


@configclass
class DeltaJointPositionActionCfg(JointPositionActionCfg):
    """Config for :class:`DeltaJointPositionAction`.

    ``scale`` is the deploy-time action scale (exported to deploy.yaml); training starts at
    ``scale_init`` and the curriculum lowers it towards ``scale``.
    """

    class_type: type = DeltaJointPositionAction

    scale: float = 0.25
    scale_init: float = 1.0
    action_clip: float = 100.0
    recompute_every_substep: bool = True
    unactuated_steps: int = 30
    max_delay_substeps: int = 4
    assist_body_name: str = "torso_link"
    # HoST: 2 x 100 N on the torso, ~60% of the G1 weight
    assist_frac_init: float = 0.6
    assist_only_upright: bool = True
    assist_upright_gravity_z: float = -0.8


##
# Curriculum
##


def update_getup_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    head_height_threshold: float = 0.9,
    assist_frac_step: float = 0.06,
    scale_step: float = 0.02,
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> dict[str, torch.Tensor]:
    """HoST ``update_force_curriculum``: if the resetting envs ended with their head above
    the threshold on average, lower their pulling force and action scale.

    Runs before the scene reset, so the state is the one the episode ended in.
    """
    term = _delta_action_term(env)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    # the very first reset happens before any episode ran
    env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(env_ids) > 0:
        head = _head_height(env, torso_cfg, feet_cfg)[env_ids]
        if head.mean() > head_height_threshold:
            term.assist_frac[env_ids] = (term.assist_frac[env_ids] - assist_frac_step).clamp(min=0.0)
            term.action_scale[env_ids] = (term.action_scale[env_ids] - scale_step).clamp(min=term.cfg.scale)
    return {"assist_frac": term.assist_frac.mean(), "action_scale": term.action_scale.mean()}


##
# Events
##


def reset_lying_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos_scale: tuple[float, float] = (0.9, 1.1),
    joint_pos_offset: tuple[float, float] = (-0.1, 0.1),
):
    """HoST ``_reset_dofs`` / ``_reset_root_states``: the root goes back to the configured
    lying pose (``init_state``), joints to ``default * U(scale) + U(offset)``. The robot then
    falls and settles during the unactuated steps.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    default = asset.data.default_joint_pos[env_ids]
    joint_pos = default * math_utils.sample_uniform(*joint_pos_scale, default.shape, env.device)
    joint_pos += math_utils.sample_uniform(*joint_pos_offset, default.shape, env.device)
    limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


##
# Observations
##


def action_scale_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Current per-env action scale (HoST feeds it to the policy)."""
    return _delta_action_term(env).action_scale


##
# Terminations
##


def joint_vel_out_of_bound(
    env: ManagerBasedRLEnv, limit: float = 300.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out = asset.data.joint_vel.abs().max(dim=1).values > limit
    return out & _delta_action_term(env)._actuated()


def base_vel_out_of_bound(
    env: ManagerBasedRLEnv, limit: float = 20.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.norm(asset.data.root_lin_vel_b, dim=1) > limit
    return out & _delta_action_term(env)._actuated()


##
# Rewards -- task (HoST multiplies these together)
##


def task_orientation_head_height(
    env: ManagerBasedRLEnv,
    orientation_threshold: float = 0.99,
    target_head_height: float = 1.0,
    target_head_margin: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """HoST task group: ``orientation * head_height``."""
    asset: Articulation = env.scene[asset_cfg.name]
    orientation = tolerance(-asset.data.projected_gravity_b[:, 2], orientation_threshold, math.inf, 1.0, 0.05)
    head = _head_height(env, torso_cfg, feet_cfg)
    head_height = tolerance(head, target_head_height, math.inf, target_head_margin, 0.1)
    return orientation * head_height


##
# Rewards -- regularization
##


def delta_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = _delta_action_term(env)
    return torch.sum(torch.square(term.raw_actions - term.prev_actions), dim=1)


def delta_action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action difference (HoST ``_reward_smoothness``)."""
    term = _delta_action_term(env)
    return torch.sum(torch.square(term.raw_actions - 2 * term.prev_actions + term.prev_prev_actions), dim=1)


def joint_tracking_error_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = _delta_action_term(env)
    q = term._asset.data.joint_pos[:, term._joint_ids]
    return torch.sum(torch.square(term.processed_actions - q), dim=1)


##
# Rewards -- style
##


def joint_abs_exceeds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_limit: float, min_limit: float | None = None
) -> torch.Tensor:
    """HoST ``*_deviation``: 1 if any |q| > ``max_limit``, or if every |q| > ``min_limit``."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids].abs()
    out = q.max(dim=1).values > max_limit
    if min_limit is not None:
        out = out | (q.min(dim=1).values > min_limit)
    return out.float()


def shoulder_roll_inward(
    env: ManagerBasedRLEnv, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg, margin: float = 0.02
) -> torch.Tensor:
    """HoST ``shoulder_roll_deviation``: arms crossing in front of the body."""
    asset: Articulation = env.scene[left_cfg.name]
    left = asset.data.joint_pos[:, left_cfg.joint_ids[0]]
    right = asset.data.joint_pos[:, right_cfg.joint_ids[0]]
    return ((left < -margin) | (right > margin)).float()


def knee_out_of_range(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_limit: float = 2.85, min_limit: float = -0.06
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return ((q.abs().max(dim=1).values > max_limit) | (q.min(dim=1).values < min_limit)).float()


def foot_displacement(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    sigma: float = -2.0,
    standup_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST ``left/right_foot_displacement``: keep the foot under the pelvis once standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot = asset.data.body_pos_w[:, foot_cfg.body_ids[0]]
    err = torch.sum(torch.square(asset.data.root_pos_w[:, :2] - foot[:, :2]), dim=1).clamp(min=0.3)
    on_ground = (foot[:, 2] - env.scene.env_origins[:, 2]) < 0.3
    standup = _base_height(env, asset_cfg) > standup_height
    return torch.exp(err * sigma) * on_ground * standup


def shank_orientation(
    env: ManagerBasedRLEnv,
    knee_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    phase1_height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST ``shank_orientation``: shanks vertical once the pelvis is off the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    shank = asset.data.body_pos_w[:, knee_cfg.body_ids] - asset.data.body_pos_w[:, foot_cfg.body_ids]
    vertical = (shank[..., 2] / torch.norm(shank, dim=-1)).mean(dim=1)
    return tolerance(vertical, 0.8, math.inf, 1.0, 0.1) * (_base_height(env, asset_cfg) > phase1_height)


def feet_ground_parallel(env: ManagerBasedRLEnv, feet_cfg: SceneEntityCfg, threshold: float = 0.05) -> torch.Tensor:
    """HoST ``ground_parallel``: soles flat (low height variance over the sole points)."""
    asset: Articulation = env.scene[feet_cfg.name]
    pos = asset.data.body_pos_w[:, feet_cfg.body_ids]  # (n, 2, 3)
    quat = asset.data.body_quat_w[:, feet_cfg.body_ids]  # (n, 2, 4)
    pts = torch.tensor(FOOT_SOLE_POINTS, device=env.device)  # (5, 3)
    n, f = pos.shape[:2]
    world = pos.unsqueeze(2) + math_utils.quat_apply(
        quat.unsqueeze(2).expand(n, f, len(pts), 4), pts.expand(n, f, len(pts), 3)
    )
    var = (world[..., 2] * 10).var(dim=2).mean(dim=1)
    return (var < threshold).float()


def feet_too_far(env: ManagerBasedRLEnv, feet_cfg: SceneEntityCfg, max_distance: float = 0.9) -> torch.Tensor:
    asset: Articulation = env.scene[feet_cfg.name]
    feet = asset.data.body_pos_w[:, feet_cfg.body_ids]
    return (torch.norm(feet[:, 0] - feet[:, 1], dim=-1) > max_distance).float()


def ang_vel_xy_exp_gated(
    env: ManagerBasedRLEnv, min_base_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``exp(-2 |w_xy|^2)`` once the pelvis is above ``min_base_height`` (HoST
    ``style_ang_vel_xy`` with phase1, ``target_ang_vel_xy`` with phase3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    r = torch.exp(-2.0 * torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1))
    return r * (_base_height(env, asset_cfg) > min_base_height)


##
# Rewards -- target (post-task, only once standing)
##


def lin_vel_xy_exp_gated(
    env: ManagerBasedRLEnv, min_base_height: float = 0.65, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    r = torch.exp(-5.0 * torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1))
    return r * (_base_height(env, asset_cfg) > min_base_height)


def feet_height_diff_exp(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    min_base_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST ``feet_height_var``: both feet at the same height."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2] * 10
    diff = (z[:, 0] - z[:, 1]).abs().clamp(min=0.2)
    return torch.exp(-2.0 * diff) * (_base_height(env, asset_cfg) > min_base_height)


def upper_joint_pos_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: dict[str, float],
    sigma: float = -0.1,
    min_base_height: float = 0.65,
) -> torch.Tensor:
    """HoST ``target_upper_dof_pos``: arms/waist back to a neutral pose once standing.
    ``target`` maps joint-name regex -> angle; unmatched joints target 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    tgt = torch.zeros(len(names), device=env.device)
    for i, name in enumerate(names):
        for pattern, value in target.items():
            if re.fullmatch(pattern, name):
                tgt[i] = value
    mse = torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - tgt), dim=1)
    return torch.exp(mse * sigma) * (_base_height(env, SceneEntityCfg(asset_cfg.name)) > min_base_height)


def flat_orientation_exp_gated(
    env: ManagerBasedRLEnv, min_base_height: float = 0.65, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    r = torch.exp(-5.0 * torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1))
    return r * (_base_height(env, asset_cfg) > min_base_height)


def base_height_exp_gated(
    env: ManagerBasedRLEnv,
    target_height: float = 0.75,
    min_base_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    h = _base_height(env, asset_cfg)
    return torch.exp(-20.0 * (h - target_height).abs()) * (h > min_base_height)


##
# Rewards -- standing still (not in HoST)
#
# With a single critic the task term dominates everything else, and HoST's target terms
# (exp kernels on base velocity) give almost no gradient near zero, so the policy keeps
# shuffling its feet and swinging its arms once up. These pin the whole body to the
# nominal pose once the pelvis is above ``min_base_height``.
##


def stand_pose_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    min_base_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-sum((q - q_default)^2) / std^2)`` over ``asset_cfg``'s joints, once standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    err = torch.sum(torch.square(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    return torch.exp(-err / std**2) * (_base_height(env, asset_cfg) > min_base_height)


def stand_joint_vel_l2(
    env: ManagerBasedRLEnv, min_base_height: float = 0.65, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1) * (_base_height(env, asset_cfg) > min_base_height)


def stand_feet_vel_l2(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    min_base_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Horizontal foot speed once standing (feet should stay planted)."""
    asset: Articulation = env.scene[feet_cfg.name]
    vel = asset.data.body_lin_vel_w[:, feet_cfg.body_ids, :2]
    return torch.sum(torch.square(vel), dim=(1, 2)) * (_base_height(env, asset_cfg) > min_base_height)


##
# Rewards -- speed limits (not in HoST)
#
# The task term pays for every step spent upright, so getting up faster always pays more than
# HoST's small regu penalties cost; without HoST's per-group critics and smoothness loss the
# policy learned a ~0.25 s kip-up. These make speeds above a limit expensive instead.
##


def joint_vel_over_limit_l2(
    env: ManagerBasedRLEnv, max_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    over = (asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - max_vel).clamp(min=0.0)
    # not while unactuated: the passive fall from the spawn height is not the policy's doing
    return torch.sum(torch.square(over), dim=1) * _delta_action_term(env)._actuated()


def base_lin_vel_over_limit_l2(
    env: ManagerBasedRLEnv, max_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    over = (torch.norm(asset.data.root_lin_vel_w, dim=1) - max_vel).clamp(min=0.0)
    return torch.square(over) * _delta_action_term(env)._actuated()


##
# Events -- reset from a recorded state bank (stand-up from crouch)
##

CROUCH_BANK_PATH = os.path.join(os.path.dirname(__file__), "data", "crouch_states.pt")
_BANK_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _load_bank(path: str, device: str) -> dict[str, torch.Tensor]:
    key = f"{path}:{device}"
    if key not in _BANK_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No state bank at '{path}'. Generate it with scripts/make_g1_crouch_bank.py.")
        bank = torch.load(path, map_location=device)
        _BANK_CACHE[key] = {k: v.to(device) for k, v in bank.items()}
    return _BANK_CACHE[key]


def reset_from_state_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    bank_path: str = CROUCH_BANK_PATH,
    joint_noise: float = 0.02,
):
    """Reset to a settled state sampled from a bank (root pose relative to the env origin,
    joint positions), with a small joint offset so the policy does not see exact copies."""
    asset: Articulation = env.scene[asset_cfg.name]
    bank = _load_bank(bank_path, env.device)
    idx = torch.randint(0, bank["root_pose"].shape[0], (len(env_ids),), device=env.device)

    root_pose = bank["root_pose"][idx].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(root_pose, env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)

    joint_pos = bank["joint_pos"][idx].clone()
    joint_pos += math_utils.sample_uniform(-joint_noise, joint_noise, joint_pos.shape, env.device)
    limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)



##
# Observations -- HoST zeroes the whole observation while unactuated
##


def unactuated_masked(env: ManagerBasedEnv, term_func, term_params: dict | None = None) -> torch.Tensor:
    """``term_func(env, **term_params)``, zeroed while the robot is held limp (HoST
    ``current_obs *= real_episode_length_buf > unactuated_time``)."""
    return term_func(env, **(term_params or {})) * _delta_action_term(env)._actuated().unsqueeze(1)


##
# Rewards -- leg alignment (not in HoST)
#
# HoST only penalizes hip yaw / roll past 1.4 rad (or both past 0.9 rad) and feet further
# than 0.9 m apart, so a policy can rise and stand bow-legged, with the hips twisted out
# and the feet wide apart. These keep the feet hip-width apart once the pelvis is up.
##


def feet_lateral_distance_tol(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    lower: float = 0.18,
    upper: float = 0.30,
    margin: float = 0.15,
    min_base_height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``tol`` on the left-right foot distance in the pelvis heading frame, once the
    pelvis is above ``min_base_height``."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, feet_cfg.body_ids]  # (n, 2, 3), left then right
    heading = math_utils.yaw_quat(asset.data.root_quat_w)
    delta = math_utils.quat_apply_inverse(heading, feet[:, 0] - feet[:, 1])
    r = tolerance(delta[:, 1].abs(), lower, upper, margin, 0.1)
    return r * (_base_height(env, asset_cfg) > min_base_height)


##
# Events -- joint range restriction (not in HoST)
##


def restrict_joint_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    limits: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Narrow joint position limits at startup (``limits``: joint-name regex -> (lower, upper)).

    Each new range is intersected with the robot's own one. Used to keep the hips out of the
    G1's far roll range (up to ~170 deg): HoST-style training found a standing pose with
    both hip rolls at their limits and the twist undone by the other joints, and once there
    it never came back.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_pos_limits[0].clone()
    for i, name in enumerate(asset.joint_names):
        for pattern, (lower, upper) in limits.items():
            if re.fullmatch(pattern, name):
                current[i, 0] = max(lower, current[i, 0].item())
                current[i, 1] = min(upper, current[i, 1].item())
    new = current.unsqueeze(0).repeat(env.num_envs, 1, 1)
    asset.write_joint_position_limit_to_sim(new, warn_limit_violation=False)


def kneeling_upright(
    env: ManagerBasedRLEnv,
    knee_cfg: SceneEntityCfg,
    max_knee_height: float = 0.15,
    upright_gravity_z: float = -0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1 while the torso is upright but a knee is near the floor (kneeling / sitting on the
    heels). Not in HoST: a symmetric run settled into kneeling upright, which already meets
    the head-height target."""
    asset: Articulation = env.scene[asset_cfg.name]
    knee_z = asset.data.body_pos_w[:, knee_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    upright = asset.data.projected_gravity_b[:, 2] < upright_gravity_z
    return ((knee_z.min(dim=1).values < max_knee_height) & upright).float()
