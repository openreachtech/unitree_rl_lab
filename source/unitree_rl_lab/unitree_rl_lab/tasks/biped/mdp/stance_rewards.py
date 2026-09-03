"""Reward terms for the two-legged stance.

Two sources, deliberately kept apart in what follows:

*The balance terms* -- ``pendulum_angle``, ``pendulum_instability``, ``handle_length`` -- come from
TumblerNet (Xiao et al. 2025) by way of ``feat/biped``, which trained both a hind-leg and a
front-leg stance on them and took them to hardware. They are reproduced here unchanged, including
their weights, because they are the part of that work that is already validated.

*The task terms* -- ``stance_pitch``, ``upright_balance``, ``support_polygon`` -- follow "Bipedalism
for Quadrupedal Robots" (Zhang et al.), whose Table I is the reward set this stage was asked to
take as its reference. Its ablation is the reason ``support_polygon`` is here at all: removing it
cost more linear-velocity tracking than removing anything else they tried.

Everything in this module describes a robot standing on two legs, and none of it should apply to a
robot doing anything else. That is what the gate is for: the config wraps each of these in
``gated`` with ``GATE_HANDSTAND`` or ``GATE_HANDSTAND_UPRIGHT`` (``..multitask.mdp.gating``), so the
whole set switches off with the command and can be carried into the merged policy without touching
the locomotion or acrobatics rewards.
"""

from __future__ import annotations

import math
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # depends on the installed isaaclab version
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =================================================================================================
# CoM-CoP geometry
# =================================================================================================


def _body_masses(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """Per-body mass, cached. Fixed after startup randomisation, so querying physx per call would
    be pure overhead in a quantity several rewards and one observation all evaluate every step."""
    if not hasattr(env, "_biped_body_masses"):
        env._biped_body_masses = asset.root_physx_view.get_masses().to(env.device)
    return env._biped_body_masses


def com_cop_vector_world(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """World-frame vector from the centre of pressure to the centre of mass.

    CoP is the vertical-force-weighted mean of the candidate feet; CoM is the mass-weighted mean of
    every rigid body. Pass *all four* feet: the weighting collapses the CoP onto whichever feet are
    actually loaded, so one call describes a quadruped gait and either bipedal stance without being
    told which is in progress.

    World frame, not body frame, and the distinction is the whole point of the balance rewards
    below. The pendulum angle is defined against gravity; once the trunk is pitched 70-90 degrees
    the body's own z-axis is nearly horizontal, so measuring against it would score alignment with
    the spine rather than balance against gravity -- backwards, for a stability term. Root
    translation cancels in the difference, so no frame change is needed at all.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    masses = _body_masses(env, asset)
    com_w = torch.sum(asset.data.body_pos_w * masses.unsqueeze(-1), dim=1) / masses.sum(dim=1, keepdim=True)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # The epsilon keeps the CoP defined through a flight phase, where every candidate foot is off
    # the ground and the weights would otherwise all be zero.
    force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0) + 1.0e-6
    cop_w = torch.sum(foot_pos_w * force_z.unsqueeze(-1), dim=1) / force_z.sum(dim=1, keepdim=True)
    return com_w - cop_w


def com_cop_vector(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """:func:`com_cop_vector_world`, rotated into the base frame. The critic observation.

    Body frame here rather than world because the observation has to be independent of the robot's
    heading, which the world-frame vector is not.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return quat_apply_inverse(asset.data.root_quat_w, com_cop_vector_world(env, asset_cfg, sensor_cfg))


def pendulum_angle_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Squared tilt of the CoM-CoP vector away from vertical (TumblerNet Eq. 10).

    Treats the stance as an inverted pendulum: a large angle means the mass is falling away from
    the point holding it up.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    norm = torch.linalg.norm(c, dim=-1).clamp(min=1.0e-6)
    return torch.square(torch.acos((c[:, 2] / norm).clamp(-1.0, 1.0)))


def pendulum_instability_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``sin^2(theta) / ||c||^2`` (TumblerNet Eq. 11): a proxy for how fast the tilt is diverging.

    Grows both as the pendulum tilts and as it shortens -- a short tilted pendulum is nearer to
    tipping than a long one at the same angle.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    norm_sq = torch.sum(torch.square(c), dim=-1).clamp(min=1.0e-4)
    cos_theta = (c[:, 2] / norm_sq.sqrt()).clamp(-1.0, 1.0)
    return (1.0 - torch.square(cos_theta)) / norm_sq


def handle_length_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Horizontal CoM-CoP offset (TumblerNet Eq. 12): the lever arm gravity has to tip the robot."""
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    return torch.linalg.norm(c[:, :2], dim=-1)


# =================================================================================================
# Posture
# =================================================================================================


def stance_pitch_reward(env: ManagerBasedRLEnv, command_name: str = "handstand") -> torch.Tensor:
    """How far the trunk has pitched into the commanded stance, in ``[-1, 1]``.

    The paper's *Base Pitch* term with the target at +-90 degrees, read straight off projected
    gravity so there is no angle to extract and no wrap-around to get wrong. The command supplies
    the sign, which is what lets one term serve both ends of the robot.

    Deliberately not ``flat_orientation_l2`` with a positive weight, which is how ``feat/biped``
    expressed the same intent: that term is ``g_x^2 + g_y^2``, so it pays exactly as well for
    toppling sideways as for rising, and the hardware trace shows the robot doing precisely that --
    43 degrees of sagittal lean carrying 33 degrees of lateral, the lateral rise arriving first.
    Separating the two axes is what :func:`stance_roll_penalty` is for.
    """
    return env.command_manager.get_term(command_name).pitch_alignment


def stance_roll_penalty(env: ManagerBasedRLEnv, command_name: str = "handstand") -> torch.Tensor:
    """Squared lateral lean. See :func:`stance_pitch_reward` for why it is a separate term."""
    return torch.square(env.command_manager.get_term(command_name).roll_error)


def upright_balance_reward(
    env: ManagerBasedRLEnv,
    linear_std: float = 0.5,
    pitch_rate_std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-v_z^2/sigma) + exp(-pitch_rate^2/sigma)`` -- the paper's *Upright Balance*.

    Rewards holding the stance still in the two directions a two-legged robot loses it: bobbing
    vertically and rotating about the pitch axis it is balancing on. Gate this on the robot
    actually being upright (the paper's "if is upright, else 0"), or it pays full marks to a
    quadruped standing quietly on four legs -- which satisfies both exponentials perfectly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vertical = torch.square(asset.data.root_lin_vel_w[:, 2])
    pitch_rate = torch.square(asset.data.root_ang_vel_b[:, 1])
    return torch.exp(-vertical / linear_std**2) + torch.exp(-pitch_rate / pitch_rate_std**2)


def support_polygon_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    handstand_command_name: str = "handstand",
) -> torch.Tensor:
    """Penalise leaning the wrong way for the commanded direction (the paper's *Support Polygon*).

    A two-legged robot changes speed by moving its mass relative to its feet: ahead of the support
    point it accelerates, behind it it decelerates. The paper turns that into a reward by comparing
    the sign of the lean angle against the sign of the commanded forward speed and charging only
    when they disagree, weighted by ``|v_x^c|^2`` -- so the term is silent at a standstill, where
    there is no direction to agree with -- and by ``(pi/2 - |theta|)^2``, which is zero when the
    pendulum is exactly upright and grows as the lean does. Their ablation removed this term and
    lost more linear-velocity tracking than any other single change they tried.

    The angle is measured here from the world-frame CoM-CoP vector, as its tilt from vertical in
    the plane of travel, rather than from the printed ``arctan(dx_b/dz_b)`` ratio of stance-foot
    positions in the body frame. Same quantity, different route: at the 70-90 degrees of trunk
    pitch this stance runs at, the body frame no longer separates "how far the mass has leaned"
    from "where the feet sit under the hips", and the world-frame vector -- already computed for
    the three balance terms above -- states it directly.
    """
    c = com_cop_vector_world(env, asset_cfg, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    # Heading, not body x: the body's own x-axis points nearly straight down in this stance, so the
    # direction the velocity command is expressed in has to come from the yaw alone.
    c_heading = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), c)
    theta = torch.atan2(c_heading[:, 0], c_heading[:, 2].clamp(min=1.0e-6))

    stance = env.command_manager.get_term(handstand_command_name).stance
    forward_command = env.command_manager.get_command(command_name)[:, 0]
    # The front stance faces the other way, so "mass ahead of the feet" carries the opposite sign
    # of theta. Folding the stance in here keeps one term correct for both ends.
    leaning = stance * theta

    disagrees = leaning * forward_command < 0.0
    penalty = torch.square(forward_command) * torch.square(math.pi / 2.0 - leaning.abs())
    return torch.where(disagrees, penalty, torch.zeros_like(penalty))


def front_body_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    standstill_boost: float = 1.0,
) -> torch.Tensor:
    """Keep named body points (the front hips) at ``target_height`` in world z.

    ``base_height_l2`` constrains only the root link, which leaves the front of the body free to
    droop until it scrapes: the root sits exactly on target while the nose is on the floor, and
    nothing in the reward reports a problem. A symmetric penalty rather than a one-sided floor, so
    the policy cannot park right on the danger threshold for free.

    ``standstill_boost`` multiplies the penalty while the commanded speed is below
    ``command_threshold``. Around 90% of training happens at a non-zero command, so a
    standstill-only failure is diluted into a population average that looks healthy -- which is
    exactly what happened in ``feat/biped``: the average stayed flat across 7000 resumed iterations
    while the ground contact at a standstill was still plainly visible in play mode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    err = torch.sum(torch.square(heights - target_height), dim=-1)
    if command_name is not None:
        speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        err = torch.where(speed < command_threshold, err * standstill_boost, err)
    return err


def lifted_foot_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Count of lifted-end feet touching the ground.

    A bounded 0..2 count, not a force magnitude, and that is not a detail. ``feat/biped`` first
    wrote this as raw Newtons, matching its reference implementation -- which also clips each
    step's total reward at zero, something Isaac Lab's ``RewardManager`` has no equivalent of.
    Without that clamp a raw-force penalty made ending the episode immediately cheaper than
    enduring it, and every episode collapsed within 5-8 steps for 2000 iterations straight. The
    same failure returned later at a merely *large* bounded weight (-0.3 was safe, -0.6 was not),
    which is why ``termination_penalty`` belongs in any config using this term.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(forces, dim=-1) > threshold).float().sum(dim=-1)


def handstand_success(env: ManagerBasedRLEnv, command_name: str = "handstand") -> torch.Tensor:
    """1 while the robot is in the commanded stance with the lifted end clear of the ground.

    The completion term. Everything else in this module shapes the way there; this is what says the
    robot has arrived, and it is what a later curriculum would read to decide the task is solved.
    """
    return env.command_manager.get_term(command_name).success.float()
