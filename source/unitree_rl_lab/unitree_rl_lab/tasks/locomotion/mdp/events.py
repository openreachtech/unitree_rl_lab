"""Domain-randomisation events specific to this project."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Attribute names on UnitreeActuator that define the torque-speed curve, paired with the
# scale that applies to each. See unitree_rl_lab/assets/robots/unitree_actuators.py.
_TORQUE_ATTRS = ("_effort_y1", "_effort_y2")
_SPEED_ATTRS = ("_velocity_x1", "_velocity_x2")
_NOMINAL_CACHE = "_tn_curve_nominal"


def randomize_actuator_torque_speed_curve(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    torque_scale_range: tuple[float, float] = (0.8, 1.2),
    speed_scale_range: tuple[float, float] = (0.8, 2.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scale each environment's actuator torque-speed curve by a random factor.

    2026-09-07. Every other actuator property is already randomised -- PD gains +/-20%,
    body mass +/-10%, friction -- but the torque-speed curve itself has always been a
    single fixed set of numbers (Y1 20.2 / Y2 23.4 for hip and thigh, 39.22 / 45.43 for
    the knee, with X1 = 13.5 rad/s and X2 = 30 rad/s for all of them). The policy could
    therefore learn a take-off that depends on exactly where the torque runs out.

    It did. Measured on model_24600 in mujoco, whose ``ctrlrange`` is flat in speed and
    imposes no derate at all: the knees reach 37-58 rad/s during the push-off, a region
    where Isaac supplies no torque, and the jump leaves the ground carrying ~2 rad/s of
    pitch that nothing in flight can remove (Go2's legs hold 79% of the pitch inertia,
    but swinging them is a one-shot trade, and folding vs extending moves I by only 18%).
    Isaac's own flight rotation for the same policy is 0.48 rad/s RMS. Same policy, two
    different manoeuvres.

    Porting the curve INTO mujoco was tried and rejected on 2026-09-06 -- it made the
    landings worse (2 of 4 attempts ended inverted), and matching the evaluator to the
    trainer defeats the point of having a second simulator. The evaluator stays honest;
    the training distribution is what widens. ``speed_scale_range`` reaching 2.5 puts X2
    at 75 rad/s, i.e. effectively no derate across the observed operating range, so the
    mujoco case sits inside the training distribution rather than outside it.

    Torque and speed are drawn per environment and shared across that environment's
    joints: a machine is uniformly strong or weak, not strong in one knee and weak in the
    other. Nominal values are cached on the actuator on first call, so this is safe to run
    at every reset without compounding.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=asset.device)
    if env_ids.numel() == 0:
        return

    n = env_ids.numel()
    torque_scale = torch.empty(n, 1, device=asset.device).uniform_(*torque_scale_range)
    speed_scale = torch.empty(n, 1, device=asset.device).uniform_(*speed_scale_range)

    for actuator in asset.actuators.values():
        if not all(hasattr(actuator, a) for a in _TORQUE_ATTRS + _SPEED_ATTRS):
            continue  # not a UnitreeActuator -- nothing with a torque-speed curve to scale

        nominal = getattr(actuator, _NOMINAL_CACHE, None)
        if nominal is None:
            nominal = {a: getattr(actuator, a).clone() for a in _TORQUE_ATTRS + _SPEED_ATTRS}
            setattr(actuator, _NOMINAL_CACHE, nominal)

        for attr, scale in ((a, torque_scale) for a in _TORQUE_ATTRS):
            getattr(actuator, attr)[env_ids] = nominal[attr][env_ids] * scale
        for attr, scale in ((a, speed_scale) for a in _SPEED_ATTRS):
            getattr(actuator, attr)[env_ids] = nominal[attr][env_ids] * scale
