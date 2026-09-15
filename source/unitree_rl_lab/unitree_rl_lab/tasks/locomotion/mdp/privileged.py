"""Privileged critic observations, following Lee et al. 2020 (Science Robotics).

That paper's teacher sees a state ``s_t = <o_t, x_t>`` where ``o_t`` is what the real
robot can measure and ``x_t`` is read straight out of the physics engine. Table S4
lists ``x_t`` as 71 numbers, and its own description of them is the design brief for
this module: "information related to foot-ground interactions such as terrain profile,
foot contact states and forces, friction coefficients, and external disturbance
forces".

    Terrain normal at each foot            12
    Height scan around each foot           36   (9 points on a 10 cm circle, per foot)
    Foot contact forces                     4
    Foot contact states                     4
    Thigh contact states                    4
    Shank contact states                    4
    Foot-ground friction coefficients       4
    External force applied to the base      3

The terms here cover the first six rows (64 of the 71). The last two are deliberately
left out, for reasons that are about this project rather than the paper:

* friction needs the value ``randomize_rigid_body_material`` actually wrote, which
  means reading back per-shape physics material buffers and mapping foot bodies to
  shape indices -- doable, but fragile enough to want its own change.
* external force would be four constant zeros here: this env disturbs the robot with
  ``push_by_setting_velocity`` (an instantaneous velocity change) and leaves
  ``apply_external_force_torque`` at ``force_range=(0.0, 0.0)``. There is no sustained
  force to report, so the term would carry no information until that event is enabled.

Why bother, given the critic already gets a body-centered ``height_scan``: that grid
says what the terrain looks like near the robot, not what each foot is standing on.
The paper's per-foot rings and normals are attached to the feet, so they follow the
swing and land where the contact actually happens -- which is what the value function
needs to explain why a step succeeded or failed. Fig. 6 of the paper is exactly this:
the quantities their decoder reconstructs are these, not a body-frame grid.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ring_pattern(cfg: RingPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays dropped straight down from a horizontal circle around the sensor origin.

    The paper's terrain profile is "the elevation of 9 scan points around each foot,
    which are symmetrically placed along a circle with a 10 cm radius". Isaac Lab's
    stock patterns are rectangular grids or LiDAR fans, so the ring is defined here.
    """
    angles = torch.arange(cfg.num_points, device=device) * (2.0 * math.pi / cfg.num_points)
    ray_starts = torch.stack(
        [
            cfg.radius * torch.cos(angles),
            cfg.radius * torch.sin(angles),
            torch.full_like(angles, cfg.height),
        ],
        dim=-1,
    )
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0
    return ray_starts, ray_directions


@configclass
class RingPatternCfg(PatternBaseCfg):
    """Configuration for :func:`ring_pattern`."""

    func: Callable = ring_pattern

    radius: float = 0.10
    """Circle radius (in m). The paper's value."""
    num_points: int = 9
    """Points on the circle. The paper's value."""
    height: float = 20.0
    """Ray start height above the sensor origin (in m), as the height-scan raycasters use."""


def foot_height_scan(env: ManagerBasedRLEnv, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """Terrain elevation at the ring of scan points around each foot, relative to that foot.

    One sensor per foot, concatenated in the order given. Positive means the terrain at
    that scan point sits below the foot. ``sensor_cfgs`` fixes the foot ordering, so keep
    it consistent with the other per-foot terms in the same observation group.
    """
    out = []
    for cfg in sensor_cfgs:
        sensor: RayCaster = env.scene.sensors[cfg.name]
        # The ring rays start 20 m up, so pos_w is the foot pose, not the ray origin.
        heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
        out.append(torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.cat(out, dim=-1)


def foot_terrain_normal(env: ManagerBasedRLEnv, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """Unit normal of the best-fit plane through each foot's ring of scan points.

    This is the *terrain slope* under the foot, fitted from nine downward rays -- not the
    contact normal the physics engine reports. The two differ where it matters most: the
    paper uses its normal to identify a frontal collision with a step riser, which points
    roughly horizontally, whereas a fit over downward samples can tilt steeply but never
    past horizontal. A step edge crossing the ring still registers strongly (25 cm over
    the 10 cm radius fits a gradient of ~2.5, tilting the normal about 70 degrees off
    vertical), so the signal is there; its saturation is the deviation to remember.

    Because the ring is symmetric, the least-squares plane through it has a closed form:
    ``sum(x*z)/sum(x^2)`` and ``sum(y*z)/sum(y^2)`` are exactly the fitted gradients
    (the cross terms ``sum(x)``, ``sum(y)`` and ``sum(x*y)`` all vanish), so the normal
    follows without an eigendecomposition.
    """
    out = []
    for cfg in sensor_cfgs:
        sensor: RayCaster = env.scene.sensors[cfg.name]
        z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        # Ring offsets are fixed in the sensor frame; recover them from the ray starts.
        xy = sensor.ray_starts[0, :, :2]  # (num_points, 2)
        x, y = xy[:, 0], xy[:, 1]
        grad_x = (z * x).sum(dim=-1) / x.square().sum().clamp(min=1e-9)
        grad_y = (z * y).sum(dim=-1) / y.square().sum().clamp(min=1e-9)
        normal = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=-1)
        out.append(normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-9))
    return torch.cat(out, dim=-1)


def contact_states(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Binary contact flag per body in ``sensor_cfg`` (1.0 = touching something).

    Thresholded on force magnitude rather than read from ``current_contact_time`` so that
    a body resting with negligible load reads as free, matching what "contact state" is
    meant to tell the critic.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.norm(forces, dim=-1) > threshold).float()


def contact_force_magnitude(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact force magnitude per body in ``sensor_cfg`` (in N)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
