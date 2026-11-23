"""Privileged (critic-only) observations, ported from humanoid-gym.

humanoid-gym gives the critic the push perturbation, friction, mass and foot
contact state. None of these are available on the real robot, so they go to
the critic group only; the actor's input (and thus deploy) is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "push_by_force",
    "push_force",
    "body_static_friction",
    "body_added_mass",
    "feet_contact",
]


# ---------------------------------------------------------------------------
# push: a sustained horizontal force on the torso, and the matching observation
# ---------------------------------------------------------------------------


class push_by_force(ManagerTermBase):
    """Random horizontal pushes applied as a force held for a duration (van Marum et al.).

    Each env waits ``U(push_interval_s)`` seconds, then pushes ``body_name`` with a
    world-frame horizontal force of magnitude ``U(force_range)`` N in a uniformly random
    direction for ``U(duration_range)`` seconds. Register with ``mode="interval"`` and an
    interval of one policy step: the term keeps its own per-env timers and rewrites the
    force every step. The current force is exposed as ``env._priv_push_force_w`` for
    :func:`push_force`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.asset: Articulation = env.scene[p.get("asset_cfg", SceneEntityCfg("robot")).name]
        body_ids, _ = self.asset.find_bodies(p["body_name"])
        assert len(body_ids) == 1, body_ids
        self._body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=env.device)
        n = env.num_envs
        self.force_w = torch.zeros(n, 3, device=env.device)
        self.remaining = torch.zeros(n, device=env.device)
        self.next_push = math_utils.sample_uniform(*p["push_interval_s"], (n,), device=env.device)
        env._priv_push_force_w = self.force_w

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.force_w[env_ids] = 0.0
        self.remaining[env_ids] = 0.0
        n = self.next_push[env_ids].shape[0]
        self.next_push[env_ids] = math_utils.sample_uniform(
            *self.cfg.params["push_interval_s"], (n,), device=self.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        body_name: str,
        force_range: tuple[float, float],
        duration_range: tuple[float, float],
        push_interval_s: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        dt = env.step_dt
        self.next_push -= dt
        self.remaining -= dt
        start = torch.nonzero(self.next_push <= 0.0).flatten()
        if start.numel() > 0:
            k = start.numel()
            mag = math_utils.sample_uniform(*force_range, (k,), device=self.device)
            ang = math_utils.sample_uniform(0.0, 2.0 * math.pi, (k,), device=self.device)
            dur = math_utils.sample_uniform(*duration_range, (k,), device=self.device)
            self.force_w[start, 0] = mag * torch.cos(ang)
            self.force_w[start, 1] = mag * torch.sin(ang)
            self.remaining[start] = dur
            self.next_push[start] = dur + math_utils.sample_uniform(*push_interval_s, (k,), device=self.device)
        self.force_w[self.remaining <= 0.0] = 0.0
        # write directly to the composer: set_external_force_and_torque is deprecated and warns every call
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(self.force_w.unsqueeze(1).contiguous(), dtype=wp.vec3f),
            torques=wp.from_torch(torch.zeros_like(self.force_w).unsqueeze(1).contiguous(), dtype=wp.vec3f),
            body_ids=self._body_ids_wp,
            env_ids=self.asset._ALL_INDICES_WP,
            is_global=True,
        )


def push_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(N, 2) push force currently applied by :class:`push_by_force`, in the yaw frame, in units of 100 N."""
    force_w = getattr(env, "_priv_push_force_w", None)
    if force_w is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    heading = env.scene[asset_cfg.name].data.heading_w
    c, s = torch.cos(heading), torch.sin(heading)
    fx = c * force_w[:, 0] + s * force_w[:, 1]
    fy = -s * force_w[:, 0] + c * force_w[:, 1]
    return torch.stack([fx, fy], dim=-1) / 100.0


# ---------------------------------------------------------------------------
# friction / mass: randomized at startup only, so read once from PhysX and cache
# ---------------------------------------------------------------------------


class body_static_friction(ManagerTermBase):
    """(N, len(bodies)) static friction of the given bodies (mean over their collision shapes).

    The value is read from PhysX on the first real env reset and cached, since
    PhysX material access is a CPU round-trip. ObservationManager also calls
    ``reset()`` (no env_ids) right after building the term, which is *before*
    the ``startup`` randomization runs, so that call is ignored.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # shape index range per body (same workaround as randomize_rigid_body_material)
        shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            view = self.asset._physics_sim_view.create_rigid_body_view(link_path)
            shapes_per_body.append(view.max_shapes)
        body_ids = self.asset_cfg.body_ids
        if body_ids == slice(None):
            body_ids = list(range(len(shapes_per_body)))
        self._shape_ranges = [
            (sum(shapes_per_body[:b]), sum(shapes_per_body[:b]) + shapes_per_body[b]) for b in body_ids
        ]
        self._cache = torch.ones(env.num_envs, len(body_ids), device=env.device)
        self._loaded = False

    def reset(self, env_ids=None):
        if env_ids is not None and not self._loaded:
            self._load()

    def _load(self):
        materials = self.asset.root_physx_view.get_material_properties()  # (N, shapes, 3) on CPU
        vals = [materials[:, s:e, 0].mean(dim=1) for s, e in self._shape_ranges]
        self._cache[:] = torch.stack(vals, dim=-1).to(self._cache.device)
        self._loaded = True

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return self._cache


class body_added_mass(ManagerTermBase):
    """(N, len(bodies)) mass offset [kg] of the given bodies relative to the nominal asset. Cached like friction."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        n = self.asset.num_bodies if self.asset_cfg.body_ids == slice(None) else len(self.asset_cfg.body_ids)
        self._cache = torch.zeros(env.num_envs, n, device=env.device)
        self._loaded = False

    def reset(self, env_ids=None):
        # see body_static_friction.reset: skip the construction-time call
        if env_ids is not None and not self._loaded:
            masses = self.asset.root_physx_view.get_masses()  # (N, bodies) on CPU
            delta = masses - self.asset.data.default_mass.cpu()
            self._cache[:] = delta[:, self.asset_cfg.body_ids].to(self._cache.device)
            self._loaded = True

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return self._cache


# ---------------------------------------------------------------------------
# contact state
# ---------------------------------------------------------------------------


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """(N, len(bodies)) 1.0 if the body is in contact (net force norm > threshold)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > threshold).float()
