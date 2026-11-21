"""Privileged (critic-only) observations, ported from humanoid-gym.

humanoid-gym gives the critic the push perturbation, friction, mass and foot
contact state. None of these are available on the real robot, so they go to
the critic group only; the actor's input (and thus deploy) is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "push_by_setting_velocity_logged",
    "push_velocity",
    "body_static_friction",
    "body_added_mass",
    "feet_contact",
]


# ---------------------------------------------------------------------------
# push: event that records what it applied, and the matching observation
# ---------------------------------------------------------------------------


def _push_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_priv_push_vel"):
        env._priv_push_vel = torch.zeros(env.num_envs, 2, device=env.device)
        env._priv_push_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    return env._priv_push_vel, env._priv_push_step


def push_by_setting_velocity_logged(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Same as ``push_by_setting_velocity``, but records the xy velocity delta it applied."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids].clone()
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    kick = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    asset.write_root_velocity_to_sim(vel_w + kick, env_ids=env_ids)

    push_vel, push_step = _push_buffers(env)
    push_vel[env_ids] = kick[:, :2]
    push_step[env_ids] = getattr(env, "common_step_counter", 0)


def push_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 2) world-frame xy velocity kick applied this step, zero otherwise.

    Interval events run right before observations in ``step``, so the critic
    sees the kick on the step it happened; with critic history_length=5 it stays
    visible for 5 steps. (humanoid-gym holds the last push until the next one;
    an impulse is the more faithful description of what happened.)
    """
    push_vel, push_step = _push_buffers(env)
    active = (push_step == env.common_step_counter).unsqueeze(-1)
    return push_vel * active


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
