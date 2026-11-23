"""Left-right symmetry augmentation for G1 (rsl_rl ``symmetry_cfg``).

Mirrors observations and actions about the robot's sagittal (x-z) plane, so PPO
trains on every sample and its mirror image (Mittal et al. 2024).

The mirror is built at runtime from the env, not hard-coded:
  * joints: ``left_*`` <-> ``right_*`` swapped; roll / yaw joints change sign
  * observation terms: looked up by term name in ``_TERM_RULES``; an unknown
    term raises, so adding an observation forces a decision here
  * history: IsaacLab stores a term's history as (H * D) flattened, oldest first;
    the same per-step mirror is applied to each of the H slices
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


# term name -> rule. Rules:
#   ("sign", [..])   multiply the per-step vector elementwise
#   "joints"         per-joint vector in robot joint order: permute + sign
#   "lr_pair"        two values ordered [left, right]: swap
#   "keep"           unchanged (scalar / symmetric quantity)
_TERM_RULES: dict[str, object] = {
    "base_lin_vel": ("sign", [1, -1, 1]),
    "base_ang_vel": ("sign", [-1, 1, -1]),
    "projected_gravity": ("sign", [1, -1, 1]),
    "velocity_commands": ("sign", [1, -1, -1]),
    "joint_pos_rel": "joints",
    "joint_vel_rel": "joints",
    "last_action": "joints",
    # G1-MLP-Phase1 privileged critic terms
    "push_force": ("sign", [1, -1]),  # yaw-frame xy force
    "feet_friction": "lr_pair",
    "torso_added_mass": "keep",
    "feet_contact": "lr_pair",
}


def _joint_mirror(joint_names: list[str]) -> tuple[list[int], list[float]]:
    """Permutation and sign so that ``mirrored = x[:, perm] * sign``."""
    index = {n: i for i, n in enumerate(joint_names)}
    perm, sign = [], []
    for name in joint_names:
        if name.startswith("left_"):
            partner = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            partner = "left_" + name[len("right_") :]
        else:
            partner = name  # waist joints
        if partner not in index:
            raise ValueError(f"symmetry: no mirror joint for '{name}'")
        perm.append(index[partner])
        sign.append(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0)
    return perm, sign


class _GroupMirror:
    """Precomputed (index, sign) so that ``mirrored = obs[:, index] * sign`` for one obs group."""

    def __init__(self, env: ManagerBasedRLEnv, group: str, joint_perm: list[int], joint_sign: list[float]):
        om = env.observation_manager
        names = om.active_terms[group]
        dims = om.group_obs_term_dim[group]
        cfgs = om._group_obs_term_cfgs[group]
        index, sign = [], []
        start = 0
        for name, dim, cfg in zip(names, dims, cfgs):
            total = int(torch.tensor(dim).prod())
            hist = cfg.history_length if cfg.history_length > 0 else 1
            step = total // hist
            local_idx, local_sign = self._term_rule(name, step, joint_perm, joint_sign)
            for h in range(hist):
                base = start + h * step
                index += [base + i for i in local_idx]
                sign += local_sign
            start += total
        self.index = torch.tensor(index, dtype=torch.long, device=env.device)
        self.sign = torch.tensor(sign, dtype=torch.float, device=env.device)
        self.dim = start

    @staticmethod
    def _term_rule(name, step, joint_perm, joint_sign):
        rule = _TERM_RULES.get(name)
        if rule is None:
            raise KeyError(f"symmetry: no mirror rule for observation term '{name}'")
        if rule == "joints":
            if step != len(joint_perm):
                raise ValueError(f"symmetry: term '{name}' has {step} dims, expected {len(joint_perm)} joints")
            return list(joint_perm), list(joint_sign)
        if rule == "lr_pair":
            if step != 2:
                raise ValueError(f"symmetry: term '{name}' must be [left, right], got {step} dims")
            return [1, 0], [1.0, 1.0]
        if rule == "keep":
            return list(range(step)), [1.0] * step
        kind, s = rule
        if len(s) != step:
            raise ValueError(f"symmetry: term '{name}' has {step} dims, sign rule has {len(s)}")
        return list(range(step)), [float(v) for v in s]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.index] * self.sign


def _mirrors(env: ManagerBasedRLEnv) -> dict[str, _GroupMirror]:
    cache = getattr(env, "_symmetry_mirrors", None)
    if cache is None:
        robot = env.scene["robot"]
        perm, sign = _joint_mirror(robot.joint_names)
        cache = {g: _GroupMirror(env, g, perm, sign) for g in env.observation_manager.active_terms}
        cache["_actions"] = (
            torch.tensor(perm, dtype=torch.long, device=env.device),
            torch.tensor(sign, dtype=torch.float, device=env.device),
        )
        env._symmetry_mirrors = cache
    return cache


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return ``[original; left-right mirrored]`` observations and actions (batch doubled)."""
    env = env.unwrapped
    mirrors = _mirrors(env)

    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][n:] = mirrors[group](obs[group])

    actions_aug = None
    if actions is not None:
        perm, sign = mirrors["_actions"]
        actions_aug = torch.cat([actions, actions[:, perm] * sign], dim=0)

    return obs_aug, actions_aug
