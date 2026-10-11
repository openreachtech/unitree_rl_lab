"""Left/right mirroring of the G1 get-up observations and actions (sagittal plane, y -> -y).

Used by ``unitree_rl_lab.utils.host_rl`` for a symmetry loss: the policy should output the
mirrored action for the mirrored observation, so it gets up the same way on both sides.

Mirroring rules:
* base angular velocity (wx, wy, wz) -> (-wx, wy, -wz)
* projected gravity (gx, gy, gz) -> (gx, -gy, gz)
* joints: swap left_* and right_*; roll and yaw joints change sign, pitch / knee / elbow keep it
* action scale: unchanged

The policy observation is Isaac Lab's history layout: each term is ``history_length`` frames of
its own dimension, concatenated term after term.
"""

from __future__ import annotations

import torch

SIGN_FLIP_KEYWORDS = ("_roll_", "_yaw_")


def _joint_mirror(joint_names: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Index permutation and sign so that ``mirrored[:, i] = sign[i] * x[:, perm[i]]``."""
    index = {n: i for i, n in enumerate(joint_names)}
    perm, sign = [], []
    for name in joint_names:
        if name.startswith("left_"):
            other = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            other = "left_" + name[len("right_") :]
        else:
            other = name
        perm.append(index[other])
        sign.append(-1.0 if any(k in name for k in SIGN_FLIP_KEYWORDS) else 1.0)
    return torch.tensor(perm, device=device), torch.tensor(sign, device=device)


def build_mirror(env) -> tuple[callable, callable]:
    """Return ``(mirror_obs, mirror_actions)`` for the env's ``policy`` observation group."""
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    perm, sign = _joint_mirror(robot.joint_names, device)

    om = unwrapped.observation_manager
    names = om.active_terms["policy"]
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    history = unwrapped.cfg.observations.policy.history_length or 1

    vec_sign = {
        "base_ang_vel": torch.tensor([-1.0, 1.0, -1.0], device=device),
        "projected_gravity": torch.tensor([1.0, -1.0, 1.0], device=device),
    }
    joint_terms = {"joint_pos", "joint_vel", "last_action"}
    for name in names:
        if name not in vec_sign and name not in joint_terms and name != "action_scale":
            raise ValueError(f"No mirroring rule for policy observation term '{name}'")

    def mirror_actions(a: torch.Tensor) -> torch.Tensor:
        return a[:, perm] * sign

    def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
        out, start = [], 0
        for name, dim in zip(names, dims):
            x = obs[:, start : start + dim].reshape(obs.shape[0], history, dim // history)
            if name in vec_sign:
                x = x * vec_sign[name]
            elif name in joint_terms:
                x = x[:, :, perm] * sign
            out.append(x.reshape(obs.shape[0], dim))
            start += dim
        assert start == obs.shape[1], (start, obs.shape)
        return torch.cat(out, dim=1)

    return mirror_obs, mirror_actions
