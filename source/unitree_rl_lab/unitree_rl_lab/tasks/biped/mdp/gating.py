"""Gate a reward term on the robot actually being up in the commanded stance.

The velocity-tracking and balance rewards are all satisfiable from a four-legged stance: a robot
that never attempts the rise tracks its command and holds still perfectly. Scaling them by
``HandstandCommand.upright_ramp`` makes the rise the only way to collect them (the paper's "if is
upright, else 0", ramped so the rise has a gradient).

Ported from ``feat/multi-task``'s ``gated`` wrapper, keeping only the "is upright" gate, plus its
``stance_aware`` router for the two-stance task.

The wrapped term's own arguments are nested in ``term_params``: Isaac Lab rejects a ``**kwargs``
catch-all. It also resolves ``SceneEntityCfg`` only at the top level of ``params``, and an
unresolved one does not fail -- ``body_ids`` stays ``slice(None)`` and silently selects every body.
:func:`resolve_nested_term_params` fixes that once at startup.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def when_upright(
    env: ManagerBasedRLEnv,
    term: Callable[..., torch.Tensor],
    term_params: dict | None = None,
    command_name: str = "handstand",
) -> torch.Tensor:
    """``term(env, **term_params)`` scaled by the handstand command's upright ramp."""
    ramp = env.command_manager.get_term(command_name).upright_ramp
    return term(env, **(term_params or {})) * ramp


def stance_aware(
    env: ManagerBasedRLEnv,
    term: Callable[..., torch.Tensor],
    front_params: dict,
    hind_params: dict,
    command_name: str = "handstand",
) -> torch.Tensor:
    """Evaluate ``term`` with each stance's params and take the commanded one per environment.

    For the two-stance task, where half the reward set names a side -- which wheels carry the robot,
    which legs are tucked, which links may not touch. Both branches are computed; the terms cost
    microseconds. Nests inside :func:`when_upright` (``term_params`` -> ``front_params``), which
    :func:`resolve_nested_term_params` handles by recursing.
    """
    is_front = env.command_manager.get_term(command_name).stance > 0
    return torch.where(is_front, term(env, **front_params), term(env, **hind_params))


def when_stance(
    env: ManagerBasedRLEnv,
    term: Callable[..., torch.Tensor],
    stance: float,
    term_params: dict | None = None,
    command_name: str = "handstand",
) -> torch.Tensor:
    """``term(env, **term_params)`` in environments commanded into ``stance``, zero elsewhere.

    For a term that only one stance has a counterpart for -- ``front_hip_height`` holds the front
    stance's shoulders up and would pull the hind stance's raised end down.
    """
    active = (env.command_manager.get_term(command_name).stance == stance).float()
    return term(env, **(term_params or {})) * active


def resolve_nested_term_params(env: ManagerBasedRLEnv, env_ids=None) -> None:
    """Resolve every ``SceneEntityCfg`` nested in a wrapper's params. A startup event."""
    for name in env.reward_manager.active_terms:
        params = env.reward_manager.get_term_cfg(name).params
        for key in ("term_params", "front_params", "hind_params"):
            if isinstance(params.get(key), dict):
                _resolve(env, params[key], f"{name}:{key}")


def _resolve(env: ManagerBasedRLEnv, params: dict, where: str) -> None:
    for key, value in params.items():
        if isinstance(value, SceneEntityCfg):
            try:
                value.resolve(env.scene)
            except ValueError as error:
                raise ValueError(f"Error while parsing '{where}:{key}'. {error}")
        elif isinstance(value, dict):
            _resolve(env, value, f"{where}:{key}")
