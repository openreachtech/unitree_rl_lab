"""Reward and termination wrappers that switch a source task's terms on and off by command state.

The merged environment reuses the locomotion and acrobatics reward sets *unchanged* -- same
functions, same weights. That is not tidiness: the value function is initialised from critics
trained against those exact rewards, so altering a weight makes the inherited value wrong by a scale
factor everywhere, and the initialisation stops being worth having. What changes is only *when* each
term applies.

Why the wrapped term's arguments are nested
-------------------------------------------
Isaac Lab validates a term against its function signature and rejects anything it cannot account
for. A ``**kwargs`` catch-all is read as a *mandatory parameter literally named "params"*, so a
wrapper that forwards arbitrary arguments cannot be expressed that way::

    ValueError: The term 'bad_orientation' expects mandatory parameters: ['gate', 'term', 'params']

So the wrapped term's own arguments live in a nested ``term_params`` dict. The cost is that Isaac
Lab resolves ``SceneEntityCfg`` entries only at the top level of ``params``
(``manager_base.py:_process_term_cfg_at_play``), which would leave a nested one without its
``joint_ids``/``body_ids``. :func:`resolve_gated_term_params` does that resolution once at startup.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING, Callable

from .gating import gate_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gated(
    env: ManagerBasedRLEnv,
    gate: str,
    term: Callable[..., torch.Tensor],
    term_params: dict | None = None,
    gate_command_name: str = "jump",
    gate_window_s: float = 1.5,
    gate_crossfade_s: float = 0.25,
    gate_standing_speed: float = 0.1,
) -> torch.Tensor:
    """Scale an existing reward term by a command-state gate.

    Wrap a term rather than editing it::

        feet_slide = RewTerm(func=mdp.gated, weight=-0.1, params={
            "gate": "locomotion",
            "term": mdp.feet_slide,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        })
    """
    mask = gate_mask(env, gate, gate_command_name, gate_window_s, gate_crossfade_s, gate_standing_speed)
    return term(env, **(term_params or {})) * mask


def gated_termination(
    env: ManagerBasedRLEnv,
    gate: str,
    term: Callable[..., torch.Tensor],
    term_params: dict | None = None,
    gate_command_name: str = "jump",
    gate_window_s: float = 1.5,
    gate_standing_speed: float = 0.1,
) -> torch.Tensor:
    """Suppress a termination outside its gate. Same wrapping contract as :func:`gated`.

    Used for ``bad_orientation``, which has to be off while the robot is deliberately inverted. No
    crossfade -- a termination is boolean, so the gate is thresholded rather than faded.

    Note what this gives for free: a flip that has *failed* is still inverted when the window
    closes, at which point the orientation limit comes back and ends the episode. No separate
    "failed the commanded move" detector is needed.
    """
    mask = gate_mask(
        env, gate, gate_command_name, gate_window_s, crossfade_s=0.0, standing_speed=gate_standing_speed
    )
    return term(env, **(term_params or {})) & (mask > 0.5)


def resolve_gated_term_params(env: ManagerBasedRLEnv, env_ids=None) -> None:
    """Resolve ``SceneEntityCfg`` entries nested inside ``term_params``.

    Registered as a ``startup`` event, which runs once after every manager exists and before any
    term is evaluated. Isaac Lab only resolves scene entities at the top level of a term's
    ``params``; a nested one would reach the wrapped function with ``joint_ids``/``body_ids`` still
    ``None``, which fails as an indexing error deep inside an unrelated reward rather than as
    anything that names the real cause.
    """
    managers = [
        manager
        for manager in (getattr(env, "reward_manager", None), getattr(env, "termination_manager", None))
        if manager is not None
    ]
    for manager in managers:
        for name in manager.active_terms:
            params = manager.get_term_cfg(name).params.get("term_params")
            if not isinstance(params, dict):
                continue
            for key, value in params.items():
                if isinstance(value, SceneEntityCfg):
                    try:
                        value.resolve(env.scene)
                    except ValueError as error:
                        raise ValueError(f"Error while parsing '{name}:term_params:{key}'. {error}")
