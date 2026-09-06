"""Reward wrappers specific to the bipedal stances.

Sits here rather than beside ``gated`` in ``..multitask.mdp`` because the concept is not shared:
``gated`` switches the locomotion, acrobatics and bipedal reward sets on and off and every family
uses it, while this one reads a stance sign that only the bipedal command has. Keeping it here
leaves the dependency running one way -- the merged environment knows about the skills it is built
from, and the skills do not need to know about each other.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stance_aware(
    env: ManagerBasedRLEnv,
    term: Callable[..., torch.Tensor],
    front_params: dict,
    hind_params: dict,
    command_name: str = "handstand",
) -> torch.Tensor:
    """Evaluate ``term`` for both stances and take the one each environment is commanded into.

    Half the bipedal reward set names a side -- which feet carry the robot, which legs are tucked,
    which links may not touch the floor. A single policy serving both stances needs those terms to
    follow the command per environment, and the alternative to this is a bespoke mode-aware
    function per term, which is how ``feat/biped`` did it and why its reward module grew a
    ``lifted_leg_*`` family.

    Both branches are computed and one is selected. That is twice the arithmetic for terms costing
    microseconds, in exchange for every side-naming term reusing the function it already had.

    Note what does *not* belong here: the CoM-CoP terms. Given all four feet as candidates, the
    force-weighted centre of pressure collapses onto whichever ones are loaded, so those terms are
    already stance-agnostic and wrapping them would only pick between two identical answers.
    """
    is_front = env.command_manager.get_term(command_name).stance > 0
    return torch.where(is_front, term(env, **front_params), term(env, **hind_params))
