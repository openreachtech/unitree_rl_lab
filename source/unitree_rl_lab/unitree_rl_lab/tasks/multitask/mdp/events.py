"""Event terms specific to the multi-task environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..obs_spec import CRITIC_UNIFIED, POLICY_UNIFIED, assert_layout_matches

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def assert_observation_layout(env: ManagerBasedRLEnv, env_ids=None) -> None:
    """Check the built observation groups against the layout in :mod:`..obs_spec`.

    Registered as a ``startup`` event, which runs once after every manager exists. The expert
    weights are placed by column index, and nothing in Isaac Lab ties the config's declaration
    order to the layout those indices assume -- reordering two ``ObsTerm`` attributes would keep
    training happily while feeding each expert the wrong inputs. This turns that into an immediate
    failure at env construction.
    """
    manager = env.observation_manager
    for group_name, layout in (("policy", POLICY_UNIFIED), ("critic", CRITIC_UNIFIED)):
        term_dims = {
            name: math.prod(shape)
            for name, shape in zip(manager.active_terms[group_name], manager.group_obs_term_dim[group_name])
        }
        assert_layout_matches(layout, term_dims, group_name)
