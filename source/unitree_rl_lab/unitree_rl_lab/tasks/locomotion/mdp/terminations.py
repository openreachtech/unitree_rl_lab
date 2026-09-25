from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation_grounded(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    command_name: str,
    relax_window_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Same check as (isaaclab) mdp.bad_orientation, relaxed only *briefly* while jumping.

    A jump attempt naturally pitches the body away from flat during the airborne phase
    and landing recovery, so the stock termination would end the episode on every
    attempt. See リファレンス_ExplicitEstimator実装仕様.md Stage B design.

    2026-09-02: the relaxation is now bounded by ``relax_window_s`` measured from the
    trigger, instead of applying for as long as jump_command==1. The unbounded version
    was an exploit: the old JumpCommand could only clear the command once the robot was
    upright again, so a robot that fell over kept jump_command==1 forever and was
    therefore permanently immune to this termination -- one of the three mechanisms
    behind the Stage B v2 collapse (the policy discovered that falling was both cheap
    and un-terminated). The command side is fixed too (max_jump_duration_s), so this is
    defence in depth. See プロジェクト_StageB崩壊の原因分析.md.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    bad_orientation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    command_term = env.command_manager.get_term(command_name)
    jumping = command_term.command[:, 0] > 0.0
    within_relax_window = command_term.time_since_trigger < relax_window_s
    return bad_orientation & (~(jumping & within_relax_window))
