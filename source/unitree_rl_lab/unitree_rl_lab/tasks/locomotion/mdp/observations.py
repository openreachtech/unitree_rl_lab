from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


"""
Explicit Estimator labels (Unitree-Go2-LongJump-v1, Stage B).

See リファレンス_ExplicitEstimator実装仕様.md for the design this implements
(ported from genesis_lr's go2_ee, itself an implementation of Ji et al.
2022's "Concurrent Training of a Control Policy and a State Estimator",
arXiv:2202.05481). These are critic-only / estimator-target observations,
never exposed to the policy group directly.
"""


def link_contact_bool(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Per-link contact flag (1.0 if in contact, else 0.0) for the bodies in ``sensor_cfg``.

    Default selects every body ("contact_forces" is configured with
    ``prim_path="{ENV_REGEX_NS}/Robot/.*"``, so it already covers
    hip/thigh/calf/foot/base for all four legs -- no new sensor config is
    needed for the 17-dim label set used by Stage B).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()


def foot_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot")
) -> torch.Tensor:
    """Foot height above ground (world-frame z) for the bodies in ``asset_cfg``.

    This task's terrain is flat with ground z=0 at spawn (see
    longjump_base_env_cfg.py), so the foot's world z coordinate already *is*
    its height above ground -- unlike the terrain-aware
    ``foot_clearance_terrain_adaptive`` reward (used by the blind-stair
    task), no height-scanner raycast against local terrain is needed here.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


def jump_time_encoding(env: ManagerBasedRLEnv, command_name: str = "jump_command") -> torch.Tensor:
    """Normalised time since the jump was triggered, in [0, 1]; 0 while not jumping.

    2026-09-02: added because the policy previously saw the jump command as a bare 0/1
    bit with no phase information -- it could not tell "the jump was just commanded, load
    the legs" from "I have been airborne for 300 ms, prepare to land", so a single
    feed-forward action mapping had to serve both. tak's working Go2-Jump task feeds the
    same quantity (mdp.jump_time_encoding, tag ``jump-demo``). Normalised by
    ``max_jump_duration_s`` so the value stays O(1) alongside the other observations.
    """
    command_term = env.command_manager.get_term(command_name)
    jumping = command_term.command[:, 0] > 0.0
    progress = (command_term.time_since_trigger / command_term.cfg.max_jump_duration_s).clamp(0.0, 1.0)
    return (progress * jumping.float()).unsqueeze(-1)
