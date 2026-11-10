"""MDP terms for whole-body end-effector pose tracking.

Rewards, reset and curriculum of "Whole-Body End-Effector Pose Tracking"
(doc/papers/Whole-Body_End-Effector_Pose_Tracking.md, III-E/F/G). They work with
:class:`EePoseKeypointCommand`, which holds the goal, the keypoints and the per-episode errors.

The paper's printed R_t and R_q have ``exp(+d / sigma)`` and sum k = 0..3 over 3 keypoints; both are
read as the evident intent, ``exp(-d / sigma)`` over the 3 keypoints.
"""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands.ee_pose_command import EePoseKeypointCommand


# ---------------------------------------------------------------------------
# Rewards (III-E)
# ---------------------------------------------------------------------------
def ee_keypoint_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """R_t: delayed keypoint tracking, (1 / T_r) * sum_k exp(-d_k / std) in the last T_r of each command."""
    command: EePoseKeypointCommand = env.command_manager.get_term(command_name)
    reward = torch.exp(-command.keypoint_distance / std).sum(dim=-1) / command.cfg.tracking_window
    return reward * command.tracking_active.float()


def ee_keypoint_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """R_p: mean decrease of the keypoint distances below their smallest value since the command started."""
    command: EePoseKeypointCommand = env.command_manager.get_term(command_name)
    return command.progress()


class feet_contact_force(ManagerTermBase):
    """R_f: sum_i max(F_i - threshold, 0), only when every foot pushes harder than ``threshold``.

    Divided by the robot's nominal weight (m g), so standing on all feet gives about 1 whatever the
    robot. Unnormalized, the paper's weight 0.015 gives Aliengo+Z1 (~25 kg) about 3.7 per step just
    for standing, which outweighs the tracking reward and makes standing still the best policy.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        self.weight = (robot.data.default_mass.sum(dim=-1) * 9.81).to(env.device)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        all_in_contact = (force > threshold).all(dim=-1)
        return torch.clamp(force - threshold, min=0.0).sum(dim=-1) / self.weight * all_in_contact.float()


class joint_pos_initial_exp(ManagerTermBase):
    """R_q: sum_i exp(-|q_i - q_i^init| / std) over the selected joints; q^init is the reset configuration."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.q_init = self.asset.data.default_joint_pos[:, self.asset_cfg.joint_ids].clone()

    def reset(self, env_ids: Sequence[int] | None = None):
        # Called after the reset events, so joint_pos is the configuration the episode starts from.
        if env_ids is None:
            env_ids = slice(None)
        self.q_init[env_ids] = self.asset.data.joint_pos[env_ids][:, self.asset_cfg.joint_ids]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
        q = self.asset.data.joint_pos[:, asset_cfg.joint_ids]
        return torch.exp(-(q - self.q_init).abs() / std).sum(dim=-1)


# ---------------------------------------------------------------------------
# Reset from recorded locomotion states (III-G)
# ---------------------------------------------------------------------------
class reset_from_init_states(ManagerTermBase):
    """Reset the robot to a state recorded from the locomotion policy on the same sub-terrain.

    The dataset (``scripts/tools/collect_init_states.py``) holds, per sample, the sub-terrain
    (``terrain_level``, ``terrain_type``), the base position relative to that sub-terrain's origin, the
    base orientation and the joint positions. A sub-terrain without samples borrows those of the
    nearest level of the same type. Velocities are zero.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        path = cfg.params["dataset_path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Initial-state dataset not found: {path}\n"
                "Collect it with: python scripts/tools/collect_init_states.py --headless"
            )
        data = torch.load(path, map_location="cpu")
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        terrain = env.scene.terrain
        rows, cols = terrain.terrain_origins.shape[:2]
        meta = data["meta"]
        if (meta["num_rows"], meta["num_cols"]) != (rows, cols) or meta["seed"] != terrain.cfg.terrain_generator.seed:
            raise ValueError(
                f"Initial states were recorded on a different terrain (rows/cols/seed {meta['num_rows']}/"
                f"{meta['num_cols']}/{meta['seed']}, scene {rows}/{cols}/{terrain.cfg.terrain_generator.seed})."
            )

        self.pos_rel = data["root_pos_rel"].to(self.device)
        self.quat = data["root_quat"].to(self.device)
        joint_ids, _ = self.asset.find_joints(data["joint_names"], preserve_order=True)
        self.joint_ids = torch.tensor(joint_ids, device=self.device)
        self.joint_pos = data["joint_pos"].to(self.device)

        # samples grouped by sub-terrain cell = level * cols + type
        cell = data["terrain_level"].long() * cols + data["terrain_type"].long()
        counts = torch.bincount(cell, minlength=rows * cols)
        self.order = torch.argsort(cell).to(self.device)
        start = torch.cumsum(counts, 0) - counts
        # empty cells borrow the nearest non-empty level of the same type
        source = torch.arange(rows * cols)
        for c in range(cols):
            filled = [r for r in range(rows) if counts[r * cols + c] > 0]
            if not filled:
                raise ValueError(f"No initial states for terrain type {c}.")
            for r in range(rows):
                if counts[r * cols + c] == 0:
                    nearest = min(filled, key=lambda f: abs(f - r))
                    source[r * cols + c] = nearest * cols + c
        self.cell_start = start[source].to(self.device)
        self.cell_count = counts[source].to(self.device)
        self.num_cols = cols

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, dataset_path: str, asset_cfg=None):
        terrain = env.scene.terrain
        cell = terrain.terrain_levels[env_ids] * self.num_cols + terrain.terrain_types[env_ids]
        pick = (torch.rand(len(env_ids), device=self.device) * self.cell_count[cell]).long()
        idx = self.order[self.cell_start[cell] + pick]

        root = torch.zeros(len(env_ids), 13, device=self.device)
        root[:, :3] = terrain.env_origins[env_ids] + self.pos_rel[idx]
        root[:, 3:7] = self.quat[idx]
        self.asset.write_root_state_to_sim(root, env_ids=env_ids)

        joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.joint_ids] = self.joint_pos[idx]
        self.asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


# ---------------------------------------------------------------------------
# Terrain curriculum (III-F)
# ---------------------------------------------------------------------------
def terrain_levels_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    up_pos: float = 0.2,
    up_rot: float = 0.349,
    down_pos: float = 0.8,
    down_rot: float = 2.094,
) -> torch.Tensor:
    """Move to harder terrain when the episode's mean tracked errors are below ``up_pos`` [m] and
    ``up_rot`` [rad] (20 cm / 20 deg), to easier terrain when above ``down_pos`` and ``down_rot``
    (80 cm / 120 deg). An episode that ended before any tracked step (a fall) also moves down."""
    command: EePoseKeypointCommand = env.command_manager.get_term(command_name)
    pos_err, rot_err, tracked = command.episode_errors(env_ids)
    move_up = tracked & (pos_err < up_pos) & (rot_err < up_rot)
    move_down = ~tracked | ((pos_err > down_pos) & (rot_err > down_rot))
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


__all__ = [
    "ee_keypoint_tracking",
    "ee_keypoint_progress",
    "feet_contact_force",
    "joint_pos_initial_exp",
    "reset_from_init_states",
    "terrain_levels_pose",
]
