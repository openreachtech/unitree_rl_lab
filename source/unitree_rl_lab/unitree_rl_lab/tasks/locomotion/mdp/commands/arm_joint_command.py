"""Scripted arm motion for locomotion pre-training of a legged manipulator.

The locomotion policy only drives the legs. This command moves the arm underneath it so the
legs learn to walk with the arm in any configuration and while it is moving, as the
locomotion policy of "Whole-Body End-Effector Pose Tracking" (III-G) requires.

Every ``resampling_time_range`` seconds a goal configuration is drawn from a pre-sampled set
of collision-free arm configurations (``scripts/tools/sample_z1_workspace.py``), or, with
probability ``default_pose_prob``, the default (folded) pose. The arm's PD position target
is moved toward the goal on a straight line in joint space, synchronized so all joints
arrive together and no joint exceeds ``max_joint_vel``. The arm's PD actuators track it.
"""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ArmJointCommand(CommandTerm):
    """Moves the arm's joint position targets toward randomly drawn goal configurations.

    ``command`` is the current goal configuration (num_envs, num_arm_joints), in the joint
    order of the dataset.
    """

    cfg: ArmJointCommandCfg

    def __init__(self, cfg: ArmJointCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        if not os.path.isfile(cfg.dataset_path):
            raise FileNotFoundError(
                f"Arm configuration dataset not found: {cfg.dataset_path}\n"
                "Generate it with: python scripts/tools/sample_z1_workspace.py --headless"
            )
        data = torch.load(cfg.dataset_path, map_location="cpu")
        self.joint_ids, self.joint_names = self.robot.find_joints(data["arm_joint_names"], preserve_order=True)
        self.goal_pool = data["arm_joint_pos"].to(self.device)  # (num_samples, num_joints)

        default = self.robot.data.default_joint_pos[:, self.joint_ids]
        self.goal = default.clone()  # current goal configuration
        self.target = default.clone()  # PD target sent to the arm, moving toward the goal

        self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"ArmJointCommand: {len(self.joint_ids)} joints, {self.goal_pool.shape[0]} goal configurations,"
            f" max_joint_vel={self.cfg.max_joint_vel}, default_pose_prob={self.cfg.default_pose_prob}"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.goal

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Start moving from where the arm is after the reset events, not from the old target.
        if env_ids is None:
            env_ids = slice(None)
        self.target[env_ids] = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        idx = torch.randint(0, self.goal_pool.shape[0], (n,), device=self.device)
        goal = self.goal_pool[idx]
        use_default = torch.rand(n, device=self.device) < self.cfg.default_pose_prob
        default = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids]
        self.goal[env_ids] = torch.where(use_default.unsqueeze(-1), default, goal)

    def _update_command(self):
        # Synchronized straight-line step: the joint with the largest remaining distance
        # moves at max_joint_vel, the others proportionally slower.
        delta = self.goal - self.target
        max_step = self.cfg.max_joint_vel * self._env.step_dt
        scale = torch.clamp(max_step / delta.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
        self.target += delta * scale
        self.robot.set_joint_position_target(self.target, joint_ids=self.joint_ids)

    def _update_metrics(self):
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        self.metrics["goal_error"] = (self.goal - joint_pos).abs().amax(dim=-1)


@configclass
class ArmJointCommandCfg(CommandTermCfg):
    """Configuration for :class:`ArmJointCommand`."""

    class_type: type = ArmJointCommand

    asset_name: str = MISSING
    """Name of the robot in the scene."""

    dataset_path: str = MISSING
    """``.pt`` file with ``arm_joint_names`` and ``arm_joint_pos`` (num_samples, num_joints)."""

    max_joint_vel: float = 0.7
    """Speed limit of the joint-space motion toward the goal [rad/s]."""

    default_pose_prob: float = 0.2
    """Probability that a resampled goal is the default (folded) pose instead of a dataset sample."""
