"""End-effector twist command along interpolated trajectories.

Implements 3.2 and 7.2.1 of "Multi-critic Learning for Whole-body End-effector Twist Tracking"
(doc/papers/Multi-critic_Learning_for_Whole-body_End-effector_Twist_Tracking.md):

* Goal (3.2, "End-effector trajectory sampling"). The paper's region: a sphere around the shoulder,
  rejecting the torso/leg region and points under the ground. Not the paper's sampling: the
  end-effector pose at random arm joint positions (forward kinematics), kept if in the region, so
  every goal is reachable (the paper's outward-facing orientations mostly are not for the Z1).
* Trajectory. Starts at the current end-effector pose and ends at the goal. Waypoints are interpolated
  linearly in position and by slerp in orientation over a random duration (a fraction of the
  command's resampling time, so the goal is held for the rest of it).
* Local trajectories. A fraction ``local_prob`` of the trajectories re-start from the current
  end-effector pose at every step (same interpolation factor), which keeps the policy close to
  the trajectory early on.
* Twist. ``v = (r_i - r_EE) / dt`` and ``w = (theta_i [-] theta_EE) / dt`` towards the next
  waypoint, in the task frame.
* Task frame. The base frame until ``control_frame_start_step`` env steps, then the control frame:
  the base yaw only, origin at the base x/y and ``nominal_height`` above the mean foot height,
  so the task does not move with the torso's height, roll and pitch. Trajectories are stored in
  the task frame and travel with the robot.

``command`` is the twist (6). The goal pose (position + 6D rotation, 9) and the error to the
current waypoint are read through :mod:`..twist_tracking`.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix (Zhou et al. 2019). Shape (N, 6)."""
    return math_utils.matrix_from_quat(quat)[..., :2].transpose(-1, -2).reshape(quat.shape[0], 6)


class EeTwistCommand(CommandTerm):
    """End-effector twist towards the next waypoint of a start-to-goal trajectory."""

    cfg: EeTwistCommandCfg

    def __init__(self, cfg: EeTwistCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_body_id = self.robot.find_bodies(cfg.body_name)[0][0]
        self.feet_body_ids = self.robot.find_bodies(cfg.feet_body_names, preserve_order=True)[0]

        # -- goal sampling: region and arm kinematics
        self.sphere_center = torch.tensor(cfg.sphere_center, device=self.device)
        self.reject_box = torch.tensor(cfg.reject_box, device=self.device)  # (3, 2): lo, hi
        arm_ids = self.robot.find_joints(cfg.arm_joint_names, preserve_order=True)[0]
        self.arm_limits = self.robot.data.soft_joint_pos_limits[0, arm_ids].clone()  # (J, 2)
        self.chain_offsets = torch.tensor([o for o, _ in cfg.arm_chain], device=self.device)
        self.chain_axes = [a for _, a in cfg.arm_chain]
        self.chain_mount = torch.tensor(cfg.arm_mount, device=self.device)
        self.chain_tip = torch.tensor(cfg.arm_tip, device=self.device)

        # -- trajectory, in the task frame
        n = self.num_envs
        self.start_pos = torch.zeros(n, 3, device=self.device)
        self.start_quat = torch.zeros(n, 4, device=self.device)
        self.goal_pos = torch.zeros(n, 3, device=self.device)
        self.goal_quat = torch.zeros(n, 4, device=self.device)
        self.target_pos = torch.zeros(n, 3, device=self.device)
        """Current waypoint (the one the last twist pointed at)."""
        self.target_quat = torch.zeros(n, 4, device=self.device)
        self.start_quat[:, 0] = self.goal_quat[:, 0] = self.target_quat[:, 0] = 1.0
        self.duration = torch.ones(n, device=self.device)
        self.elapsed = torch.zeros(n, device=self.device)
        self.is_local = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.needs_start = torch.ones(n, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        """The waypoint has been set from a fresh end-effector pose (rewards and metrics are masked otherwise)."""
        self.twist = torch.zeros(n, 6, device=self.device)

        # -- metrics: errors to the waypoint, averaged over the episode
        self._err_pos_sum = torch.zeros(n, device=self.device)
        self._err_rot_sum = torch.zeros(n, device=self.device)
        self._err_count = torch.zeros(n, device=self.device)
        self.metrics["position_error"] = torch.zeros(n, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            f"EeTwistCommand: goals in a {self.cfg.sphere_radius} m sphere around {self.cfg.sphere_center},"
            f" local trajectories {self.cfg.local_prob:.0%},"
            f" control frame from env step {self.cfg.control_frame_start_step}"
        )

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """End-effector linear and angular velocity in the task frame. Shape (num_envs, 6)."""
        return self.twist

    @property
    def use_control_frame(self) -> bool:
        return self._env.common_step_counter >= self.cfg.control_frame_start_step

    def frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Task frame in the world: the base frame, or the control frame (see module docstring)."""
        base_pos = self.robot.data.root_link_pos_w
        base_quat = self.robot.data.root_link_quat_w
        if not self.use_control_frame:
            return base_pos, base_quat
        pos = base_pos.clone()
        pos[:, 2] = self.robot.data.body_link_pos_w[:, self.feet_body_ids, 2].mean(dim=1) + self.cfg.nominal_height
        return pos, math_utils.yaw_quat(base_quat)

    def ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current end-effector pose in the task frame."""
        frame_pos, frame_quat = self.frame_pose_w()
        ee = self.robot.data.body_link_pose_w[:, self.ee_body_id]
        pos = math_utils.quat_apply_inverse(frame_quat, ee[:, :3] - frame_pos)
        quat = math_utils.quat_mul(math_utils.quat_conjugate(frame_quat), ee[:, 3:7])
        return pos, quat

    def waypoint_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current waypoint minus current end-effector pose, in the task frame: position (N, 3) and
        rotation vector (N, 3). Measured after the physics step, this is the error the last twist
        asked to close."""
        pos, quat = self.ee_pose()
        return self.target_pos - pos, math_utils.quat_box_minus(self.target_quat, quat)

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._err_pos_sum[env_ids] = 0.0
        self._err_rot_sum[env_ids] = 0.0
        self._err_count[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)
        self.goal_pos[env_ids], self.goal_quat[env_ids] = self._sample_goals(n)
        # the goal is reached after a random part of the command time and held for the rest
        lo, hi = self.cfg.duration_fraction
        self.duration[env_ids] = self.time_left[env_ids] * (lo + (hi - lo) * torch.rand(n, device=self.device))
        self.elapsed[env_ids] = 0.0
        self.is_local[env_ids] = torch.rand(n, device=self.device) < self.cfg.local_prob
        # the start is set from the end-effector pose in _update_command
        self.needs_start[env_ids] = True

    def _update_command(self):
        pos, quat = self.ee_pose()
        # right after a reset (episode length 0) the body poses are stale until the next physics step
        ready = self.needs_start & (self._env.episode_length_buf > 0)
        reset_start = ready | (self.is_local & ~self.needs_start)
        self.start_pos[reset_start] = pos[reset_start]
        self.start_quat[reset_start] = quat[reset_start]
        self.needs_start &= ~ready
        self.valid[:] = ~self.needs_start

        # next waypoint: the elapsed fraction of the trajectory (eq. 3). A local trajectory re-starts at
        # the current pose with the same alpha, so its waypoint moves towards the goal as time goes on
        # and a lagging end-effector is pulled towards it (the paper's denser reward landscape).
        dt = self._env.step_dt
        alpha = ((self.elapsed + dt) / self.duration).clamp(max=1.0).unsqueeze(-1)
        self.target_pos[:] = self.start_pos + alpha * (self.goal_pos - self.start_pos)
        self.target_quat[:] = math_utils.quat_box_plus(
            self.start_quat, alpha * math_utils.quat_box_minus(self.goal_quat, self.start_quat)
        )
        self.elapsed += dt

        self.twist[:, :3] = (self.target_pos - pos) / dt
        self.twist[:, 3:] = math_utils.quat_box_minus(self.target_quat, quat) / dt
        self.twist[~self.valid] = 0.0

    def _update_metrics(self):
        pos_err, rot_err = self.waypoint_error()
        valid = self.valid.float()
        self._err_pos_sum += valid * pos_err.norm(dim=-1)
        self._err_rot_sum += valid * rot_err.norm(dim=-1)
        self._err_count += valid
        count = self._err_count.clamp_min(1.0)
        self.metrics["position_error"] = self._err_pos_sum / count
        self.metrics["orientation_error"] = self._err_rot_sum / count

    """
    Helpers.
    """

    def _arm_fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector pose in the base frame for arm joint positions ``q`` (M, J), from the
        kinematic chain in the config (all link frames unrotated at zero joint position)."""
        m = q.shape[0]
        pos = self.chain_mount.expand(m, 3).clone()
        quat = torch.zeros(m, 4, device=self.device)
        quat[:, 0] = 1.0
        zero = torch.zeros(m, device=self.device)
        for j, (offset, axis) in enumerate(zip(self.chain_offsets, self.chain_axes)):
            pos = pos + math_utils.quat_apply(quat, offset.expand(m, 3))
            euler = [zero, zero, zero]
            euler["xyz".index(axis)] = q[:, j]
            quat = math_utils.quat_mul(quat, math_utils.quat_from_euler_xyz(*euler))
        pos = pos + math_utils.quat_apply(quat, self.chain_tip.expand(m, 3))
        return pos, quat

    def _in_region(self, p: torch.Tensor) -> torch.Tensor:
        """In the goal sphere, outside the torso/leg box and above the minimum height."""
        in_ball = (p - self.sphere_center).norm(dim=-1) <= self.cfg.sphere_radius
        in_box = ((p >= self.reject_box[:, 0]) & (p <= self.reject_box[:, 1])).all(dim=-1)
        return in_ball & ~in_box & (p[:, 2] >= self.cfg.min_height)

    def _sample_goals(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Goal poses reachable by construction: the end-effector pose at random joint positions
        within the soft limits, kept if inside the paper's region (sphere, box, height)."""
        pos = torch.empty(n, 3, device=self.device)
        quat = torch.empty(n, 4, device=self.device)
        filled = 0
        lo, hi = self.arm_limits[:, 0], self.arm_limits[:, 1]
        while filled < n:
            m = 4 * (n - filled)
            q = lo + torch.rand(m, len(lo), device=self.device) * (hi - lo)
            p, r = self._arm_fk(q)
            ok = self._in_region(p)
            k = min(int(ok.sum()), n - filled)
            pos[filled : filled + k] = p[ok][:k]
            quat[filled : filled + k] = r[ok][:k]
            filled += k
        return pos, quat

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.target_visualizer = VisualizationMarkers(self.cfg.target_pose_visualizer_cfg)
                self.ee_visualizer = VisualizationMarkers(self.cfg.ee_pose_visualizer_cfg)
            for v in (self.goal_visualizer, self.target_visualizer, self.ee_visualizer):
                v.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            for v in (self.goal_visualizer, self.target_visualizer, self.ee_visualizer):
                v.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        frame_pos, frame_quat = self.frame_pose_w()
        for visualizer, pos, quat in (
            (self.goal_visualizer, self.goal_pos, self.goal_quat),
            (self.target_visualizer, self.target_pos, self.target_quat),
        ):
            visualizer.visualize(
                frame_pos + math_utils.quat_apply(frame_quat, pos), math_utils.quat_mul(frame_quat, quat)
            )
        ee = self.robot.data.body_link_pose_w[:, self.ee_body_id]
        self.ee_visualizer.visualize(ee[:, :3], ee[:, 3:7])


@configclass
class EeTwistCommandCfg(CommandTermCfg):
    """Configuration for :class:`EeTwistCommand`."""

    class_type: type = EeTwistCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    """End-effector body."""

    feet_body_names: list[str] = MISSING
    """Feet, whose mean height sets the control frame's height."""

    sphere_center: tuple[float, float, float] = MISSING
    """Shoulder (the arm's first pitch joint) in the base frame [m]: center of the goal sphere."""

    sphere_radius: float = MISSING
    """Goal sphere radius [m]."""

    reject_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = MISSING
    """Torso and leg region in the base frame, ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) [m]: goals
    inside it are rejected."""

    min_height: float = MISSING
    """Lowest goal height in the base frame [m] (the ground is about ``-nominal_height``)."""

    arm_joint_names: list[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    """Arm joints, base to tip."""

    arm_mount: tuple[float, float, float] = (0.2535, 0.0, 0.056)
    """Arm base (first link) in the base frame [m] (Z1 on Aliengo)."""

    arm_chain: tuple = (
        ((0.0, 0.0, 0.0585), "z"),
        ((0.0, 0.0, 0.045), "y"),
        ((-0.35, 0.0, 0.0), "y"),
        ((0.218, 0.0, 0.057), "y"),
        ((0.07, 0.0, 0.0), "z"),
        ((0.0492, 0.0, 0.0), "x"),
    )
    """Per arm joint: offset from the previous joint [m] and rotation axis (Z1 URDF)."""

    arm_tip: tuple[float, float, float] = (0.051, 0.0, 0.0)
    """End-effector body from the last joint [m] (Z1 gripperStator)."""

    duration_fraction: tuple[float, float] = (0.3, 0.9)
    """Trajectory duration as a fraction of the command's resampling time."""

    local_prob: float = 0.3
    """Fraction of trajectories re-started from the current end-effector pose at every step."""

    control_frame_start_step: int = 0
    """Env step (``common_step_counter``) from which the task frame is the control frame (7.2.1).
    The counter restarts with every run, including resumed ones."""

    nominal_height: float = 0.37
    """Control frame height above the mean foot height [m]: the base's in the default stance."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_goal_pose"
    )
    target_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_target_pose"
    )
    ee_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pose"
    )

    def __post_init__(self):
        self.goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.target_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_pose_visualizer_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
