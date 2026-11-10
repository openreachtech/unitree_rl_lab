"""End-effector pose command with a keypoint representation.

Implements III-B/C of "Whole-Body End-Effector Pose Tracking"
(doc/papers/Whole-Body_End-Effector_Pose_Tracking.md):

* Pre-sampled poses. End-effector poses reachable by the arm alone on a fixed base
  (``scripts/tools/sample_z1_workspace.py``), in the base frame.
* Uniform workspace coverage. The poses are binned into ``num_bins`` concentric cylinders around
  the base z-axis (equal radial width up to the largest radius); a bin is chosen uniformly, then a
  pose within it.
* Whole-body reach. A random body transform (``base_offset_ranges``) is applied to the pose, as if
  the base had moved, so reaching it needs the legs as well.
* Terrain awareness. The goal is fixed in the world at sampling time (from the current base pose).
  A goal lower than ``terrain_margin`` above the terrain is re-drawn; the terrain height comes from
  a coarse map built once at start-up, holding the highest point within ``height_map_patch``.
* Keypoints. The pose is represented by 3 vertices of a cube of side ``cube_side`` centred on the
  end-effector and aligned with it. ``command`` is target minus current keypoints in the base frame
  (9 values).

The command also carries the state of the delayed tracking and progress rewards (``tracking_active``,
``keypoint_distance``, ``progress``) and the per-episode errors read by the terrain curriculum.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Cube vertices (in units of half the side) used as keypoints: any 3 non-collinear vertices fix the pose.
_KEYPOINT_VERTICES = ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0))


class EePoseKeypointCommand(CommandTerm):
    """Samples end-effector pose goals and exposes them as keypoint differences."""

    cfg: EePoseKeypointCommandCfg

    def __init__(self, cfg: EePoseKeypointCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_body_id = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- pre-sampled poses and their cylinder bins
        if not os.path.isfile(cfg.dataset_path):
            raise FileNotFoundError(
                f"End-effector pose dataset not found: {cfg.dataset_path}\n"
                "Generate it with: python scripts/tools/sample_z1_workspace.py --headless"
            )
        data = torch.load(cfg.dataset_path, map_location="cpu")
        self.pool_pos = data["ee_pos_b"].to(self.device)
        self.pool_quat = data["ee_quat_b"].to(self.device)
        radius = self.pool_pos[:, :2].norm(dim=-1)
        bins = torch.clamp((radius / radius.max() * cfg.num_bins).long(), max=cfg.num_bins - 1)
        order = torch.argsort(bins)
        counts = torch.bincount(bins, minlength=cfg.num_bins)
        self.bin_order = order
        self.bin_start = torch.cumsum(counts, 0) - counts
        self.bin_count = counts
        self.nonempty_bins = torch.nonzero(counts > 0).flatten()

        # -- keypoint offsets in the end-effector frame
        self.keypoint_offsets = torch.tensor(_KEYPOINT_VERTICES, device=self.device) * (cfg.cube_side / 2)

        # -- goal pose (world frame, fixed between resamples)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 0] = 1.0
        self.keypoint_diff_b = torch.zeros(self.num_envs, 3 * len(_KEYPOINT_VERTICES), device=self.device)
        # smallest keypoint distances since the last resample (NaN: not measured yet)
        self.min_distance = torch.full((self.num_envs, len(_KEYPOINT_VERTICES)), float("nan"), device=self.device)

        # -- coarse terrain height map
        self._build_height_map()

        # -- metrics: errors averaged over the steps in which the tracking reward is active
        self._err_pos_sum = torch.zeros(self.num_envs, device=self.device)
        self._err_rot_sum = torch.zeros(self.num_envs, device=self.device)
        self._err_count = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["resample_failures"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"EePoseKeypointCommand: {self.pool_pos.shape[0]} poses in bins {self.bin_count.tolist()},"
            f" cube side {self.cfg.cube_side} m, tracking window {self.cfg.tracking_window} s"
        )

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Target minus current keypoints in the base frame. Shape (num_envs, 9)."""
        return self.keypoint_diff_b

    @property
    def tracking_active(self) -> torch.Tensor:
        """True during the last ``tracking_window`` seconds of each command."""
        return self.time_left <= self.cfg.tracking_window

    @property
    def ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.robot.data.body_link_pose_w[:, self.ee_body_id]
        return pose[:, :3], pose[:, 3:7]

    def keypoints_w(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Keypoints of the pose(s) (num_envs, 3) / (num_envs, 4) in the world. Shape (num_envs, 3, 3)."""
        n, k = pos.shape[0], self.keypoint_offsets.shape[0]
        offsets = math_utils.quat_apply(
            quat.unsqueeze(1).expand(n, k, 4).reshape(-1, 4), self.keypoint_offsets.repeat(n, 1)
        ).view(n, k, 3)
        return pos.unsqueeze(1) + offsets

    @property
    def keypoint_distance(self) -> torch.Tensor:
        """Distance between each current and target keypoint. Shape (num_envs, 3)."""
        return (self.keypoints_w(self.goal_pos_w, self.goal_quat_w) - self.keypoints_w(*self.ee_pose_w)).norm(dim=-1)

    def progress(self) -> torch.Tensor:
        """Mean decrease of the keypoint distances below their smallest value so far (III-E, R_p).

        Stateful: updates the running minimum, so call it once per step (from the progress reward).
        """
        d = self.keypoint_distance
        first = torch.isnan(self.min_distance)
        self.min_distance = torch.where(first, d, self.min_distance)
        gain = torch.clamp(self.min_distance - d, min=0.0)
        self.min_distance = torch.minimum(self.min_distance, d)
        return gain.mean(dim=-1)

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
        pending = env_ids
        for _ in range(self.cfg.max_resample_tries):
            pos_w, quat_w = self._sample_goals(pending)
            self.goal_pos_w[pending] = pos_w
            self.goal_quat_w[pending] = quat_w
            too_low = pos_w[:, 2] < self._terrain_height(pos_w[:, :2]) + self.cfg.terrain_margin
            pending = pending[too_low]
            if len(pending) == 0:
                break
        # the remaining few keep their last (too low) goal
        self.metrics["resample_failures"][env_ids] = 0.0
        self.metrics["resample_failures"][pending] = 1.0
        self.min_distance[env_ids] = float("nan")

    def _update_command(self):
        base_quat = self.robot.data.root_link_quat_w
        diff_w = self.keypoints_w(self.goal_pos_w, self.goal_quat_w) - self.keypoints_w(*self.ee_pose_w)
        k = diff_w.shape[1]
        diff_b = math_utils.quat_apply_inverse(
            base_quat.unsqueeze(1).expand(-1, k, 4).reshape(-1, 4), diff_w.reshape(-1, 3)
        )
        self.keypoint_diff_b[:] = diff_b.view(self.num_envs, -1)

    def _update_metrics(self):
        ee_pos, ee_quat = self.ee_pose_w
        active = self.tracking_active.float()
        self._err_pos_sum += active * (self.goal_pos_w - ee_pos).norm(dim=-1)
        self._err_rot_sum += active * math_utils.quat_error_magnitude(self.goal_quat_w, ee_quat)
        self._err_count += active
        count = self._err_count.clamp_min(1.0)
        self.metrics["position_error"] = self._err_pos_sum / count
        self.metrics["orientation_error"] = self._err_rot_sum / count

    def episode_errors(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean position [m] / orientation [rad] errors over the tracked steps of the episode, and
        whether any step was tracked. Read by the terrain curriculum before the reset."""
        count = self._err_count[env_ids]
        return self.metrics["position_error"][env_ids], self.metrics["orientation_error"][env_ids], count > 0

    """
    Helpers.
    """

    def _sample_goals(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        # bin first, then a pose inside the bin
        b = self.nonempty_bins[torch.randint(0, len(self.nonempty_bins), (n,), device=self.device)]
        offset = (torch.rand(n, device=self.device) * self.bin_count[b]).long()
        idx = self.bin_order[self.bin_start[b] + offset]
        pos_b, quat_b = self.pool_pos[idx], self.pool_quat[idx]

        # random body transform
        r = self.cfg.base_offset_ranges
        lo = torch.tensor([r.x[0], r.y[0], r.z[0], r.roll[0], r.pitch[0], r.yaw[0]], device=self.device)
        hi = torch.tensor([r.x[1], r.y[1], r.z[1], r.roll[1], r.pitch[1], r.yaw[1]], device=self.device)
        t = lo + (hi - lo) * torch.rand(n, 6, device=self.device)
        t_quat = math_utils.quat_from_euler_xyz(t[:, 3], t[:, 4], t[:, 5])
        pos_b = math_utils.quat_apply(t_quat, pos_b) + t[:, :3]
        quat_b = math_utils.quat_mul(t_quat, quat_b)

        # fix the goal in the world, relative to the current base pose
        base_pos = self.robot.data.root_link_pos_w[env_ids]
        base_quat = self.robot.data.root_link_quat_w[env_ids]
        pos_w = base_pos + math_utils.quat_apply(base_quat, pos_b)
        quat_w = math_utils.quat_mul(base_quat, quat_b)
        return pos_w, quat_w

    def _build_height_map(self):
        """Highest terrain point within ``height_map_patch`` around each cell of a grid over the terrain."""
        terrain = self._env.scene.terrain
        self.height_map = None
        if terrain.cfg.terrain_type != "generator":
            return  # flat plane at z = 0
        from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
        from pxr import UsdGeom

        import omni.usd

        mesh_prim = sim_utils.get_first_matching_child_prim(
            terrain.cfg.prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        tf = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = points @ tf[:3, :3].T + tf[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)

        # grid over the sub-terrains (robots never leave them by more than a few metres)
        origins = terrain.terrain_origins.reshape(-1, 3)
        margin = 0.5 * max(terrain.cfg.terrain_generator.size) + 2.0
        res = self.cfg.height_map_resolution
        x0, y0 = origins[:, 0].min().item() - margin, origins[:, 1].min().item() - margin
        x1, y1 = origins[:, 0].max().item() + margin, origins[:, 1].max().item() + margin
        xs = torch.arange(x0, x1, res, device=self.device)
        ys = torch.arange(y0, y1, res, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        starts = torch.stack([gx.flatten(), gy.flatten(), torch.full_like(gx.flatten(), 100.0)], dim=-1)
        dirs = torch.zeros_like(starts)
        dirs[:, 2] = -1.0
        hits = raycast_mesh(starts.unsqueeze(0), dirs.unsqueeze(0), wp_mesh)[0][0]
        heights = torch.nan_to_num(hits[:, 2], nan=0.0, posinf=0.0, neginf=0.0).view(len(xs), len(ys))
        k = max(1, int(round(self.cfg.height_map_patch / res)) | 1)  # odd kernel
        heights = F.max_pool2d(heights[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0]
        self.height_map = heights
        self.height_map_origin = (x0, y0)

    def _terrain_height(self, xy: torch.Tensor) -> torch.Tensor:
        if self.height_map is None:
            return torch.zeros(xy.shape[0], device=self.device)
        res = self.cfg.height_map_resolution
        ix = ((xy[:, 0] - self.height_map_origin[0]) / res).long().clamp(0, self.height_map.shape[0] - 1)
        iy = ((xy[:, 1] - self.height_map_origin[1]) / res).long().clamp(0, self.height_map.shape[1] - 1)
        return self.height_map[ix, iy]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.ee_visualizer = VisualizationMarkers(self.cfg.ee_pose_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.ee_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.ee_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_visualizer.visualize(self.goal_pos_w, self.goal_quat_w)
        self.ee_visualizer.visualize(*self.ee_pose_w)


@configclass
class EePoseKeypointCommandCfg(CommandTermCfg):
    """Configuration for :class:`EePoseKeypointCommand`."""

    class_type: type = EePoseKeypointCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    """End-effector body."""

    dataset_path: str = MISSING
    """``.pt`` file with ``ee_pos_b`` (N, 3) and ``ee_quat_b`` (N, 4, wxyz): poses in the base frame."""

    @configclass
    class Ranges:
        x: tuple[float, float] = (0.0, 0.0)
        y: tuple[float, float] = (0.0, 0.0)
        z: tuple[float, float] = (0.0, 0.0)
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (0.0, 0.0)

    base_offset_ranges: Ranges = Ranges()
    """Random body transform applied to each pre-sampled pose [m, rad]."""

    num_bins: int = 5
    """Concentric cylinders for uniform workspace coverage."""

    cube_side: float = 0.2
    """Side of the keypoint cube [m]."""

    tracking_window: float = 2.0
    """The tracking reward is active during the last ``tracking_window`` seconds of each command [s]."""

    terrain_margin: float = 0.08
    """Goals lower than this above the terrain are re-drawn [m]."""

    height_map_resolution: float = 0.04
    height_map_patch: float = 0.2
    """Each height-map cell holds the highest terrain point within this square patch [m]."""

    max_resample_tries: int = 10

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_goal_pose"
    )
    ee_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pose"
    )

    def __post_init__(self):
        # the goal frame is drawn larger than the current end-effector frame
        self.goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.ee_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
