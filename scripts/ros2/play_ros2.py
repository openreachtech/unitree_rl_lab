# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy while bridging the sim to ROS 2 (VLFM nav stack).

Runs one environment of a Go2 velocity task and exchanges the robot-layer contract
(doc/design/vlfm_nav.md §3) with the ROS side:

Publishes (Isaac's bundled rclpy, py3.11 -- so standard messages only):
  /clock               rosgraph_msgs/Clock       sim time; ROS side runs use_sim_time
  /sim/joint_states    sensor_msgs/JointState    12 joints, named; effort = applied torque
  /sim/imu             sensor_msgs/Imu           base orientation / gyro / accelerometer
  /sim/gt_odom         nav_msgs/Odometry         ground truth (topic only), for evaluating
                                                 whatever odometry actually runs
  /sim/applied_cmd     geometry_msgs/Twist       the command applied to the policy this step
  /utlidar/cloud       sensor_msgs/PointCloud2   full MID-360 returns, `radar` frame -- LIO input
  /utlidar/cloud_band  sensor_msgs/PointCloud2   height-band subset for pointcloud_to_laserscan
  /vlfm/camera/image_raw       sensor_msgs/Image        front RGB, `camera_optical` frame (M5 VLM input)
  /vlfm/camera/camera_info     sensor_msgs/CameraInfo   intrinsics, derived from the cfg's focal/aperture

The two camera topics are only filled while something is subscribed -- see
SimBridge.image_wanted. Plain frontier exploration subscribes to neither, so the camera
costs nothing until a VLM scorer or a bag recorder asks for it.

Subscribes:
  /cmd_vel             geometry_msgs/Twist       written into the base_velocity command term:
                                                 clamped to the training limit ranges, small
                                                 nonzero commands lifted past the policy's
                                                 stand deadband (--min_walk_speed)

With --gt_odom the ground truth additionally becomes THE odometry (odom->base TF +
/odometry/filtered); the default is to leave odometry to the ROS side
(go2_sim.launch.py runs RKO-LIO, the same odometry the hardware uses).

The task needs the MID-360 scanner in its scene (default task
Go2-Blind-GRU-Mid360-Explore). There is no mid360/explore experiment folder, so pass
the base phase's checkpoint explicitly:

    python scripts/ros2/play_ros2.py \
        --checkpoint logs/rsl_rl/go2_blind_gru_phase4/<run>/model_7300.pt

Run from the default shell (env_isaaclab venv). Do NOT source ROS: the
isaacsim.ros2.bridge extension ships its own Humble libraries built for py3.11 and
uses them exactly when no system ROS is on the environment. Both sides speak the
default FastDDS on the same ROS_DOMAIN_ID, which is how they meet.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import os
import sys

# --- bundled-ROS environment, set by re-exec ---------------------------------------
# The bridge's internal Humble libraries resolve through LD_LIBRARY_PATH, which the
# dynamic loader reads once at process start -- os.environ changes after that are
# invisible to it. So if the env is not prepared yet, prepare it and exec ourselves.
_spec = importlib.util.find_spec("isaacsim")
_ros2_ext = os.path.join(os.path.dirname(_spec.origin), "exts", "isaacsim.ros2.bridge")
_ros2_lib = os.path.join(_ros2_ext, "humble", "lib")
if _ros2_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _env = dict(os.environ)
    _env["ROS_DISTRO"] = "humble"
    _env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    _env["LD_LIBRARY_PATH"] = (_env.get("LD_LIBRARY_PATH", "") + ":" + _ros2_lib).lstrip(":")
    print("[INFO] Re-exec with the bundled ROS 2 (humble, FastDDS) environment.")
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
# ------------------------------------------------------------------------------------

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a policy and bridge the sim to ROS 2.")
parser.add_argument("--task", type=str, default="Go2-Blind-GRU-Mid360-Explore", help="Name of the task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--cloud_topic", type=str, default="/utlidar/cloud", help="PointCloud2 topic.")
parser.add_argument(
    "--cloud_frame",
    type=str,
    default="radar",
    help="TF frame of the published cloud. `radar` is the L1 link in go2.urdf, so"
    " robot_state_publisher provides base->radar and the chain closes.",
)
parser.add_argument("--cmd_vel_timeout", type=float, default=0.5, help="Zero the command this long after the last /cmd_vel (s).")
parser.add_argument(
    "--gt_odom",
    action="store_true",
    default=False,
    help="Publish ground-truth odometry (odom->base TF + /odometry/filtered) straight"
    " from the sim, instead of leaving odometry to RKO-LIO on the ROS side. Debug/eval"
    " tool. Pair with go2_sim.launch.py odom:=external so there is exactly one"
    " odom->base publisher.",
)
parser.add_argument(
    "--min_walk_speed",
    type=float,
    default=0.0,
    help="Deadband shaping: a nonzero /cmd_vel with a smaller linear norm is scaled up"
    " to this, keeping direction. 0 (the default) disables it."
    " ONLY needed by policies with a stand deadband. The Phase3->Phase4 lineage had a"
    " severe one -- measured 2026-09-28 (model_7300, raw commands): vx<=0.5 stood"
    " COMPLETELY still, 0.6 crept at 0.14 m/s, 0.7 tracked 67%% -- and 0.8 was the"
    " workaround. Go2-Blind-GRU-Phase4-VLFM does not: measured 2026-10-01 on flat"
    " ground, 64 envs, it tracks 92%% at vx 0.1 and 102-109%% from 0.2 up. Phase 1 is"
    " also clean (54%% at 0.1, >=88%% from 0.3). Inflating a command such a policy can"
    " already follow just makes it overshoot what Nav2 asked for, so leave this at 0"
    " unless a measurement says otherwise.",
)
parser.add_argument(
    "--min_walk_wz",
    type=float,
    default=0.0,
    help="Same shaping for pure rotations, applied only when the linear command is"
    " ~zero. 0 (the default) disables it. Same story as --min_walk_speed: the old"
    " Phase 4 did not rotate in place at all (wz<=0.5 gave 0.04 rad/s over 8 s) and 0.9"
    " was the workaround; Phase4-VLFM tracks 80%% at wz 0.2 and ~89%% at 0.9, within a"
    " few points of Phase 1 across the range. This matters more than it used to --"
    " MPPI now runs in DiffDrive, so in-place rotation is the only way to turn.",
)
parser.add_argument(
    "--cloud_z_band",
    type=float,
    nargs=2,
    default=(0.15, 0.85),
    metavar=("MIN", "MAX"),
    help="WORLD-z band (m) for the *_band cloud used by pointcloud_to_laserscan."
    " Gravity-aligned, standing in for the LIO-posed height filter of the real"
    " pipeline: a base-frame band tilts with body pitch and far ground returns leak"
    " past the 1 m walls, ray-tracing phantom free space outside the building."
    " The 0.15 m floor assumes a flat-ground policy (no step climbing): anything"
    " taller than 0.15 m counts as a wall for Nav2/SLAM. History: 0.30 (Phase4"
    " climbs clean 0.30 m steps) -> 0.20 (a ~0.20 m sofa seat defeated it,"
    " 2026-09-28) -> 0.15 (flat-walking policies)."
    " The full unfiltered cloud is always published too (on --cloud_topic) -- that"
    " is what LiDAR odometry consumes; a 0.7 m slab has no vertical structure to"
    " register against. Pass equal values to disable the band topic.",
)
parser.add_argument(
    "--cloud_accum_steps",
    type=int,
    default=5,
    help="Env steps of MID-360 returns per PointCloud2. 5 steps = 0.1 s = one full"
    " elevation sweep, matching the real driver's 10 Hz frames; single 20 ms windows"
    " cover only a slice of the elevation band and starve pointcloud_to_laserscan.",
)
parser.add_argument(
    "--camera_topic",
    type=str,
    default="/vlfm/camera/image_raw",
    help="sensor_msgs/Image topic for the front RGB camera. CameraInfo goes to the"
    " sibling `.../camera_info`. Silently inactive on tasks with no `front_cam` in"
    " the scene (the phase configs).",
)
parser.add_argument(
    "--camera_hz",
    type=float,
    default=0.0,
    help="Publish rate for the front camera. 0 (default) follows the sensor's own"
    " update_period from the env cfg (GO2_CAM_HZ). Lower it to measure how much of"
    " the frame budget rendering costs; the VLM scores at <=2 Hz behind a motion"
    " gate, so nothing downstream needs more than a few Hz.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera sensors do not render at all in headless mode without this, and the nav tasks
# (explore/kujiale) all carry a front_cam. Harmless on tasks that have none.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import numpy as np
import time
import torch

from isaacsim.core.utils.extensions import enable_extension

# Load the bundled ROS 2 libraries (py3.11 Humble). Must happen after the app is up and
# before `import rclpy`.
enable_extension("isaacsim.ros2.bridge")

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2, PointField
from tf2_ros import TransformBroadcaster

from unitree_rl_lab.assets.models.modules.runners import UnitreeOnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# SDK (unitree_sdk2 / unitree_go) joint order. The composer on the ROS side maps the
# named /sim/joint_states back into this order; publishing names here keeps the wire
# format self-describing.
SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

GRAVITY = 9.81
FALL_TILT_RAD = math.radians(45.0)
"""Trunk tilt that counts as "over" for the notice above. Not a termination."""


def _stamp(t: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t)
    msg.nanosec = int((t - int(t)) * 1e9)
    return msg


class SimBridge(Node):
    """Publishes the robot-layer contract topics; holds the latest /cmd_vel."""

    def __init__(self):
        super().__init__("isaac_sim_bridge")
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_joints = self.create_publisher(JointState, "/sim/joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "/sim/imu", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, args_cli.cloud_topic, qos_profile_sensor_data)
        self.pub_cloud_band = self.create_publisher(
            PointCloud2, args_cli.cloud_topic + "_band", qos_profile_sensor_data
        )
        # Ground truth is always published on /sim/gt_odom (topic only, no TF) so any
        # odometry source can be evaluated against it. --gt_odom additionally makes it
        # THE odometry: odom->base TF + the contract topic /odometry/filtered.
        self.pub_gt = self.create_publisher(Odometry, "/sim/gt_odom", 10)
        if args_cli.gt_odom:
            self.pub_odom = self.create_publisher(Odometry, "/odometry/filtered", 10)
            self.tf_broadcaster = TransformBroadcaster(self)
        self.cmd_vel = np.zeros(3)
        self.cmd_vel_time = None  # wall time of last message
        self._cmd_seen = False
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        # Observability: the command actually applied to the policy this step.
        self.pub_applied = self.create_publisher(Twist, "/sim/applied_cmd", 10)
        # Front RGB (M5). Created unconditionally -- the main loop only writes to it
        # when the task's scene actually has a `front_cam`.
        #
        # RELIABLE, not the sensor-data profile the clouds use. A 768x384 rgb8 frame is
        # 884 kB, which DDS fragments across many UDP datagrams; under BEST_EFFORT one
        # lost fragment drops the whole sample, and measured 2026-09-30 that cost 70% of
        # the frames (42 of 143 reached a bag, while the CameraInfo published in the very
        # same call -- tiny, so never fragmented -- arrived complete). At 5 Hz the
        # retransmits are cheap, and a dropped frame here is a hole in the VLM's evidence,
        # not a stale scan the next sweep replaces.
        img_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_image = self.create_publisher(Image, args_cli.camera_topic, img_qos)
        self.pub_caminfo = self.create_publisher(
            CameraInfo, args_cli.camera_topic.rsplit("/", 1)[0] + "/camera_info", img_qos
        )
        self.image_msg = None  # built once by init_camera()

        # Static parts of the messages
        self.joint_msg = JointState()
        self.joint_msg.name = SDK_JOINT_NAMES
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu"
        self.cloud_msg = PointCloud2()
        self.cloud_msg.header.frame_id = args_cli.cloud_frame
        self.cloud_msg.height = 1
        self.cloud_msg.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")
        ]
        self.cloud_msg.is_bigendian = False
        self.cloud_msg.point_step = 12
        self.cloud_msg.is_dense = True

    def _on_cmd_vel(self, msg: Twist):
        self.cmd_vel[:] = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_vel_time = time.time()
        if not self._cmd_seen:
            self._cmd_seen = True
            print(f"[INFO] first /cmd_vel received: {self.cmd_vel}", flush=True)

    def command(self) -> np.ndarray:
        """Latest command, zeroed once it goes stale (Nav2 died, teleop closed)."""
        if self.cmd_vel_time is None or time.time() - self.cmd_vel_time > args_cli.cmd_vel_timeout:
            return np.zeros(3)
        return self.cmd_vel

    def publish_clock(self, t: float):
        msg = Clock()
        msg.clock = _stamp(t)
        self.pub_clock.publish(msg)

    def publish_applied_cmd(self, cmd: np.ndarray):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(cmd[0]), float(cmd[1]), float(cmd[2])
        self.pub_applied.publish(msg)

    def publish_state(self, t: float, q, dq, tau, quat_wxyz, gyro, acc):
        stamp = _stamp(t)
        self.joint_msg.header.stamp = stamp
        self.joint_msg.position = q.tolist()
        self.joint_msg.velocity = dq.tolist()
        self.joint_msg.effort = tau.tolist()
        self.pub_joints.publish(self.joint_msg)

        m = self.imu_msg
        m.header.stamp = stamp
        m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = quat_wxyz.tolist()
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gyro.tolist()
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = acc.tolist()
        self.pub_imu.publish(m)


    def publish_gt_odom(self, t: float, pos, quat_wxyz, lin_vel_b, ang_vel_b):
        stamp = _stamp(t)
        w, x, y, z = quat_wxyz.tolist()
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "odom"
        od.child_frame_id = "base"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = pos.tolist()
        od.pose.pose.orientation.w, od.pose.pose.orientation.x = w, x
        od.pose.pose.orientation.y, od.pose.pose.orientation.z = y, z
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = lin_vel_b.tolist()
        od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = ang_vel_b.tolist()
        self.pub_gt.publish(od)
        if not args_cli.gt_odom:
            return

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos.tolist()
        tf.transform.rotation.w, tf.transform.rotation.x = w, x
        tf.transform.rotation.y, tf.transform.rotation.z = y, z
        self.tf_broadcaster.sendTransform(tf)
        self.pub_odom.publish(od)

    def publish_cloud(self, t: float, points_xyz: np.ndarray, band_mask: np.ndarray | None):
        msg = self.cloud_msg
        msg.header.stamp = _stamp(t)
        msg.width = len(points_xyz)
        msg.row_step = msg.point_step * msg.width
        data = points_xyz.astype(np.float32)
        msg.data = data.tobytes()
        self.pub_cloud.publish(msg)
        if band_mask is not None:
            band = data[band_mask]
            msg.width = len(band)
            msg.row_step = msg.point_step * msg.width
            msg.data = band.tobytes()
            self.pub_cloud_band.publish(msg)

    def init_camera(self, width: int, height: int, fx: float):
        """Freeze the constant parts of Image and CameraInfo.

        Images go out in `camera_optical` (+z forward, +x right, +y down) because that
        is what CameraInfo means by fx/cx and what consumers such as Foxglove's frustum
        overlay assume. `camera_joint`'s x-forward `camera` frame is the one our own
        bearing maths uses; both are in the URDF.

        Pixels are square (the env cfg leaves vertical_aperture unset, so Isaac Lab
        derives it from the aspect ratio), hence fy == fx.
        """
        msg = Image()
        msg.header.frame_id = "camera_optical"
        msg.height, msg.width = height, width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        self.image_msg = msg

        info = CameraInfo()
        info.header.frame_id = "camera_optical"
        info.height, info.width = height, width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        cx, cy = width / 2.0, height / 2.0
        info.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.caminfo_msg = info

    def image_wanted(self) -> bool:
        """Is anyone actually going to read the picture?

        A frame is 884 kB and costs a GPU-to-host copy before any of it reaches the wire,
        so sending it to nobody is the most expensive no-op in this loop. Exploring
        without a target starts no VLM and records no bag, and then nothing subscribes --
        so gating on the subscriber count turns the camera off by itself, and turns it
        back on the instant a scorer or a `ros2 bag record` shows up. Both topics are
        gated together to keep their counts equal, which is the check that caught the
        2026-09-30 frame loss.
        """
        return (self.pub_image.get_subscription_count()
                + self.pub_caminfo.get_subscription_count()) > 0

    def publish_image(self, t: float, rgb: np.ndarray):
        """`rgb` is (H, W, 3) uint8, C-contiguous. No cv_bridge here -- it is not in
        Isaac's bundled py3.11 ROS -- but an Image is a header plus a byte buffer, the
        same hand-packing the point clouds already do."""
        stamp = _stamp(t)
        self.image_msg.header.stamp = stamp
        self.image_msg.data = rgb.tobytes()
        self.pub_image.publish(self.image_msg)
        self.caminfo_msg.header.stamp = stamp
        self.pub_caminfo.publish(self.caminfo_msg)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # -- overrides for a continuous, externally-commanded rollout --
    env_cfg.scene.num_envs = 1
    # SLAM/odometry cannot survive a teleport; make the episode effectively endless.
    # Fall terminations stay, but a reset now ENDS the run (see the dones[] check in
    # the loop) -- continuing after the teleport overlaid a second, misregistered
    # floor plan on the map.
    env_cfg.episode_length_s = 1.0e9
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # never resample over /cmd_vel
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0

    # limit_ranges are the policy's training envelope; clamp /cmd_vel into it.
    lim = cmd_cfg.limit_ranges
    cmd_low = np.array([lim.lin_vel_x[0], lim.lin_vel_y[0], lim.ang_vel_z[0]])
    cmd_high = np.array([lim.lin_vel_x[1], lim.lin_vel_y[1], lim.ang_vel_z[1]])

    # checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = UnitreeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # -- scene handles --
    scene = env.unwrapped.scene
    robot = scene["robot"]
    if "mid360_scanner" not in scene.sensors:
        raise RuntimeError(
            f"Task '{args_cli.task}' has no `mid360_scanner`; use a Mid360 task"
            " (e.g. Go2-Blind-GRU-Mid360-Phase4) so there is a cloud to publish."
        )
    scanner = scene.sensors["mid360_scanner"]
    # Optional: only the nav worlds (explore/kujiale) carry it; phase tasks do not.
    front_cam = scene.sensors.get("front_cam")

    device = env.unwrapped.device
    # articulation joint order -> SDK order
    sdk_joint_ids = [robot.data.joint_names.index(n) for n in SDK_JOINT_NAMES]
    # mount rotation, to express returns in the URDF `radar` link frame
    mount_quat = torch.tensor(list(scanner.cfg.offset.rot), device=device)
    min_range = float(getattr(scanner.cfg, "min_range", 0.0))
    max_range = float(scanner.cfg.max_distance)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    # -- ROS --
    rclpy.init()
    bridge = SimBridge()
    print(f"[INFO] ROS 2 bridge up: /clock /sim/* {args_cli.cloud_topic} <- /cmd_vel (domain {os.environ.get('ROS_DOMAIN_ID', '0')})")

    dt = env.unwrapped.step_dt
    # -- front camera --
    cam_every = 0
    if front_cam is not None:
        cam_cfg = front_cam.cfg
        # Same three numbers the FOV comes from: HFOV = 2*atan(aperture / (2*focal)),
        # so fx = (width/2) / tan(HFOV/2) = width * focal / aperture. Square pixels
        # (vertical_aperture left unset in the cfg) make fy == fx.
        fx = cam_cfg.width * cam_cfg.spawn.focal_length / cam_cfg.spawn.horizontal_aperture
        bridge.init_camera(cam_cfg.width, cam_cfg.height, fx)
        hz = args_cli.camera_hz or (1.0 / cam_cfg.update_period if cam_cfg.update_period else 1.0 / dt)
        cam_every = max(1, round(1.0 / (hz * dt)))
        hfov = 2.0 * math.degrees(math.atan(cam_cfg.spawn.horizontal_aperture / (2.0 * cam_cfg.spawn.focal_length)))
        print(
            f"[INFO] front camera: {cam_cfg.width}x{cam_cfg.height} HFOV {hfov:.1f} deg"
            f" fx {fx:.1f} -> {args_cli.camera_topic} every {cam_every} steps"
            f" ({1.0 / (cam_every * dt):.1f} Hz)"
        )
    else:
        print(f"[INFO] task '{args_cli.task}' has no `front_cam`; no image published.")
    sim_t = 0.0
    step_i = 0
    fallen = False
    cloud_buf: list[torch.Tensor] = []
    prev_lin_vel_w = robot.data.root_lin_vel_w[0].clone()
    gravity_w = torch.tensor([0.0, 0.0, GRAVITY], device=device)

    obs = env.get_observations()
    if isinstance(obs, tuple):  # rsl-rl-lib 2.3.x returns (obs, extras)
        obs = obs[0]

    while simulation_app.is_running():
        loop_start = time.time()
        # /cmd_vel -> command term (clamped to the training envelope)
        rclpy.spin_once(bridge, timeout_sec=0.0)
        cmd = np.clip(bridge.command(), cmd_low, cmd_high)
        # Deadband shaping: lift small nonzero commands to the speed the policy
        # actually walks at (see --min_walk_speed). Zero stays zero.
        lin = float(np.hypot(cmd[0], cmd[1]))
        if 0.05 < lin < args_cli.min_walk_speed:
            cmd[:2] *= args_cli.min_walk_speed / lin
        elif lin <= 0.05 and 0.05 < abs(cmd[2]) < args_cli.min_walk_wz:
            cmd[2] = math.copysign(args_cli.min_walk_wz, cmd[2])
        cmd = np.clip(cmd, cmd_low, cmd_high)
        bridge.publish_applied_cmd(cmd)
        with torch.inference_mode():
            cmd_term.vel_command_b[0] = torch.tensor(cmd, dtype=torch.float32, device=device)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        sim_t += dt

        # Fall notice, not a termination. Nothing stops the run any more (see
        # velocity_env_cfg_explore.py), so this is the only mark left saying when the
        # robot went over -- enough to find the moment in a bag or line it up against
        # what the camera saw.
        tilt = float(torch.acos(torch.clamp(-robot.data.projected_gravity_b[0, 2], -1.0, 1.0)))
        if tilt > FALL_TILT_RAD and not fallen:
            fallen = True
            p = robot.data.root_pos_w[0]
            print(f"[WARN] trunk tipped {math.degrees(tilt):.0f} deg at"
                  f" ({float(p[0]):.2f}, {float(p[1]):.2f}, {float(p[2]):.2f}), sim t={sim_t:.1f}s."
                  " NOT terminating -- the robot stays where it is so the fall can be looked at.")
        elif tilt < FALL_TILT_RAD * 0.6 and fallen:
            fallen = False
            print(f"[INFO] back upright at sim t={sim_t:.1f}s")

        if dones[0]:
            # A reset teleports the robot home. LIO cannot track a teleport, so every
            # scan after this would be registered at a wrong pose and overlaid on the
            # good map (two superimposed floor plans, observed 2026-09-28). The run is
            # unrecoverable -- stop cleanly instead of corrupting it further.
            print("[ERROR] Environment reset (fall/termination). The map cannot survive a"
                  " teleport -- ending the run. Restart Isaac AND the ROS launch to retry.")
            break

        # -- robot state --
        q = robot.data.joint_pos[0, sdk_joint_ids].cpu().numpy()
        dq = robot.data.joint_vel[0, sdk_joint_ids].cpu().numpy()
        tau = robot.data.applied_torque[0, sdk_joint_ids].cpu().numpy()
        quat = robot.data.root_quat_w[0]  # (w, x, y, z)
        gyro = robot.data.root_ang_vel_b[0].cpu().numpy()
        # accelerometer = specific force: finite-difference world acc minus gravity
        # (-9.81 z), rotated into the base frame. IMU-link offset from base is ignored.
        lin_vel_w = robot.data.root_lin_vel_w[0]
        acc_w = (lin_vel_w - prev_lin_vel_w) / dt + gravity_w
        prev_lin_vel_w = lin_vel_w.clone()
        acc_b = quat_apply_inverse(quat.unsqueeze(0), acc_w.unsqueeze(0))[0].cpu().numpy()

        bridge.publish_clock(sim_t)
        bridge.publish_state(sim_t, q, dq, tau, quat.cpu().numpy(), gyro, acc_b)
        bridge.publish_gt_odom(
            sim_t,
            robot.data.root_pos_w[0].cpu().numpy(),
            quat.cpu().numpy(),
            robot.data.root_lin_vel_b[0].cpu().numpy(),
            robot.data.root_ang_vel_b[0].cpu().numpy(),
        )

        # -- point cloud: accumulate one full elevation sweep, publish in `radar` frame --
        # Returns are kept as world-frame points and only projected into the sensor
        # frame at publish time, i.e. deskewed into the end-of-sweep pose (what the
        # real driver+LIO pipeline delivers, minus their residual distortion).
        hits_w = scanner.data.ray_hits_w[0]
        sensor_pos = scanner._get_true_sensor_pos()[0]
        keep = torch.isfinite(hits_w).all(dim=-1)
        dist = (hits_w - sensor_pos).norm(dim=-1)
        keep &= (dist >= min_range) & (dist < max_range - 1e-3)
        cloud_buf.append(hits_w[keep])
        if len(cloud_buf) >= args_cli.cloud_accum_steps:
            hits = torch.cat(cloud_buf)
            cloud_buf.clear()
            rel_w = hits - sensor_pos
            base_quat = scanner.data.quat_w[0]
            p_base = quat_apply_inverse(base_quat.expand(len(rel_w), 4), rel_w)
            p_radar = quat_apply_inverse(mount_quat.expand(len(rel_w), 4), p_base)
            z_lo, z_hi = args_cli.cloud_z_band
            band_mask = None
            if z_hi > z_lo:
                band_mask = ((hits[:, 2] >= z_lo) & (hits[:, 2] <= z_hi)).cpu().numpy()
            bridge.publish_cloud(sim_t, p_radar.cpu().numpy(), band_mask)

        # -- front RGB --
        # The sensor renders on its own update_period; this only decides how often the
        # rendered buffer is shipped. Drop the alpha channel if the backend hands one
        # over, and make the slice contiguous before tobytes().
        step_i += 1
        if cam_every and step_i % cam_every == 0 and bridge.image_wanted():
            rgb = front_cam.data.output["rgb"][0, ..., :3]
            bridge.publish_image(sim_t, np.ascontiguousarray(rgb.cpu().numpy(), dtype=np.uint8))

        # real-time pacing: the ROS side integrates in /clock time, but Nav2's watchdogs
        # and the human at the teleop live in wall time.
        sleep = dt - (time.time() - loop_start)
        if sleep > 0:
            time.sleep(sleep)

    bridge.destroy_node()
    rclpy.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
