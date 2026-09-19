# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy while bridging the sim to ROS 2 (VLFM nav stack, M1).

Runs one environment of a Go2 velocity task and exchanges the robot-layer contract
(doc/design/vlfm_nav.md §3) with the ROS side:

Publishes (Isaac's bundled rclpy, py3.11 -- so standard messages only):
  /clock            rosgraph_msgs/Clock       sim time; ROS side runs use_sim_time
  /sim/joint_states sensor_msgs/JointState    12 joints, named; effort = applied torque
  /sim/imu          sensor_msgs/Imu           base orientation / gyro / accelerometer
  /sim/foot_forces  std_msgs/Float32MultiArray  [FR, FL, RR, RL] contact force norms (N)
  /utlidar/cloud    sensor_msgs/PointCloud2   MID-360 returns in the URDF `radar` frame

Subscribes:
  /cmd_vel          geometry_msgs/Twist       written into the base_velocity command term,
                                              clamped to the policy's training limit ranges

`unitree_go/LowState` itself is composed on the ROS side (go2_nav_bringup
sim_lowstate_bridge) from the /sim/* topics: the unitree_go Python bindings are built
for the system Python 3.10 and cannot be imported by Isaac's bundled rclpy.

The task needs the MID-360 scanner in its scene (default task
Go2-Blind-GRU-Mid360-Phase4). There is no mid360 experiment folder, so pass the base
phase's checkpoint explicitly:

    python scripts/ros2/play_ros2.py --task Go2-Blind-GRU-Mid360-Phase4 \
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
parser.add_argument("--task", type=str, default="Go2-Blind-GRU-Mid360-Phase4", help="Name of the task.")
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import time
import torch

from isaacsim.core.utils.extensions import enable_extension

# Load the bundled ROS 2 libraries (py3.11 Humble). Must happen after the app is up and
# before `import rclpy`.
enable_extension("isaacsim.ros2.bridge")

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray

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
# unitree_go/LowState.foot_force order, matched by go2_odometry's `f_unitree[i] for i in
# [1, 0, 3, 2]` -> [FL, FR, RL, RR] against foot frames [FL, FR, RL, RR].
SDK_FOOT_ORDER = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

GRAVITY = 9.81


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
        self.pub_feet = self.create_publisher(Float32MultiArray, "/sim/foot_forces", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, args_cli.cloud_topic, qos_profile_sensor_data)
        self.cmd_vel = np.zeros(3)
        self.cmd_vel_time = None  # wall time of last message
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)

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

    def command(self) -> np.ndarray:
        """Latest command, zeroed once it goes stale (Nav2 died, teleop closed)."""
        if self.cmd_vel_time is None or time.time() - self.cmd_vel_time > args_cli.cmd_vel_timeout:
            return np.zeros(3)
        return self.cmd_vel

    def publish_clock(self, t: float):
        msg = Clock()
        msg.clock = _stamp(t)
        self.pub_clock.publish(msg)

    def publish_state(self, t: float, q, dq, tau, quat_wxyz, gyro, acc, feet):
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

        feet_msg = Float32MultiArray()
        feet_msg.data = feet.tolist()
        self.pub_feet.publish(feet_msg)

    def publish_cloud(self, t: float, points_xyz: np.ndarray):
        msg = self.cloud_msg
        msg.header.stamp = _stamp(t)
        msg.width = len(points_xyz)
        msg.row_step = msg.point_step * msg.width
        msg.data = points_xyz.astype(np.float32).tobytes()
        self.pub_cloud.publish(msg)


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
    # Fall terminations stay -- a fallen robot has to reset, and the ROS side just has
    # to be restarted after one.
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
    contact = scene.sensors["contact_forces"]

    device = env.unwrapped.device
    # articulation joint order -> SDK order
    sdk_joint_ids = [robot.data.joint_names.index(n) for n in SDK_JOINT_NAMES]
    foot_body_ids = [contact.find_bodies(n)[0][0] for n in SDK_FOOT_ORDER]
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
    sim_t = 0.0
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
        with torch.inference_mode():
            cmd_term.vel_command_b[0] = torch.tensor(cmd, dtype=torch.float32, device=device)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        sim_t += dt

        if dones[0]:
            prev_lin_vel_w = robot.data.root_lin_vel_w[0].clone()
            print("[WARN] Environment reset (fall). Odometry/SLAM state on the ROS side is now invalid; restart it.")

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
        feet = contact.data.net_forces_w[0, foot_body_ids].norm(dim=-1).cpu().numpy()

        bridge.publish_clock(sim_t)
        bridge.publish_state(sim_t, q, dq, tau, quat.cpu().numpy(), gyro, acc_b, feet)

        # -- point cloud, in the `radar` (mount) frame --
        hits_w = scanner.data.ray_hits_w[0]
        sensor_pos = scanner._get_true_sensor_pos()[0]
        base_quat = scanner.data.quat_w[0]
        rel_w = hits_w - sensor_pos
        finite = torch.isfinite(rel_w).all(dim=-1)
        dist = rel_w.norm(dim=-1)
        keep = finite & (dist >= min_range) & (dist < max_range - 1e-3)
        rel_w = rel_w[keep]
        p_base = quat_apply_inverse(base_quat.expand(len(rel_w), 4), rel_w)
        p_radar = quat_apply_inverse(mount_quat.expand(len(rel_w), 4), p_base)
        bridge.publish_cloud(sim_t, p_radar.cpu().numpy())

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
