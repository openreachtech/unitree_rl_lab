#!/usr/bin/env python3
"""Compose unitree_go/LowState from the sim's standard-message topics.

The Isaac side (scripts/ros2/play_ros2.py) runs on the bundled py3.11 rclpy, which
cannot import the unitree_go bindings (built for the system py3.10). So the sim
publishes plain sensor_msgs and this node -- running on the system Python -- folds
them back into the /lowstate that go2_odometry consumes on the real robot. With it,
the odometry stack is byte-for-byte the same in sim and on hardware.

Inputs (all published together each env step, no synchronization needed beyond
"latest wins" -- LowState is emitted on each /sim/joint_states):
  /sim/joint_states  sensor_msgs/JointState      named joints; effort = torque
  /sim/imu           sensor_msgs/Imu             orientation / gyro / accelerometer
  /sim/foot_forces   std_msgs/Float32MultiArray  [FR, FL, RR, RL] contact norms (N)

Output:
  /lowstate          unitree_go/LowState
"""

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowState

# unitree_sdk2 motor index order
SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class SimLowStateBridge(Node):
    def __init__(self):
        super().__init__("sim_lowstate_bridge")
        self._imu = None
        self._feet = None
        self._joint_index = None  # sdk idx -> position in the JointState arrays

        self._pub = self.create_publisher(LowState, "/lowstate", 10)
        self.create_subscription(Imu, "/sim/imu", self._on_imu, 10)
        self.create_subscription(Float32MultiArray, "/sim/foot_forces", self._on_feet, 10)
        self.create_subscription(JointState, "/sim/joint_states", self._on_joints, 10)

    def _on_imu(self, msg: Imu):
        self._imu = msg

    def _on_feet(self, msg: Float32MultiArray):
        self._feet = msg.data

    def _on_joints(self, msg: JointState):
        if self._imu is None or self._feet is None:
            return
        if self._joint_index is None:
            names = list(msg.name)
            try:
                self._joint_index = [names.index(n) for n in SDK_JOINT_NAMES]
            except ValueError as e:
                self.get_logger().error(f"/sim/joint_states is missing a joint: {e}")
                return
            self.get_logger().info("First full state received; publishing /lowstate.")

        out = LowState()
        for sdk_i, src_i in enumerate(self._joint_index):
            m = out.motor_state[sdk_i]
            m.q = float(msg.position[src_i])
            m.dq = float(msg.velocity[src_i])
            m.tau_est = float(msg.effort[src_i])

        imu = self._imu
        out.imu_state.quaternion = [
            float(imu.orientation.w),
            float(imu.orientation.x),
            float(imu.orientation.y),
            float(imu.orientation.z),
        ]
        out.imu_state.gyroscope = [
            float(imu.angular_velocity.x),
            float(imu.angular_velocity.y),
            float(imu.angular_velocity.z),
        ]
        out.imu_state.accelerometer = [
            float(imu.linear_acceleration.x),
            float(imu.linear_acceleration.y),
            float(imu.linear_acceleration.z),
        ]
        out.imu_state.rpy = list(_quat_to_rpy(*out.imu_state.quaternion))
        # int16, same convention the hardware uses (go2_odometry thresholds at >= 20)
        out.foot_force = [int(max(-32768, min(32767, round(f)))) for f in self._feet]
        out.tick = (msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000) % (2**32)

        self._pub.publish(out)


def _quat_to_rpy(w: float, x: float, y: float, z: float):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = SimLowStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
