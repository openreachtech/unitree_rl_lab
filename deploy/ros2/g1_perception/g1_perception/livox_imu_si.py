"""/livox/imu (g, frame livox_frame) -> /livox/imu_si (m/s^2, frame livox_imu_frame).

livox_ros_driver2 (and unitree_mujoco -l 1, which copies its output) reports acceleration
in g; DLIO expects m/s^2. Ported from tsubame_perception.livox_imu_si.
"""
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

STANDARD_GRAVITY = 9.80665


class LivoxImuSi(Node):
    def __init__(self):
        super().__init__("livox_imu_si")
        self.pub = self.create_publisher(Imu, "/livox/imu_si", qos_profile_sensor_data)
        self.create_subscription(Imu, "/livox/imu", self.on_imu, 256)

    def on_imu(self, msg: Imu) -> None:
        msg.header.frame_id = "livox_imu_frame"  # DLIO frames/imu
        a = msg.linear_acceleration
        a.x, a.y, a.z = a.x * STANDARD_GRAVITY, a.y * STANDARD_GRAVITY, a.z * STANDARD_GRAVITY
        msg.orientation_covariance[0] = -1.0  # no orientation (REP 145)
        self.pub.publish(msg)
