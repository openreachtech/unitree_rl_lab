"""Puts the G1 model where DLIO says the torso is, for rviz.

  /lowstate (unitree_hg/LowState, rt/lowstate over DDS) -> /joint_states (29 joints)
  TF dlio_torso -> pelvis = torso_link -> pelvis (from robot_state_publisher)

DLIO tracks the torso but publishes it as dlio_torso, because robot_state_publisher already
makes torso_link a child of pelvis (via the waist) and a TF frame has one parent. Hanging
pelvis under dlio_torso by the same waist transform gives odom -> dlio_torso -> pelvis -> ...
"""
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from unitree_hg.msg import LowState

# SDK motor order (= g1_29dof URDF joint order).
JOINTS = [
    f"{side}_{j}_joint" for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"{side}_{j}_joint" for side in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]
PERIOD_NS = 20_000_000  # 50 Hz out (lowstate comes at 500 Hz)


class G1State(Node):
    def __init__(self):
        super().__init__("g1_state")
        self.js_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(LowState, "lowstate", self.on_lowstate, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)
        self.last_ns = 0

    def on_lowstate(self, msg):
        now = self.get_clock().now()
        if now.nanoseconds - self.last_ns < PERIOD_NS:
            return
        self.last_ns = now.nanoseconds
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = JOINTS
        js.position = [float(msg.motor_state[i].q) for i in range(len(JOINTS))]
        self.js_pub.publish(js)

        try:
            t = self.tf_buffer.lookup_transform("torso_link", "pelvis", Time())
        except TransformException:
            return
        t.header.frame_id = "dlio_torso"
        t.header.stamp = js.header.stamp
        self.tf_pub.sendTransform(t)
