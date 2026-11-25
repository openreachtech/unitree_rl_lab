"""livox_imu_si and g1_state in one process.

Each process is one DDS participant, and on lo (no multicast) CycloneDDS gives a domain only
about 10 of them. unitree_mujoco and g1_ctrl need theirs too, so the small nodes share one.
"""
import rclpy
from rclpy.executors import SingleThreadedExecutor

from .g1_state import G1State
from .livox_imu_si import LivoxImuSi


def main():
    rclpy.init()
    executor = SingleThreadedExecutor()
    for node in (LivoxImuSi(), G1State()):
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
