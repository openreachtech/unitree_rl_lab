#!/usr/bin/env python3
"""VLFM core node (skeleton -- M4/M5).

Robot-agnostic by contract (doc/design/vlfm_nav.md §3): consumes /map, TF and the
RGB camera, produces Nav2 goals. No robot-specific types anywhere in this package.

Planned structure (paper: VLFM, Yokoyama et al., ICRA 2024):
  frontier.py   -- frontier detection/clustering on the occupancy grid (pure logic,
                   numpy in / numpy out, ROS-free so it can be unit-tested)
  value_map.py  -- language-grounded value map from VLM scores (VLM runs in a
                   separate process behind a service; stub scorer until M5)
  decision.py   -- explore / approach / done state machine
  nav_bridge.py -- nav2_simple_commander wrapper (goToPose, cancel, feedback)

M4 milestone runs frontier-only exploration: nearest/most-informative frontier with
the value map stubbed to a constant, which reduces to classic frontier exploration.
"""

import rclpy
from rclpy.node import Node


class VlfmNode(Node):
    def __init__(self):
        super().__init__("vlfm")
        self.get_logger().info("vlfm_nav skeleton -- exploration logic lands in M4.")


def main(args=None):
    rclpy.init(args=args)
    node = VlfmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
