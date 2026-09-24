#!/usr/bin/env python3
"""VLFM exploration node -- M4: frontier-only (value map stubbed to uniform).

Loop (paper section IV): initialize with a full in-place turn, then repeatedly
pick the best frontier of the live /map and send it to Nav2, until no frontier
of meaningful size remains. Robot-agnostic: consumes /map + TF, produces
NavigateToPose goals and (during the initial spin only) /cmd_vel.

    ros2 run vlfm_nav vlfm_node --ros-args -p use_sim_time:=true

Observability: state transitions are logged; /vlfm/markers (RViz MarkerArray)
shows frontier cells (cyan), candidate goals (blue), current goal (green).
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from vlfm_nav.decision import FrontierSelector
from vlfm_nav.frontier import clearance_cells, find_clusters, goal_for_cluster
from vlfm_nav.nav_bridge import NavBridge, NavState
from vlfm_nav.value_map import UniformValueMap


class VlfmNode(Node):
    def __init__(self):
        super().__init__("vlfm")
        p = self.declare_parameters(
            namespace="",
            parameters=[
                ("base_frame", "base"),
                ("min_cluster_cells", 8),
                ("min_clearance_m", 0.40),
                ("goal_search_radius_m", 0.9),
                ("goal_timeout_s", 90.0),
                ("spin_speed", 0.5),
                ("spin_duration_s", 14.0),
                ("done_patience", 3),
            ],
        )
        self._pget = lambda n: self.get_parameter(n).value

        self._tf_buf = tf2_ros.Buffer()
        self._tf = tf2_ros.TransformListener(self._tf_buf, self)
        self._nav = NavBridge(self)
        self._selector = FrontierSelector()
        self._value_map = UniformValueMap()

        self._grid = None
        self._grid_meta = None  # (resolution, origin_x, origin_y)
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/vlfm/markers", 1)

        self._state = "WAIT"
        self._spin_end = None
        self._goal_xy = None
        self._goal_deadline = None
        self._empty_cycles = 0
        self._goals_sent = 0
        self._goals_reached = 0

        self._spin_timer = self.create_timer(0.1, self._spin_tick)
        self._timer = self.create_timer(1.0, self._tick)
        self.get_logger().info("VLFM exploration node up (M4: frontier-only).")

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid):
        self._grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._grid_meta = (
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )

    def _robot_xy(self) -> np.ndarray | None:
        try:
            tr = self._tf_buf.lookup_transform("map", self._pget("base_frame"), rclpy.time.Time())
            return np.array([tr.transform.translation.x, tr.transform.translation.y])
        except Exception:
            return None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- state machine
    def _transition(self, state: str, why: str):
        self.get_logger().info(f"{self._state} -> {state}: {why}")
        self._state = state

    def _spin_tick(self):
        if self._state != "SPIN":
            return
        cmd = Twist()
        cmd.angular.z = float(self._pget("spin_speed"))
        self._cmd_pub.publish(cmd)

    def _tick(self):
        if self._state == "WAIT":
            if self._grid is not None and self._robot_xy() is not None and self._nav.server_ready():
                self._spin_end = self._now() + float(self._pget("spin_duration_s"))
                self._transition("SPIN", "map/TF/Nav2 ready; initial 360")
            else:
                self.get_logger().info(
                    f"WAIT: map={self._grid is not None}"
                    f" tf={self._robot_xy() is not None} nav={self._nav.server_ready()}",
                    throttle_duration_sec=10.0,
                )
        elif self._state == "SPIN":
            if self._now() >= self._spin_end:
                self._cmd_pub.publish(Twist())  # stop
                self._transition("SELECT", "spin complete")
        elif self._state == "SELECT":
            self._select()
        elif self._state == "NAVIGATE":
            self._monitor()
        # DONE: terminal

    def _select(self):
        robot = self._robot_xy()
        if robot is None or self._grid is None:
            return
        grid = self._grid
        res, ox, oy = self._grid_meta
        clearance = clearance_cells(grid)
        min_clear_cells = float(self._pget("min_clearance_m")) / res
        search_cells = int(float(self._pget("goal_search_radius_m")) / res)

        clusters = find_clusters(grid, int(self._pget("min_cluster_cells")))
        goals, sizes = [], []
        for c in clusters:
            g = goal_for_cluster(c, grid, clearance, min_clear_cells, search_cells)
            if g is not None:
                goals.append([ox + (g[1] + 0.5) * res, oy + (g[0] + 0.5) * res])
                sizes.append(c.size)
        goals = np.array(goals) if goals else np.zeros((0, 2))
        self._publish_markers(clusters, goals, res, ox, oy)

        if len(goals) == 0:
            self._empty_cycles += 1
            if self._empty_cycles >= int(self._pget("done_patience")):
                self._finish("no frontiers left")
            return
        idx = self._selector.choose(goals, self._value_map.score(goals), robot, self._now())
        if idx is None:
            self._empty_cycles += 1
            if self._empty_cycles >= int(self._pget("done_patience")):
                self._finish("all remaining frontiers blacklisted or adjacent")
            return
        self._empty_cycles = 0

        gx, gy = goals[idx]
        yaw = math.atan2(gy - robot[1], gx - robot[0])
        self._goal_xy = (gx, gy)
        self._goal_deadline = self._now() + float(self._pget("goal_timeout_s"))
        self._nav.go_to(gx, gy, yaw)
        self._goals_sent += 1
        self._transition(
            "NAVIGATE",
            f"goal {self._goals_sent}: ({gx:.2f}, {gy:.2f}), {len(goals)} frontiers"
            f" (largest {max(sizes)} cells)",
        )

    def _monitor(self):
        s = self._nav.state
        if s is NavState.SUCCEEDED:
            self._goals_reached += 1
            self._transition("SELECT", "goal reached")
        elif s in (NavState.ABORTED, NavState.REJECTED):
            self._selector.blacklist(*self._goal_xy, self._now())
            self._transition("SELECT", f"goal {s.value}; blacklisted")
        elif self._now() > self._goal_deadline:
            self._nav.cancel()
            self._selector.blacklist(*self._goal_xy, self._now())
            self._transition("SELECT", "goal timeout; blacklisted")

    def _finish(self, why: str):
        known = int((self._grid != -1).sum()) if self._grid is not None else 0
        self._transition("DONE", why)
        self.get_logger().info(
            f"Exploration finished: {self._goals_reached}/{self._goals_sent} goals reached,"
            f" {known} known cells ({known * self._grid_meta[0]**2:.1f} m^2)."
        )

    # ------------------------------------------------------------------ markers
    def _publish_markers(self, clusters, goals, res, ox, oy):
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        def base_marker(mid, mtype):
            m = Marker()
            m.header.frame_id = "map"
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        m = base_marker(1, Marker.POINTS)
        m.scale.x = m.scale.y = res
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 0.8, 0.8
        for c in clusters:
            for row, col in c.cells[:: max(1, len(c.cells) // 200)]:
                m.points.append(_pt(ox + (col + 0.5) * res, oy + (row + 0.5) * res))
        arr.markers.append(m)

        m = base_marker(2, Marker.SPHERE_LIST)
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.2, 1.0, 0.9
        for gx, gy in goals:
            m.points.append(_pt(gx, gy))
        arr.markers.append(m)

        if self._goal_xy is not None:
            m = base_marker(3, Marker.SPHERE)
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.g, m.color.a = 1.0, 0.9
            m.pose.position.x, m.pose.position.y = self._goal_xy
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def _pt(x, y):
    p = Point()
    p.x, p.y = float(x), float(y)
    return p


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
