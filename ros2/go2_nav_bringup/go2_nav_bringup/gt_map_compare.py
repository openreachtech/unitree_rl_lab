#!/usr/bin/env python3
"""Publish the ground-truth floorplan as /gt_map and score /map against it.

Sim-only evaluation tooling for the Go2-Blind-GRU-Mid360-Explore world. The GT
occupancy grid is rasterized from the same layout MeshIndoorRoomsTerrainCfg bakes
into the terrain (keep LAYOUT below in sync with
unitree_rl_lab/tasks/locomotion/terrains.py -- one deterministic floorplan, two
representations).

Alignment: the SLAM map frame is anchored at the robot's start pose, the GT lives
in world coordinates. T_map_world = T_map_base * inv(T_world_base), taken from TF
(map->base) and /sim/gt_odom (world->base, published by play_ros2.py regardless of
odometry mode). The GT grid is published in the map frame with that transform in
its origin pose, so in Foxglove /gt_map and /map overlay directly -- alignment
error is visible as walls not coinciding.

Metrics, logged whenever a new /map arrives (throttled):
  wall recall     GT wall cells that have a SLAM occupied cell within `tol`
  wall precision  SLAM occupied cells within `tol` of a GT wall
  coverage        GT free cells known (not -1) in the SLAM map

    ros2 run go2_nav_bringup gt_map_compare --ros-args -p use_sim_time:=true
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy import ndimage

RES = 0.05

# ---- floorplan, mirroring MeshIndoorRoomsTerrainCfg defaults -------------------
SX = SY = 12.0
WALL_T = 0.10
DOOR = 1.5
TILE_CORNER_WORLD = (-SX / 2.0, -SY / 2.0)  # single tile centered on the world origin
FURNITURE = [  # (x-frac, y-frac, w, d)
    (0.125, 0.375, 0.8, 0.8),
    (0.790, 0.150, 1.2, 0.6),
    (0.375, 0.850, 0.6, 0.6),
    (0.630, 0.875, 1.0, 0.8),
]


def build_gt_grid() -> np.ndarray:
    """Occupancy in tile coordinates: 100 occupied, 0 free. (H, W), row = y."""
    w, h = int(SX / RES), int(SY / RES)
    g = np.zeros((h, w), dtype=np.int8)

    def fill(x0, y0, x1, y1):
        g[int(y0 / RES) : int(np.ceil(y1 / RES)), int(x0 / RES) : int(np.ceil(x1 / RES))] = 100

    t = WALL_T
    fill(0, 0, SX, t)
    fill(0, SY - t, SX, SY)
    fill(0, 0, t, SY)
    fill(SX - t, 0, SX, SY)

    # N-S wall at x = SX/2, doors centered at 0.26*SY and 0.74*SY
    for y0, y1 in [(0, 0.26 * SY - DOOR / 2), (0.26 * SY + DOOR / 2, 0.74 * SY - DOOR / 2), (0.74 * SY + DOOR / 2, SY)]:
        fill(SX / 2 - t / 2, y0, SX / 2 + t / 2, y1)
    # E-W wall at y = SY/2, doors centered at 0.24*SX and 0.76*SX
    for x0, x1 in [(0, 0.24 * SX - DOOR / 2), (0.24 * SX + DOOR / 2, 0.76 * SX - DOOR / 2), (0.76 * SX + DOOR / 2, SX)]:
        fill(x0, SY / 2 - t / 2, x1, SY / 2 + t / 2)

    for fx, fy, fw, fd in FURNITURE:
        cx, cy = fx * SX, fy * SY
        fill(cx - fw / 2, cy - fd / 2, cx + fw / 2, cy + fd / 2)
    return g


def _yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class GtMapCompare(Node):
    def __init__(self):
        super().__init__("gt_map_compare")
        self.declare_parameter("tolerance_m", 0.10)
        self._gt = build_gt_grid()
        self._tf_buf = tf2_ros.Buffer()
        self._tf = tf2_ros.TransformListener(self._tf_buf, self)
        self._gt_odom = None
        self._slam = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(OccupancyGrid, "/gt_map", latched)
        self.create_subscription(Odometry, "/sim/gt_odom", self._on_gt_odom, 10)
        self.create_subscription(OccupancyGrid, "/map", self._on_map, latched)
        self.create_timer(5.0, self._tick)
        self._published = False

    def _on_gt_odom(self, msg):
        self._gt_odom = msg

    def _on_map(self, msg):
        self._slam = msg

    def _map_from_world(self):
        """SE2 T_map_world, or None."""
        if self._gt_odom is None:
            return None
        try:
            tr = self._tf_buf.lookup_transform("map", "base", rclpy.time.Time())
        except Exception:
            return None
        t, q = tr.transform.translation, tr.transform.rotation
        mb = np.array([t.x, t.y, _yaw(q.w, q.x, q.y, q.z)])
        p, o = self._gt_odom.pose.pose.position, self._gt_odom.pose.pose.orientation
        wb = np.array([p.x, p.y, _yaw(o.w, o.x, o.y, o.z)])
        yaw = mb[2] - wb[2]
        c, s = math.cos(yaw), math.sin(yaw)
        tx = mb[0] - (c * wb[0] - s * wb[1])
        ty = mb[1] - (s * wb[0] + c * wb[1])
        return np.array([tx, ty, yaw])

    def _tick(self):
        T = self._map_from_world()
        if T is None:
            self.get_logger().info("waiting for TF map->base and /sim/gt_odom", throttle_duration_sec=10.0)
            return
        self._publish_gt(T)
        if self._slam is not None:
            self._score(T)

    def _publish_gt(self, T):
        tx, ty, yaw = T
        c, s = math.cos(yaw), math.sin(yaw)
        cx, cy = TILE_CORNER_WORLD
        gx = tx + c * cx - s * cy
        gy = ty + s * cx + c * cy
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = RES
        msg.info.width = self._gt.shape[1]
        msg.info.height = self._gt.shape[0]
        msg.info.origin.position.x = gx
        msg.info.origin.position.y = gy
        msg.info.origin.orientation.z = math.sin(yaw / 2)
        msg.info.origin.orientation.w = math.cos(yaw / 2)
        msg.data = self._gt.flatten().tolist()
        self._pub.publish(msg)
        if not self._published:
            self._published = True
            self.get_logger().info(f"/gt_map published, aligned to map frame (yaw {math.degrees(yaw):.1f} deg)")

    def _score(self, T):
        m = self._slam
        slam = np.array(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        res, ox, oy = m.info.resolution, m.info.origin.position.x, m.info.origin.position.y
        tol = float(self.get_parameter("tolerance_m").value)

        # world coords of GT cells -> map frame -> slam indices
        tx, ty, yaw = T
        c, s = math.cos(yaw), math.sin(yaw)
        ys, xs = np.mgrid[0 : self._gt.shape[0], 0 : self._gt.shape[1]]
        wx = TILE_CORNER_WORLD[0] + (xs + 0.5) * RES
        wy = TILE_CORNER_WORLD[1] + (ys + 0.5) * RES
        mx = tx + c * wx - s * wy
        my = ty + s * wx + c * wy
        ix = ((mx - ox) / res).astype(int)
        iy = ((my - oy) / res).astype(int)
        inside = (ix >= 0) & (ix < m.info.width) & (iy >= 0) & (iy < m.info.height)

        occ_slam = slam >= 65
        d_occ = ndimage.distance_transform_edt(~occ_slam) * res  # dist to slam wall
        gt_wall = (self._gt == 100) & inside
        recall = float((d_occ[iy[gt_wall], ix[gt_wall]] <= tol).mean()) if gt_wall.any() else 0.0

        # precision: slam occupied cells near a GT wall (distance field on GT, in map frame -> approximate by inverse mapping)
        d_gt = ndimage.distance_transform_edt(self._gt != 100) * RES
        sy_, sx_ = np.nonzero(occ_slam)
        smx = ox + (sx_ + 0.5) * res
        smy = oy + (sy_ + 0.5) * res
        # map -> world
        cw, sw = math.cos(-yaw), math.sin(-yaw)
        wx2 = cw * (smx - tx) - sw * (smy - ty)
        wy2 = sw * (smx - tx) + cw * (smy - ty)
        gx2 = ((wx2 - TILE_CORNER_WORLD[0]) / RES).astype(int)
        gy2 = ((wy2 - TILE_CORNER_WORLD[1]) / RES).astype(int)
        ins2 = (gx2 >= 0) & (gx2 < self._gt.shape[1]) & (gy2 >= 0) & (gy2 < self._gt.shape[0])
        prec = float((d_gt[gy2[ins2], gx2[ins2]] <= tol).mean()) if ins2.any() else 0.0

        gt_free = (self._gt == 0) & inside
        known = slam[iy[gt_free], ix[gt_free]] != -1
        self.get_logger().info(
            f"wall recall {recall * 100:.1f}% | wall precision {prec * 100:.1f}%"
            f" | GT free space known {known.mean() * 100:.1f}% (tol {tol} m)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GtMapCompare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
