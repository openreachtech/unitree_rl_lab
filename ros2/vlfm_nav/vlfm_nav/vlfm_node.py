#!/usr/bin/env python3
"""VLFM exploration node -- M4: frontier-only (value map stubbed to uniform).

The loop (paper section IV): initialize with a full in-place turn, then repeatedly
pick the best frontier of the live /map and send it to Nav2, until exploration is
finished. Robot-agnostic: consumes /map + TF + /scan, produces NavigateToPose
goals, /cmd_vel directly in the open-loop states (SPIN, ESCAPE), and
/vlfm/contact_obstacles for the costmaps' contact_layer.

State machine (every arrow is logged as "FROM -> TO: why"):

    WAIT ----> SPIN ----> SELECT <====> NAVIGATE ----> DONE
      ^          (once)      ^             |
      |                      +-- ESCAPE <--+  (stuck watchdog: the ONLY entry)
      +--- goal REJECTED

  WAIT      until /map, TF and Nav2 are ready. "Ready" includes bt_navigator's
            lifecycle state being 'active' -- its action server existing is not
            enough (a half-started one rejects every goal) -- and WAIT drives the
            missing CONFIGURE/ACTIVATE transitions itself when the lifecycle
            manager lost the startup race (_maybe_heal_bt).
  SPIN      one full turn (spin_duration_s * spin_speed) to seed the map; runs
            once per node lifetime, not on every WAIT re-entry.
  SELECT    frontier pipeline over the latest /map (see _select):
            detect -> cluster -> place goals -> score -> filter -> send best.
            Nothing eligible for a while -> one blacklist amnesty, then DONE.
  NAVIGATE  monitors the active goal (see _monitor). The USUAL way a goal ends is
            frontier consumption: the scans made en route reveal what was behind
            the frontier, and the goal -- only ever an excuse to look there -- is
            done without being reached. Arrival (goal_reached_m) is the fallback
            for frontiers that persist (seen through a window). Failures: Nav2
            abort (-> blacklist) and the stuck watchdog (-> ESCAPE). Deliberately
            NO goal timeout and NO progress watchdog: as long as the robot moves,
            it may take any route however long; an unreachable goal ends via
            physical evidence -- contact marks wall the route off, Nav2 aborts.
  ESCAPE    Nav2-free recovery, entered ONLY from the stuck watchdog (see
            _start_escape): raw velocity probe bursts in scan-ordered directions,
            success judged purely by TF displacement.
            Verified-blocked directions become contact obstacles. Nav2's own
            recoveries cannot do this job: they refuse to move from a cell the
            costmap calls lethal, which is exactly where a stuck robot stands.
  DONE      terminal; reached via "no frontiers left" (none detected, or all
            remaining ones blacklisted) or a failed escape (physically stuck).

Two kinds of memory (decision.FrontierSelector), both permanent:
BLACKLIST (failures) -- a goal Nav2 aborted while the robot's own cell was
feasible (an abort with an inscribed/lethal start is the start's fault and
records nothing); a small disc (blacklist_radius_m) that goal selection skips.
VISITED (successes) -- a goal the robot arrived at; its retire_radius_m
neighborhood is excluded from FRONTIER DETECTION itself, so it stops producing
frontier cells and goals entirely. A goal consumed en route leaves no entry --
the robot never stood there; the map records that outcome. Getting stuck en
route is not final either -- escape + contact marks wall the route off until
Nav2 aborts. The one amnesty: when everything is
blacklisted, the most recent entries (likely accrued while the robot's own
position was bad, not the goals' fault) are dropped once before concluding.

    ros2 run vlfm_nav vlfm_node --ros-args -p use_sim_time:=true

Observability: /vlfm/markers shows frontier cells (cyan points), candidate goals
(blue spheres), the current goal (green sphere), contact obstacles (red cubes),
blacklist zones (orange discs, true radius) and visited zones (large translucent
green discs, the areas retired from frontier detection).
"""

from __future__ import annotations

import math
import random

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point, Twist
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from vlfm_nav.decision import FrontierSelector
from vlfm_nav.frontier import (
    exclusion_mask,
    find_clusters,
    goal_for_cluster,
    has_frontier_near,
)
from vlfm_nav.nav_bridge import NavBridge, NavState
from vlfm_nav.value_map import GridValueMap

# Escape probe motions as unit commands (label, vx, vy, wz, priority). Scaled by the
# escape_* speed parameters at runtime. Ordering intent: reversing out is the most
# likely exit (the robot drove IN forward), forward variants can push deeper into
# whatever we hit, and in-place rotation is the most likely to tangle legs while
# wedged -- so rotations go last regardless of what the scan says.
# NO strafe probes: step-response measurement (2026-09-28, Phase4 model_7300)
# showed the policy does not track pure lateral commands AT ALL (vy 0.2-0.4 ->
# 0.00 m/s) -- a strafe probe always "failed" and painted a FALSE contact wall
# on the robot's flank.
ESCAPE_MOTIONS = [
    ("back", -1.0, 0.0, 0.0, 0),
    ("back-left", -1.0, 0.0, 1.0, 0),
    ("back-right", -1.0, 0.0, -1.0, 0),
    ("fwd-left", 1.0, 0.0, 1.0, 2),
    ("fwd-right", 1.0, 0.0, -1.0, 2),
    ("rot-left", 0.0, 0.0, 1.0, 3),
    ("rot-right", 0.0, 0.0, -1.0, 3),
]


class VlfmNode(Node):
    def __init__(self):
        super().__init__("vlfm")
        p = self.declare_parameters(
            namespace="",
            parameters=[
                ("base_frame", "base"),
                # -------- frontier detection & goal selection (SELECT) --------
                # 10 cells @5 cm = 0.5 m of frontier: the Go2's width (0.3 m) plus
                # margin -- the narrowest opening the robot could actually pass, so
                # anything smaller (window-frame slivers, gaps under furniture) is
                # not worth a goal. Discarded cells revive once the map grows them
                # into a longer segment.
                ("min_cluster_cells", 10),
                # Cost-utility selection: score = value + size_weight*min(cells, cap)
                # - 0.1*distance. A full room behind a doorway (hundreds of cells)
                # now outbids a nearby scrap, cutting cross-apartment backtracking.
                ("frontier_size_weight", 0.0015),
                ("frontier_size_cap_cells", 600),
                # The goal IS the frontier cell (goal_for_cluster), so both radii
                # below are centred ON the frontier -- no goal<->frontier gap to span.
                # "Consumed en route": the goal is done when NO frontier remains
                # within this radius of it -- i.e. the scans en route filled the
                # unknown and it stopped being a frontier. A read only; removes
                # nothing, so small is fine.
                ("consumed_radius_m", 0.15),
                # VISITED retire: an arrived-at (unconsumable) frontier excludes this
                # radius from frontier detection so it is never re-selected. Set equal
                # to goal_reached_m (0.40) so "what counted as reached" == "what gets
                # retired": a smaller radius leaves an un-retired ring that choose()
                # skips as already-reached yet detection keeps as frontier, so the
                # robot drifts back to it. Visits are permanent, so a long unconsumable
                # frontier is chewed off 0.40 m per arrival and never grows back.
                ("retire_radius_m", 0.40),
                # -------- goal monitoring (NAVIGATE) --------
                # Deliberately NO goal timeout and NO progress watchdog: a goal that
                # cannot be reached stops the robot sooner or later, and then stuck
                # -> escape -> contact marks wall the route off -> Nav2 aborts ->
                # blacklist, all on physical evidence. (An earlier progress watchdog
                # existed for MPPI spin-dithering; that was a symptom of the Phase4
                # deadband, fixed at the policy/shaping layer, not here.)
                #
                # goal_reached_m is the arrival radius AND the selection floor (a
                # goal closer than this is by definition already reached). 0.40 m:
                # the goal IS a frontier cell (in/near inflation), so the robot must
                # count as arrived while still standing off it in clear space -- it
                # never docks on the frontier itself. Usually consumption ends the
                # goal first; this is the fallback for a frontier that persists
                # (glass/wall) so the robot doesn't grind toward an unreachable cell.
                ("goal_reached_m", 0.40),
                # -------- initial spin --------
                # Slow, deliberately: rotation is the worst case for a lagging LIO
                # (angular error amplifies with range -- the fast 0.9 rad/s spin
                # smeared wall returns 0.5-0.7 m sideways and sprinkled phantom
                # occupied cells around the spawn, 2026-09-29). 0.45 rad/s x 16 s
                # is a bit over one turn. NOTE: assumes a policy that tracks slow
                # rotation (Phase1); Phase4 does not rotate below |0.7|.
                ("spin_speed", -0.45),
                ("spin_duration_s", 16.0),
                # -------- failure memory & termination --------
                ("done_patience", 3),
                # ~half a Go2 (0.4 m long, 0.3 m wide): blocks re-picking the same
                # spot without swallowing neighboring, possibly viable goals.
                # Blacklist entries are PERMANENT (see decision.py): reached =
                # consumed, failed = written off, and "no frontiers left" is the
                # single termination concept.
                ("blacklist_radius_m", 0.15),
                # How much detour a point of VLM value is worth: 1 / 0.25 = 4 m. VLFM's
                # own trade-off works out at ~10 m, which is most of kujiale's diagonal
                # -- enough for a marginally better score to send the robot across the
                # flat and back. 4 m is about "the next room".
                ("distance_cost_per_m", 0.25),
                # Frontier cells jitter as the map grows, so sample a neighbourhood
                # rather than one cell of the value grid.
                ("value_sample_radius_m", 0.5),
                # Amnesty window for the sealed-start case: entries younger than
                # this are dropped after an all-blocked escape (see _select).
                ("amnesty_window_s", 180.0),
                # -------- stuck detection & escape --------
                # Stuck detection + escape. Nav2's own recoveries check the costmap
                # before moving, so from a cell the map calls lethal (wedged into a
                # bookshelf) they all refuse ("Collision Ahead") and the robot never
                # gets out. The escape bypasses Nav2: raw /cmd_vel probe bursts, each
                # judged by the only ground truth available -- did the robot actually
                # move away (TF displacement)? No direction is assumed to work.
                ("stuck_window_s", 5.0),
                ("stuck_min_travel_m", 0.15),
                ("escape_burst_s", 2.5),
                ("escape_resume_s", 1.5),
                ("escape_success_m", 0.20),
                ("escape_max_rounds", 3),
                # -------- contact obstacles --------
                # Verified blocked directions only: an escape probe that produced no
                # displacement, or a stuck window whose mean commanded direction
                # moved nothing. Both are direct physical evidence of an obstacle --
                # often one no sensor shows (glass, a sill below the scan band, a
                # thin shelf). A short 3-point segment is placed at the FOOTPRINT
                # EDGE along that direction (rectangle support: hl*|fwd|+hw*|lat|,
                # plus margin); a dedicated costmap layer (contact_layer, marking
                # only, never raytraced away) keeps the planner off it.
                ("footprint_half_length_m", 0.40),
                ("footprint_half_width_m", 0.20),
                ("contact_mark_margin_m", 0.12),
                # marks expire: stale ones from long-fixed situations otherwise
                # accumulate into phantom pens (see _mark_contact)
                ("contact_ttl_s", 600.0),
                # SELECT finding nothing eligible waits this long before acting (the
                # map may still change); then one amnesty, then DONE (see _select)
                ("all_blocked_wait_s", 45.0),
                # at/above the policy's real deadband (see ESCAPE_MOTIONS note)
                ("escape_vx", 0.8),
                ("escape_vy", 0.0),  # policy cannot strafe; kept for other robots
                ("escape_wz", 0.9),
            ],
        )
        self._pget = lambda n: self.get_parameter(n).value

        self._tf_buf = tf2_ros.Buffer()
        self._tf = tf2_ros.TransformListener(self._tf_buf, self)
        self._nav = NavBridge(self)
        self._selector = FrontierSelector(
            blacklist_radius_m=float(self._pget("blacklist_radius_m")),
            distance_cost_per_m=float(self._pget("distance_cost_per_m")),
        )
        # Reads the two grids the VLM node publishes. With no VLM running it returns 0
        # for everything, leaving selection to distance -- so the stack needs no switch
        # to run without the model.
        self._value_map = GridValueMap(radius_m=float(self._pget("value_sample_radius_m")))
        self.create_subscription(
            OccupancyGrid, "/vlfm/value_map",
            lambda m: self._value_map.set_value(
                np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width),
                (m.info.origin.position.x, m.info.origin.position.y), m.info.resolution), 1)
        self.create_subscription(
            OccupancyGrid, "/vlfm/value_confidence",
            lambda m: self._value_map.set_confidence(
                np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width),
                (m.info.origin.position.x, m.info.origin.position.y), m.info.resolution), 1)

        self._grid = None
        self._grid_meta = None  # (resolution, origin_x, origin_y)
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Nav2's global costmap, only to judge WHOSE fault an abort was: NavFn
        # fails instantly for EVERY goal when the robot's own cell is inscribed/
        # lethal (start infeasible). Such an abort says nothing about the goal.
        self._costmap = None
        self._costmap_meta = None
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self._on_costmap,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/vlfm/markers", 1)
        self._contact_pub = self.create_publisher(PointCloud2, "/vlfm/contact_obstacles", 1)
        self._contact_pts: list[tuple[np.ndarray, float]] = []  # (map xy, stamp); expire after contact_ttl_s
        self._all_blocked_since = None  # when SELECT first found nothing eligible
        self._amnesty_used = False  # one blacklist amnesty per all-blocked episode
        self.create_timer(1.0, self._publish_contacts)
        self._scan = None
        self.create_subscription(
            LaserScan, "/scan", lambda m: setattr(self, "_scan", m), qos_profile_sensor_data
        )
        # Recent commanded velocities (whoever publishes -- MPPI during NAVIGATE).
        # Read only at stuck detection: the mean commanded direction over the stuck
        # window is where the robot was pushed and did not move.
        self._cmd_log: list[tuple[float, float, float]] = []  # (sim time, vx, vy)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)

        self._state = "WAIT"
        self._spin_end = None
        self._goal_xy = None
        self._empty_cycles = 0
        self._goals_sent = 0
        self._goals_reached = 0
        self._travel_log: list[tuple[float, np.ndarray]] = []  # (sim time, xy) while NAVIGATE
        self._escape = None  # active escape attempt state, see _start_escape
        self._spun = False  # the initial 360 runs once, not on every WAIT re-entry

        # bt_navigator lifecycle supervision. server_ready() only proves the action
        # server EXISTS (it is created at configure); a bt_navigator that never
        # activated rejects every goal. Seen 2026-09-26: its change_state response to
        # the lifecycle manager timed out under start-up CPU load, the manager gave
        # up, and the node sat configured-but-inactive while exploration blindly
        # escaped its way around the apartment. So WAIT checks the actual lifecycle
        # state, and heals a stuck node by driving the missing transitions itself.
        self._bt_state = None
        self._bt_state_pending = False
        self._bt_heal_at = None
        self._gs_cli = self.create_client(GetState, "/bt_navigator/get_state")
        self._cs_cli = self.create_client(ChangeState, "/bt_navigator/change_state")

        self._cmd_timer = self.create_timer(0.1, self._cmd_tick)
        self._timer = self.create_timer(0.5, self._tick)
        self.get_logger().info("VLFM exploration node up (M4: frontier-only).")

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid):
        self._grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._grid_meta = (
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )

    def _on_costmap(self, msg: OccupancyGrid):
        self._costmap = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._costmap_meta = (
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )

    def _start_infeasible(self) -> bool:
        """True when the robot's own cell in the global costmap is inscribed or
        lethal -- the planner then fails for ANY goal, instantly."""
        robot = self._robot_xy()
        if robot is None or self._costmap is None:
            return False
        res, ox, oy = self._costmap_meta
        r = int((robot[1] - oy) / res)
        c = int((robot[0] - ox) / res)
        if not (0 <= r < self._costmap.shape[0] and 0 <= c < self._costmap.shape[1]):
            return False
        return int(self._costmap[r, c]) >= 99

    def _robot_xy(self) -> np.ndarray | None:
        pose = self._robot_pose()
        return None if pose is None else pose[0]

    def _robot_pose(self) -> tuple[np.ndarray, float] | None:
        try:
            tr = self._tf_buf.lookup_transform("map", self._pget("base_frame"), rclpy.time.Time())
            q = tr.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return np.array([tr.transform.translation.x, tr.transform.translation.y]), yaw
        except Exception:
            return None

    def _on_cmd(self, msg: Twist):
        now = self._now()
        self._cmd_log.append((now, msg.linear.x, msg.linear.y))
        window = float(self._pget("stuck_window_s"))
        self._cmd_log = [c for c in self._cmd_log if now - c[0] <= window]

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- state machine
    def _transition(self, state: str, why: str):
        self.get_logger().info(f"{self._state} -> {state}: {why}")
        self._state = state

    def _cmd_tick(self):
        """10 Hz command source for the open-loop states (SPIN, ESCAPE)."""
        if self._state == "SPIN":
            cmd = Twist()
            cmd.angular.z = float(self._pget("spin_speed"))
            self._cmd_pub.publish(cmd)
        elif self._state == "ESCAPE" and self._escape is not None:
            self._escape_tick()

    def _tick(self):
        if self._state == "WAIT":
            self._poll_bt_state()
            nav_ok = self._nav.server_ready() and self._bt_state == "active"
            if self._grid is not None and self._robot_xy() is not None and nav_ok:
                self._bt_heal_at = None
                if self._spun:
                    self._transition("SELECT", "stack healthy again")
                else:
                    self._spin_end = self._now() + float(self._pget("spin_duration_s"))
                    self._transition("SPIN", "map/TF/Nav2 ready; initial 360")
            else:
                self._maybe_heal_bt()
                self.get_logger().info(
                    f"WAIT: map={self._grid is not None}"
                    f" tf={self._robot_xy() is not None} nav={self._nav.server_ready()}"
                    f" bt_state={self._bt_state}",
                    throttle_duration_sec=10.0,
                )
        elif self._state == "SPIN":
            if self._now() >= self._spin_end:
                self._cmd_pub.publish(Twist())  # stop
                self._spun = True
                self._transition("SELECT", "spin complete")
        elif self._state == "SELECT":
            self._select()
        elif self._state == "NAVIGATE":
            self._monitor()
        # ESCAPE: driven by _cmd_tick; DONE: terminal

    def _select(self):
        robot = self._robot_xy()
        if robot is None or self._grid is None:
            return
        grid = self._grid
        res, ox, oy = self._grid_meta

        visited = self._selector.visited_entries()
        excl = None
        if visited:
            cells = [(int((vy - oy) / res), int((vx - ox) / res)) for vx, vy in visited]
            excl = exclusion_mask(grid.shape, cells, int(float(self._pget("retire_radius_m")) / res))
        clusters = find_clusters(grid, int(self._pget("min_cluster_cells")), exclusion=excl)
        robot_cell = np.array([(robot[1] - oy) / res, (robot[0] - ox) / res])
        goals, sizes = [], []
        for c in clusters:
            g = goal_for_cluster(c, near=robot_cell)
            goals.append([ox + (g[1] + 0.5) * res, oy + (g[0] + 0.5) * res])
            sizes.append(c.size)
        goals = np.array(goals) if goals else np.zeros((0, 2))

        if len(goals) == 0:
            self._publish_markers(clusters, goals, res, ox, oy)
            self._empty_cycles += 1
            if self._empty_cycles >= int(self._pget("done_patience")):
                self._finish("no frontiers left")
            return
        values = self._value_map.score(goals) + float(self._pget("frontier_size_weight")) * np.minimum(
            np.array(sizes, dtype=float), float(self._pget("frontier_size_cap_cells"))
        )
        idx = self._selector.choose(goals, values, robot, float(self._pget("goal_reached_m")))
        if idx is None:
            self._publish_markers(clusters, goals, res, ox, oy)
            # Everything blacklisted. Two possible truths: (a) the robot's own
            # position was temporarily bad (sealed by inflation/contact marks) and
            # the recent failures were the START's fault -- grant one amnesty on
            # the recent entries and retry; (b) the remaining frontiers really are
            # all written off -- with permanent blacklists nothing will change, so
            # exploration is done. ESCAPE is deliberately NOT an option here: it is
            # reserved for physical stuck (its single entry point).
            now = self._now()
            if self._all_blocked_since is None:
                self._all_blocked_since = now
            elif now - self._all_blocked_since > float(self._pget("all_blocked_wait_s")):
                self._all_blocked_since = None
                if self._amnesty_used:
                    self._finish("no frontiers left (all remaining blacklisted)")
                    return
                self._amnesty_used = True
                self._selector.clear_recent(now, float(self._pget("amnesty_window_s")))
                self.get_logger().warn(
                    f"all goals blacklisted; amnesty on the last {self._pget('amnesty_window_s'):.0f} s of entries"
                )
                return
            self.get_logger().info(
                f"all {len(goals)} frontier goals blacklisted or already reached",
                throttle_duration_sec=30.0,
            )
            return
        self._empty_cycles = 0
        self._all_blocked_since = None
        self._amnesty_used = False

        gx, gy = goals[idx]
        yaw = math.atan2(gy - robot[1], gx - robot[0])
        self._goal_xy = (gx, gy)
        self._nav.go_to(gx, gy, yaw)
        self._goals_sent += 1
        # NOTE: _travel_log is NOT reset here -- stuck detection must span goal
        # churn. When the start cell is infeasible every goal insta-aborts in ~2 s,
        # so a per-goal window would never fill; "hasn't moved for stuck_window_s"
        # is the truth regardless of how many goals were tried in that time.
        # markers go out AFTER the new goal is set -- published before selection,
        # the green sphere always showed the PREVIOUS goal (one update behind)
        self._publish_markers(clusters, goals, res, ox, oy)
        self._transition(
            "NAVIGATE",
            f"goal {self._goals_sent}: ({gx:.2f}, {gy:.2f}), {len(goals)} frontiers"
            f" (largest {max(sizes)} cells)",
        )

    def _goal_reached(self, why: str, arrived: bool):
        """Wrap up the current goal and select the next immediately.

        ``arrived`` = the robot actually stood at the goal: the spot is VISITED --
        recorded separately from the blacklist, and excluded at the FRONTIER level
        (its retire_radius_m neighborhood stops being frontier at all -- that radius
        reaches the frontier the goal was placed off of).
        Without this, two adjacent unconsumable frontiers ping-pong forever --
        observed 2026-09-29: goals 48-62 alternated between two spots 0.35 m apart
        with the map not growing a single cell. A goal consumed EN ROUTE leaves no
        entry: the robot never stood there; the map already records the outcome."""
        self._goals_reached += 1
        if arrived:
            self._selector.visit(*self._goal_xy, self._now())
            why += " -> visited; frontier around it retired"
        self._transition("SELECT", why)
        self._select()  # no dead tick: pick the next frontier immediately

    def _frontier_consumed(self) -> bool:
        """True when no frontier cell remains near the current goal."""
        if self._grid is None or self._goal_xy is None:
            return False
        res, ox, oy = self._grid_meta
        row = int((self._goal_xy[1] - oy) / res)
        col = int((self._goal_xy[0] - ox) / res)
        radius = int(float(self._pget("consumed_radius_m")) / res)
        return not has_frontier_near(self._grid, row, col, radius)

    def _monitor(self):
        s = self._nav.state
        robot = self._robot_xy()
        if s in (NavState.ACTIVE, NavState.PENDING) and self._frontier_consumed():
            # the scans made en route revealed what was behind the frontier; the
            # goal has served its purpose without being reached
            self._nav.cancel()
            self._goal_reached("frontier consumed en route", arrived=False)
            return
        # Arrival fallback: standing AT the goal while its frontier persists (e.g.
        # visible through a window) -- nothing more to gain here, move on.
        if (
            s in (NavState.ACTIVE, NavState.PENDING)
            and robot is not None
            and self._goal_xy is not None
            and float(np.hypot(robot[0] - self._goal_xy[0], robot[1] - self._goal_xy[1]))
            < float(self._pget("goal_reached_m"))
        ):
            self._nav.cancel()
            self._goal_reached("goal reached (proximity)", arrived=True)
            return
        if s is NavState.SUCCEEDED:
            self._goal_reached("goal reached", arrived=True)
        elif s is NavState.ABORTED:
            if self._start_infeasible():
                # The START is the problem (robot cell inscribed/lethal): the goal
                # was never genuinely attempted, so it is NOT blacklisted. But if
                # the robot has ALSO not moved for stuck_window_s, it is wedged --
                # every goal insta-aborts and no NAVIGATE runs long enough for the
                # ordinary stuck check, so drive the escape from here.
                if self._stuck():
                    self._begin_escape_from_stuck()
                    return
                self._transition("SELECT", "goal aborted -- START infeasible, goal NOT blacklisted")
            else:
                self._selector.blacklist(*self._goal_xy, self._now())
                self._transition("SELECT", "goal aborted; blacklisted")
        elif s is NavState.REJECTED:
            # A rejection is an infrastructure fault (bt_navigator inactive, or its
            # BT failed to load), never a statement about the goal or the terrain:
            # no blacklist, no escape. Back to WAIT, which supervises and heals
            # bt_navigator's lifecycle.
            self.get_logger().error("goal REJECTED by Nav2 -- bt_navigator not active?")
            self._bt_state = None
            self._transition("WAIT", "goal rejected; waiting for a healthy Nav2")
        elif self._stuck():
            self._begin_escape_from_stuck()

    def _begin_escape_from_stuck(self):
        """Mark the mean commanded direction of the stuck window as a contact, then
        escape. "Pushed that way for stuck_window_s, did not move" is the same grade
        of physical evidence as a failed escape probe. Without the mark the loop
        never converges: the escape usually exits backward, Nav2 replans the same
        route (nothing in the costmap changed), and the robot re-stucks at the same
        spot. Skipped when the window was mostly rotation (mean drive < 0.1 m/s --
        direction meaningless); never a blind "forward" mark (that variant
        false-marked under the Phase4 deadband and penned the robot in, 2026-09-28)."""
        if self._cmd_log:
            a = np.array([(vx, vy) for _, vx, vy in self._cmd_log])
            mvx, mvy = a.mean(axis=0)
            if math.hypot(mvx, mvy) >= 0.1:
                self._mark_contact(float(mvx), float(mvy))
        self._start_escape(
            f"stuck: moved <{self._pget('stuck_min_travel_m')} m"
            f" in {self._pget('stuck_window_s'):.0f} s"
        )

    # ---------------------------------------------------------- bt_navigator watch
    def _poll_bt_state(self):
        if self._bt_state_pending or not self._gs_cli.service_is_ready():
            return
        self._bt_state_pending = True

        def done(fut):
            self._bt_state_pending = False
            try:
                self._bt_state = fut.result().current_state.label
            except Exception:
                self._bt_state = None

        self._gs_cli.call_async(GetState.Request()).add_done_callback(done)

    def _maybe_heal_bt(self):
        """Drive a stuck bt_navigator to 'active'.

        10 s of grace first (the lifecycle manager usually gets there itself), then
        the missing transitions in quick succession -- unconfigured needs CONFIGURE
        *and* ACTIVATE, and making each step wait the full grace left the stack dead
        for ~40 s (observed 2026-09-28). A transient/unknown state keeps the
        deadline instead of resetting it."""
        if not self._cs_cli.service_is_ready():
            return
        now = self._now()
        if self._bt_heal_at is None:
            self._bt_heal_at = now + 10.0
            return
        if now < self._bt_heal_at:
            return
        transition = {
            "unconfigured": Transition.TRANSITION_CONFIGURE,
            "inactive": Transition.TRANSITION_ACTIVATE,
        }.get(self._bt_state)
        if transition is None:
            return  # configuring/activating right now, or state unknown: re-check next tick
        self.get_logger().warn(
            f"bt_navigator stuck '{self._bt_state}'; driving the missing lifecycle transition myself"
        )
        req = ChangeState.Request()
        req.transition.id = transition
        self._cs_cli.call_async(req)
        self._bt_heal_at = now + 3.0  # the next step of the sequence, promptly

    # ------------------------------------------------------------------ escape
    def _stuck(self) -> bool:
        """True when net displacement over stuck_window_s is below the threshold.

        Endpoint displacement, deliberately: a robot oscillating on the spot (Nav2
        recovery loops included) covers distance but goes nowhere, and that is
        exactly the situation the escape should interrupt.
        """
        xy = self._robot_xy()
        if xy is None:
            return False
        now = self._now()
        window = float(self._pget("stuck_window_s"))
        self._travel_log.append((now, xy))
        self._travel_log = [(t, p) for t, p in self._travel_log if now - t <= window]
        t0, p0 = self._travel_log[0]
        if now - t0 < window * 0.95:
            return False  # not enough history yet
        return float(np.hypot(*(xy - p0))) < float(self._pget("stuck_min_travel_m"))

    def _escape_order(self) -> list[tuple[str, float, float, float]]:
        """Probe order: translations sorted by scan openness (widest exit first),
        rotations always last. The scan only orders the attempts -- success is
        judged by displacement alone, because a stuck robot is usually stuck
        precisely where the map/scan and reality disagree."""
        translations = [m for m in ESCAPE_MOTIONS if m[4] < 3]
        rotations = [m for m in ESCAPE_MOTIONS if m[4] == 3]
        translations.sort(key=lambda m: (-self._openness(math.atan2(m[2], m[1])), m[4]))
        return [(m[0], m[1], m[2], m[3]) for m in translations + rotations]

    def _openness(self, theta: float) -> float:
        """Robust free range (20th percentile) in a +-30 deg sector of the base
        frame around ``theta``; 0 when no scan has arrived yet."""
        s = self._scan
        if s is None:
            return 0.0
        ang = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
        r = np.array(s.ranges, dtype=float)
        r[~np.isfinite(r)] = s.range_max
        sector = r[np.abs((ang - theta + math.pi) % (2 * math.pi) - math.pi) < math.radians(30)]
        return float(np.percentile(sector, 20)) if len(sector) else 0.0

    def _start_escape(self, why: str):
        # Entered from exactly one place: the stuck watchdog. No blacklist here --
        # getting stuck en route says nothing final about the goal. If it truly
        # cannot be reached, the escape's contact marks wall the route off and
        # Nav2 aborts -- THE one place that blacklists.
        self._nav.cancel()
        self._escape = {
            "anchor": self._robot_xy(),
            "cands": self._escape_order(),
            "idx": 0,
            "round": 1,
            "phase": "settle",  # settle -> burst -> (resume | next candidate)
            "until": self._now() + 0.5,
            "burst": float(self._pget("escape_burst_s")),
            "scale": 1.0,
        }
        order = ", ".join(c[0] for c in self._escape["cands"])
        self._transition("ESCAPE", f"{why}; probe order: {order}")

    def _escape_cmd(self) -> Twist:
        _, ux, uy, uw = self._escape["cands"][self._escape["idx"]]
        k = self._escape["scale"]
        cmd = Twist()
        cmd.linear.x = ux * float(self._pget("escape_vx")) * k
        cmd.linear.y = uy * float(self._pget("escape_vy")) * k
        cmd.angular.z = uw * float(self._pget("escape_wz")) * k
        return cmd

    def _escape_tick(self):
        e = self._escape
        now = self._now()
        xy = self._robot_xy()
        if xy is None:
            return
        dist = float(np.hypot(*(xy - e["anchor"])))
        success = dist >= float(self._pget("escape_success_m"))

        if e["phase"] == "burst":
            if success:
                e["phase"] = "resume"
                e["until"] = now + float(self._pget("escape_resume_s"))
                self.get_logger().info(
                    f"ESCAPE: {e['cands'][e['idx']][0]} works ({dist:.2f} m from anchor); pushing on"
                )
            elif now >= e["until"]:
                # a full burst with no displacement: that direction is physically
                # blocked -- record the contact (rotations carry no direction)
                _, ux, uy, _ = e["cands"][e["idx"]]
                if ux or uy:
                    self._mark_contact(ux, uy)
                e["idx"] += 1
                e["phase"] = "settle"
                e["until"] = now + 0.5
                self._cmd_pub.publish(Twist())
                return
            self._cmd_pub.publish(self._escape_cmd())
        elif e["phase"] == "resume":
            if now >= e["until"]:
                self._cmd_pub.publish(Twist())
                self._escape = None
                self._transition("SELECT", "escaped")
                return
            self._cmd_pub.publish(self._escape_cmd())
        else:  # settle: stop between probes so displacement attributes to one motion
            if now >= e["until"]:
                if e["idx"] >= len(e["cands"]):
                    e["round"] += 1
                    if e["round"] > int(self._pget("escape_max_rounds")):
                        self._escape = None
                        self._finish("escape failed -- physically stuck, human help needed")
                        return
                    # next round: stronger and longer bursts, shuffled order so the
                    # same failing pattern is not just replayed (seeded: reproducible)
                    random.Random(e["round"]).shuffle(e["cands"])
                    e["idx"] = 0
                    e["burst"] *= 1.5
                    e["scale"] = min(e["scale"] * 1.25, 2.0)
                    self.get_logger().info(f"ESCAPE: round {e['round']}, stronger bursts")
                e["phase"] = "burst"
                e["until"] = now + e["burst"]
                self.get_logger().info(f"ESCAPE: trying {e['cands'][e['idx']][0]}")

    def _finish(self, why: str):
        known = int((self._grid != -1).sum()) if self._grid is not None else 0
        self._transition("DONE", why)
        self.get_logger().info(
            f"Exploration finished: {self._goals_reached}/{self._goals_sent} goals reached,"
            f" {known} known cells ({known * self._grid_meta[0]**2:.1f} m^2)."
        )

    # ----------------------------------------------------------- contact obstacles
    def _mark_contact(self, ux: float, uy: float):
        """Record a blocked direction (base-frame unit motion) as a short wall
        segment just outside the footprint, in map frame."""
        pose = self._robot_pose()
        if pose is None:
            return
        xy, yaw = pose
        n = math.hypot(ux, uy)
        bx, by = ux / n, uy / n  # base-frame unit direction
        # how far the body reaches along that direction (rectangle support function)
        reach = float(self._pget("footprint_half_length_m")) * abs(bx) + float(
            self._pget("footprint_half_width_m")
        ) * abs(by)
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.array([c * bx - s * by, s * bx + c * by])
        perp = np.array([-d[1], d[0]])
        center = xy + d * (reach + float(self._pget("contact_mark_margin_m")))
        now = self._now()
        added = 0
        for off in (-0.15, 0.0, 0.15):
            p = center + perp * off
            if all(float(np.hypot(*(p - q))) > 0.12 for q, _ in self._contact_pts):
                self._contact_pts.append((p, now))
                added += 1
        if added:
            self.get_logger().info(
                f"contact obstacle marked at ({center[0]:.2f}, {center[1]:.2f}) ({len(self._contact_pts)} pts total)"
            )

    def _publish_contacts(self):
        """1 Hz re-publish so the costmaps' contact_layer keeps marking (and any
        costmap started later still receives the accumulated points)."""
        now = self._now()
        self._contact_pts = [(p, t) for p, t in self._contact_pts if now - t <= float(self._pget("contact_ttl_s"))]
        if not self._contact_pts:
            return
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        self._contact_pub.publish(
            point_cloud2.create_cloud_xyz32(header, [(float(p[0]), float(p[1]), 0.4) for p, _ in self._contact_pts])
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

        m = base_marker(4, Marker.CUBE_LIST)
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r, m.color.a = 1.0, 0.9
        for p, _ in self._contact_pts:
            m.points.append(_pt(p[0], p[1]))
        arr.markers.append(m)

        # Blacklist (failures): flat orange discs at the true exclusion radius --
        # a blue candidate inside one is ineligible. Visited (arrivals): larger
        # translucent green discs at retire_radius_m, the area retired from
        # frontier detection. Lifted off the floor (z-fights the /map plane in
        # Foxglove otherwise). Ids 100+/200+ never collide with fixed markers.
        for i, (bx, by) in enumerate(self._selector.blacklist_entries()):
            m = base_marker(100 + i, Marker.CYLINDER)
            m.scale.x = m.scale.y = 2.0 * self._selector.radius
            m.scale.z = 0.10
            m.color.r, m.color.g, m.color.a = 1.0, 0.55, 0.8
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(bx), float(by), 0.05
            arr.markers.append(m)
        for i, (vx, vy) in enumerate(self._selector.visited_entries()):
            m = base_marker(200 + i, Marker.CYLINDER)
            m.scale.x = m.scale.y = 2.0 * float(self._pget("retire_radius_m"))
            m.scale.z = 0.04
            m.color.g, m.color.b, m.color.a = 0.8, 0.3, 0.35
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(vx), float(vy), 0.02
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
