"""Non-blocking NavigateToPose client.

nav2_simple_commander insists on spinning its own node and on an AMCL-shaped
bringup; this is the same few calls without the assumptions, designed to be
driven from a timer in the caller's executor.
"""

from __future__ import annotations

import math
from enum import Enum

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class NavState(Enum):
    IDLE = "idle"
    PENDING = "pending"  # goal sent, waiting for acceptance
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    REJECTED = "rejected"


class NavBridge:
    def __init__(self, node: Node):
        self._node = node
        self._client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self._state = NavState.IDLE
        self._goal_handle = None
        # Generation counter: every go_to()/cancel() bumps it, and in-flight
        # callbacks of older goals are ignored. Without it, the CANCELED result of
        # a preempted goal arrived ~1 s after the NEXT goal became active and was
        # booked as that goal's ABORT -- each false abort triggered a reselection,
        # which preempted again, a self-sustaining cascade that blacklisted dozens
        # of untried goals (observed 2026-09-29).
        self._seq = 0

    @property
    def state(self) -> NavState:
        return self._state

    def server_ready(self) -> bool:
        return self._client.server_is_ready()

    def go_to(self, x: float, y: float, yaw: float) -> None:
        """Send a map-frame goal; track progress via ``state``."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        # Leave the stamp zero: Nav2 then transforms with the latest TF instead of
        # pinning the goal to a stamp that ages out of the TF buffer while the BT
        # retries (bitten by this in M3).
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._state = NavState.PENDING
        self._goal_handle = None
        self._seq += 1
        seq = self._seq
        send = self._client.send_goal_async(goal)
        send.add_done_callback(lambda f: self._on_goal_response(f, seq))

    def cancel(self) -> None:
        self._seq += 1  # everything still in flight is now stale
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._goal_handle = None
        self._state = NavState.IDLE

    def _on_goal_response(self, future, seq: int) -> None:
        if seq != self._seq:
            return  # response for a goal that was superseded meanwhile
        handle = future.result()
        if not handle.accepted:
            self._state = NavState.REJECTED
            return
        self._goal_handle = handle
        self._state = NavState.ACTIVE
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, future, seq: int) -> None:
        if seq != self._seq:
            return  # a stale goal's terminal result (preempted/canceled), not ours
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._state = NavState.SUCCEEDED
        elif self._state is NavState.ACTIVE:
            self._state = NavState.ABORTED
