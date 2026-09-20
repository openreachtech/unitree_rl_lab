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
        send = self._client.send_goal_async(goal)
        send.add_done_callback(self._on_goal_response)

    def cancel(self) -> None:
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._state = NavState.IDLE

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._state = NavState.REJECTED
            return
        self._goal_handle = handle
        self._state = NavState.ACTIVE
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._state = NavState.SUCCEEDED
        elif self._state is NavState.ACTIVE:
            self._state = NavState.ABORTED
