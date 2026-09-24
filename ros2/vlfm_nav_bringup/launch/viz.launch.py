"""Remote visualization: foxglove_bridge for Foxglove Studio on another machine.

The bridge serves the ROS graph over one websocket; rendering happens client-side,
so this works smoothly over SSH-only setups (e.g. a Mac on the LAN).

  ros2 launch vlfm_nav_bringup viz.launch.py            # sim (default)
  ros2 launch vlfm_nav_bringup viz.launch.py use_sim_time:=false   # real robot

Client side: Foxglove Studio -> Open connection -> Foxglove WebSocket
  - direct:      ws://<server>:8765  (needs the port reachable, e.g. ufw allow)
  - ssh tunnel:  ssh -L 8765:localhost:8765 <server>, then ws://localhost:8765

Panels worth adding: 3D (/map, /scan, /utlidar/cloud, /vlfm/markers, /plan, TF),
Raw Messages on /odometry/filtered, and later an Image panel for the M5 camera.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "port": 8765,
                        # localhost binding would break direct LAN access; the ufw
                        # default-deny still gates who can actually reach it.
                        "address": "0.0.0.0",
                        "send_buffer_limit": 50_000_000,  # /utlidar/cloud is 10k pts at 10 Hz
                    }
                ],
            ),
        ]
    )
