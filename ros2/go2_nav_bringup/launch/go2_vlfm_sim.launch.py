"""Everything-ROS in one launch: robot layer + mapping + Nav2 + viz + exploration.

The Isaac side stays a separate process (scripts/ros2/play_ros2.py); start it first,
then this single launch replaces terminals 2..6:

  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py            # explore
  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py explore:=false   # stack only (teleop etc.)
  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py viz:=false       # no foxglove bridge

Composition (all sim-time):
  go2_sim.launch.py          robot_state_publisher + RKO-LIO
  mapping.launch.py          pointcloud_to_laserscan (+_band cloud) + slam_toolbox
  nav.launch.py [+5 s]       Nav2 (delayed: its lifecycle bringup lost a service
                             response under the start-up CPU spike when everything
                             launched at once -- seen 2026-09-24; widened 3->5 s
                             after it recurred. vlfm_node also self-heals a
                             bt_navigator this race leaves un-activated)
  viz.launch.py              foxglove_bridge :8765
  vlfm_node [+6 s]           exploration; waits internally for map/TF/Nav2 anyway,
                             the delay just keeps its logs out of the boot noise

Remember: if the Isaac process restarts, restart this launch too (/clock rolls
back and every TF buffer goes silent with TF_OLD_DATA otherwise).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    explore = LaunchConfiguration("explore")
    viz = LaunchConfiguration("viz")

    def pkg_launch(pkg, name):
        return PathJoinSubstitution([FindPackageShare(pkg), "launch", name])

    go2_params = PathJoinSubstitution(
        [FindPackageShare("go2_nav_bringup"), "params", "nav2_go2.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("explore", default_value="true", description="start the VLFM exploration node"),
            DeclareLaunchArgument("viz", default_value="true", description="start foxglove_bridge"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([pkg_launch("go2_nav_bringup", "go2_sim.launch.py")])
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([pkg_launch("vlfm_nav_bringup", "mapping.launch.py")]),
                launch_arguments={
                    "pointcloud_topic": "/utlidar/cloud_band",
                    "base_frame": "base",
                    "min_height": "-0.4",
                    "max_height": "0.7",
                    "use_sim_time": "true",
                }.items(),
            ),
            TimerAction(
                period=5.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([pkg_launch("vlfm_nav_bringup", "nav.launch.py")]),
                        launch_arguments={
                            "robot_params_file": go2_params,
                            "use_sim_time": "true",
                        }.items(),
                    )
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([pkg_launch("vlfm_nav_bringup", "viz.launch.py")]),
                condition=IfCondition(viz),
            ),
            TimerAction(
                period=6.0,
                actions=[
                    Node(
                        package="vlfm_nav",
                        executable="vlfm_node",
                        name="vlfm",
                        output="screen",
                        parameters=[{"use_sim_time": True}],
                        condition=IfCondition(explore),
                    )
                ],
            ),
        ]
    )
