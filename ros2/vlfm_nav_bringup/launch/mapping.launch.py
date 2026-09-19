"""Robot-agnostic mapping stack (M2): point cloud -> /scan -> slam_toolbox -> /map.

Everything robot-specific arrives as launch arguments; a robot bringup package
(go2_nav_bringup, later anaguma_nav_bringup) supplies them from its config yaml.

  ros2 launch vlfm_nav_bringup mapping.launch.py \
      pointcloud_topic:=/utlidar/cloud base_frame:=base \
      min_height:=0.0 max_height:=0.5 use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    min_height = LaunchConfiguration("min_height")
    max_height = LaunchConfiguration("max_height")
    use_sim_time = LaunchConfiguration("use_sim_time")

    slam_params = PathJoinSubstitution(
        [FindPackageShare("vlfm_nav_bringup"), "params", "slam_toolbox.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("pointcloud_topic"),
            DeclareLaunchArgument("base_frame"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("min_height", description="scan band lower bound, in base_frame (m)"),
            DeclareLaunchArgument("max_height", description="scan band upper bound, in base_frame (m)"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", pointcloud_topic), ("scan", "/scan")],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        # Flatten in the base frame: the band rides the body but stays
                        # body-relative, which is what the config's numbers mean.
                        "target_frame": base_frame,
                        "transform_tolerance": 0.05,
                        "min_height": min_height,
                        "max_height": max_height,
                        "angle_min": -3.14159,
                        "angle_max": 3.14159,
                        "angle_increment": 0.0087,  # 0.5 deg
                        "scan_time": 0.1,  # MID-360 full elevation sweep
                        "range_min": 0.2,
                        "range_max": 15.0,
                        "use_inf": True,
                    }
                ],
            ),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[
                    slam_params,
                    {
                        "use_sim_time": use_sim_time,
                        "base_frame": base_frame,
                        "odom_frame": odom_frame,
                    },
                ],
            ),
        ]
    )
