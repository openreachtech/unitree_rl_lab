"""Go2 robot layer, sim flavor.

Pairs with scripts/ros2/play_ros2.py on the Isaac side:

  [Isaac] /clock /sim/joint_states /sim/imu /utlidar/cloud{,_band}
     -> robot_state_publisher (go2.urdf: base->radar/imu and leg TF,
        fed by /sim/joint_states directly)
     -> RKO-LIO on /utlidar/cloud + /sim/imu
     -> odom->base TF, /odometry/filtered

Odometry is RKO-LIO (the same system the hardware runs on both Go2 and Anaguma);
odom:=external instead leaves odometry to the sim (play_ros2.py --gt_odom).

The earlier leg-kinematics InEKF path (go2_odometry + /lowstate composition) was
removed 2026-09-24: its foot-contact assumption breaks when the blind policy
brushes walls, the resulting yaw jumps forked the SLAM map on every long run,
and the hardware plan never used it. See doc/design/vlfm_nav.md §6 and git
history if it needs resurrecting.

  ros2 launch go2_nav_bringup go2_sim.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from unitree_description import GO2_DESCRIPTION_URDF_PATH


def generate_launch_description():
    with open(GO2_DESCRIPTION_URDF_PATH) as f:
        robot_desc = f.read()

    lio_launch = PathJoinSubstitution([FindPackageShare("rko_lio"), "launch", "odometry.launch.py"])
    lio_config = PathJoinSubstitution(
        [FindPackageShare("go2_nav_bringup"), "params", "rko_lio_go2.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("odom", default_value="lio", description="lio | external"),
            # Applies to every node below, included launch files too.
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_desc}],
                # The sim publishes named joint states directly; no LowState detour.
                remappings=[("joint_states", "/sim/joint_states")],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([lio_launch]),
                condition=LaunchConfigurationEquals("odom", "lio"),
                # rko_lio's launch honors per-param overrides only from its own CLI
                # (context.argv); everything therefore lives in the config file, the
                # one argument that passes through an include.
                launch_arguments={
                    "config_file": lio_config,
                    "use_sim_time": "true",
                }.items(),
            ),
        ]
    )
