"""Go2 robot layer, sim flavor.

Pairs with scripts/ros2/play_ros2.py on the Isaac side:

  [Isaac] /clock /sim/joint_states /sim/imu /sim/foot_forces /utlidar/cloud{,_band}
     -> sim_lowstate_bridge -> /lowstate
     -> robot_state_publisher (base->radar/imu TF) + state_converter (/joint_states)
     -> odometry (see below) -> odom->base TF, /odometry/filtered

Odometry source (odom:=...):
  lio      RKO-LIO on /utlidar/cloud + /sim/imu -- the deployment-matching path
           (hardware runs RKO-LIO on both Go2 and Anaguma). Default.
  inekf    go2_odometry's leg-kinematics InEKF on /lowstate. Works for short runs;
           yaw drifts under wall contact and forks the SLAM map on long ones
           (measured 2026-09-20) -- kept for comparison.
  external Nothing here publishes odometry; the sim does (play_ros2.py --gt_odom).

  ros2 launch go2_nav_bringup go2_sim.launch.py odom:=lio
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    odom = LaunchConfiguration("odom")
    inekf_launch = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_inekf_odometry.launch.py"]
    )
    state_pub_launch = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_state_publisher.launch.py"]
    )
    lio_launch = PathJoinSubstitution([FindPackageShare("rko_lio"), "launch", "odometry.launch.py"])
    lio_config = PathJoinSubstitution(
        [FindPackageShare("go2_nav_bringup"), "params", "rko_lio_go2.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("odom", default_value="lio", description="lio | inekf | external"),
            # Applies to every node below, included launch files too.
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="go2_nav_bringup",
                executable="sim_lowstate_bridge",
                name="sim_lowstate_bridge",
                output="screen",
            ),
            # inekf's launch already includes the state publisher; the other modes
            # need it separately (base->radar TF and /joint_states).
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([inekf_launch]),
                condition=LaunchConfigurationEquals("odom", "inekf"),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([state_pub_launch]),
                condition=LaunchConfigurationNotEquals("odom", "inekf"),
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
