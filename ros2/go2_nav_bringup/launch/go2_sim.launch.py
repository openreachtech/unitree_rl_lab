"""Go2 robot layer, sim flavor (M1).

Pairs with scripts/ros2/play_ros2.py running on the Isaac side:

  [Isaac] /clock /sim/joint_states /sim/imu /sim/foot_forces /utlidar/cloud
     -> sim_lowstate_bridge -> /lowstate
     -> go2_odometry (state_converter + robot_state_publisher + inekf)
     -> odom->base TF, /odometry/filtered, base->radar via URDF

Everything runs on sim time (/clock from Isaac).

  ros2 launch go2_nav_bringup go2_sim.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    inekf_launch = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_inekf_odometry.launch.py"]
    )

    return LaunchDescription(
        [
            # Applies to every node below, included launch files too.
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="go2_nav_bringup",
                executable="sim_lowstate_bridge",
                name="sim_lowstate_bridge",
                output="screen",
            ),
            IncludeLaunchDescription(PythonLaunchDescriptionSource([inekf_launch])),
        ]
    )
