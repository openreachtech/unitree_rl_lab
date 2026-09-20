"""Robot-agnostic Nav2 stack (M3): planner + MPPI controller + behaviors + BT.

Parameters layer as [nav2_common.yaml, robot_params_file, use_sim_time], later
entries winning -- the robot file (e.g. go2_nav_bringup/params/nav2_go2.yaml)
carries frames, footprint and velocity limits.

Expects /map + map->odom from mapping.launch.py, odom->base + /odometry/filtered
from the robot layer, and publishes /cmd_vel.

  ros2 launch vlfm_nav_bringup nav.launch.py \
      robot_params_file:=<pkg>/params/nav2_go2.yaml use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

NAV_NODES = [
    ("nav2_controller", "controller_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_behaviors", "behavior_server"),
    ("nav2_bt_navigator", "bt_navigator"),
]


def generate_launch_description():
    robot_params_file = LaunchConfiguration("robot_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    common_params = PathJoinSubstitution(
        [FindPackageShare("vlfm_nav_bringup"), "params", "nav2_common.yaml"]
    )
    params = [common_params, robot_params_file, {"use_sim_time": use_sim_time}]

    nodes = [
        Node(
            package=pkg,
            executable=exe,
            name=exe,
            output="screen",
            respawn=False,
            parameters=params,
        )
        for pkg, exe in NAV_NODES
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_params_file", description="robot overlay yaml (frames, footprint, vel limits)"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            *nodes,
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": [exe for _, exe in NAV_NODES],
                    }
                ],
            ),
        ]
    )
