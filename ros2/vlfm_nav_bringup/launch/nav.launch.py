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

# No behavior_server: the fail-fast BT's only recovery is ClearEntireCostmap
# (a costmap service), and the exploration layer owns real failure handling
# (blacklist + displacement-verified ESCAPE). Launching it only added a node whose
# lifecycle-race failure to activate blocked bt_navigator from loading its tree
# (the spin recovery it pulled in was never available). bt_navigator is likewise
# pinned to the navigate_to_pose navigator only (see nav2_common.yaml).
NAV_NODES = [
    ("nav2_controller", "controller_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_bt_navigator", "bt_navigator"),
]


def generate_launch_description():
    robot_params_file = LaunchConfiguration("robot_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    common_params = PathJoinSubstitution(
        [FindPackageShare("vlfm_nav_bringup"), "params", "nav2_common.yaml"]
    )
    # Fail-fast trees: one retry, costmap clears as the only recovery. The exploration
    # layer owns failure handling (blacklist + escape); see the XML header. Passed
    # here because a yaml file cannot carry a package-relative path.
    # Both trees are overridden: bt_navigator loads BOTH at activate, and the stock
    # navigate_through_poses tree needs behavior_server's spin action -- which we do
    # not run -- so leaving it stock makes bt_navigator fail to activate even though
    # we only ever send NavigateToPose goals.
    bt_dir = FindPackageShare("vlfm_nav_bringup")
    bt_to_pose = PathJoinSubstitution([bt_dir, "behavior_trees", "navigate_to_pose_fail_fast.xml"])
    bt_through_poses = PathJoinSubstitution(
        [bt_dir, "behavior_trees", "navigate_through_poses_fail_fast.xml"]
    )
    params = [
        common_params,
        robot_params_file,
        {
            "use_sim_time": use_sim_time,
            "default_nav_to_pose_bt_xml": bt_to_pose,
            "default_nav_through_poses_bt_xml": bt_through_poses,
        },
    ]

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
