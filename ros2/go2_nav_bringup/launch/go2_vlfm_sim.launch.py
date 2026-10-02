"""Everything-ROS in one launch: robot layer + mapping + Nav2 + viz + exploration.

The Isaac side stays a separate process (scripts/ros2/play_ros2.py); start it first,
then this single launch replaces terminals 2..6:

  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py                  # frontier explore
  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py target:=chair    # + VLM value map
  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py explore:=false   # stack only (teleop etc.)
  ros2 launch go2_nav_bringup go2_vlfm_sim.launch.py viz:=false       # no foxglove bridge

``target`` is the one switch that matters: without it nothing loads a model and the
robot explores nearest-first, exactly as it did before M5. With it, the SigLIP 2 scorer
comes up alongside and the exploration node starts preferring frontiers that look like
they lead to the thing named.

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
  vlm_node  [+6 s]           only with target:=; py3.11 + Isaac's bundled ROS, so it is
                             an ExecuteProcess rather than a Node

Remember: if the Isaac process restarts, restart this launch too (/clock rolls
back and every TF buffer goes silent with TF_OLD_DATA otherwise).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    explore = LaunchConfiguration("explore")
    viz = LaunchConfiguration("viz")
    target = LaunchConfiguration("target")
    vlm_python = LaunchConfiguration("vlm_python")
    vlm_script = LaunchConfiguration("vlm_script")
    # Empty target = plain frontier exploration, which is the default and needs no model.
    want_vlm = IfCondition(PythonExpression(["'", target, "' != ''"]))

    def pkg_launch(pkg, name):
        return PathJoinSubstitution([FindPackageShare(pkg), "launch", name])

    go2_params = PathJoinSubstitution(
        [FindPackageShare("go2_nav_bringup"), "params", "nav2_go2.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("explore", default_value="true", description="start the VLFM exploration node"),
            DeclareLaunchArgument("viz", default_value="true", description="start foxglove_bridge"),
            DeclareLaunchArgument(
                "target",
                default_value="",
                description="What to look for, in English (e.g. chair, toilet, bed)."
                " Empty -- the default -- runs plain frontier exploration with no model"
                " loaded. Setting it also starts the SigLIP 2 scorer, which publishes"
                " /vlfm/value_map and /vlfm/value_confidence; the exploration node reads"
                " those and prefers frontiers that look promising.",
            ),
            # The scorer is a py3.11 process using Isaac's bundled ROS, so it cannot be a
            # Node() here -- hence ExecuteProcess with its own interpreter. Both paths
            # are arguments because the Anaguma port keeps its venv and checkout
            # elsewhere; nothing below assumes this machine's layout beyond the default.
            DeclareLaunchArgument(
                "vlm_python",
                default_value=[EnvironmentVariable("HOME"), "/isaacsim/env_isaaclab/bin/python"],
                description="interpreter for the VLM scorer (needs torch + transformers"
                " + isaacsim, i.e. the Isaac venv)",
            ),
            DeclareLaunchArgument(
                "vlm_script",
                default_value=[EnvironmentVariable("HOME"), "/isaacsim/unitree_rl_lab/scripts/ros2/vlm_node.py"],
                description="path to vlm_node.py",
            ),
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
                    ),
                    ExecuteProcess(
                        cmd=[vlm_python, vlm_script, "--target", target, "--use-sim-time"],
                        output="screen",
                        condition=want_vlm,
                        # Blank out this shell's ROS. We are launched from a sourced
                        # py3.10 Humble, and the scorer resolves a py3.11 Humble out of
                        # the Isaac extension; leaving both on the loader path mixes two
                        # ABIs of the same libraries and fails as missing symbols.
                        # ROS_DOMAIN_ID and the RMW choice are inherited, which is what
                        # lets the two halves find each other.
                        additional_env={
                            "PYTHONPATH": "",
                            "LD_LIBRARY_PATH": "",
                            "AMENT_PREFIX_PATH": "",
                            "CMAKE_PREFIX_PATH": "",
                        },
                    ),
                ],
            ),
        ]
    )
