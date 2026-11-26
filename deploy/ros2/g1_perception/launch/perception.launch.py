"""G1 MID-360 perception: /livox/lidar + /livox/imu -> DLIO -> elevation_mapping_cupy -> rviz.

Same for unitree_mujoco -l 1 and the real livox_ros_driver2; nothing here knows which.

  helpers                 livox_imu_si: /livox/imu (g) -> /livox/imu_si (m/s^2)
                          g1_state: /lowstate -> /joint_states, TF dlio_torso -> pelvis
  DLIO                    TF odom -> dlio_torso (+ livox_frame, livox_imu_frame),
                          /dlio/odom_node/{odom,path,pointcloud/deskewed}
  DLIO map                /dlio/map_node/map  (dlio_map:=true; off to save a DDS participant)
  elevation_mapping_cupy  /elevation_mapping_node/elevation_map_{raw,filter}  (map:=false to skip)
  robot_state_publisher   G1 model (urdf:=..., default unitree_ros g1_29dof_rev_1_0) at DLIO's pose
  rviz2                   (rviz:=false to skip)

elevation_mapping_cupy is built in the same workspace and runs with the workspace's .venv
python (cupy; see deploy/ros2/README.md), or the system python when there is no .venv.
"""
import glob
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

G1_URDF = Path.home() / "unitree" / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"


def venv_python_and_libs(ws):
    """cupy lives in <ws>/.venv; point the loader at the CUDA libraries pip put there."""
    python = ws / ".venv" / "bin" / "python3"
    if not python.exists():
        return None, {}
    site = ws / ".venv" / "lib" / "python3*" / "site-packages" / "nvidia"
    libs = sorted(glob.glob(str(site / "*" / "lib")))
    env = {"LD_LIBRARY_PATH": ":".join(libs + [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p])}
    runtime = sorted(glob.glob(str(site / "cuda_runtime")))
    if runtime and "CUDA_PATH" not in os.environ:
        env["CUDA_PATH"] = runtime[0]
    return str(python), env


def robot_model(context):
    """robot_state_publisher with the URDF's relative mesh paths made absolute (rviz needs that)."""
    urdf = Path(LaunchConfiguration("urdf").perform(context)).expanduser()
    text = urdf.read_text().replace('filename="meshes/', f'filename="file://{urdf.parent}/meshes/')
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": text}]),
    ]


def generate_launch_description():
    share = get_package_share_directory("g1_perception")
    dlio_share = get_package_share_directory("direct_lidar_inertial_odometry")
    cfg = lambda name: os.path.join(share, "config", name)

    # livox_imu_si + g1_state: one process, one DDS participant (see helpers.py)
    helpers = Node(package="g1_perception", executable="helpers", output="log")

    dlio_remaps = [
        ("pointcloud", "/livox/lidar"),
        ("imu", "/livox/imu_si"),
        ("odom", "dlio/odom_node/odom"),
        ("pose", "dlio/odom_node/pose"),
        ("path", "dlio/odom_node/path"),
        ("kf_pose", "dlio/odom_node/keyframes"),
        ("kf_cloud", "dlio/odom_node/pointcloud/keyframe"),
        ("deskewed", "dlio/odom_node/pointcloud/deskewed"),
    ]
    dlio_params = [os.path.join(dlio_share, "cfg", "dlio.yaml"), os.path.join(dlio_share, "cfg", "params.yaml"), cfg("dlio.yaml")]
    dlio_odom = Node(
        package="direct_lidar_inertial_odometry", executable="dlio_odom_node",
        output="screen", parameters=dlio_params, remappings=dlio_remaps,
    )
    dlio_map = Node(
        package="direct_lidar_inertial_odometry", executable="dlio_map_node",
        output="screen", parameters=dlio_params,
        remappings=[("keyframes", "dlio/odom_node/pointcloud/keyframe"), ("map", "dlio/map_node/map")],
        condition=IfCondition(LaunchConfiguration("dlio_map")),
    )

    python, env = venv_python_and_libs(Path(share).parents[3])  # <ws>/install/g1_perception/share/g1_perception
    elevation_mapping = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        prefix=python,
        additional_env=env,
        parameters=[
            os.path.join(get_package_share_directory("elevation_mapping_cupy"), "config", "core", "core_param.yaml"),
            cfg("elevation_mapping.yaml"),
            {"plugin_config_file": cfg("elevation_mapping_plugins.yaml")},
        ],
        condition=IfCondition(LaunchConfiguration("map")),
    )

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2_g1_perception", output="log",
        arguments=["-d", os.path.join(share, "rviz", "perception.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="true", description="Run elevation_mapping_cupy"),
        DeclareLaunchArgument("rviz", default_value="true", description="Launch rviz2"),
        DeclareLaunchArgument("dlio_map", default_value="false", description="DLIO keyframe map (dlio_map:=true)"),
        DeclareLaunchArgument("urdf", default_value=str(G1_URDF), description="G1 URDF for the rviz model"),
        helpers, OpaqueFunction(function=robot_model), dlio_odom, dlio_map, elevation_mapping, rviz,
    ])
