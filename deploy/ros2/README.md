# ROS 2 packages for the deploy side

`g1_perception`: G1 head MID-360 -> DLIO odometry -> elevation_mapping_cupy -> rviz.
Input is `/livox/lidar` + `/livox/imu` (livox_ros_driver2 format), from the real driver or
from unitree_mujoco `-l 1`.

## Workspace (`~/unitree/ros2_ws`)

```bash
sudo apt install -y ros-humble-pcl-ros ros-humble-grid-map-rviz-plugin \
    ros-humble-rviz2 ros-humble-rviz-default-plugins ros-humble-rosidl-generator-dds-idl
git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree/unitree_ros    # G1 URDF + meshes
git clone https://github.com/unitreerobotics/unitree_ros2.git ~/unitree/unitree_ros2  # unitree_hg msgs

mkdir -p ~/unitree/ros2_ws/src && cd ~/unitree/ros2_ws/src
ln -s ~/isaacsim/unitree_rl_lab/deploy/ros2/g1_perception g1_perception
ln -s ~/unitree/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg unitree_hg
git clone -b feature/ros2 https://github.com/vectr-ucla/direct_lidar_inertial_odometry.git
cd .. && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

elevation_mapping_cupy comes from the tsubame workspace (`~/tsubame/ros2_ws`, with its
`.venv` for cupy / torch); `TSUBAME_WS` overrides the path.

## Run (MuJoCo)

```bash
g1sim -s testbench_v3_room.xml -l 1   # 1. sim with the MID-360
g1lidar                               # 2. perception + rviz, while the robot hangs still
                                      #    (DLIO calibrates the IMU for the first 3 s)
g1m                                   # 3. walk: 1 -> Enter -> 9
```

`g1lidar` (in `~/.bashrc`) sources ROS 2 + both workspaces, sets CycloneDDS on `lo`
(`-n eth0` for the robot's network) with `ROS_DOMAIN_ID=0`, and runs
`ros2 launch g1_perception perception.launch.py` with the rest of its args
(`map:=false`, `rviz:=false`, `dlio_map:=true`, `urdf:=...`):

```bash
g1lidar() ( n=lo; if [ "$1" = -n ]; then n=$2; shift 2; fi
            source /opt/ros/humble/setup.bash; source ~/tsubame/ros2_ws/install/setup.bash; source ~/unitree/ros2_ws/install/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$n\"/></Interfaces></General></Domain></CycloneDDS>"
            exec ros2 launch g1_perception perception.launch.py "$@" )
```

Notes:
- Map heights are in `odom`, whose origin is the torso at start (the floor reads about -1.2 m).
  Restart step 2 after resetting the sim.
- DLIO publishes the torso as `dlio_torso`; `g1_state` hangs `pelvis` under it, so the URDF
  model (driven by `/lowstate`) stands where DLIO says.
- On `lo` CycloneDDS has no multicast and allows about 10 participants (processes) per
  domain, sim and g1_ctrl included. Too many and g1_ctrl dies with "Failed to find a free
  participant index"; check for leftover nodes from an earlier run.
