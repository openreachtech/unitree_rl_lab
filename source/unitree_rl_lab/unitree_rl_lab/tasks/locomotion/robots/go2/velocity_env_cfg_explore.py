"""Go2-Blind-GRU-Mid360-Phase4 in an indoor exploration world, for the VLFM nav stack.

Not a training config: one robot, one fixed four-room floorplan
(``terrains.MeshIndoorRoomsTerrainCfg``), driven over ROS via
``scripts/ros2/play_ros2.py``. The MID-360 and its held elevation map come from
``RobotEnvCfgMid360Phase4`` unchanged, so Go2-Blind-GRU-Phase4 checkpoints load as-is.

What changes against the phase config, and why:

* **Terrain**: the wall-hurdle grid is replaced by flat rooms with 1 m walls and a few
  knee-height boxes. The scan z band (base +0.0..0.5 m) needs structure that crosses it
  for slam_toolbox to have anything to map; the phase's 0.25 m hurdles all sit below it
  by design.
* **push_robot off**: a 0.5 m/s shove every 5-10 s is a robustness event for training;
  under SLAM it is an odometry insult with no upside.
* **Terminations: the trunk hitting the floor, nothing else** (2026-09-30). A reset
  teleports the robot home, which silently invalidates the map and odometry, and
  play_ros2.py then ends the whole run rather than overlay a second misregistered floor
  plan on the SLAM map. That is expensive enough that only a real fall should trigger it.

  - ``bad_orientation`` off: a 45.8 deg tilt is a posture, not a fall. Ending a
    15-minute exploration for one is a bad trade.
  - ``base_contact`` off too (2026-10-02). It was briefly raised 1 -> 300 N so only a
    real impact would fire, and it duly fired -- but a termination is not a diagnosis.
    Ending the episode teleports the robot home, which is precisely what destroys the
    evidence: the posture it landed in and what it was tangled with. With nothing left to
    terminate on, a fallen robot simply stays down where it fell, the map stays valid,
    and the fall can be looked at. play_ros2.py prints a (non-terminating) notice when
    the trunk tips past 45 deg, so the moment is still findable in a log or a bag.

    The cost is that a run no longer ends itself -- exploration just stops making
    progress. That is the right trade while falls are the thing under investigation; put
    one of these back when they are not.
* **Curriculum off**: one terrain level, commands come from /cmd_vel, nothing to ratchet.
* **Front RGB camera on**: the VLM layer (M5) scores what the robot sees. Only the
  nav worlds carry it -- the phase configs train with thousands of environments and
  would pay the render cost for an image nothing reads. Needs ``--enable_cameras``,
  which play_ros2.py forces on.

Episode length, command resampling and standing envs are handled by play_ros2.py at
runtime (endless episode, /cmd_vel writes) rather than here, so ordinary play on this
task still behaves like play.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_mid360 import (
    RobotEnvCfgMid360Phase4,
    _attach_front_camera,
)

EXPLORE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    use_cache=False,
    curriculum=False,
    sub_terrains={"indoor_rooms": terrains.MeshIndoorRoomsTerrainCfg(proportion=1.0)},
)
"""One 12 x 12 m four-room tile. Spawn is the SW room center (the sub-terrain's origin)."""


@configclass
class RobotEnvCfgMid360Explore(RobotEnvCfgMid360Phase4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = EXPLORE_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = None
        # /cmd_vel spans the full training envelope, same as play
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # see module docstring
        self.events.push_robot = None
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        # Front RGB camera for the VLM layer (M5). Attached here rather than in the
        # phase config so training never pays the render cost; kujiale inherits it.
        _attach_front_camera(self)


@configclass
class RobotPlayEnvCfgMid360Explore(RobotEnvCfgMid360Explore):
    """Same config; registered separately so play.py's entry-point lookup works."""

    pass
