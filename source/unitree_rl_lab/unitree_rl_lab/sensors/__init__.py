"""OmniPerception's LidarSensor (https://github.com/aCodeDog/OmniPerception), carried
inside unitree_rl_lab instead of being installed into the IsaacLab checkout.

The port is as-is apart from import paths: ``LidarSensor`` extends the stock
:class:`isaaclab.sensors.RayCaster`, and the scene instantiates it through
``LidarSensorCfg.class_type``, so nothing needs to be registered inside isaaclab.
``scan_patterns/`` holds the Livox scan sequences the pattern loader falls back to;
only ``mid360.npy`` is checked in -- copy further sensors' files from
``OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/``
next to it when needed.

``rolling_livox_sensor.py`` is *not* part of the port. It is a subclass that advances the
scan window each step instead of freezing it at init, which is the port's one substantive
bug; see that module for what the port does and why it matters.
"""

from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg
from .lidar_sensor_data import LidarSensorData
from .patterns import livox_pattern
from .patterns_cfg import LivoxPatternCfg
from .rolling_livox_sensor import RollingLivoxSensor, RollingLivoxSensorCfg
