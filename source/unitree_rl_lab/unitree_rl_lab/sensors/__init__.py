"""OmniPerception's LidarSensor (https://github.com/aCodeDog/OmniPerception), carried inside
unitree_rl_lab, ported from feat/mid360 as-is (apart from import paths). ``LidarSensor``
extends the stock :class:`isaaclab.sensors.RayCaster`, and the scene instantiates it through
``LidarSensorCfg.class_type``.

``rolling_livox_sensor.py`` advances the scan window each step instead of freezing it at init
(the port's one substantive bug). ``scan_patterns/mid360.npy`` is the recorded MID-360 scan
sequence. feat/mid360's robot self-occluder is not carried: on the G1 the head-mounted
sensor's blind cone already contains the legs and arms.
"""

from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg
from .lidar_sensor_data import LidarSensorData
from .patterns import livox_pattern
from .patterns_cfg import LivoxPatternCfg
from .rolling_livox_sensor import RollingLivoxSensor, RollingLivoxSensorCfg
