"""Go2-Blind-GRU-Phase1 with an experimental Livox MID-360 sensor in the scene.

This is the OmniPerception ``LidarSensor`` (https://github.com/aCodeDog/OmniPerception)
ported as-is into ``unitree_rl_lab.sensors`` -- the IsaacLab checkout itself is
untouched. The scan pattern is a 4,000-ray slice of the real MID-360
non-repetitive sequence (``sensors/scan_patterns/mid360.npy``): at the 0.02 s env step
that is 200k points/s, exactly the real sensor's data rate.

The base task is untouched: these classes extend ``RobotEnvCfgPhase1`` /
``RobotPlayEnvCfgPhase1`` and register separately as ``Go2-Blind-GRU-Mid360-Phase1``.
Nothing reads the ``mid360`` observation group: like ``LidarMapObsCfg`` in
``velocity_env_cfg_lidar.py``, it exists so the observation manager touches the sensor
each step, which is what makes it raycast at all. The policy and critic inputs are
unchanged, so Go2-Blind-GRU-Phase1 checkpoints still load. In play the scene carries
both fans: Phase 1's own ``lidar_scanner`` (drawn by the lidar-view observations) and
the ``mid360_scanner`` added here.

The sensor sits at the real L1's pose (nose tip, pitched -- see ``GO2_L1_MOUNT`` /
``GO2_L1_ROT`` below), so the scan band sweeps the ground ahead instead of the horizon.

Known as-is limitations of the port, accepted for this experiment:
  * The 4,000-ray slice is fixed at init (``rolling_window_start=0``). The real sensor
    advances through the 800k-point sequence every frame; the ported sensor only drifts
    the fixed slice slowly in yaw (``_update_dynamic_rays``).
  * Rearward the real nose mount is blind behind the head and trunk; the RayCaster
    only sees the static ground mesh, so the sim looks straight through the body and
    is more optimistic than hardware there (same caveat as velocity_env_cfg_lidar.py).
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.sensors import LidarSensorCfg, LivoxPatternCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase1 import (
    RobotEnvCfgPhase1,
    RobotPlayEnvCfgPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    GO2_LIDAR_OFFSET_X,
    GO2_LIDAR_OFFSET_Y,
    GO2_LIDAR_OFFSET_Z,
)

MID360_SAMPLES_PER_STEP = 4000
"""Rays per env step. 4,000 x 50 Hz = 200k points/s, the real MID-360's rate."""

# ---------------------------------------------------------------------------
# Mount: the real L1 (utlidar) pose from go2_description.urdf's ``radar_joint``,
# ``xyz="0.28945 0 -0.046825" rpy="0 2.8782 0"`` -- the nose tip, pitched 164.9 deg
# so the sensor hangs nearly upside down looking out and down through the nose
# aperture. The translation is already in the repo as GO2_LIDAR_OFFSET_*.
#
# Under this tilt the MID-360's -7..+52 deg elevation band lands as roughly a
# horizon ring (tilted 15 deg, front-down) down to ~52 deg below it: the dense
# upper-hemisphere coverage that pointed at the sky on the old flat trunk mount
# now sweeps the ground from ~0.2 m ahead out to max_distance.
# ---------------------------------------------------------------------------
GO2_L1_MOUNT = (GO2_LIDAR_OFFSET_X, GO2_LIDAR_OFFSET_Y, GO2_LIDAR_OFFSET_Z)
_L1_PITCH = 2.8782  # rad, from the URDF radar_joint rpy
GO2_L1_ROT = (math.cos(_L1_PITCH / 2), 0.0, math.sin(_L1_PITCH / 2), 0.0)
"""(w, x, y, z) quaternion of the L1 mount: a pure pitch of 2.8782 rad."""


def _mid360_scanner_cfg(debug_vis: bool) -> LidarSensorCfg:
    """A fresh sensor cfg per env-cfg instance, so play tweaks never leak into train."""
    return LidarSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=LidarSensorCfg.OffsetCfg(pos=GO2_L1_MOUNT, rot=GO2_L1_ROT),
        # Bolted to the nose like the real L1: pitch and roll swing the pattern
        # with the body.
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=MID360_SAMPLES_PER_STEP),
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
        min_range=0.2,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        update_frequency=50.0,
        debug_vis=debug_vis,
    )


@configclass
class Mid360ObsCfg(ObsGroup):
    """Display-only group: nothing reads it, it exists so the sensor updates each step."""

    distances = ObsTerm(
        func=mdp.lidar_distances,
        params={"sensor_cfg": SceneEntityCfg("mid360_scanner")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _attach_mid360(cfg: RobotEnvCfgPhase1, debug_vis: bool) -> None:
    cfg.scene.mid360_scanner = _mid360_scanner_cfg(debug_vis)
    cfg.scene.mid360_scanner.update_period = cfg.decimation * cfg.sim.dt
    cfg.observations.mid360 = Mid360ObsCfg()


@configclass
class RobotEnvCfgMid360Phase1(RobotEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        _attach_mid360(self, debug_vis=False)


@configclass
class RobotPlayEnvCfgMid360Phase1(RobotPlayEnvCfgPhase1):
    def __post_init__(self):
        super().__post_init__()
        _attach_mid360(self, debug_vis=True)
