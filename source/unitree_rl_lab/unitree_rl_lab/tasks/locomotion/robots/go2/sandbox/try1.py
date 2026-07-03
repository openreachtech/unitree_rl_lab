from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase3 import (
    RewardsCfgPhase3,
    RobotEnvCfgPhase3,
    RobotPlayEnvCfgPhase3,
)


@configclass
class RewardsCfgPhase3StairClimb(RewardsCfgPhase3):
    # Body orientation: relax heavily — robot must pitch while ascending/descending stairs
    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.3)

    # Vertical velocity penalty: stairs require upward body motion
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.2)

    # Joint position: relax to allow larger joint excursions needed to step up
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.3)

    # Undesired contacts: reduce — calves naturally brush stair edges
    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)

    # Foot air time: longer swing needed to clear stair steps
    feet_air_time = RewardsCfgPhase3().feet_air_time.replace(weight=0.15)

    # Foot clearance: raise target_clearance to 0.12 m so feet clear stair risers (up to 0.23 m)
    wild_foot_clearance = RewardsCfgPhase3().wild_foot_clearance.replace(
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "radius": 0.1,
            "target_clearance": 0.12,
        },
    )

    # Forward progress: provides a linear gradient when track_lin_vel_xy saturates
    # near a stalled robot; critical for bootstrapping onto and climbing stairs
    forward_command_progress = RewardsCfgPhase3().forward_command_progress.replace(weight=0.5)


@configclass
class RobotEnvCfgPhase3StairClimb(RobotEnvCfgPhase3):
    """Phase 3 sandbox: tuned rewards for stair climbing."""

    rewards: RewardsCfgPhase3StairClimb = RewardsCfgPhase3StairClimb()


@configclass
class RobotPlayEnvCfgPhase3StairClimb(RobotPlayEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = RewardsCfgPhase3StairClimb()
