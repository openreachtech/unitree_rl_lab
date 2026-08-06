"""Bipedal (2-leg stance) locomotion task for Go2 using the FRONT legs as the
stance/support legs (hind legs lifted/tucked) -- the mirror-image stance the paper
describes: "By changing the contact penalty for front legs to hind legs, the robot
can perform bipedal locomotion with front legs" (Xiao et al. 2025 / "TumblerNet").

Scene, domain randomization, commands, action space, terminations, and the
estimator/actor architecture are identical to the hind-leg-stance ``biped_env_cfg``
(reused via inheritance); only the stance feet and the tucked-leg motion penalties
swap sides.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import CommandsCfg, ObservationsCfg, RewardsCfg, RobotEnvCfg

# Hind legs swing/tuck (lifted); front legs are the stance/support legs -- mirror
# image of the hind-leg-stance biped_env_cfg's STANCE_FOOT_NAMES / *_motion penalties.
HIND_CALF_JOINT_NAMES = ["RR_calf_joint", "RL_calf_joint"]
STANCE_FOOT_NAMES = ["FR_foot", "FL_foot"]


@configclass
class ObservationsCfgFront(ObservationsCfg):
    """Same layout as ``ObservationsCfg``, but ``com_cop`` uses the front feet as stance."""

    @configclass
    class CriticCfgFront(ObservationsCfg.CriticCfg):
        com_cop = ObservationsCfg.CriticCfg().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
            }
        )

    critic: CriticCfgFront = CriticCfgFront()

    @configclass
    class EstimatorTargetCfgFront(ObservationsCfg.EstimatorTargetCfg):
        com_cop = ObservationsCfg.EstimatorTargetCfg().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
            }
        )

    estimator_target: EstimatorTargetCfgFront = EstimatorTargetCfgFront()


@configclass
class RewardsCfgFront(RewardsCfg):
    """Same reward terms as ``RewardsCfg``, with stance/swing legs swapped front<->hind."""

    # -- keep the swing (now hind) legs near their default pose instead of flailing --
    front_hip_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_hip_joint", "RL_hip_joint"])},
    )
    front_thigh_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"])},
    )
    front_calf_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIND_CALF_JOINT_NAMES)},
    )

    # -- CoM-CoP balance (TumblerNet); stance = front feet --
    pendulum_angle = RewardsCfg().pendulum_angle.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )
    pendulum_instability = RewardsCfg().pendulum_instability.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )
    handle_length = RewardsCfg().handle_length.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )


@configclass
class CommandsCfgFront(CommandsCfg):
    """Same as ``CommandsCfg``, but ``gait_mode`` is pinned to front-biped instead
    of hind-biped (inert either way -- this file's reward set never reads
    gait_mode -- but keeps the observation semantically correct)."""

    gait_mode = mdp.PinnedGaitModeCommandCfg(asset_name="robot", pinned_mode=mdp.MODE_FRONT_BIPED)


@configclass
class RobotEnvCfgFront(RobotEnvCfg):
    """Front-leg-stance variant of the Go2 bipedal env."""

    commands: CommandsCfgFront = CommandsCfgFront()
    observations: ObservationsCfgFront = ObservationsCfgFront()
    rewards: RewardsCfgFront = RewardsCfgFront()


@configclass
class RobotPlayEnvCfgFront(RobotEnvCfgFront):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
