"""Hind-leg handstand for Go2w: up on the two hind wheels, front legs tucked, rolling there.

The mirror of ``biped_env_cfg_front``, ported from ``feat/multi-task``'s ``Go2-Multitask-Biped-Hind``
the same way: everything the stances share is inherited, and only the stance/lifted sides swap.
``stance = STANCE_HIND`` (-1) does the sign work by itself -- ``stance_pitch_reward`` is
``stance * projected_gravity_b[:, 0]``, so the same term asks for nose-up instead of nose-down.

On go2 this was the harder stance (seven sandbox tries against three), for two reasons that the
wheeled port sidesteps rather than solves:

- *Tripod shuffle and a front foot touching down while walking forward* -- a stepping-gait
  problem: a dragged foot was cheaper than a real two-leg gait. Go2w does not step; it rolls on
  the hind wheels, and ``lifted_contact`` still prices a front wheel on the floor.
- *Sidestepping* -- go2's hind legs sit side by side, so it drifted sideways. go2 narrowed
  ``lin_vel_y`` to +-0.5; here it is already 0 (wheels cannot roll sideways).

Dropped, not mirrored: ``front_hip_height``. It pulls the FR/FL hips to 0.35 m, which in the front
stance holds the shoulders up; here those hips are the raised end, and the same term would pull
the robot back down. ``head_contact`` is inherited and expected to stay at zero with the nose up --
kept because it costs nothing and guards a fall onto the nose.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_front import (
    BipedFrontCommandsCfg,
    BipedFrontRewardsCfg,
    RobotEnvCfgBipedFront,
    RobotPlayEnvCfgBipedFront,
    upright_params,
)

# Hind legs stand, front legs lift -- the mirror of the front stance's lists.
STANCE_FEET = ["RR_foot", "RL_foot"]
LIFTED_FEET = ["FR_foot", "FL_foot"]
LIFTED_HIP_JOINTS = ["FR_hip_joint", "FL_hip_joint"]
LIFTED_THIGH_JOINTS = ["FR_thigh_joint", "FL_thigh_joint"]
LIFTED_CALF_JOINTS = ["FR_calf_joint", "FL_calf_joint"]
LIFTED_WHEEL_JOINTS = ["FR_foot_joint", "FL_foot_joint"]
STANCE_LEG_LINKS = ["RR_thigh", "RL_thigh", "RR_calf", "RL_calf"]
LIFTED_LEG_LINKS = ["FR_thigh", "FL_thigh", "FR_calf", "FL_calf"]


@configclass
class BipedHindCommandsCfg(BipedFrontCommandsCfg):
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.handstand.stance = mdp.STANCE_HIND


@configclass
class BipedHindRewardsCfg(BipedFrontRewardsCfg):
    """The front stance's rewards with every term that names a side restated for the hind legs.

    Terms that name neither side -- smoothness and limits, ``stance_pitch``/``roll``,
    ``base_height``, ``head_contact``, ``upright_balance``, the tracking terms, ``stance_held``,
    ``termination_penalty`` -- are inherited and correct for either stance as written.
    """

    lifted_wheel_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_WHEEL_JOINTS)}
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_LEG_LINKS)},
    )
    # The measurement that put this term on go2 was taken on this stance: a shin on the floor 8.7%
    # of the time at up to 3215 N -- kneeling on one leg and sliding.
    stance_leg_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_LEG_LINKS)},
    )

    lifted_contact = RewTerm(
        func=mdp.lifted_foot_contact,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_FEET)},
    )
    lifted_hip_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.15, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_HIP_JOINTS)}
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_THIGH_JOINTS)}
    )
    lifted_calf_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_CALF_JOINTS)}
    )

    pendulum_angle = RewTerm(
        func=mdp.pendulum_angle_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )
    pendulum_instability = RewTerm(
        func=mdp.pendulum_instability_penalty,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )
    handle_length = RewTerm(
        func=mdp.handle_length_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )
    support_polygon = RewTerm(
        func=mdp.when_upright,
        weight=-0.1,
        params=upright_params(
            mdp.support_polygon_penalty,
            asset_cfg=SceneEntityCfg("robot", body_names=STANCE_FEET),
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
            command_name="base_velocity",
        ),
    )

    front_hip_height = None


@configclass
class RobotEnvCfgBipedHind(RobotEnvCfgBipedFront):
    commands: BipedHindCommandsCfg = BipedHindCommandsCfg()
    rewards: BipedHindRewardsCfg = BipedHindRewardsCfg()


@configclass
class RobotPlayEnvCfgBipedHind(RobotPlayEnvCfgBipedFront):
    commands: BipedHindCommandsCfg = BipedHindCommandsCfg()
    rewards: BipedHindRewardsCfg = BipedHindRewardsCfg()
