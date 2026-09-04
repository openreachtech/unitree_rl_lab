"""Hind-leg biped walking, on the unified multi-task observation.

The mirror of ``biped_front_env_cfg_multitask`` -- the robot rises onto its hind legs, front legs
tucked, and walks there. Everything the two stances share is inherited rather than restated; only
the stance/lifted sides swap, plus the handful of places where ``feat/biped``'s record says the two
genuinely differ.

One line of that inheritance does the sign work by itself. ``stance_pitch_reward`` is
``stance * projected_gravity_b[:, 0]``, and gravity's body-x component is positive when the nose is
down and negative when it is up -- so setting ``stance`` to :data:`~...mdp.handstand.STANCE_HIND`
(-1) turns the same expression from "pitch nose-down" into "pitch nose-up" with nothing else
changed. Same for the ``handstand_command`` observation, which carries that sign to the policy.

**This is the harder stance, and the branch that trained both says so.** The front stance reached a
working policy in three sandbox tries; the hind one took seven, and two of its problems were never
solved:

- *The tripod shuffle.* The robot lifts one front foot, freezes, and collects the tilt reward
  without ever walking. Tilt alone does not produce a gait. What broke it was not a reward weight
  but the **commanded pace**: widening the velocity range (``lin_vel_x`` +-0.4 -> +-1.0) is what
  finally forced a real two-leg gait, because a dragged foot cannot sustain the faster command.
  Lifted-foot ground contact fell from a 5% plateau to 0.5-0.8%. That range is inherited here.
- *Sidestepping.* The hind legs sit side by side, so lateral weight-shift is the mechanically
  stable direction and the robot prefers to walk sideways. Splitting the tracking reward into
  separate x and y terms did **not** fix it, and slowed lifted-foot progress by diluting the
  contact penalty's share of the total. Still open; do not spend the first round on it.
- *A front foot touching down during forward walking.* Reward shaping never removed it and twice
  collapsed training when made strict enough to matter. What worked was a separate forward-only
  phase with a decaying assist force, gated on front-foot contact rather than on not falling --
  a robot that "didn't fall" by leaning on a front foot decayed the assist away before learning
  anything. That belongs in a Phase 2, not here.

A trap from the same record, worth stating because this task inherits the same numbers: the
tracking reward's ``std`` is ``sqrt(0.25)`` ~ 0.5 m/s. If a later phase narrows the command range
below that, tracking pays near-maximally regardless of what the robot does, and the policy stops
tracking. Narrow the range and the std together or not at all.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped.robots.go2.biped_front_env_cfg_multitask import (
    BASE_HEIGHT_TARGET,
    BipedFrontCommandsCfg,
    BipedFrontRewardsCfg,
    RobotEnvCfgBipedFront,
    RobotPlayEnvCfgBipedFront,
)
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.mdp.gating import GATE_HANDSTAND, GATE_HANDSTAND_UPRIGHT

# Hind legs stand, front legs lift -- the mirror of the front stance's lists.
STANCE_FEET = ["RR_foot", "RL_foot"]
LIFTED_FEET = ["FR_foot", "FL_foot"]
LIFTED_HIP_JOINTS = ["FR_hip_joint", "FL_hip_joint"]
LIFTED_THIGH_JOINTS = ["FR_thigh_joint", "FL_thigh_joint"]
LIFTED_CALF_JOINTS = ["FR_calf_joint", "FL_calf_joint"]


@configclass
class BipedHindCommandsCfg(BipedFrontCommandsCfg):
    """The front stance's commands, mirrored to the hind legs.

    ``lin_vel_y`` narrows to +-0.5 from the front stance's +-1.0. Not a copy of the front recipe:
    +-0.5 is what ``feat/biped`` validated for *this* stance, and the front task's own docstring
    records widening it as a change made for the front stance specifically. Given that sideways is
    already the direction this stance drifts toward, opening the lateral command to full width
    would be inviting the failure mode rather than testing for it.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.base_velocity.limit_ranges.lin_vel_y = (-0.5, 0.5)
        self.handstand.stance = mdp.STANCE_HIND
        self.handstand.stance_foot_names = tuple(STANCE_FEET)
        self.handstand.lifted_foot_names = tuple(LIFTED_FEET)


@configclass
class BipedHindRewardsCfg(BipedFrontRewardsCfg):
    """The front stance's reward set with the stance and lifted sides swapped.

    Every term that names a foot, a leg or a contact sensor is restated below. Terms that name
    neither -- the smoothness and limit penalties, ``stance_pitch``, ``stance_roll``,
    ``base_height``, ``upright_balance``, ``stance_held``, ``termination_penalty`` -- are inherited
    untouched and are correct for either stance as written.
    """

    # -- the lifted legs: off the ground, and tucked ---------------------------------------------
    lifted_contact = RewTerm(
        func=mdp.gated,
        weight=-0.6,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.lifted_foot_contact,
            "term_params": {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_FEET)},
        },
    )
    lifted_hip_motion = RewTerm(
        func=mdp.gated,
        weight=-0.15,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.joint_deviation_l1,
            "term_params": {"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_HIP_JOINTS)},
        },
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.gated,
        weight=-0.05,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.joint_deviation_l1,
            "term_params": {"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_THIGH_JOINTS)},
        },
    )
    lifted_calf_motion = RewTerm(
        func=mdp.gated,
        weight=-0.05,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.joint_deviation_l1,
            "term_params": {"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_CALF_JOINTS)},
        },
    )

    # -- CoM-CoP balance, measured against the feet that are actually carrying the robot ----------
    pendulum_angle = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.pendulum_angle_penalty,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
            },
        },
    )
    pendulum_instability = RewTerm(
        func=mdp.gated,
        weight=-0.0001,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.pendulum_instability_penalty,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
            },
        },
    )
    handle_length = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.handle_length_penalty,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
            },
        },
    )
    support_polygon = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params={
            "gate": GATE_HANDSTAND_UPRIGHT,
            "gate_command_name": "handstand",
            "term": mdp.support_polygon_penalty,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
                "command_name": "base_velocity",
            },
        },
    )
    feet_air_time = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={
            "gate": GATE_HANDSTAND_UPRIGHT,
            "gate_command_name": "handstand",
            "term": mdp.feet_air_time_positive_biped,
            "term_params": {
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
                "threshold": 0.4,
            },
        },
    )

    # -- dropped, not mirrored --------------------------------------------------------------------
    # `front_hip_height` pulls the FR/FL hips toward 0.30 m. In the front stance those are the
    # shoulders carrying the robot, and the term is what stops the head scraping. Here they are at
    # the *raised* end, well above 0.30, so the same term would pull the robot's front end back
    # down -- the opposite of what it is for. `feat/biped`'s own hind-stance config has no
    # equivalent, and neither does this one.
    front_hip_height = None

    # `head_contact` at -20.0 is inherited, and is expected to sit at zero here: with the nose up,
    # the lowest part of the trunk is the rear of `base`, which `base_contact` already terminates
    # on at 1 N. Kept rather than removed because it costs nothing while inert and still guards a
    # fall onto the nose.
    #
    # Do not treat that reasoning as settled. The front stance spent four training runs on a
    # symptom that turned out to be an unpoliced 4050 N contact nobody had measured, and the
    # cheapest possible check is to read `head/trunk contact` and the `base` contact force out of
    # `measure_stance.py` on the first run of this task rather than to assume the geometry.


@configclass
class RobotEnvCfgBipedHind(RobotEnvCfgBipedFront):
    commands: BipedHindCommandsCfg = BipedHindCommandsCfg()
    rewards: BipedHindRewardsCfg = BipedHindRewardsCfg()


@configclass
class RobotPlayEnvCfgBipedHind(RobotPlayEnvCfgBipedFront):
    commands: BipedHindCommandsCfg = BipedHindCommandsCfg()
    rewards: BipedHindRewardsCfg = BipedHindRewardsCfg()
