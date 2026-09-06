"""Both bipedal stances in one policy.

``Go2-Multitask-Biped-Front`` and ``-Hind`` each train a network that only ever sees one stance.
This trains a single network that does whichever the command asks for, so the mixture of experts
gets one bipedal expert rather than two -- which is the point: an expert per stance would double
the mixture's width for two behaviours that share almost all of their mechanics.

The observation already carries what is needed. ``handstand_command`` is ``(enabled, stance)`` and
the stance column has been +-1 since the layout was widened, precisely so this task could exist
without another re-widen. What changes here is that the column *varies*: it is drawn per episode
instead of pinned, and the reward terms that name a side follow it through
:func:`~..mdp.rewards.stance_aware`.

Why this is not the experiment `feat/biped` failed
--------------------------------------------------
That branch spent a full round trying to make one policy respond to a mode command -- probabilistic
resampling, a scripted schedule with an assist force, one transition per episode, a direct
multimodal edit -- and every variant produced a policy that provably received the observation and
ignored it, out to 9900 iterations. Its conclusion was that the paper's own transition demo trains
two specialists and swaps between them at deployment.

Two things differ, and both matter:

*Its modes were quadruped against biped.* One of those is far easier and collects most of the
reward regardless of what the command says, so ignoring the command was close to optimal. Front
against hind are mirror siblings of comparable difficulty -- there is no easy option to collapse
onto, and a policy that ignores the stance column gets the wrong end of the robot off the ground
half the time.

*It had nothing to start from.* Both stances here already work: 0.547 m and 0.569 m of settled base
height, one fall in 64 environments each, contact on the stance feet and nothing else. Initialising
from either one starts this task with half the answer already in the weights.

That does not make it certain. If the merged policy converges to one stance and refuses the other,
the diagnosis is in ``handstand/success`` split by stance -- which is why the command tallies it
that way -- and the fallback is the same one that branch reached: keep the two experts separate.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped.robots.go2.biped_front_env_cfg_multitask import (
    BipedFrontCommandsCfg,
    BipedFrontRewardsCfg,
    RobotEnvCfgBipedFront,
    RobotPlayEnvCfgBipedFront,
)
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.mdp.gating import GATE_HANDSTAND, GATE_HANDSTAND_UPRIGHT

FRONT_FEET = ["FR_foot", "FL_foot"]
HIND_FEET = ["RR_foot", "RL_foot"]
FRONT_LEG_LINKS = ["FR_thigh", "FL_thigh", "FR_calf", "FL_calf"]
HIND_LEG_LINKS = ["RR_thigh", "RL_thigh", "RR_calf", "RL_calf"]


def _by_stance(term, front_params: dict, hind_params: dict, gate: str = GATE_HANDSTAND) -> dict:
    """Params for a reward term that follows the commanded stance, under the bipedal gate."""
    return {
        "gate": gate,
        "gate_command_name": "handstand",
        "term": mdp.stance_aware,
        "term_params": {"term": term, "front_params": front_params, "hind_params": hind_params},
    }


@configclass
class BipedCommandsCfg(BipedFrontCommandsCfg):
    """The front stance's commands, with the stance drawn per episode instead of pinned."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.handstand.stance_front_probability = 0.5
        # The lateral command splits the difference between the two single-stance tasks (+-1.0 for
        # the front, +-0.5 for the hind, narrowed there because that stance drifts sideways). Ask
        # for what the weaker of the two can do; widening it later is a change with an owner.
        self.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.base_velocity.limit_ranges.lin_vel_y = (-0.5, 0.5)


@configclass
class BipedRewardsCfg(BipedFrontRewardsCfg):
    """The single-stance reward set, with every side-naming term routed by the commanded stance.

    Untouched from the front config, because nothing in them names a side: the smoothness and limit
    penalties, ``stance_pitch`` (the command supplies its sign), ``stance_roll``, ``base_height``,
    ``upright_balance``, ``stance_held``, ``head_contact``, ``termination_penalty``, and the two
    velocity-tracking terms.
    """

    # -- contacts: which links may touch depends on which end is standing ------------------------
    undesired_contacts = RewTerm(
        func=mdp.gated,
        weight=-1.0,
        params=_by_stance(
            mdp.undesired_contacts,
            {"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_LEG_LINKS)},
            {"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_LEG_LINKS)},
        ),
    )
    stance_leg_contact = RewTerm(
        func=mdp.gated,
        weight=-20.0,
        params=_by_stance(
            mdp.undesired_contacts,
            {"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_LEG_LINKS)},
            {"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_LEG_LINKS)},
        ),
    )
    lifted_contact = RewTerm(
        func=mdp.gated,
        weight=-0.6,
        params=_by_stance(
            mdp.lifted_foot_contact,
            {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET)},
            {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET)},
        ),
    )

    # -- the tucked legs, whichever pair they are ------------------------------------------------
    # One term where the single-stance configs had three. Splitting hip, thigh and calf let each
    # carry its own weight (-0.15 / -0.05 / -0.05); folding them together would lose that, so the
    # split is kept and only the side is routed.
    lifted_hip_motion = RewTerm(
        func=mdp.gated,
        weight=-0.15,
        params=_by_stance(
            mdp.joint_deviation_l1,
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_hip_joint", "RL_hip_joint"])},
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_hip_joint", "FL_hip_joint"])},
        ),
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.gated,
        weight=-0.05,
        params=_by_stance(
            mdp.joint_deviation_l1,
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"])},
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh_joint", "FL_thigh_joint"])},
        ),
    )
    lifted_calf_motion = RewTerm(
        func=mdp.gated,
        weight=-0.05,
        params=_by_stance(
            mdp.joint_deviation_l1,
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_calf_joint", "RL_calf_joint"])},
            {"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_calf_joint", "FL_calf_joint"])},
        ),
    )

    # -- cadence, on whichever pair is carrying the robot -----------------------------------------
    feet_air_time = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params=_by_stance(
            mdp.feet_air_time_positive_biped,
            {"command_name": "base_velocity", "threshold": 0.4,
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET)},
            {"command_name": "base_velocity", "threshold": 0.4,
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET)},
            gate=GATE_HANDSTAND_UPRIGHT,
        ),
    )

    # -- CoM-CoP balance, on the commanded stance's feet only --------------------------------------
    # These were handed all four feet in the first version of this task, on the reasoning that the
    # force-weighted centre of pressure settles onto whichever feet are loaded and is therefore
    # already stance-agnostic. The premise is true and the conclusion was backwards.
    #
    # A tripod -- both stance feet plus one lifted foot still on the floor -- is exactly a case
    # where a foot outside the commanded stance is carrying load. Given all four as candidates, the
    # centre of pressure moves forward onto that foot, toward the centre of mass, and the CoM-CoP
    # vector straightens out: `pendulum_angle` measured -0.0048 in that run against the hind
    # specialist's -0.0065, so the tripod scored *better balanced* than the real two-legged stance
    # it was avoiding. Restricted to the stance pair, the same tripod puts the centre of pressure
    # between two feet while the mass hangs forward of them, and the term charges for it properly.
    #
    # That left `lifted_contact` at -0.25 per step as the only thing pricing the tripod, against a
    # -200 termination penalty for falling: worth it if attempting the real stance falls more than
    # about 2.5% of the time. The policy took that trade, in every hind-commanded environment, and
    # held the pose an operator recognised on sight as a temple-lion statue.
    #
    # This is what the single-stance configs do, and it is why neither of them tripods.
    pendulum_angle = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params=_by_stance(
            mdp.pendulum_angle_penalty,
            {"asset_cfg": SceneEntityCfg("robot", body_names=FRONT_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET)},
            {"asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET)},
        ),
    )
    pendulum_instability = RewTerm(
        func=mdp.gated,
        weight=-0.0001,
        params=_by_stance(
            mdp.pendulum_instability_penalty,
            {"asset_cfg": SceneEntityCfg("robot", body_names=FRONT_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET)},
            {"asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET)},
        ),
    )
    handle_length = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params=_by_stance(
            mdp.handle_length_penalty,
            {"asset_cfg": SceneEntityCfg("robot", body_names=FRONT_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET)},
            {"asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET)},
        ),
    )
    support_polygon = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params=_by_stance(
            mdp.support_polygon_penalty,
            {"asset_cfg": SceneEntityCfg("robot", body_names=FRONT_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FEET),
             "command_name": "base_velocity"},
            {"asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET),
             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET),
             "command_name": "base_velocity"},
            gate=GATE_HANDSTAND_UPRIGHT,
        ),
    )

    # `front_hip_height` is dropped, as it is for the hind stance: its 0.30 m target describes the
    # front hips when they are the shoulders, and nothing when they are the raised end. A term that
    # is right for half the episodes is worse than no term.
    front_hip_height = None


@configclass
class RobotEnvCfgBiped(RobotEnvCfgBipedFront):
    commands: BipedCommandsCfg = BipedCommandsCfg()
    rewards: BipedRewardsCfg = BipedRewardsCfg()


@configclass
class RobotPlayEnvCfgBiped(RobotPlayEnvCfgBipedFront):
    commands: BipedCommandsCfg = BipedCommandsCfg()
    rewards: BipedRewardsCfg = BipedRewardsCfg()
