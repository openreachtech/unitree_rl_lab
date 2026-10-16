"""Both handstands in one policy: front or hind, as the command says.

Ported from ``feat/multi-task``'s ``Go2-Multitask-Biped``. The stance is drawn per episode and the
policy reads it from the ``handstand_command`` observation, ``(enabled, stance)`` with stance +1
front / -1 hind -- the one observation the single-stance tasks do not carry. Every reward term
that names a side is routed by the commanded stance through :func:`~...mdp.stance_aware`; the rest
are identical in the two single-stance configs and are inherited unchanged.

How it is trained -- each step was needed (sandbox Try1-3):

1. *Start from the Go2w-Biped-Hind policy*, widened for the stance column
   (``scripts/rsl_rl/widen_policy_inputs.py``). From scratch, the policy learned the front stance
   by iteration 400 and stayed propped on its front wheels at ~58 degrees in the hind stance to
   iteration 2000: whichever stance comes second gets stuck on the lifted end.
2. *Balance the running population of stances* (``balance_stance_population``). Started from the
   hind policy with plain 50/50 draws, front-commanded episodes fell in a few steps and were 2% of
   the environments at any moment, so the front stance never got enough samples to start. With
   the population held at 50/50, front success went from 0 to 0.97 around iteration 1150 (hind
   stays at 0.98).
3. *Keep ``front_hip_height`` for the front stance* (:func:`~...mdp.when_stance`). go2's
   two-stance task drops it -- in the hind stance the front hips are the raised end and the term
   would pull them down -- but without it the front stance hangs its head just above the floor.
   Active only where the front stance is commanded, it brings the shoulders back to the
   single-stance height, at the single-stance weight.

Not ported: go2's left-right mirror augmentation (``BipedPPORunnerCfg``), which needs a mirror
function for Go2w's 16-joint layout -- worth adding if the policy develops a preferred side.
"""
from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2w import biped_env_cfg_front as front
from unitree_rl_lab.tasks.biped.robots.go2w import biped_env_cfg_hind as hind
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_front import (
    BipedFrontCommandsCfg,
    BipedFrontRewardsCfg,
    BipedObservationsCfg,
    RobotEnvCfgBipedFront,
    RobotPlayEnvCfgBipedFront,
)


_FRONT_HIP_HEIGHT = BipedFrontRewardsCfg().front_hip_height


def _by_stance(term, front_params: dict, hind_params: dict) -> dict:
    return {"term": term, "front_params": front_params, "hind_params": hind_params}


def _feet(names: list[str]) -> dict:
    return {
        "asset_cfg": SceneEntityCfg("robot", body_names=names),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=names),
    }


def _contacts(links: list[str]) -> dict:
    return {"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=links)}


def _joints(names: list[str]) -> dict:
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=names)}


@configclass
class BipedCommandsCfg(BipedFrontCommandsCfg):
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.handstand.stance_front_probability = 0.5
        self.handstand.balance_stance_population = True


@configclass
class BipedUnifiedObservationsCfg(BipedObservationsCfg):
    """The single-stance observations plus ``(enabled, stance)``, so the policy knows which end."""

    @configclass
    class PolicyCfg(BipedObservationsCfg.PolicyCfg):
        handstand_command = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "handstand"}
        )

    @configclass
    class CriticCfg(BipedObservationsCfg.CriticCfg):
        handstand_command = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "handstand"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BipedRewardsCfg(BipedFrontRewardsCfg):
    """The single-stance reward set with every side-naming term routed by the commanded stance.

    Inherited unchanged because nothing in them names a side: the smoothness and limit penalties,
    ``head_contact``, ``termination_penalty``, ``stance_pitch`` (the command supplies its sign),
    ``stance_roll``, ``base_height``, ``upright_balance``, the tracking terms and ``stance_held``.
    """

    lifted_wheel_vel = RewTerm(
        func=mdp.stance_aware,
        weight=-1e-3,
        params=_by_stance(mdp.joint_vel_l2, _joints(front.LIFTED_WHEEL_JOINTS), _joints(hind.LIFTED_WHEEL_JOINTS)),
    )

    undesired_contacts = RewTerm(
        func=mdp.stance_aware,
        weight=-1.0,
        params=_by_stance(
            mdp.undesired_contacts, _contacts(front.LIFTED_LEG_LINKS), _contacts(hind.LIFTED_LEG_LINKS)
        ),
    )
    stance_leg_contact = RewTerm(
        func=mdp.stance_aware,
        weight=-20.0,
        params=_by_stance(
            mdp.undesired_contacts, _contacts(front.STANCE_LEG_LINKS), _contacts(hind.STANCE_LEG_LINKS)
        ),
    )

    lifted_contact = RewTerm(
        func=mdp.stance_aware,
        weight=-0.6,
        params=_by_stance(
            mdp.lifted_foot_contact,
            {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=front.LIFTED_FEET)},
            {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=hind.LIFTED_FEET)},
        ),
    )
    # Hip, thigh and calf kept as three terms so each keeps its own weight; only the side is routed.
    lifted_hip_motion = RewTerm(
        func=mdp.stance_aware,
        weight=-0.15,
        params=_by_stance(mdp.joint_deviation_l1, _joints(front.LIFTED_HIP_JOINTS), _joints(hind.LIFTED_HIP_JOINTS)),
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.stance_aware,
        weight=-0.05,
        params=_by_stance(
            mdp.joint_deviation_l1, _joints(front.LIFTED_THIGH_JOINTS), _joints(hind.LIFTED_THIGH_JOINTS)
        ),
    )
    lifted_calf_motion = RewTerm(
        func=mdp.stance_aware,
        weight=-0.05,
        params=_by_stance(mdp.joint_deviation_l1, _joints(front.LIFTED_CALF_JOINTS), _joints(hind.LIFTED_CALF_JOINTS)),
    )

    # CoM-CoP balance on the commanded stance's wheels only. go2 first handed these all four feet,
    # reasoning that the force-weighted CoP settles on whichever are loaded; a tripod then scored as
    # *better* balanced than the real stance, and the policy held one in every hind episode.
    pendulum_angle = RewTerm(
        func=mdp.stance_aware,
        weight=-0.1,
        params=_by_stance(mdp.pendulum_angle_penalty, _feet(front.STANCE_FEET), _feet(hind.STANCE_FEET)),
    )
    pendulum_instability = RewTerm(
        func=mdp.stance_aware,
        weight=-0.0001,
        params=_by_stance(mdp.pendulum_instability_penalty, _feet(front.STANCE_FEET), _feet(hind.STANCE_FEET)),
    )
    handle_length = RewTerm(
        func=mdp.stance_aware,
        weight=-0.1,
        params=_by_stance(mdp.handle_length_penalty, _feet(front.STANCE_FEET), _feet(hind.STANCE_FEET)),
    )
    support_polygon = RewTerm(
        func=mdp.when_upright,
        weight=-0.1,
        params={
            "term": mdp.stance_aware,
            "term_params": _by_stance(
                mdp.support_polygon_penalty,
                {**_feet(front.STANCE_FEET), "command_name": "base_velocity"},
                {**_feet(hind.STANCE_FEET), "command_name": "base_velocity"},
            ),
        },
    )

    # The front stance's own term, unchanged, in front-commanded environments only.
    front_hip_height = RewTerm(
        func=mdp.when_stance,
        weight=_FRONT_HIP_HEIGHT.weight,
        params={"term": _FRONT_HIP_HEIGHT.func, "stance": mdp.STANCE_FRONT, "term_params": _FRONT_HIP_HEIGHT.params},
    )


@configclass
class RobotEnvCfgBiped(RobotEnvCfgBipedFront):
    observations: BipedUnifiedObservationsCfg = BipedUnifiedObservationsCfg()
    commands: BipedCommandsCfg = BipedCommandsCfg()
    rewards: BipedRewardsCfg = BipedRewardsCfg()


@configclass
class RobotPlayEnvCfgBiped(RobotPlayEnvCfgBipedFront):
    observations: BipedUnifiedObservationsCfg = BipedUnifiedObservationsCfg()
    commands: BipedCommandsCfg = BipedCommandsCfg()
    rewards: BipedRewardsCfg = BipedRewardsCfg()
