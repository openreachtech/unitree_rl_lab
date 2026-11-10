"""Stance on command: drive on four wheels, rise into a handstand when told, come back down.

``Go2w-Biped`` with the stance scheduled instead of pinned (``HandstandCommandCfg.scheduled``):
every episode starts on four wheels; the stance -- front or hind -- is switched on at a drawn
time, held, released, and called for again after a drawn rest, and a share of episodes never stand
at all. The policy reads the switch from ``handstand_command``'s ``enabled`` column, which
``Go2w-Biped`` already observes (always 1 there), so the observation layout is unchanged and the
``Go2w-Biped`` policy is the starting point as it is.

Which reward set is in force follows the command (``HandstandCommand.stance_weight``):

* *the stance's*, wherever the two-stance task names a posture -- pitch, roll, height, the lifted
  legs, the balance terms -- wrapped in :func:`~...mdp.in_stance`. Its tracking and balance terms
  are already gated on being upright, which the release ends.
* *four-wheel driving*, the Go2w velocity task's terms (yaw-frame tracking, level trunk, default
  leg pose, wheels down at a standstill), wrapped in :func:`~...mdp.on_wheels`.
* *shared*: the leg-joint smoothness and limit penalties, the contact penalties on the trunk end
  and the termination penalty.

Nothing pays for coming back down directly: the pressure is the four-wheel rewards returning at
the release (faded in over 0.25 s) and the orientation limit returning after a 1.5 s grace
(:func:`~...mdp.when_on_wheels`), which ends the episode -- at the termination penalty's price --
for a robot still up when it runs out. ``handstand/descent_success`` counts the releases that got
back to a level four-wheel stance inside the grace.

Train from the Go2w-Biped policy (copied, iteration counter reset):

    python scripts/rsl_rl/widen_policy_inputs.py logs/rsl_rl/go2w_biped/<run>/model_999.pt \\
        logs/rsl_rl/go2w_biped_switch/0000_init_from_biped/model_0.pt --extra 0
    python scripts/rsl_rl/train.py --task Go2w-Biped-Switch --headless \\
        --resume --load_run 0000_init_from_biped --checkpoint model_0.pt
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg import (
    BipedCommandsCfg,
    BipedRewardsCfg,
    RobotEnvCfgBiped,
    RobotPlayEnvCfgBiped,
)
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_front import BipedTerminationsCfg
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg import LEG_JOINT_NAMES

STANCE_ONLY_TERMS = (
    "lifted_wheel_vel",
    "undesired_contacts",
    "lifted_contact",
    "lifted_hip_motion",
    "lifted_thigh_motion",
    "lifted_calf_motion",
    "pendulum_angle",
    "pendulum_instability",
    "handle_length",
    "stance_pitch",
    "stance_roll",
    "base_height",
    "front_hip_height",
)
"""``BipedRewardsCfg`` terms that describe the stance and must not act on four wheels. The tracking,
``upright_balance``, ``support_polygon`` and ``stance_held`` terms are left as they are: they are
gated on being upright in the commanded stance, which a release ends at once."""


def _wheels(term, weight: float, **term_params) -> RewTerm:
    return RewTerm(func=mdp.on_wheels, weight=weight, params={"term": term, "term_params": term_params})


@configclass
class BipedSwitchCommandsCfg(BipedCommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.handstand.scheduled = True


@configclass
class BipedSwitchRewardsCfg(BipedRewardsCfg):
    # -- four-wheel driving: the Go2w velocity task's terms (velocity_env_cfg.RewardsCfg) ----------
    wheels_track_lin_vel_xy = _wheels(
        mdp.track_lin_vel_xy_yaw_frame_exp, 1.5, command_name="base_velocity", std=math.sqrt(0.25)
    )
    wheels_track_ang_vel_z = _wheels(mdp.track_ang_vel_z_exp, 1.0, command_name="base_velocity", std=math.sqrt(0.25))
    wheels_lin_vel_z = _wheels(mdp.lin_vel_z_l2, -2.0)
    wheels_ang_vel_xy = _wheels(mdp.ang_vel_xy_l2, -0.05)
    wheels_flat_orientation = _wheels(mdp.flat_orientation_l2, -2.5)
    wheels_joint_pos = _wheels(
        mdp.joint_position_penalty,
        -0.7,
        asset_cfg=SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        stand_still_scale=5.0,
        velocity_threshold=0.3,
    )
    wheels_contact_without_cmd = _wheels(
        mdp.feet_contact_without_cmd,
        0.1,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_foot"),
        command_name="base_velocity",
    )
    wheels_undesired_contacts = _wheels(
        mdp.undesired_contacts,
        -1.0,
        threshold=1.0,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        for name in STANCE_ONLY_TERMS:
            term = getattr(self, name)
            setattr(
                self,
                name,
                RewTerm(func=mdp.in_stance, weight=term.weight, params={"term": term.func, "term_params": term.params}),
            )


@configclass
class BipedSwitchTerminationsCfg(BipedTerminationsCfg):
    # The velocity task's limit, on four wheels only: off while a stance is commanded and over the
    # grace after a release.
    bad_orientation = DoneTerm(
        func=mdp.when_on_wheels, params={"term": mdp.bad_orientation, "term_params": {"limit_angle": 0.8}}
    )


@configclass
class RobotEnvCfgBipedSwitch(RobotEnvCfgBiped):
    commands: BipedSwitchCommandsCfg = BipedSwitchCommandsCfg()
    rewards: BipedSwitchRewardsCfg = BipedSwitchRewardsCfg()
    terminations: BipedSwitchTerminationsCfg = BipedSwitchTerminationsCfg()


@configclass
class RobotPlayEnvCfgBipedSwitch(RobotPlayEnvCfgBiped):
    commands: BipedSwitchCommandsCfg = BipedSwitchCommandsCfg()
    rewards: BipedSwitchRewardsCfg = BipedSwitchRewardsCfg()
    terminations: BipedSwitchTerminationsCfg = BipedSwitchTerminationsCfg()
