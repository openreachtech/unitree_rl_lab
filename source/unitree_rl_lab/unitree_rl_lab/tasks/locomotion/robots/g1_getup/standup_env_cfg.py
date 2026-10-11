"""G1 (23dof) stand up from a deep crouch, then stand still.

Step 2 of a two-step get-up (step 1, lying -> crouch, comes later), and on its own the
"unfold from the shipping posture" motion. Same observations, actions and rewards as the
supine task (``getup_env_cfg``); what changes:

* Reset from a bank of settled crouch states (``scripts/make_g1_crouch_bank.py``): deep
  squat, feet flat, pelvis 0.23-0.45 m, torso leaning forward. On flat ground the G1 cannot
  crouch with an upright torso (ankle dorsiflexion limit ~0.87 rad), it tips backwards.
* No unactuated phase, no pulling force, no action-scale curriculum: the policy starts at
  the deploy scale from the first step, holding the crouch until it acts.
* Extra terms on top of HoST's (same group prefixes): joint / base speed limits (regu;
  base 0.5 m/s, rising ~0.45 m takes about a second) and stand-still terms once up
  (target: nominal pose, joint and foot speed).
* Falling over ends the episode: this policy is not trained to get up from lying.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import getup_env_cfg as base
from . import mdp as getup_mdp

MAX_JOINT_VEL = 6.0  # rad/s
MAX_BASE_VEL = 0.5  # m/s


@configclass
class EventCfg(base.EventCfg):
    reset_pose = EventTerm(
        func=getup_mdp.reset_from_state_bank,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg(base.RewardsCfg):
    # -- speed limits: squared excess over the limit
    regu_joint_speed_limit = RewTerm(
        func=getup_mdp.joint_vel_over_limit_l2, weight=-1.0, params={"max_vel": MAX_JOINT_VEL}
    )
    regu_base_speed_limit = RewTerm(
        func=getup_mdp.base_lin_vel_over_limit_l2, weight=-50.0, params={"max_vel": MAX_BASE_VEL}
    )
    # -- stand still once the pelvis is above PHASE3_HEIGHT
    target_stand_pose = RewTerm(
        func=getup_mdp.stand_pose_exp, weight=10.0, params={"std": 0.5, "min_base_height": base.PHASE3_HEIGHT}
    )
    target_stand_feet_vel = RewTerm(
        func=getup_mdp.stand_feet_vel_l2,
        weight=-20.0,
        params={"feet_cfg": base.FEET, "min_base_height": base.PHASE3_HEIGHT},
    )


@configclass
class TerminationsCfg(base.TerminationsCfg):
    # crouch starts at up to ~0.9 rad of forward lean
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.3})


@configclass
class RobotEnvCfg(base.RobotEnvCfg):
    """G1 23dof stand-up from crouch."""

    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0

        action = self.actions.JointPositionDeltaAction
        action.unactuated_steps = 0
        action.assist_frac_init = 0.0
        action.scale_init = action.scale
        self.curriculum.getup = None


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
