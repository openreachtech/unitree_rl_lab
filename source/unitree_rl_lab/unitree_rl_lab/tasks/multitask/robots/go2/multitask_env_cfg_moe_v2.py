"""The merged policy, second version: locomotion, acrobatics and a bipedal stance in three experts.

Two changes from ``multitask_env_cfg_moe``, and everything else is inherited from it so the
difference stays readable.

*The third expert is the bipedal policy.* It used to hold a randomly initialised transition expert,
put there to absorb the run/take-off and landing/run states neither pre-trained policy had visited.
It never did: measured on the finished two-expert policy its routing weight read 0.000 while
running, 0.000 inside an acrobatic window, and 0.001 in the hand-back bin where it had the most to
contribute. `Go2-Multitask-Biped` takes the slot, and every expert now starts from a trained policy.

*The bipedal stance is commanded.* Once per episode in a share of them, held for ten seconds, with
the stance chosen by the commanded heading.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg_multitask import BipedRewardsCfg
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.mdp.gating import GATE_LOCOMOTION_OR_BIPED_UPRIGHT
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import apply_multitask_post_init
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg_moe import (
    MoeRewardsCfg,
    ACRO_SPEED_CEILING,
    MAX_LOCOMOTION_ERROR,
    MoeCommandsCfg,
    MoeCurriculumCfg,
    RobotEnvCfgMoe,
    RobotPlayEnvCfgMoe,
)

BIPED_HOLD_S = 10.0
"""How long a commanded stance is held.

Matched to the velocity command's own 10 s resampling period, so a stance spans exactly one
commanded gait and the episode contains one rise, one hold and one descent.
"""

BIPED_TRIGGER_RANGE = (2.0, 8.0)
"""When the stance starts, within the 20 s episode.

The lower bound leaves time to be genuinely running or standing first, so the rise is entered from
a settled state rather than from the spawn transient. The upper bound leaves at least two seconds
of quadruped afterwards, which is what makes the descent learnable at all: the bipedal expert has
never come down -- its own episodes end with it still up -- so the only pressure is the locomotion
rewards returning when the window closes.
"""

BIPED_EPISODE_SHARE = 0.5
"""Share of episodes that carry a stance.

Ten seconds of a twenty second episode is half the budget, and the acrobatic moves interleave at
1.5-3 s intervals and were already the weaker half of the merged policy -- backflip 0.543,
handspring 0.405, left sideflip 0.272 against 1.000 for each standalone. Splitting per episode
keeps the stance long enough to contain a whole rise-hold-descent while leaving half the episodes
at the acrobatics' original exposure. Net cost to the acrobatics is about a quarter of their
windows rather than half.
"""


@configclass
class MoeV2CommandsCfg(MoeCommandsCfg):
    """The merged commands, with the bipedal stance switched on.

    Mutual exclusion is handled on the acrobatics side: a flip defers while a stance is commanded.
    The asymmetry decides itself -- a flip re-arms every 1.5-3 s and loses little by waiting, a
    stance fires once per episode and holds for ten seconds.
    """

    handstand = mdp.HandstandCommandCfg(
        asset_name="robot",
        pinned=False,
        direction_matched=True,
        episode_probability=BIPED_EPISODE_SHARE,
        trigger_time_range=BIPED_TRIGGER_RANGE,
        hold_duration_range=(BIPED_HOLD_S, BIPED_HOLD_S),
        velocity_command_name="base_velocity",
        # Offered only below a speed the curriculum raises. The expert has only ever risen from
        # rest, so a stance commanded at the full ceiling is a state it has never seen.
        initial_takeoff_speed_limit=0.3,
        debug_vis=False,
    )


# Where the commanded speed range starts, copied from Go2-Multitask-Gallop-Phase2. v2 had no
# velocity curriculum at all: `ranges` was set equal to `limit_ranges`, so the very first iteration
# commanded up to 3.5 m/s. The locomotion expert arrives able to run 3.36 m/s and the merged policy
# is not that expert, so the range was out of reach from the start -- and this is the failure mode
# `lin_vel_cmd_levels` is written about: the exponential tracking reward goes numerically flat over
# most of the command distribution, the gradient disappears while every penalty stays live, and
# standing still becomes reward-maximising. Measured on the finished v2: it tracks 1.5 and 2.0 m/s
# fine, then delivers 0.05 m/s at 2.5 and 0.00 at 3.0 and 3.4, using 10% of available torque. Not
# saturation -- the cliff.
INITIAL_LIN_VEL_X = (-0.5, 1.5)
INITIAL_LIN_VEL_Y = (-0.5, 0.5)
INITIAL_ANG_VEL_Z = (-0.5, 0.5)

# And where it stops. The locomotion expert runs 3.36 m/s alone; the mixture does not, and asking
# it to chase a speed it cannot reach costs the other two skills outright.
#
# Measured across the run that first added the ratchet, as the commanded ceiling climbed:
#
#     range  jump attempt-success  handstand success
#      2.40              0.420           0.119   <- the best either skill measured, anywhere
#      2.90              0.453           0.076
#      3.10              0.191           0.060
#      3.10 (later)      0.007           0.031
#      3.10 (later)      0.000           0.000
#
# ``attempting_fraction`` held at 0.021-0.031 throughout, so the skills were still being offered at
# the same rate and still being tried -- they stopped *working*. Past about 2.9 the locomotion
# gradient takes the shared parameters and the acrobatics and bipedal experts are overwritten;
# ``track_lin_vel_xy`` fell to 0.84, under its own 0.90 down-threshold, at the same iterations.
#
# The velocity ratchet cannot find this level by itself: it judges on the tracking reward alone and
# has no way to see that a flip stopped landing. So the ceiling is set here instead, inside the
# band where all three behaviours measured well together. Raising it is a change with an owner --
# and a reason to re-measure both skills, not just the speed.
LIN_VEL_X_CEILING = 2.5


@configclass
class MoeV2CurriculumCfg(MoeCurriculumCfg):
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            # Two-way from the start. The one-way version cost this repository two Gallop retrains:
            # once the range is past what the robot can do, nothing lowers it again.
            "decrease_threshold": 0.6,
            "state_file": "logs/rsl_rl/go2_multitask_v2/lin_vel_cmd_state.json",
            # No `assist_free_only`: unlike the gallop tasks, v2 has no tow assist, so every
            # environment is already an honest sample of unaided ability.
        },
    )

    handstand_takeoff_speed = CurrTerm(
        func=mdp.handstand_takeoff_speed_levels,
        params={
            "command_name": "handstand",
            "jump_command_name": "jump",
            "maximum_speed": ACRO_SPEED_CEILING,
            "max_velocity_error": MAX_LOCOMOTION_ERROR,
            "state_file": "logs/rsl_rl/go2_multitask_v2/handstand_takeoff_state.json",
        },
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Its own state file: sharing the v1 path would have the two versions writing one another's limits.
        self.takeoff_speed.params["state_file"] = "logs/rsl_rl/go2_multitask_v2/takeoff_speed_state.json"
        # Promoted from sandbox Try X2, and the single largest measured win in this task.
        #
        # The inherited raise threshold is 0.6 -- a per-attempt success rate the merged policy never
        # reached, because a mixture is not the specialist it was built from: measured 0.22-0.40
        # across every run. So the take-off limit sat at its 0.3 m/s floor for all 3000 iterations
        # of every v2 run, an acrobatic move was only ever offered below 0.3 m/s, and the expert got
        # a third of the attempts it would otherwise have had (349 under the play command
        # distribution against 1103 with the velocity command pinned to zero). Too few attempts kept
        # the success rate down, which kept the threshold out of reach. A closed loop.
        #
        # Lowering the bar to something reachable opens it. Probed identically at 3000 iterations,
        # per-attempt flip success went 0.079 -> 0.785 and the limit climbed to its 1.0 ceiling, at
        # no cost to the other two skills: top speed 2.43 -> 2.45 m/s, bipedal stance 16 windows ->
        # 69, all successful. Against the *best* checkpoint the old setting produced anywhere
        # (model_1500, 0.445) it is still a 1.8x improvement.
        #
        # The same shape as `lin_vel_cmd_levels`'s own recorded failure, one level up: there a
        # command range past what the robot could do flattened the tracking reward, here a success
        # threshold past what the mixture could reach cut off the supply of attempts. Sandbox Try X1
        # reached the same place from the other end -- protecting the acrobatics expert with a lower
        # learning rate raised its success rate until it cleared the original 0.6 -- which is the
        # evidence that this loop, and not forgetting on its own, was the thing to fix.
        self.takeoff_speed.params["increase_threshold"] = 0.45
        # Kept clear of the raise threshold. The inherited 0.4 would leave a 0.05-wide dead band and
        # the limit would oscillate on sampling noise instead of settling.
        self.takeoff_speed.params["decrease_threshold"] = 0.30


# Every bipedal reward term, taken from the expert rather than restated here. Restating them would
# be fifteen terms of nested SceneEntityCfg, each one an opportunity to transcribe a body list
# wrong, and each wrong one silent: an unresolved or mis-scoped SceneEntityCfg does not raise, it
# evaluates over the wrong links. Reading them off the expert makes "the same content as the
# expert" true by construction instead of by review.
_BIPED_REWARDS = BipedRewardsCfg()


@configclass
class MoeV2RewardsCfg(MoeRewardsCfg):
    """The three-expert reward set: the mixture's own terms, plus the bipedal expert's.

    The first v2 run had none of these. It commanded the stance, routed the gate to the bipedal
    expert, and paid nothing for arriving -- while ``GATE_LOCOMOTION`` correctly switched the
    quadruped rewards off for the duration. Entering the stance therefore *lost* the tracking
    reward and kept every ungated effort penalty, so it was strictly negative, and the policy
    unlearned it exactly as it should have: ``handstand/success`` peaked at 0.134 near iteration
    400 and decayed to 0.000, with the gate weight on the bipedal expert following it down
    (``biped_tilted`` 0.86 -> 0.004). Measured afterwards, all four feet carried 38-56% ground
    contact through the whole stance window -- it simply walked.

    Names that already exist on ``MoeRewardsCfg`` are left alone. ``undesired_contacts`` there is
    ungated and covers head, hips, thighs and calves at -1.0, which is a superset of the expert's
    stance-routed version, so nothing goes unpriced by not overriding it -- and the two -20.0 terms
    below are what actually price a collapse, for the reason the expert's own notes give: a bounded
    count shared across eight links cannot charge a 3 kN kneel differently from a calf brushing the
    floor. ``feet_air_time`` is carried under its own name because the quadruped one is still
    needed outside the stance.
    """

    # -- posture and balance, all gated on the stance being commanded -----------------------------
    stance_pitch = _BIPED_REWARDS.stance_pitch
    stance_roll = _BIPED_REWARDS.stance_roll
    base_height = _BIPED_REWARDS.base_height
    upright_balance = _BIPED_REWARDS.upright_balance
    stance_held = _BIPED_REWARDS.stance_held

    # -- resting part of itself on the floor, priced properly -------------------------------------
    head_contact = _BIPED_REWARDS.head_contact
    stance_leg_contact = _BIPED_REWARDS.stance_leg_contact
    lifted_contact = _BIPED_REWARDS.lifted_contact

    # -- the tucked pair, whichever it is ---------------------------------------------------------
    lifted_hip_motion = _BIPED_REWARDS.lifted_hip_motion
    lifted_thigh_motion = _BIPED_REWARDS.lifted_thigh_motion
    lifted_calf_motion = _BIPED_REWARDS.lifted_calf_motion

    # -- CoM-CoP balance, on the commanded stance's feet only -------------------------------------
    pendulum_angle = _BIPED_REWARDS.pendulum_angle
    pendulum_instability = _BIPED_REWARDS.pendulum_instability
    handle_length = _BIPED_REWARDS.handle_length
    support_polygon = _BIPED_REWARDS.support_polygon

    # -- cadence on the stance pair. Distinct name: the quadruped `feet_air_time` above is gated on
    #    GATE_LOCOMOTION and so is already off for the duration, leaving no double count.
    biped_feet_air_time = _BIPED_REWARDS.feet_air_time

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # One tracking term covering both regimes rather than a second, parallel one. See
        # GATE_LOCOMOTION_OR_BIPED_UPRIGHT: the stance keeps a velocity command to follow -- which
        # is what the expert was trained to do -- but only once the robot is actually up on two
        # legs, because tracking is fully satisfiable from a quadruped stance and paying it from
        # there is what makes never rising the better trade.
        self.track_lin_vel_xy.params["gate"] = GATE_LOCOMOTION_OR_BIPED_UPRIGHT
        self.track_ang_vel_z.params["gate"] = GATE_LOCOMOTION_OR_BIPED_UPRIGHT


@configclass
class RobotEnvCfgMoeV2(RobotEnvCfgMoe):
    commands: MoeV2CommandsCfg = MoeV2CommandsCfg()
    curriculum: MoeV2CurriculumCfg = MoeV2CurriculumCfg()
    rewards: MoeV2RewardsCfg = MoeV2RewardsCfg()

    def __post_init__(self):
        self.episode_length_s = 20.0
        apply_multitask_post_init(self)
        # `limit_ranges` is left at the full (-1.0, 3.5); this is only where the ratchet starts.
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = INITIAL_LIN_VEL_X
        ranges.lin_vel_y = INITIAL_LIN_VEL_Y
        ranges.ang_vel_z = INITIAL_ANG_VEL_Z
        limits = self.commands.base_velocity.limit_ranges
        limits.lin_vel_x = (limits.lin_vel_x[0], LIN_VEL_X_CEILING)


@configclass
class RobotPlayEnvCfgMoeV2(RobotPlayEnvCfgMoe):
    commands: MoeV2CommandsCfg = MoeV2CommandsCfg()
    curriculum: MoeV2CurriculumCfg = MoeV2CurriculumCfg()
    rewards: MoeV2RewardsCfg = MoeV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Both ceilings stay for the same reason: they are not training restrictions to be lifted
        # at the end but the speeds above which the move is not offered at all.
        self.curriculum.handstand_takeoff_speed = None
        self.commands.handstand.initial_takeoff_speed_limit = ACRO_SPEED_CEILING
        # The velocity range is a training restriction, unlike the two ceilings above: play should
        # exercise the whole commandable range. The state file goes off for the reason spelled out
        # on RobotPlayEnvCfgGo2Run -- the curriculum manager still runs at play time, and the line
        # below has just set the range to its ceiling, so a single play session would otherwise
        # rewrite the training state with 3.5.
        # The play config does not inherit RobotEnvCfgMoeV2, so the ceiling has to be applied here
        # too -- otherwise play would command 3.5 m/s at a policy only ever trained to 2.5.
        limits = self.commands.base_velocity.limit_ranges
        limits.lin_vel_x = (limits.lin_vel_x[0], LIN_VEL_X_CEILING)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.lin_vel_cmd_levels.params["state_file"] = None
