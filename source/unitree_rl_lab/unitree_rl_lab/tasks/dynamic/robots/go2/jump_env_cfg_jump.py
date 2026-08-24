"""Go2-Jump: Phase 2's jump, on its own, at 0.40m, with the knee actually used.

Phase 2 trains jump + backflip + sideflip together at a fixed 0.20m and keeps the full
EFGCL assist schedule. This task keeps that recipe -- assist included -- and changes four
things, each for a measured reason.

WHY THE ASSIST STAYS ON
-----------------------
The sandbox ``Go2-Jump-MaxHeight*`` tasks set ``initial_assist_scale = 0.0``, which was
right for their purpose (resuming from a policy that already jumps, where stacking assist
on top teaches the policy to stop contributing). But ``assist_scale`` multiplies the
*crouch* pulse as well as the launch force (``JumpCommand._apply_assistance``), so turning
it off also removed the downward shove that physically teaches the robot to load its legs
before pushing.

The MuJoCo capture of the 0.40m no-assist policy shows what that costs. Over the push-off:

    net work    thigh 60.2 J (68%)   calf 21.3 J (24%)   hip 7.4 J (8%)
    calf travel FL -1.44 -> -1.29, FR -1.43 -> -1.47, RL -1.50 -> -1.51

The knee ends where it started -- it is acting as a strut, not a motor -- and the crouch
only reaches -1.71 rad against a -2.72 rad joint limit. The robot jumps by swinging its
thighs, which are the *weaker* joints (23.7 N*m vs the knee's 45.43) and which run into
the speed derate: FR_thigh sat at its speed-limited ceiling for 198 ms of a 351 ms
push-off. So this task starts from Phase 1 with the assist at full scale, exactly as
Phase 2 does, to get the crouch demonstrated physically.

WHY joint_torques IS RELAXED
----------------------------
``joint_torques_l2`` penalises the SQUARE of torque, so the stronger a joint is, the more
it costs to use:

    thigh at 20 N*m -> 400 units      knee at 45 N*m -> 2025 units  (5.1x)

That is a direct disincentive against the one joint with torque to spare, and it matches
what the capture shows the policy doing. Weight is cut 10x here (-2.0e-4 -> -2.0e-5).

SUMMARY.md records an earlier attempt that halved ``joint_torques`` and ``action_rate``
and still plateaued at ~0.20m. That does not contradict this: back then ``target_height``
was pinned at 0.20 and ``motion_progress`` scores ``|max_height - target|`` two-sided, so
the robot was penalised for exceeding 0.20 no matter what the effort penalties said. The
plateau was the target, not the penalty.

``action_rate`` (-0.1) is left alone deliberately -- it also opposes an explosive
extension, but changing both at once would make the result unattributable. It is the next
knob if the knee still under-contributes.
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg_phase2 import (
    CommandsCfgPhase2,
    EventCfgPhase2,
    FlipRewardsCfg,
    RobotEnvCfgPhase2,
)

TARGET_HEIGHT = 0.60
EXPERIMENT_DIR = "logs/rsl_rl/go2_jump_60"

# The height every other height quantity is measured against: height_delta is
# `root_pos_w[2] - nominal_standing_height`. The 0.40 default is wrong for this robot.
#
# Measured with the trained policy holding its idle pose:
#     Isaac Lab  root_pos_w[2] = 0.2870
#     MuJoCo     base_link z   = 0.3239   (imu site 0.3662 - go2.xml site offset 0.0423)
#
# At 0.40 the consequences compound, because three separate things key off this constant:
#
#   max_height              under-reported the true rise by 0.113 m
#   motion_progress_standing height_term = exp(-0.113^2/0.01) = 0.28, so the robot forfeits
#                           72% of that reward simply by standing correctly
#   landed                  requires |height_delta| < 0.10, but standing already gives
#                           0.113 -- a correctly standing robot was classed as NOT landed,
#                           which is why it had to stretch unnaturally tall to score at all
#
# 0.30 sits just above Isaac Lab's own 0.287 and splits the difference with MuJoCo's 0.324,
# which still differ by 3.7 cm for reasons that are NOT this constant -- same policy, same
# joint offsets, different leg extension, i.e. a USD/MJCF geometry mismatch that has to be
# chased separately.
NOMINAL_STANDING_HEIGHT = 0.30

# --- actuator model -------------------------------------------------------------------
# Carried over from the MaxHeight experiments, where both halves were established against
# external references rather than guessed.
#
# TORQUE: the stock UnitreeActuatorCfg_Go2HV gives every joint the bare GO-M8010-6 curve
# (Y1 20.2 / Y2 23.4 N*m). That is right for hip and thigh, which sit on the motor, and
# wrong for the calf, which reaches roughly double. The project's MuJoCo model agrees:
# go2.xml clamps abduction/hip at +/-23.7 and the knee at +/-45.43. So the calf's Y is
# raised to 45.43, keeping the stock Y1/Y2 ratio (0.863) for the push-off peak.
#
# SPEED: X1/X2 are deliberately LEFT at the motor's 13.5 / 30.0 rad/s for the calf too.
#
# An earlier version divided them by r = 45.43/23.4 = 1.94, on the reasoning that a
# linkage reduction trades speed for torque and must preserve peak power. That is correct
# IF the extra torque comes from a reduction -- but that was inferred, never measured, and
# it made the knee unusable for jumping: at X1 = 6.95 rad/s the ceiling collapsed to
# 7-16 N*m over a third of the push-off, and the MuJoCo capture showed the policy
# responding rationally by leaving the knee idle (21 J of work against the thigh's 60 J).
# Modelling the knee as slow was itself a plausible cause of the knee never being used.
#
# Note what this implies: 39.22 N*m at 13.5 rad/s is ~529 W of peak joint power against
# ~273 W for hip and thigh. That is only physical if the knee is genuinely a different
# actuator rather than the same motor geared down. It is, however, exactly what the
# project's MuJoCo model already assumes -- a flat +/-45.43 clamp with no speed derate at
# all -- so this choice moves the two simulators closer together, not further apart.
#
# ARMATURE: MuJoCo's 0.01 on every joint. An earlier version used 0.0122 * r^2 = 0.046 for
# the calf, which is what a genuine 1.94:1 reduction implies, but it left the simulators
# 4.6x apart on knee inertia and the heavier training joint taught the policy to command
# harder than the lighter MuJoCo knee could take -- that mismatch produced a full tumble in
# sim2sim. Measured cost of matching: 0.381m -> 0.325m at the same 0.40 target, because
# heavier joints stay below the speed derate longer and deliver more impulse.
CALF_PEAK_TORQUE = 45.43
CALF_PUSH_TORQUE = 45.43 * (20.2 / 23.4)  # keep the stock Y1/Y2 ratio -> 39.22

GO2_CORRECTED_ACTUATOR_CFG: ArticulationCfg = UNITREE_GO2_CFG.replace(
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            # JOINT FRICTION, matched to the MuJoCo model. UnitreeActuatorCfg_Go2HV leaves
            # Fs/Fd unset, i.e. zero, so Isaac Lab's joints had no friction of any kind
            # while go2.xml gives every joint `frictionloss="0.2"` and `damping="0.1"`.
            # UnitreeActuator.compute() applies
            #     applied_effort -= Fs * tanh(joint_vel / Va) + Fd * joint_vel
            # so Fs is the dry (Coulomb) torque in N*m and Fd the viscous coefficient in
            # N*m*s/rad -- a direct one-to-one mapping onto MuJoCo's two fields.
            #
            # `friction` (the PhysX joint-friction coefficient) drops to 0: it is a second
            # dry-friction mechanism, and leaving it at 0.01 alongside Fs = 0.2 would
            # double-count what MuJoCo models once.
            #
            # Why this matters here: the policy pitches forward ~160 deg and lands on its
            # back in MuJoCo while scoring success 0.998 in Isaac Lab. Frictionless joints
            # respond faster and more symmetrically than damped ones, and the divergence
            # only appeared once the motion became violent enough (deep crouch, 0.685 m)
            # for that difference to matter.
            friction=0.0,
            Fs=0.2,
            Fd=0.1,
            Y1={
                ".*_hip_joint": 20.2,
                ".*_thigh_joint": 20.2,
                ".*_calf_joint": CALF_PUSH_TORQUE,  # 39.22
            },
            Y2={
                ".*_hip_joint": 23.4,
                ".*_thigh_joint": 23.4,
                ".*_calf_joint": CALF_PEAK_TORQUE,  # 45.43
            },
            # Same speed envelope on every joint -- see the SPEED note above.
            X1={
                ".*_hip_joint": 13.5,
                ".*_thigh_joint": 13.5,
                ".*_calf_joint": 13.5,
            },
            X2={
                ".*_hip_joint": 30.0,
                ".*_thigh_joint": 30.0,
                ".*_calf_joint": 30.0,
            },
            armature={
                ".*_hip_joint": 0.01,
                ".*_thigh_joint": 0.01,
                ".*_calf_joint": 0.01,
            },
        ),
    },
)


@configclass
class CommandsCfgJump(CommandsCfgPhase2):
    """Phase 2's jump command, jump-only, at 0.60m. Assist schedule inherited untouched.

    The target is a fixed point, never a range: the height reward's e-folding distance is
    0.10m, so environments given targets spread wider than that share no good behaviour and
    hedging scores ~0 everywhere. A ranged target collapsed learning outright once already
    (max_height 0.233 -> 0.026, success 0.000 by iteration 700).
    """

    jump = CommandsCfgPhase2().jump.replace(
        enable_jump=True,
        enable_backflip=False,
        enable_sideflip=False,
        target_height_range=(TARGET_HEIGHT, TARGET_HEIGHT),
        target_pitch_turns_range=(0.0, 0.0),
        target_roll_turns_range=(0.0, 0.0),
        nominal_standing_height=NOMINAL_STANDING_HEIGHT,
        # 26 deg instead of the 37 deg default. At 37 the gate was measurably not doing its
        # job: every environment peaked at 33.7 deg mean tilt, under the threshold by 0.9
        # deg, so `success` 0.997 meant "scraped past" rather than "landed upright".
        landing_upright_threshold=-0.90,
        state_file=f"{EXPERIMENT_DIR}/jump_curriculum_state.json",
    )


@configclass
class JumpRewardsCfg(FlipRewardsCfg):
    """Phase 2's reward set with the effort penalty on the strongest joint relaxed."""


    # An explosive knee extension is a large action change by definition, so this term
    # opposes the very strategy under test. Cut 10x rather than removed, so the policy is
    # still discouraged from pure chatter.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Removed outright. Being an L2 term it charged 5.1x more for the knee at capacity
    # (45 N*m) than the thigh (20 N*m) -- backwards for this experiment. The actuator's own
    # torque-speed envelope still bounds what any joint can produce.
    joint_torques = None

    # Back to the plain gate. `pre_jump_standing_reward_windup` extends the standing reward
    # from the trigger until `assist_delay_s` elapses -- and that reward is
    # upright * stillness, where stillness is exp(-|v|^2/0.25^2) on the body velocity. It
    # therefore pays the robot to hold still through exactly the window a countermovement
    # would occupy, and at linear_std = 0.25 a 0.5 m/s descent scores essentially zero.
    # Its docstring says as much: it exists to tell the robot "to simply hold still until
    # the assist force lands".
    #
    # That is very likely why the body dips 0.000 m in the MuJoCo capture despite the
    # crouch pulse, and the failed 0.60 attempt made it worse by lengthening
    # assist_delay_s, which lengthened this reward window in step with it -- pushing the
    # robot down harder while paying it more to not move.
    #
    # `pre_jump_standing_reward` gates on `~enabled` alone, so the stillness reward stops
    # the instant the command fires. Crouching is then neither paid for nor penalised,
    # which is the whole intent: remove the obstacle, do not reward the mechanism.
    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )

    # -0.05 -> -0.15. Purely a side-effect suppressor, not a means: it penalises angular
    # velocity about the axes the commanded motion does NOT target, which for a pure
    # vertical jump is all of them. The MuJoCo capture shows why it needs strengthening --
    # the robot is already tilted 28 deg by 0.32 s and leaves the ground at 31 deg, so it
    # is airborne with no way to correct and lands at 45 deg, then tips (94 deg by 1.40 s).
    # Suppressing that costs nothing in height: tilting is not how the robot gets up, it is
    # an asymmetry in how it pushes.
    # -0.15 -> -1.0. The take-off rotation is systematic, not noise: measured across 64
    # environments the policy leaves the ground at -0.671 rad/s pitch and +0.366 rad/s roll
    # with the SAME sign every time and std under 0.11. At -0.15 that costs 0.15 * 0.67^2 =
    # 0.067, two orders of magnitude under motion_progress's ~1.0, so there was never a
    # reason to stop twisting. At -1.0 it is ~0.45 and finally competes.
    #
    # Still not a reward on a mechanism: twisting is not how the robot gains height, it is
    # an asymmetry in how it pushes. The previous increase (-0.05 -> -0.15) cost no height.
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-1.0,
        params={"command_name": "jump", "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfgJump(EventCfgPhase2):
    """Phase 2's events plus mass randomisation.

    Until now the robot was a single fixed 16.087 kg specimen: only friction was
    randomised (0.4-1.2, added earlier when a backflip transferred poorly to MuJoCo).
    Mass then turned out to differ between the simulators by 5.8%:

        URDF (the source of truth)  16.087 kg
        Isaac Lab                   16.087 kg   -- matches the URDF exactly
        MuJoCo MJCF                 15.206 kg   -- 0.88 kg light

    and the per-body split differs too (MJCF folds the foot into the calf, and its
    hip/thigh masses do not match the URDF either). With kp = 25, that showed up directly
    as posture: Isaac Lab's knees sag 0.11-0.12 rad further and it stands 3.7 cm lower,
    which in turn is a plausible share of the 29% jump-height gap between the two.

    Chasing an exact match is the wrong fix. Friction never became a problem precisely
    because MuJoCo's value sits inside the randomised range, and hardware will be a third
    specimen again -- payload and battery move the mass on their own. +/-10% brackets
    MuJoCo's -5.8% with margin.

    This costs height: the policy has to launch the heaviest specimen it might get.
    """

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # PD gains, for the same reason as the mass. Randomising mass took the MuJoCo height
    # gap from +29% to +6% and left the policy jumping HIGHER, not lower -- the variation
    # acted as a regulariser rather than a tax. What remains is that the robot still leaves
    # the ground with pitch rate it cannot correct in flight: 2 deg at 0.20 s, 19 deg by
    # 0.36 s, past 100 deg by 1.16 s, while Isaac Lab scores success 1.000 with
    # base_contact 0.001 on the same policy.
    #
    # That is the same signature as before -- fine in one simulator, not the other -- so an
    # unabsorbed difference remains rather than a reward that needs tuning. Gains are the
    # next candidate: they set how hard the leg resists at touchdown and how fast it tracks
    # during push-off, both of which feed the take-off rotation, and neither simulator's
    # value is verified against hardware.
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.80, 1.20),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RobotEnvCfgJump(RobotEnvCfgPhase2):
    """Phase 2 jump-only at 0.40m, assist on, relaxed torque penalty, corrected actuator."""

    commands: CommandsCfgJump = CommandsCfgJump()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    events: EventCfgJump = EventCfgJump()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GO2_CORRECTED_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotPlayEnvCfgJump(RobotEnvCfgJump):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # Never write back over the training curriculum state, and show what the policy
        # does unaided rather than what the assist is still doing for it.
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = 0.0




# --- sideflip, two rotations ------------------------------------------------------------
# Trained from Phase 1. Two full rotations unassisted, landing upright on 64/64 randomised
# robots, sim2sim-validated in MuJoCo. It jumps ~0.42 m, which is a means to the rotation
# rather than the goal.
#
# What it inherits, and why:
#
#   from Go2-Jump-60   the crouch-and-extend machinery -- pre_jump_standing_reward instead
#                      of the windup variant (which paid the robot to hold still through
#                      exactly the window a crouch needs), joint_torques removed,
#                      action_rate relaxed, and mass/gain randomisation
#   from Phase 2       non_target_rotation at -0.05. Under MOTION_SIDEFLIP that penalty is
#                      omega_y^2 + omega_z^2, and a sideflip cannot avoid some pitch/yaw
#                      coupling. At the -1.0 that fixed the jump's crooked take-off, a
#                      sideflip measured -0.464 against motion_progress of 0.030 and the
#                      policy stopped jumping at all (max_height 0.609 -> 0.010,
#                      base_contact on every episode). What helps a vertical jump actively
#                      prevents this motion.
#
# It took ten attempts to get assist_scale off 1.000, and every one of the nine failures had
# the same shape: the assist could not itself demonstrate two rotations, so `success` never
# fired and the 60%-success decay gate never opened. The policy was never the blocker. Three
# separate things were wrong with the assist and all three had to be fixed:
#
#   1. the couple fired while the feet were still planted, and the ground absorbed it
#   2. the couple block sat inside `assist_active`, so delaying it past that window stopped
#      it firing at all
#   3. it was sized against a roll inertia inferred from a one-sided force, understated 2.6x
#
# And the sizing itself was then measured wrong twice more, by taking the peak rotation over
# the whole episode -- which counts the tumble after landing. Measured properly, over the
# airborne phase only, on the Phase 1 standing policy:
#
#    900 N  -1.91 turns   grav_z at rest -0.23   lands on its side
#   1100 N  -2.04 turns   grav_z at rest -0.88   very nearly upright
#   1300 N  -2.49 turns   grav_z at rest +0.51   past vertical, lands inverted
#
# Completing the two turns and arriving upright are the same condition, not two: stopping a
# tenth of a turn short IS landing on your flank. That is why every run that fell short of
# the rotation also failed `landed`, and why the fix was one number rather than a reward
# change.
#
# Verify the assist before training, always, and measure it rather than trusting the eye or
# the arithmetic -- both were wrong here, repeatedly. Nine training runs were spent on an
# assist that a five-minute check would have ruled out. The check: play this task with a
# Phase 1 checkpoint and initial_assist_scale forced to 1.0. Phase 1 does nothing but hold a
# stand, so whatever the robot does is the external force's doing, and the rotation it
# produces has to reach the target before any policy is asked to learn from it.
#
#   cfg = parse_env_cfg("Go2-Sideflip-Double", num_envs=16)
#   cfg.commands.jump.initial_assist_scale = 1.0
#   cfg.commands.jump.state_file = None
#
# What the assist applies, in order, on the four hip bodies:
#
#   0.00 - 0.12 s   crouch   150 N down, triangular
#   0.12 - 0.34 s   launch   derived from flip_launch_height = 1.50, symmetric (no roll)
#   0.26 - 0.36 s   couple   1100 N up on FR/RR, 1100 N down on FL/RL (roll, no lift)
SIDEFLIP_TURNS = -2.0


@configclass
class CommandsCfgSideflipDouble(CommandsCfgJump):
    jump = CommandsCfgJump().jump.replace(
        enable_jump=False,
        enable_sideflip=True,
        target_height_range=(0.0, 0.0),
        target_roll_turns_range=(SIDEFLIP_TURNS, SIDEFLIP_TURNS),
        # Lift and spin come from separately sized mechanisms:
        #
        #   flip_launch_height    -> the launch force, symmetric across all four hips
        #   sideflip_couple_force -> pure torque, no translation at all
        #   sideflip_assist_force -> 0. The one-sided force does both at once, and mixing it
        #                            in re-adds uncontrolled lift on top of the launch;
        #                            doubling it for a second turn once reached 2.351 m.
        #
        # flip_target_height is what the reward asks for and flip_launch_height is what the
        # force is sized for. They were a single number, and raising it to buy flight time
        # moved the reward target out of reach at the same time: height_progress is
        # exp(-(max_height - target)^2 / 0.16), which reads 0.975 against 0.60 m and 0.012
        # against 1.50 m. The launch at 1.50 delivers 0.669 m on the standing policy.
        flip_target_height=0.60,
        flip_launch_height=1.50,
        sideflip_assist_force=0.0,
        sideflip_couple_force=1100.0,
        # Take-off is at 0.24 s under this launch, and the couple waits for it. Spending the
        # couple against the ground is what wasted the first several attempts.
        sideflip_couple_delay_s=0.26,
        sideflip_couple_duration_s=0.10,
        # 0.30 rad is 4.8% of one turn but only 2.4% of two, tight enough that a rotation
        # could complete and still be scored a failure -- which would stall the assist decay.
        rotation_tolerance_rad=0.60,
        # Touchdown is at 0.94 s, well past the 0.80 s the single-rotation tasks assume.
        # Judging `landed` before the robot is down keeps success at 0.
        minimum_landing_time_s=1.0,
        state_file="logs/rsl_rl/go2_sideflip_double/jump_curriculum_state.json",
    )


# Both widened terms are needed together, and each was run alone first:
#
#   rotation_scale (2*pi)^2   motion_progress climbed 0.053 -> 0.255, so the robot did start
#                             chasing the rotation -- but max_height collapsed to 0.023 m.
#                             With a gradient on rotation and none on height, spinning on the
#                             spot is cheaper than spinning in the air.
#   height_progress at 0.16   max_height rose 0.086 -> 0.195 -> 0.322 m, confirming height
#                             only ever failed to appear because nothing rewarded it. But
#                             with rotation still at exp(-16) from a two-turn target, there
#                             was no reason to use the air once it had it.
#
# They are the two halves of one deadlock: air time is worthless without a reason to rotate,
# and rotation is unreachable without air time.
@configclass
class SideflipDoubleRewardsCfg(JumpRewardsCfg):
    motion_progress = RewTerm(
        func=mdp.motion_progress_reward,
        weight=1.0,
        params={"command_name": "jump", "rotation_scale": (2.0 * math.pi) ** 2},
    )
    height_progress = RewTerm(
        func=mdp.jump_progress_reward,
        weight=1.0,
        params={"command_name": "jump", "scale": 0.16},
    )
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-0.05,
        params={"command_name": "jump", "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RobotEnvCfgSideflipDouble(RobotEnvCfgJump):
    """Sideflip, two rotations. Train from Phase 1."""

    commands: CommandsCfgSideflipDouble = CommandsCfgSideflipDouble()
    rewards: SideflipDoubleRewardsCfg = SideflipDoubleRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Flight is longer than any previous motion here; give the episode room for the
        # landing and the settle that `landed` checks for.
        self.episode_length_s = 5.0


@configclass
class RobotPlayEnvCfgSideflipDouble(RobotEnvCfgSideflipDouble):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = 0.0

