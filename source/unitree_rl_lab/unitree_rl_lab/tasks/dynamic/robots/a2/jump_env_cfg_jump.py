"""A2-Jump-60: Go2-Jump-60's recipe on A2, as the first rung of a height ladder.

This is a port of ``dynamic/robots/go2/jump_env_cfg_jump.py`` (``Go2-Jump-60``), which is
Go2's Phase 2 jump taken on its own with the assist schedule kept, the effort penalties
that opposed an explosive knee extension relaxed or removed, and mass / PD-gain
randomisation added. Every one of those decisions was made against a measurement on Go2 and
none of them are robot-specific, so they carry over unchanged. What is reproduced here, and
why, in the order the Go2 file argues it:

  ASSIST STAYS ON.        ``assist_scale`` multiplies the crouch pulse as well as the launch
                          force, so turning it off also removes the downward shove that
                          physically teaches the robot to load its legs before pushing.
                          Training starts from Phase 1 at full assist and the curriculum
                          decays it once 60% of episodes succeed.

  joint_torques REMOVED.  Being an L2 term it charges the square of torque, so the stronger
                          a joint is the more it costs to use -- a direct disincentive
                          against the one joint with torque to spare. A2's spread is wider
                          than Go2's, not narrower (calf 180 N*m against hip/thigh 120,
                          where Go2 is 45.43 against 23.7), so the argument holds harder
                          here. The actuator's own torque-speed envelope still bounds what
                          any joint can produce.

  action_rate RELAXED.    -0.1 -> -0.01. An explosive knee extension is a large action
                          change by definition. Cut 10x rather than removed, so pure
                          chatter is still discouraged.

  pre_motion_standing     ``pre_jump_standing_reward`` rather than Go2 Phase 2's windup
  BACK TO THE PLAIN GATE. variant (``pre_jump_standing_reward_windup``, dropped from
                          ``dynamic/mdp/rewards.py`` as dead here). That variant extends a
                          reward built on *stillness* from the
                          trigger until ``assist_delay_s`` elapses -- i.e. it pays the robot
                          to hold still through exactly the window a countermovement would
                          occupy. Crouching is then neither paid for nor penalised, which is
                          the intent: remove the obstacle, do not reward the mechanism.

  non_target_rotation     -0.05 -> -1.0. On Go2 the take-off rotation was systematic rather
  STRENGTHENED.           than noise (same sign on all 64 envs, std under 0.11), and at
                          -0.05 it cost two orders of magnitude less than motion_progress,
                          so there was never a reason to stop twisting. Twisting is not how
                          a robot gains height, it is an asymmetry in how it pushes.

  MASS AND GAINS          Both were added on Go2 to close a sim2sim gap, and both turned out
  RANDOMISED.             to act as regularisers rather than taxes. A2 has the same problem
                          waiting: its stiffness/damping are the only numbers in
                          ``UNITREE_A2_CFG`` with no source anywhere (a MuJoCo model is
                          torque-controlled and carries no gain), so +/-20% on them is
                          insurance against a value that was scaled rather than measured.

WHAT IS *NOT* CARRIED OVER
--------------------------
Go2 needed ``GO2_CORRECTED_ACTUATOR_CFG``, a jump-specific actuator override, because the
stock Go2 config gave its calf the bare GO-M8010-6 curve when the real calf reaches roughly
double. ``UNITREE_A2_CFG`` was reconstructed correctly to begin with -- peak torque and
no-load speed from ``a2_description``, Fs/Fd and armature from ``a2.xml`` -- so there is no
correction to apply and the scene keeps the stock A2 articulation.

WHAT THE ASSIST ACTUALLY DELIVERS -- MEASURED, NOT ASSUMED
----------------------------------------------------------
Go2's sideflip notes are emphatic that the assist must be measured before training rather
than trusted ("nine training runs were spent on an assist that a five-minute check would
have ruled out"). Run with zero actions -- a PD holding the default pose, i.e. what Phase 1
produces -- so whatever height the robot reaches is the external force's doing. 64 envs,
assist_scale forced to 1.0, peak height_delta over one episode:

    Go2-Jump-60   target 0.60   mean 0.972  std 0.057   1.62x target   success 0/64
    A2-Jump-60    target 0.60   mean 1.126  std 0.067   1.88x target   success 0/64
    A2-Jump-100   target 1.00   mean 1.974  std 0.132   1.97x target   success 0/64

The over-delivery is INHERITED, not introduced by the scaling: the reference Go2 task
overshoots by the same kind of factor and trained to success 0.997 regardless, so a passive
0/64 does not predict a stuck curriculum -- a policy has ample authority to shape its own
take-off, and `max_height` is what the policy produces, not what the force does to a rag
doll. Two thirds of the factor is structural and affects both robots equally: the launch
force is sized as `m*v0 / assist_duration_s` but applied over `assist_ramp_s +
assist_duration_s` with a linear ramp, so the delivered impulse is (ramp/2 + duration) /
duration = 1.60x (Go2) / 1.64x (A2) of the m*v0 the formula intends.

A2 still overshoots ~16% harder than Go2 in relative terms, and the leg spring-back is the
likely reason: the crouch pulse is 0.95 body weights on both, but A2's torque-to-weight is
roughly double Go2's, so a passively PD-held leg pushes back harder against it.

That deadlock was not hypothetical here: with the launch force sized off A2's true mass,
A2-Jump-60 ran 130 iterations with `success` flat at 0.000, `assist_scale` stuck at 1.000 and
`max_height` pinned at the passive 1.07 m. It is resolved by `jump_assist_mass` below, which
is calibrated by measurement rather than set to the real mass -- see the sweep table there.

Watch `Metrics/jump/success` early in any new rung. If it stays at 0 while `max_height` sits
far above target, re-run that sweep for the rung's own target height. Do not reach for reward
tuning first.

THE LADDER
----------
A2 is 2.49x Go2's mass but its actuators are ~5x stronger (120/180 N*m against 23.7/45.43),
so the torque-to-weight ratio is about 2x Go2's and the reachable height should be well past
the 0.60 m Go2 plateaus at. Rather than guess where, each rung trains from the one below:

    python scripts/rsl_rl/train.py --task A2-Jump-Phase1
    python scriptsts/rsl_rl/train.py --task A2-Jump-60  --resume --previous-task A2-Jump-Phase1

VERIFIED so far -- 300 iterations of Phase 1, then 1200 of A2-Jump-60. Evaluated unaided
(assist_scale 0.0) on 256 randomised envs: 0.592 m above stance (median 0.592, std 0.009,
min 0.570, max 0.614), absolute apex 1.008 m, 256/256 reaching the target and landing
upright. The assist decayed to 0.000 by iteration 800 and the policy kept climbing after it.

Only that rung exists. ``TARGET_HEIGHTS_CM`` in ``jump_heights.py`` is where the next one is
added, but adding the number is not sufficient on its own -- see ``jump_assist_mass`` below.

The target is a fixed point on every rung, never a range: ``motion_progress``'s height term
has an e-folding distance of 0.10 m, so environments given targets spread wider than that
share no good behaviour and hedging scores ~0 everywhere. A ranged target collapsed learning
outright on Go2 once already (max_height 0.233 -> 0.026, success 0.000 by iteration 700).
The ladder is how the range is covered instead.

Read the ceiling off ``Metrics/jump/max_height`` rather than off the task id -- the target
is what the reward asks for, and a rung that plateaus below its target is the answer. Add
rungs by extending ``TARGET_HEIGHTS_CM`` in ``jump_heights.py``; nothing else needs touching.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.a2.jump_env_cfg import (
    MASS_RATIO,
    NOMINAL_STANDING_HEIGHT,
    TIME_RATIO,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    RobotEnvCfg,
    StandingRewardsCfg,
    TerminationsCfg,
)
from unitree_rl_lab.tasks.dynamic.robots.a2.jump_heights import TARGET_HEIGHTS_CM


@configclass
class CommandsCfgJump(CommandsCfg):
    """Go2 Phase 2's assisted jump command, jump-only, with every duration and force scaled.

    Forces scale by mass (R_m = 2.49), durations by sqrt of length (R_t = 1.136). The launch
    force itself is NOT listed here because it is derived per-environment inside
    ``JumpCommand._apply_assistance`` from projectile motion -- ``mass * sqrt(2*g*h) /
    assist_duration_s``, with the mass auto-detected from the articulation. It therefore
    scales with A2 and with each rung of the ladder on its own.
    """

    jump = CommandsCfg().jump.replace(
        auto_trigger=True,
        enable_jump=True,
        enable_backflip=False,
        enable_sideflip=False,
        # Unchanged. This window exists to make an early, precommitted crouch a worse bet by
        # being unpredictable; it is not a dynamic quantity, and ``episode_length_s`` below
        # is sized to leave room after the latest trigger.
        trigger_time_range=(0.5, 2.0),
        # Overridden per rung -- see _build_height_variant.
        target_height_range=(0.60, 0.60),
        target_pitch_turns_range=(0.0, 0.0),
        target_roll_turns_range=(0.0, 0.0),
        assist_body_names=["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
        nominal_standing_height=NOMINAL_STANDING_HEIGHT,
        # CALIBRATED BY MEASUREMENT, not the robot's real 40.071 kg. This is the mass the
        # launch force is derived from (`m * sqrt(2*g*h) / assist_duration_s`), so it is the
        # one config-side knob that scales that force without touching the shared
        # JumpCommand. Left at the true mass the assist massively over-delivers and training
        # dies -- see the measurement table below.
        #
        # Swept with the trained Phase-1 policy driving A2-Jump-60 at assist_scale 1.0,
        # 64 envs, peak height_delta over one episode:
        #
        #     jump_assist_mass   mean peak   within +/-0.10   base_contact
        #        40.071 (true)     1.055 m        0/64            0.88
        #        32.0              0.618 m       63/64            0.20
        #        28.0              0.447 m        5/64            0.02
        #        24.0              0.307 m        0/64            0.00
        #        20.0              0.195 m        0/64            0.00
        #
        # 32.0 is 0.80x the true mass. Note how sharply it bites -- a 20% force cut drops the
        # height 41% -- because over half the peak comes from the legs' own spring-back
        # against the crouch pulse, not from the external force, and that contribution falls
        # away faster than the force does.
        #
        # What the true-mass setting cost, measured in training rather than argued:
        # A2-Jump-60 ran 130 iterations with max_height pinned at 1.06-1.08 (i.e. exactly the
        # passive value above -- the policy was not shaping its take-off at all), success
        # flat 0.000, assist_scale stuck at 1.000, base_contact 0.80-0.82 and mean_reward
        # flat at -7.3. motion_progress reads exp(-(1.07-0.60)^2/0.01) = e^-22, so there was
        # no height gradient to learn from at all.
        #
        # RE-MEASURE THIS PER RUNG before adding any rung above 0.60 m: the force scales
        # with sqrt(target_height) on its own, but the leg spring-back does not, so 0.80x
        # is calibrated for 0.60 m and is not guaranteed to hold at 1.00 m. The sweep is:
        # drive the task with the trained Phase-1 policy at assist_scale 1.0 and vary this
        # value until the peak lands on the new target.
        jump_assist_mass=32.0,
        # 0.50 / 0.10 / 0.12 / 0.12 / 0.12 / 0.80 s on Go2, all x R_t.
        command_duration_s=round(0.50 * TIME_RATIO, 2),  # 0.57
        assist_duration_s=round(0.10 * TIME_RATIO, 2),  # 0.11
        # A downward pulse on all four legs right at trigger, before the launch force, so the
        # robot physically experiences a crouch-load instead of relying on reward shaping to
        # elicit the timing. 150 N on a 16.087 kg Go2 is 0.95 body weights; x R_m keeps it
        # 0.95 body weights on a 40.071 kg A2.
        crouch_assist_force=round(150.0 * MASS_RATIO, 1),  # 373.6 N
        crouch_assist_duration_s=round(0.12 * TIME_RATIO, 2),  # 0.14
        # assist_delay_s == crouch_assist_duration_s so the launch force begins ramping in
        # exactly as the crouch pulse ends -- both sides of the handoff are at ~0 force.
        # assist_ramp_s smooths the launch onset the same way; a hard step reliably broke
        # training from a cold Phase 1 resume on Go2, regardless of magnitude.
        assist_delay_s=round(0.12 * TIME_RATIO, 2),  # 0.14
        assist_ramp_s=round(0.12 * TIME_RATIO, 2),  # 0.14
        initial_assist_scale=1.0,
        # A floor on when ``landed`` may first be true, so a robot that never left the ground
        # cannot be scored as having landed. 0.80 x R_t. Sanity check against the tallest
        # rung: a 1.00 m jump is 0.90 s of flight alone, so the floor is never the binding
        # constraint on any rung here.
        minimum_landing_time_s=round(0.80 * TIME_RATIO, 2),  # 0.91
        # 26 deg instead of the -0.8 default's 37 deg. Dimensionless, and carried over for
        # the reason it was tightened on Go2: at 37 deg a policy peaking at 33.7 deg mean
        # tilt scored success 0.997, which meant "scraped past" rather than "landed upright".
        landing_upright_threshold=-0.90,
        # Overridden per rung -- see _build_height_variant.
        state_file=None,
    )


@configclass
class JumpRewardsCfg(StandingRewardsCfg):
    """Go2 Phase 2's motion reward set with Go2-Jump-60's four changes."""

    # The standing trio is replaced by pre_motion_standing (upright x stillness, gated to
    # the idle phase) plus the motion progress terms below -- rewarding a quiet stand
    # throughout would fight the jump.
    upright = None
    standing_pose = None
    stillness = None

    # Removed outright, see the module docstring. The torque-speed envelope still bounds the
    # joint.
    joint_torques = None
    # -0.1 -> -0.01.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # The plain gate (``~enabled``), not Go2's windup variant -- stillness stops being paid
    # for the instant the command fires, so the countermovement window is free.
    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress = RewTerm(
        func=mdp.motion_progress_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress_standing = RewTerm(
        func=mdp.motion_progress_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    # -1.0, Go2-Jump-60's value. The term is angular velocity squared; A2's angular rates
    # for an equivalent motion are 1.14x smaller, so at the same weight this lands 1.29x
    # softer -- inside the noise of a knob that was moved by 20x on Go2 to make it bite.
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-1.0,
        params={"command_name": "jump", "asset_cfg": SceneEntityCfg("robot")},
    )
    # Keeps the pre-jump crouch tucking the legs by flexing thigh/calf rather than splaying
    # the hips outward.
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # Cost for holding a non-default pose while the command is idle, so an anticipatory
    # crouch right after spawn is not free.
    pre_jump_pose = RewTerm(
        func=mdp.pre_jump_pose_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )


@configclass
class JumpTerminationsCfg(TerminationsCfg):
    # A jump is tilted by definition at the top of its arc; terminating on orientation would
    # end the episode mid-flight. ``base_contact`` still catches a robot that lands on its
    # trunk, and ``landing_upright_threshold`` still gates ``success`` on arriving upright.
    bad_orientation = None


@configclass
class CurriculumCfgJump(CurriculumCfg):
    """EFGCL: decay the external assist once the policy is doing the work itself."""

    assist_force = CurrTerm(
        func=mdp.assist_force_decay,
        params={
            "command_name": "jump",
            "success_threshold": 0.60,
            "decay_step": 0.01,
            "minimum_episodes": 1024,
        },
    )


@configclass
class EventCfgJump(EventCfg):
    """Ground friction, body mass and PD gains randomised, all at Go2's fractions.

    All three ranges are dimensionless, so they port without scaling. The reasons they exist
    do too: friction was added when a Go2 flip transferred poorly to MuJoCo, and mass and
    gains when a policy scoring success 1.000 in Isaac Lab tumbled in MuJoCo. A2 has the same
    exposure and then some -- its stiffness/damping (86.0 / 1.72) were scaled from Go2's by
    m*g*L rather than measured, so +/-20% brackets a number that has no source.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
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
class RobotEnvCfgJump(RobotEnvCfg):
    """Everything the rungs share. Only the target height and log directory differ."""

    commands: CommandsCfgJump = CommandsCfgJump()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    terminations: JumpTerminationsCfg = JumpTerminationsCfg()
    curriculum: CurriculumCfgJump = CurriculumCfgJump()
    events: EventCfgJump = EventCfgJump()

    def __post_init__(self):
        super().__post_init__()
        # 4.0 -> 5.0 s. The episode has to hold the latest trigger (2.0) plus the whole
        # motion plus enough standing afterwards for motion_progress_standing to be earned.
        # At the tallest rung that is 2.0 + 0.14 delay + ~0.35 push-off + 0.90 flight = 3.4 s,
        # leaving 1.6 s of post-landing stand -- against the 1.0 s Go2 gets from its 4.0 at
        # 0.60 m. Time-scaling alone would have given 4.55; the extra 0.45 is for the taller
        # rungs, whose flight time grows as sqrt(h).
        self.episode_length_s = 5.0


def _build_height_variant(height_cm: int):
    """Return the (train, play) env cfg pair for one rung of the ladder.

    Only two things vary between rungs -- the commanded height and where the assist
    curriculum's state is persisted -- so they are generated rather than written out five
    times. The state file must be per-rung: it holds ``assist_scale`` and the success
    counters, and rsl_rl checkpoints save network weights only, so sharing one file would
    let a solved 0.60 m hand its fully-decayed assist to a fresh 0.90 m that has not earned
    it. The path matches the log root ``cli_args`` derives from the task id
    (``A2-Jump-60`` -> ``logs/rsl_rl/a2_jump_60``), so the curriculum state sits next to the
    checkpoints it belongs to.
    """
    height_m = height_cm / 100.0
    experiment_dir = f"logs/rsl_rl/a2_jump_{height_cm}"

    @configclass
    class _CommandsCfg(CommandsCfgJump):
        jump = CommandsCfgJump().jump.replace(
            target_height_range=(height_m, height_m),
            state_file=f"{experiment_dir}/jump_curriculum_state.json",
        )

    @configclass
    class _RobotEnvCfg(RobotEnvCfgJump):
        commands: _CommandsCfg = _CommandsCfg()

    @configclass
    class _RobotPlayEnvCfg(_RobotEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 32
            self.observations.policy.enable_corruption = False
            # Never write back over the training curriculum's state, and show what the policy
            # does unaided rather than what the assist is still doing for it. Set
            # initial_assist_scale back up by hand to inspect the assisted motion.
            self.commands.jump.state_file = None
            self.commands.jump.initial_assist_scale = 0.0

    _RobotEnvCfg.__name__ = _RobotEnvCfg.__qualname__ = f"RobotEnvCfgJump{height_cm}"
    _RobotPlayEnvCfg.__name__ = _RobotPlayEnvCfg.__qualname__ = f"RobotPlayEnvCfgJump{height_cm}"
    return _RobotEnvCfg, _RobotPlayEnvCfg


# Module-level names the gym registrations in __init__.py resolve by string:
#   RobotEnvCfgJump60 / RobotPlayEnvCfgJump60 ... RobotEnvCfgJump100 / RobotPlayEnvCfgJump100
for _height_cm in TARGET_HEIGHTS_CM:
    _env_cfg, _play_env_cfg = _build_height_variant(_height_cm)
    globals()[f"RobotEnvCfgJump{_height_cm}"] = _env_cfg
    globals()[f"RobotPlayEnvCfgJump{_height_cm}"] = _play_env_cfg
del _height_cm, _env_cfg, _play_env_cfg
