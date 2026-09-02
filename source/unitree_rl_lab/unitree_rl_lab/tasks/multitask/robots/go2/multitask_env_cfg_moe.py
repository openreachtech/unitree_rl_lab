"""The merged multi-task environment: running and acrobatics in one episode.

Both source tasks keep their reward functions and weights exactly as trained -- what changes is
*when* each applies. That is a hard constraint, not a preference: the value function is initialised
from critics trained against those rewards, so re-tuning a weight makes the inherited value wrong by
a scale factor and the initialisation stops being worth having.

Three groups:

* **Shared effort terms** -- ``joint_vel``, ``joint_acc``, ``joint_torques``, ``action_rate``,
  ``joint_pos_limits``, ``undesired_contacts``. These carry *identical* weights and parameters in
  both source tasks (verified term by term), so they need no gate at all and stay always-on.
* **Locomotion terms**, active outside the acrobatics window.
* **Acrobatics terms**, active inside it -- with the three that assume a standing start further
  gated on commanded speed, so they apply exactly in the regime the acrobatics expert was trained in
  and retire as the take-off curriculum moves flips out of it.

Terminations follow the same idea. ``base_contact`` stays on throughout: it was already active in
the acrobatics task, where it serves as the failure detector for a flip that lands on its back.
``bad_orientation`` is suppressed inside the window, since being inverted is the commanded
behaviour -- and, usefully, comes back at the window's end, so a flip that failed and is still
inverted terminates without needing a separate detector.
"""

from __future__ import annotations

import math
from dataclasses import fields

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg_multitask import (
    MultitaskCommandsCfgPhase2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ActionsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_run import (
    _GAIT_FEET,
    _GALLOP_SPEED_THRESHOLD,
    CommandsCfgGo2GallopPhase2,
)
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.mdp.gating import (
    GATE_ACROBATICS,
    GATE_ACROBATICS_STANDING,
    GATE_LOCOMOTION,
    GATE_STANDING,
)
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    MultitaskEventCfg,
    MultitaskSceneCfg,
    UnifiedObservationsCfg,
    apply_multitask_post_init,
)

ACRO_WINDOW_S = 1.0
"""How long a commanded move counts as in progress.

Covers flight, landing and recovery: ``minimum_landing_time_s`` alone is 0.8 s, so anything shorter
ends the window with the robot still in the air. Four things read this -- the acrobatics reward
window, the terminations, the command's re-arm interval, and (through ``command_duration_s``) the
gate's expert-routing prior. They have to agree, and when they did not, the gap between a 0.5 s
command and a 1.5 s window was worth 0.55 flip success against 0.79.
"""

ACROBATIC_COMPLETION_WEIGHT = 10.0
"""Weight on the two terms that reward *completing* a commanded move.

The source tasks' weights were carried over untouched so the inherited critics stayed meaningful.
That reason has now expired -- the critic has trained for 2000 iterations and owes nothing to its
initialisation -- and leaving them produced a policy that correctly learned not to jump.

The arithmetic it was responding to: a failed move ends the episode, which in a 4 s acrobatics
episode forfeited ~2 s of standing reward and here forfeits ~12 s of running reward. The cost of
failure grew more than tenfold while the reward for success did not move, so declining to attempt
became optimal. Measured at the end of that run: locomotion +1.874 against acrobatics +0.110 per
episode, and ``max_height`` decaying to zero as the policy worked that out.

Raising these cannot make the robot flip *more often* -- the command decides that, not the policy --
so this buys effort during a window that was already going to happen, rather than more windows.
Only the completion terms are raised; the in-window penalties and the standing terms keep their
original weights, since scaling those would make attempting a move less attractive, not more.
"""


def _multi_trigger_jump_cfg():
    """The acrobatics Phase 2 jump command, re-typed to re-arm, with all assistance off.

    Copies every field from the trained configuration rather than restating them, so the trigger
    schedule, motion mix, rotation targets and tolerances are exactly the ones the expert learned
    against -- there are around twenty of them and several were tuned the hard way.
    """
    # `MultitaskCommandsCfgPhase2`, the class the acrobatics expert is actually trained under -- not
    # the `CommandsCfgPhase2` it derives from. Copying the base instead was silently wrong: the
    # merged environment then ran on the base's three motions, its 350 N sideflip force and its
    # window, none of which the loaded expert had ever seen. Most of that happened to be masked
    # here (the assist is off, so its forces do not matter, and the enables and window were
    # restated below), which is exactly what makes it worth pinning down -- the next field added to
    # the trained config would have gone missing with nothing to show for it.
    trained = MultitaskCommandsCfgPhase2().jump
    cfg = mdp.MultiTriggerJumpCommandCfg(
        **{f.name: getattr(trained, f.name) for f in fields(trained) if f.name != "class_type"}
    )
    # Both pre-trained policies were weaned off external force by their own curricula -- the jump
    # assist reached 0 and stayed there for all 2500 iterations of Phase 2 at a 1.0 success rate.
    # Nothing here should re-apply it. Setting the scale to zero is enough on its own: the crouch
    # pulse and the launch force are both gated on `assist_scale > 0`.
    cfg.initial_assist_scale = 0.0
    cfg.state_file = None
    cfg.rearm_after_s = ACRO_WINDOW_S
    # One span, one number: the command that drives the gate's routing prior now lasts exactly as
    # long as the acrobatics reward window and the re-arm interval. They disagreed before -- 0.5 s
    # against 1.5 s -- and the gap is where the policy handed an inverted robot back to the
    # locomotion expert. See ACRO_WINDOW_S.
    cfg.command_duration_s = ACRO_WINDOW_S
    # Tighter than the acrobatics task's schedule: with a 1.5 s window this gives a 3.0-4.5 s cycle,
    # so an eligible environment attempts ~5 moves per 20 s episode instead of ~3.5. Only ~13% of
    # environments are eligible at a 0.3 m/s take-off limit (the 10% commanded to stand, plus those
    # whose sampled command happens to be slow), which put acrobatics in ~3.4% of all samples --
    # against a locomotion signal collecting reward the rest of the time. This raises that without
    # touching which environments qualify, so the task itself does not get harder.
    cfg.retrigger_interval_range = (1.5, 3.0)
    cfg.velocity_command_name = "base_velocity"
    # No plain jump here, unlike the expert's own task. Every remaining move carries a direction and
    # `_select_motion_for_direction` hands the commanded heading the one that goes with it -- so the
    # move is decided by the gait rather than sampled and then patched up. The vertical jump was
    # what the earlier design substituted whenever the sampled move fought the direction, which kept
    # the interruption but discarded the rotation and left half the headings with nothing else to
    # do. With all four rotations trained (handspring mirroring the backflip, right sideflip
    # mirroring the left) the substitute is no longer needed, and dropping it means every attempt
    # trains a rotation the robot is actually being asked for.
    cfg.enable_jump = False
    return cfg


@configclass
class MoeCommandsCfg:
    """Full-range velocity commands, plus an acrobatic move that interrupts them."""

    # Start at the ceiling instead of curriculum-ing up to it. The locomotion expert is already
    # trained across this whole range; opening narrow would spend iterations re-earning ground it
    # holds, and would show it a slow-only command distribution in the meantime -- which is how a
    # fast policy drifts slow.
    base_velocity = CommandsCfgGo2GallopPhase2().base_velocity.replace(
        ranges=CommandsCfgGo2GallopPhase2().base_velocity.limit_ranges,
        limit_ranges=CommandsCfgGo2GallopPhase2().base_velocity.limit_ranges,
        # 10% standing, up from the Go2 lineage's 1%. Standing is where the acrobatics expert is
        # exactly on-distribution, and at 1% only ~41 of 4096 environments would offer it.
        rel_standing_envs=0.1,
    )

    jump = _multi_trigger_jump_cfg()


@configclass
class MoeRewardsCfg:
    """Both source reward sets, unchanged, switched by command state."""

    # -- shared: identical weights and parameters in both source tasks, so no gate ------------
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # -- locomotion: Go2-Gallop-Phase2's set, active outside the acrobatics window -------------
    track_lin_vel_xy = RewTerm(
        func=mdp.gated,
        weight=1.5,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.track_lin_vel_xy_exp,
            "term_params": {
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            },
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.track_ang_vel_z_exp,
            "term_params": {
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            },
        },
    )
    base_linear_velocity = RewTerm(
        func=mdp.gated, weight=-0.5, params={"gate": GATE_LOCOMOTION, "term": mdp.lin_vel_z_l2}
    )
    base_angular_velocity = RewTerm(
        func=mdp.gated, weight=-0.05, params={"gate": GATE_LOCOMOTION, "term": mdp.ang_vel_xy_l2}
    )
    energy = RewTerm(func=mdp.gated, weight=-2e-5, params={"gate": GATE_LOCOMOTION, "term": mdp.energy})
    flat_orientation_l2 = RewTerm(
        func=mdp.gated, weight=-0.75, params={"gate": GATE_LOCOMOTION, "term": mdp.flat_orientation_l2}
    )
    joint_pos = RewTerm(
        func=mdp.gated,
        weight=-0.7,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.joint_position_penalty,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
        },
    )
    feet_air_time = RewTerm(
        func=mdp.gated,
        weight=0.2,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.feet_air_time,
            "term_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        },
    )
    feet_slide = RewTerm(
        func=mdp.gated,
        weight=-0.1,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.feet_slide,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        },
    )
    paired_gait = RewTerm(
        func=mdp.gated,
        weight=0.2,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.paired_gait_reward,
            "term_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_GAIT_FEET, preserve_order=True),
                "velocity_command_name": "base_velocity",
                "speed_threshold": _GALLOP_SPEED_THRESHOLD,
            },
        },
    )

    # -- acrobatics: Go2-Jump-Phase2's set, active inside the window ---------------------------
    motion_progress = RewTerm(
        func=mdp.gated,
        weight=ACROBATIC_COMPLETION_WEIGHT,
        params={
            "gate": GATE_ACROBATICS,
            "term": mdp.motion_progress_reward,
            "term_params": {
                "command_name": "jump",
            },
        },
    )
    non_target_rotation = RewTerm(
        func=mdp.gated,
        weight=-0.05,
        params={
            "gate": GATE_ACROBATICS,
            "term": mdp.non_target_angular_velocity_penalty,
            "term_params": {
                "command_name": "jump",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        },
    )
    hip_deviation = RewTerm(
        func=mdp.gated,
        weight=-0.4,
        params={
            "gate": GATE_ACROBATICS,
            "term": mdp.joint_deviation_l1,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            },
        },
    )

    # -- acrobatics, standing start only -------------------------------------------------------
    # These three encode "leave a standstill, return to a standstill". pre_jump_pose in particular
    # charges for holding a non-default joint pose while the jump command is idle, which in this
    # environment would be a standing penalty applied to running. Gating them on commanded speed
    # keeps them intact where they are meaningful and lets the take-off curriculum retire them.
    pre_motion_standing = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={
            "gate": GATE_STANDING,
            "term": mdp.pre_jump_standing_reward_windup,
            "term_params": {
                "command_name": "jump",
            },
        },
    )
    motion_progress_standing = RewTerm(
        func=mdp.gated,
        weight=ACROBATIC_COMPLETION_WEIGHT,
        params={
            "gate": GATE_ACROBATICS_STANDING,
            "term": mdp.motion_progress_standing_reward,
            "term_params": {
                "command_name": "jump",
            },
        },
    )
    pre_jump_pose = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={
            "gate": GATE_STANDING,
            "term": mdp.pre_jump_pose_reward,
            "term_params": {
                "command_name": "jump",
            },
        },
    )


@configclass
class MoeTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Active throughout, as it was in the acrobatics task: a flip that puts the trunk on the floor
    # has failed, whether or not the robot was running beforehand.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    # Suppressed while a move is in progress -- inversion is the instruction there. It returns when
    # the window closes, which ends the episode for a flip that failed and stayed down.
    bad_orientation = DoneTerm(
        func=mdp.gated_termination,
        params={
            "gate": GATE_LOCOMOTION,
            "term": mdp.bad_orientation,
            "term_params": {
                "limit_angle": 0.8,
            },
        },
    )


ACRO_SPEED_CEILING = 1.0
"""Fastest commanded speed at which an acrobatic move may be triggered, in m/s.

Not the locomotion ceiling, and that distinction is the whole point. The take-off limit gates
*whether a move fires at all*, so raising it to the locomotion ceiling (3.5 m/s) tells the policy to land a
flip while being asked for 3.5 m/s -- and the only way to do that is to stop. Measured over 3000
iterations with the ceiling at 3.5: the limit reached it halfway through, and velocity tracking
error went 0.03 -> 2.11 while the flips kept landing 74% of attempts. The policy had bought flip
success by giving up running, exactly as ``takeoff_speed_levels`` warns.

Above this speed the robot simply runs; no acrobatic move is offered. That is the intended
behaviour rather than a limitation -- a flip from 3.5 m/s is not a skill being withheld, it is one
that does not exist.

Choosing 1.0 also evens out *which* move fires. The gate is on the speed magnitude, and the command
ranges are ``lin_vel_x`` [-1.0, 3.5] against ``lin_vel_y`` [-1.0, 1.0]; since the heading picks the
move by dominant axis, a fore-aft span three times wider makes lateral commands -- and so both
sideflips -- a small minority. Below 1.0 m/s the two axes span the same range and the four motions
fire about equally.
"""

MAX_LOCOMOTION_ERROR = 0.6
"""Velocity-tracking error above which the take-off limit may not rise, and below which it falls.

Without it the curriculum promotes on flip success alone, which is what let the limit climb while
running fell apart. Measured from ``MultiTriggerJumpCommand.locomotion_error``, which excludes the
steps spent mid-flip -- the robot cannot follow a ground velocity command while inverted, and
charging the gate for that made the limit oscillate instead of settle. For scale, the locomotion
expert on its own sits at 0.30-0.40, and the collapsed run reported 0.95 on this restricted figure
against 2.11 unrestricted.
"""


@configclass
class MoeCurriculumCfg:
    takeoff_speed = CurrTerm(
        func=mdp.takeoff_speed_levels,
        params={
            "command_name": "jump",
            "maximum_speed": ACRO_SPEED_CEILING,
            "max_velocity_error": MAX_LOCOMOTION_ERROR,
            "state_file": "logs/rsl_rl/go2_multitask/takeoff_speed_state.json",
        },
    )


@configclass
class RobotEnvCfgMoe(ManagerBasedRLEnvCfg):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: MoeCommandsCfg = MoeCommandsCfg()
    rewards: MoeRewardsCfg = MoeRewardsCfg()
    terminations: MoeTerminationsCfg = MoeTerminationsCfg()
    events: MultitaskEventCfg = MultitaskEventCfg()
    curriculum: MoeCurriculumCfg = MoeCurriculumCfg()

    def __post_init__(self):
        # 20 s, the locomotion episode. With a 1.5 s window and a 3-6 s cooldown that is roughly
        # three to four moves per episode -- about a quarter of the time acrobatic, the rest
        # ordinary running, which is the ratio the merged skill is meant to have.
        self.episode_length_s = 20.0
        apply_multitask_post_init(self)


@configclass
class RobotPlayEnvCfgMoe(RobotEnvCfgMoe):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.observations.policy.enable_corruption = False
        # Play exercises the finished skill, so the curriculum is off -- but the ceiling stays.
        # The ceiling still applies: the limit is not a training restriction
        # to be lifted at the end, but the speed above which an acrobatic move is not offered at
        # all. Letting Play fire one at 3.5 m/s shows a failure the deployed robot will never be
        # asked to produce.
        self.curriculum.takeoff_speed = None
        self.commands.jump.initial_takeoff_speed_limit = ACRO_SPEED_CEILING
