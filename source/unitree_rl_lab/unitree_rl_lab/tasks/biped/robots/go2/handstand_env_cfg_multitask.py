"""Front-leg handstand walking, on the unified multi-task observation.

The robot rises onto its front legs, hind legs tucked, and walks there while tracking a velocity
command. Trained as a standalone expert, on the same observation layout, control rate and actuator
model as the locomotion and acrobatics experts, so it can join the mixture later without any weight
surgery beyond the zero-padding widen that any layout change costs.

Why the front legs and not the hind ones: ``feat/biped`` trained both, and the hind-leg stance took
seven sandbox rounds to stop shuffling on a tripod while the front stance worked from a much
shorter recipe.

Two hard-won settings from that work are carried over rather than rediscovered:

*Actuation delay.* The stance's rise strategy is to hold the stance shoulders at the torque ceiling
and let the load set the speed -- in simulation they sit pinned at 20.2 N*m while turning at only
3-7 rad/s, which has almost no phase margin. On hardware, where the command reaches the motor 8-10
ms late, that became a 4-6 Hz saturated limit cycle and the robot reached 43 degrees of lean
instead of 72. A play sweep reproduced the failure in simulation purely by adding delay (8/8 falls
at 40 ms, 0/8 at 0 ms), and training with 0-30 ms of it fixed the rise at every delay from 0 to 40.
None of the other multi-task experts model delay, and they do not have to: this one training with
delay in range covers the zero-delay case too, since 0 stays in the range.

*The reset height.* The robot spawns 7.8 cm above its own standing height and the stance learns a
rise that uses the fall. Deployment never provides one -- the FSM hands over a robot standing still
on four feet -- and a single-leg stance trained on the drop reached 15.6 degrees from a settled
start against 75.9 from the drop. Sampling the spawn height across the whole gap puts both starts
in distribution.

The reward set is gated: every term describing a bipedal stance is wrapped in ``gated`` with
``GATE_HANDSTAND`` or ``GATE_HANDSTAND_UPRIGHT``, so it retires with the command. Here the command
is pinned on and the gates never close, which makes them look like dead weight -- they are not.
They are what lets this reward set be merged with the locomotion and acrobatics sets untouched,
which is the same move that made the current two-expert policy possible.
"""

from __future__ import annotations

import math
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ActionsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import CommandsCfgPhase1
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.mdp.gating import GATE_HANDSTAND, GATE_HANDSTAND_UPRIGHT
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    IDLE_JUMP_COMMAND,
    MultitaskEventCfg,
    MultitaskSceneCfg,
    UnifiedObservationsCfg,
    apply_multitask_post_init,
)

# Front legs stand, hind legs lift.
STANCE_FEET = ["FR_foot", "FL_foot"]
LIFTED_FEET = ["RR_foot", "RL_foot"]
LIFTED_HIP_JOINTS = ["RR_hip_joint", "RL_hip_joint"]
LIFTED_THIGH_JOINTS = ["RR_thigh_joint", "RL_thigh_joint"]
LIFTED_CALF_JOINTS = ["RR_calf_joint", "RL_calf_joint"]
# Physically at the front of the body whichever end is standing. In this stance they are the
# lowest part of the trunk, which is why they need a height floor of their own.
FRONT_HIPS = ["FR_hip", "FL_hip"]

BASE_HEIGHT_TARGET = 0.55
"""Well above Go2's 0.32 m quadruped stance, so the term pulls the robot up rather than describing
where it already is. Carried from ``feat/biped``, which reached 0.56-0.62 m of base height in this
stance; the target is not meant to be attainable, only to point."""

FRONT_HIP_HEIGHT_TARGET = 0.30
"""Where the front hips should sit. Go2's thigh and calf are 0.213 m each, so the shoulders cannot
physically reach much past 0.426 m -- and without this term the trunk hunches until the nose
scrapes while the root height reward reports nothing wrong."""

ACTUATOR_MIN_DELAY_STEPS = 0
ACTUATOR_MAX_DELAY_STEPS = 6
"""0-30 ms at the 5 ms physics step, drawn per environment on reset. See the module docstring."""

SPAWN_HEIGHT_DROP = 0.078
"""How far above its own standing height the robot spawns (0.400 m nominal against 0.322 m
measured). Sampled over rather than removed, so the drop the old stances learned to use and the
settled stand the deploy FSM actually provides are both in distribution."""


@configclass
class HandstandCommandsCfg:
    """A velocity command to follow, and the stance to follow it in."""

    # Built by replacing ranges on the locomotion command rather than constructing a fresh term, so
    # every field this task does not care about keeps the value the locomotion expert was trained
    # with -- the same term has to serve the merged environment later.
    #
    # +-1.0 m/s in both axes and +-1.0 rad/s, matching what `feat/biped` validated rather than the
    # paper's narrower [-0.8, 0.8] / [-0.4, 0.4]. That width is the single most load-bearing
    # finding in their whole biped history: at a slower demanded pace a tripod shuffle tracks the
    # command well enough that nothing pushes the policy to commit to a real two-leg gait, and
    # widening the range was what finally produced sustained bipedal walking (lifted-foot ground
    # contact fell from a 5% plateau to 0.5-0.8%).
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=CommandsCfgPhase1().base_velocity.ranges.replace(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
        # Equal to `ranges`, so the inherited `lin_vel_cmd_levels` curriculum has nowhere to widen
        # to. This task has no room for a command curriculum on top of learning to balance.
        limit_ranges=CommandsCfgPhase1().base_velocity.ranges.replace(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
        rel_standing_envs=0.1,
    )

    # Inert: this task never jumps, but the five observation columns have to be filled by the term
    # that defines them.
    jump = IDLE_JUMP_COMMAND

    handstand = mdp.HandstandCommandCfg(
        asset_name="robot",
        stance=mdp.STANCE_FRONT,
        pinned=True,
        stance_foot_names=tuple(STANCE_FEET),
        lifted_foot_names=tuple(LIFTED_FEET),
    )


@configclass
class HandstandEventCfg(MultitaskEventCfg):
    """The shared multi-task events, plus a spawn height that spans the drop."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.reset_base.params["pose_range"] = {
            **self.reset_base.params["pose_range"],
            "z": (-SPAWN_HEIGHT_DROP, 0.0),
        }


@configclass
class HandstandRewardsCfg:
    """The bipedal reward set, every task term gated on the stance being commanded.

    Weights on the balance and posture terms are ``feat/biped``'s, which trained this stance to
    hardware; the smoothness and limit terms are the multi-task set's, so this expert is shaped by
    the same costs as the ones it will be merged with.
    """

    # -- shared with every other multi-task expert, ungated -----------------------------------
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # -0.1, the merged set's value, not `feat/biped`'s -0.02. Matching the environment this expert
    # is being built for matters more than matching the one the recipe came from -- but if the rise
    # stalls part-way, this is the first weight to suspect: the stance's rise is a large, fast
    # motion and -0.1 is five times the damping it was developed under.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    # The stance legs' own thigh and calf are included deliberately: one of them on the ground
    # means the support leg has collapsed, which is no better than a swing leg dragging.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_thigh", ".*_calf"]),
        },
    )
    # Isaac Lab's RewardManager sums weighted terms with no clamp, unlike the reference
    # implementation this reward set descends from, which clips each step's total to >= 0. Without
    # this, `lifted_foot_contact` alone made ending the episode immediately cheaper than enduring
    # it: every episode collapsed to 5-8 steps and stayed there for 2000 iterations. The weight
    # matches isaaclab_tasks' own h1/g1/cassie configs, which share this step_dt.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- posture: get up, and stay square while doing it ---------------------------------------
    stance_pitch = RewTerm(
        func=mdp.gated,
        weight=0.8,
        params={"gate": GATE_HANDSTAND, "gate_command_name": "handstand", "term": mdp.stance_pitch_reward},
    )
    stance_roll = RewTerm(
        func=mdp.gated,
        weight=-0.5,
        params={"gate": GATE_HANDSTAND, "gate_command_name": "handstand", "term": mdp.stance_roll_penalty},
    )
    # -2.0 rather than the bipedal recipe's -0.5. At -0.5 this term contributed 0.017 per step
    # against `stance_held`'s 1.0, and the measured stance sat 0.37 m high with the head clearing
    # the floor by 45 mm -- visibly about to scrape, and agreed on by every environment (p10 to p90
    # spanned one millimetre). The reference recipe could afford -0.5 because its stance stood on
    # nearly straight legs and satisfied the term for free; this one crouches, so the term has to
    # do real work.
    #
    # Raising *this* term rather than `front_hip_height`, which regulates the shoulders directly:
    # base height can only be satisfied by rising. Standing back down on four legs does not help --
    # a 0.322 m quadruped stance is further from the 0.55 m target than the handstand already is,
    # so the penalty gets worse, not better. And extending the front legs moves the stance calf
    # away from the flexion limit it currently presses against, so `dof_pos_limits` pushes the same
    # way for once.
    base_height = RewTerm(
        func=mdp.gated,
        weight=-2.0,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.base_height_l2,
            "term_params": {"target_height": BASE_HEIGHT_TARGET},
        },
    )
    front_hip_height = RewTerm(
        func=mdp.gated,
        weight=-0.5,
        params={
            "gate": GATE_HANDSTAND,
            "gate_command_name": "handstand",
            "term": mdp.front_body_height_l2,
            "term_params": {
                "asset_cfg": SceneEntityCfg("robot", body_names=FRONT_HIPS),
                "target_height": FRONT_HIP_HEIGHT_TARGET,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "standstill_boost": 5.0,
            },
        },
    )

    # -- the lifted legs: off the ground, and tucked rather than flailing ----------------------
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

    # -- CoM-CoP balance (TumblerNet). Weights are the reference's own validated values, not the
    #    naive paper-derived ones: real single-support walking necessarily swings the CoM across
    #    the CoP, and the stricter weights biased the policy toward a balanced freeze instead of a
    #    gait -- one sandbox run reached near-maximum tilt with the robot lifting a leg and
    #    standing still. Stance feet only, so the term describes the support the robot is meant to
    #    be balancing on rather than whatever happens to be touching.
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

    # -- the task itself, gated on the robot actually being up ---------------------------------
    # Yaw-frame tracking, not body-frame: this stance pitches the trunk 70-90 degrees off flat, so
    # the body's xy plane is nowhere near the world-horizontal plane the velocity command lives in.
    track_lin_vel_xy = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={
            "gate": GATE_HANDSTAND_UPRIGHT,
            "gate_command_name": "handstand",
            "term": mdp.track_lin_vel_xy_yaw_frame_exp,
            "term_params": {"command_name": "base_velocity", "std": math.sqrt(0.25)},
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.gated,
        weight=0.5,
        params={
            "gate": GATE_HANDSTAND_UPRIGHT,
            "gate_command_name": "handstand",
            "term": mdp.track_ang_vel_z_exp,
            "term_params": {"command_name": "base_velocity", "std": math.sqrt(0.25)},
        },
    )
    upright_balance = RewTerm(
        func=mdp.gated,
        weight=0.5,
        params={"gate": GATE_HANDSTAND_UPRIGHT, "gate_command_name": "handstand", "term": mdp.upright_balance_reward},
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
    # Cadence. `feet_air_time_positive_biped` is the two-foot formulation -- it scores the minimum
    # of the stance pair's air and contact times -- which is exactly what a two-legged gait needs
    # and what the four-foot version cannot express. Zero, not merely unpenalised, while standing
    # still, so it never rewards shuffling in place.
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
    # What "done" means, kept as its own term so the training log carries the answer directly
    # rather than requiring it to be inferred from the shaping terms.
    stance_held = RewTerm(
        func=mdp.gated,
        weight=1.0,
        params={"gate": GATE_HANDSTAND, "gate_command_name": "handstand", "term": mdp.handstand_success},
    )


@configclass
class HandstandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Any hip on the ground -- all four, not just the lifted pair. In this stance the front hips
    # are the lowest part of the trunk, so this is what stops the robot from resting its shoulders
    # on the floor and calling it a handstand.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    # A full collapse. Note what is deliberately *absent*: the orientation limit every locomotion
    # task terminates on. The target posture here is a 75-degree tilt, so that termination would
    # fire on success.
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})


@configclass
class RobotEnvCfgHandstand(ManagerBasedRLEnvCfg):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: HandstandCommandsCfg = HandstandCommandsCfg()
    rewards: HandstandRewardsCfg = HandstandRewardsCfg()
    terminations: HandstandTerminationsCfg = HandstandTerminationsCfg()
    events: HandstandEventCfg = HandstandEventCfg()

    def __post_init__(self):
        self.episode_length_s = 20.0
        apply_multitask_post_init(self)
        # Modelling the deploy-time actuation lag is what makes this stance transfer; see the
        # module docstring. Scoped to this task rather than to the shared robot config, which every
        # other Go2 task also reads.
        actuator = self.scene.robot.actuators["GO2HV"]
        actuator.min_delay = ACTUATOR_MIN_DELAY_STEPS
        actuator.max_delay = ACTUATOR_MAX_DELAY_STEPS


@configclass
class RobotPlayEnvCfgHandstand(RobotEnvCfgHandstand):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
