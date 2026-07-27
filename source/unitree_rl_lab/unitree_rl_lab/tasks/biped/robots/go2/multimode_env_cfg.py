"""Single-policy multimode locomotion task for Go2: one controller that switches,
within an episode, between normal quadruped walking and the two biped stances
(hind-leg-only, front-leg-only) already validated as standalone tasks in
``biped_env_cfg.py`` / ``biped_env_cfg_front.py``.

A discrete ``gait_mode`` command (see ``unitree_rl_lab.tasks.biped.mdp.commands``)
tells the policy which stance to hold; every episode starts in quad mode (the
default), and the command resamples mid-episode to exercise the quad<->biped
transitions the TumblerNet paper itself demonstrates are learnable with the same
reward architecture ("Transition between quadrupedal and bipedal locomotion").

Design notes on how the three modes share one reward set:
  - The CoM-CoP balance rewards (``pendulum_angle_penalty`` etc.) are called with
    *all four* feet as the candidate stance set. The force-weighted CoP naturally
    collapses onto whichever 1-2 feet are actually bearing weight, so these three
    terms need no mode branching at all -- they work unmodified in every mode.
  - ``feet_air_time`` / ``air_time_variance_penalty`` / ``feet_slide`` (ported from
    the reference quadruped task, ``Unitree-Go2-Velocity-v0``) are similarly safe
    unmodified: a permanently-lifted biped-mode foot never registers a first-contact
    event (zero contribution to ``feet_air_time``) and its air/contact time simply
    freezes, bounded by the 0.5s clip in ``air_time_variance_penalty``.
  - Orientation, base-height target, and "keep the idle legs near default pose"
    fundamentally *disagree* between quad and biped (flat vs. tilted, no explicit
    height target vs. 0.55m, natural gait vs. tucked-and-still) or need to know
    which leg pair is currently idle. These use dedicated mode-aware functions in
    ``mdp.rewards`` / ``mdp.terminations`` that read the commanded mode directly.
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg_multimode import PROPRIO_HISTORY_LENGTH
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import ActionsCfg, EventCfg
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import CommandsCfg as BipedCommandsCfg
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import RobotEnvCfg as BipedRobotEnvCfg
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import RobotSceneCfg as BipedRobotSceneCfg

FRONT_FOOT_NAMES = ["FR_foot", "FL_foot"]
HIND_FOOT_NAMES = ["RR_foot", "RL_foot"]
ALL_FOOT_NAMES = FRONT_FOOT_NAMES + HIND_FOOT_NAMES

FRONT_HIP_JOINT_NAMES = ["FR_hip_joint", "FL_hip_joint"]
FRONT_THIGH_JOINT_NAMES = ["FR_thigh_joint", "FL_thigh_joint"]
FRONT_CALF_JOINT_NAMES = ["FR_calf_joint", "FL_calf_joint"]
HIND_HIP_JOINT_NAMES = ["RR_hip_joint", "RL_hip_joint"]
HIND_THIGH_JOINT_NAMES = ["RR_thigh_joint", "RL_thigh_joint"]
HIND_CALF_JOINT_NAMES = ["RR_calf_joint", "RL_calf_joint"]

BASE_HEIGHT_TARGET_BIPED = 0.55


@configclass
class RobotSceneCfg(BipedRobotSceneCfg):
    """Same flat-ground scene as the single-stance biped tasks, but with
    ``track_air_time`` enabled -- needed by the ported quadruped gait-shaping
    rewards (``feet_air_time`` / ``air_time_variance_penalty``)."""

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class CommandsCfg(BipedCommandsCfg):
    """Adds the discrete gait-mode command on top of the biped velocity command."""

    gait_mode = mdp.GaitModeCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        # More training weight on the two biped stances than on quad -- quad is
        # still the guaranteed episode-start default (see GaitModeCommand), but
        # in-episode resamples should spend most of their time on the harder
        # biped skills.
        mode_probs=(0.2, 0.4, 0.4),
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor input: [current-step proprio (incl. gait_mode) | stacked history]."""

        # -- current step (order preserved; must match PROPRIO_TERM_DIM = 48) --
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        gait_mode = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "gait_mode"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        # -- stacked history (same terms, PROPRIO_HISTORY_LENGTH frames each) --
        base_ang_vel_hist = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        projected_gravity_hist = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        velocity_commands_hist = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        gait_mode_hist = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "gait_mode"},
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos_rel_hist = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel_rel_hist = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action_hist = ObsTerm(func=mdp.last_action, clip=(-100, 100), history_length=PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic input: current-step proprio (incl. gait_mode) +
        ground-truth lin_vel / CoM-CoP (all four feet) + joint effort. No noise."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        gait_mode = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "gait_mode"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        com_cop = ObsTerm(
            func=mdp.com_cop_vector,
            clip=(-100, 100),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
            },
        )
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()

    @configclass
    class EstimatorTargetCfg(ObsGroup):
        """Ground truth for the auxiliary estimator loss (see ``BipedPPO``). Not
        fed to the actor; read directly by name, independent of ``obs_groups``."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        com_cop = ObsTerm(
            func=mdp.com_cop_vector,
            clip=(-100, 100),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
            },
        )

    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task (yaw-frame tracking works for both the ~flat quad orientation and the
    # ~70-90 degree pitched biped orientation) --
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base (mode-independent) --
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)

    # -- mode-aware posture: flat+natural-gait for quad, tilted+held for biped --
    mode_orientation = RewTerm(
        func=mdp.mode_orientation_reward,
        weight=1.0,  # passthrough -- sign/magnitude already applied inside the function
        params={"command_name": "gait_mode", "quad_weight": 2.5, "biped_weight": 0.8},
    )
    mode_base_height = RewTerm(
        func=mdp.mode_base_height_reward,
        weight=-0.5,
        params={"command_name": "gait_mode", "biped_target_height": BASE_HEIGHT_TARGET_BIPED},
    )
    quad_stand_still = RewTerm(
        func=mdp.mode_gated_joint_position_penalty,
        weight=-0.7,
        params={
            "command_name": "gait_mode",
            "vel_command_name": "base_velocity",
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- keep whichever leg pair is currently "lifted" near its default pose and
    # off the ground; zero in quad mode (see mdp.rewards docstrings) --
    lifted_hip_motion = RewTerm(
        func=mdp.lifted_leg_joint_deviation,
        weight=-0.15,
        params={
            "command_name": "gait_mode",
            "front_joint_names": FRONT_HIP_JOINT_NAMES,
            "hind_joint_names": HIND_HIP_JOINT_NAMES,
        },
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.lifted_leg_joint_deviation,
        weight=-0.05,
        params={
            "command_name": "gait_mode",
            "front_joint_names": FRONT_THIGH_JOINT_NAMES,
            "hind_joint_names": HIND_THIGH_JOINT_NAMES,
        },
    )
    lifted_calf_motion = RewTerm(
        func=mdp.lifted_leg_joint_deviation,
        weight=-0.05,
        params={
            "command_name": "gait_mode",
            "front_joint_names": FRONT_CALF_JOINT_NAMES,
            "hind_joint_names": HIND_CALF_JOINT_NAMES,
        },
    )
    lifted_leg_contact_force = RewTerm(
        func=mdp.lifted_leg_contact_force,
        weight=-0.05,
        params={
            "command_name": "gait_mode",
            "front_foot_names": FRONT_FOOT_NAMES,
            "hind_foot_names": HIND_FOOT_NAMES,
        },
    )

    # -- CoM-CoP balance (TumblerNet); mode-agnostic, see module docstring --
    pendulum_angle = RewTerm(
        func=mdp.pendulum_angle_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
        },
    )
    pendulum_instability = RewTerm(
        func=mdp.pendulum_instability_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
        },
    )
    handle_length = RewTerm(
        func=mdp.handle_length_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
        },
    )

    # -- quadruped gait shaping (ported from Unitree-Go2-Velocity-v0); mode-agnostic,
    # see module docstring for why a permanently-lifted biped foot doesn't break these --
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- other (mode-independent) --
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})
    # Hard tilt-limit termination, but only in quad mode -- biped modes need to
    # pitch ~70-90 degrees by design (see mdp.terminations.bad_orientation_quad_only).
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_quad_only, params={"command_name": "gait_mode", "limit_angle": 0.8}
    )


@configclass
class RobotEnvCfg(BipedRobotEnvCfg):
    """Configuration for the Go2 multimode (quad / hind-biped / front-biped) env."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
