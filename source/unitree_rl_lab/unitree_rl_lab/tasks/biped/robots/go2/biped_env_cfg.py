"""Bipedal (2-leg stance) locomotion task for Go2: the quadruped walks on its hind
legs only, front legs lifted/tucked.

Also carries a ``gait_mode`` one-hot observation, permanently pinned to hind-biped
(see ``mdp.PinnedGaitModeCommand``) -- the reward/termination set below is
otherwise completely unchanged from the original single-stance design (still
unconditionally biped-seeking, correct since the mode never actually varies
here). This is a deliberate foundation stage for a *future* mode-switching task:
several earlier attempts at learning "stand on two legs" and "switch modes based
on gait_mode" simultaneously, from scratch, in one step, all converged to a
policy that received the gait_mode observation correctly but never produced any
visible mode-dependent behavior change. Carrying a real (if constant) gait_mode
input through Phase1's own proven training lets a later task swap in an actually
varying gait-mode command and resume from this checkpoint -- same observation
shape, no architecture change -- instead of learning both problems at once.

Trains a single actor-critic with a jointly-trained TumblerNet-style state
estimator (see ``unitree_rl_lab.assets.models.biped_actor``), rather than the
Wild-style teacher/student distillation used by the locomotion velocity task.
Reward design follows the reference ``bipedal_dog`` implementation
(bipedal_locomotion_for_quadrupedal_robots-dev/legged_gym/envs/bipedal_dog/) --
orientation + base-height + swing-leg motion penalties to shape the bipedal
posture -- plus CoM-CoP balance rewards from the paper it implements ("Learning
stable bipedal locomotion skills for quadrupedal robots on challenging terrains
with automatic fall recovery", Xiao et al. 2025 / "TumblerNet"), which the
reference code defines but leaves at zero weight in its baseline config.

Flat ground only for now (matches the reference's actual training setup;
robustness comes from domain randomization, not a terrain curriculum). No
explicit phase variable / Hermite-spline foot trajectory (PGTT-style) -- gait
timing is left entirely to the learned policy and the CoM-CoP + posture rewards.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.agents.rsl_rl_ppo_cfg import PROPRIO_HISTORY_LENGTH

# Front legs swing/tuck (lifted); hind legs are the stance/support legs.
FRONT_CALF_JOINT_NAMES = ["FR_calf_joint", "FL_calf_joint"]
STANCE_FOOT_NAMES = ["RR_foot", "RL_foot"]
FRONT_FOOT_NAMES = ["FR_foot", "FL_foot"]
ALL_FOOT_NAMES = [".*_foot"]

# Well above Go2's normal quadruped stance height (~0.32-0.34 m), so this term
# pulls the robot to rise up onto its hind legs. Matches the reference's
# base_height_target (0.55 m for its dog, whose stance height is comparable to
# Go2's).
BASE_HEIGHT_TARGET = 0.55

# Actuation delay, in physics steps (``sim.dt`` = 5 ms, so one policy step is 4 of these).
# Drawn per environment on reset, so the population spans the whole range.
#
# Left at zero, the biped stances learn a rise that holds the stance shoulders at the
# actuator's torque ceiling and lets the load set the speed -- in simulation those joints
# sit pinned at exactly Y1 = 20.2 N*m while turning at only 3-7 rad/s. That has almost no
# phase margin, and on hardware, where the command reaches the motor ~8-10 ms late
# (measured: cross-correlating the published position command against the motor's torque
# response in a 1 kHz deploy trace), it degenerates into a 4-6 Hz saturated limit cycle:
# the shoulders swing +-1.4 rad with 0.9 rad of tracking error, spin at up to 30 rad/s,
# burn ~39 W each against the ~8.5 W the lift actually needs, and -- oscillating
# incoherently -- produce a roll moment instead of a pitch-up. `Go2-Biped-Front` reached
# only 43 deg of sagittal lean on hardware, half its tilt being a lateral lean, while
# rising to 72 deg in both simulators. A play sweep reproduced that failure in Isaac Lab
# purely by adding delay (8/8 environments fell at 40 ms; 0/8 at 0 ms), and ruled out the
# actuator envelope and the initial condition as causes.
#
# 0..6 (0-30 ms) covers the measured command latency plus the unmeasured sensing lag, with
# margin into the band where a zero-delay policy visibly breaks. Keeping 0 in the range
# lets part of the population keep training the nominal case, which is what gives a resume
# from a zero-delay checkpoint somewhere to start.
ACTUATOR_MIN_DELAY_STEPS = 0
ACTUATOR_MAX_DELAY_STEPS = 6


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization (no terrain curriculum -- flat ground only)."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # z is an offset from init_state.pos (0.400 m), so this range is
            # [0.3235, 0.400] m of base height. Its lower end is exactly where the feet
            # touch: at the default joint pose (which is what reset_joints_by_scale
            # reproduces, position_range = 1.0) the front feet' contact spheres sit
            # 0.3235 m below the base origin and the rear ones 0.3139 m, so a base at
            # 0.3235 m is settled on the ground and anything lower would penetrate it.
            #
            # Without this the reset always dropped the base 7.65 cm, and the biped
            # stances learned a rise that *depends* on that fall -- the downward momentum
            # and the leg compression it buys. Deployment never provides it: the FSM hands
            # the policy a robot standing still on four feet (FixStand). The 2-leg stances
            # happen not to care, but Go2-Biped-Single could only rise from the drop --
            # from a settled stand it reached 16 deg of lean instead of 76 and fell 37
            # times in 8 environments over 5 s, which is exactly what it does in MuJoCo
            # through the deploy stack. Randomizing the drop height puts both the training
            # condition and the deployment condition in distribution.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "z": (-0.0765, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )

    gait_mode = mdp.PinnedGaitModeCommandCfg(asset_name="robot", pinned_mode=mdp.MODE_HIND_BIPED)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor input: [current-step proprio | stacked proprio history].

        The estimator (inside ``BipedPolicy``) consumes only the history block;
        the policy MLP consumes the current-step block plus the estimator's own
        [lin_vel, com_cop] prediction -- it never sees ground-truth privileged
        values, matching what is available at deployment time.
        """

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
        # -- fed only to the estimator; independent noise draw from the current-step
        # -- block above is intentional (matches independently-noisy sensor reads).
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
        """Privileged critic input: current-step proprio + ground-truth lin_vel /
        CoM-CoP + joint effort. No noise (privileged / training-only)."""

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
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
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
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
            },
        )

    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task --
    # Yaw-frame (not body-frame) tracking: the biped stance pitches the body ~70-90
    # degrees off flat, so body-frame xy no longer approximates the world-horizontal
    # plane the velocity command is defined in (same reasoning as the h1 humanoid task).
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base --
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)

    # -- bipedal posture: tilt away from flat/quadruped stance, rise onto hind legs --
    # NOTE: positive weight -- the inverse of the usual "stay flat" use of this term.
    tilt_reward = RewTerm(func=mdp.flat_orientation_l2, weight=0.8)
    upright_penalty = RewTerm(func=mdp.gravity_z_l2, weight=-0.03)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": BASE_HEIGHT_TARGET})

    # -- keep the swing (front) legs near their default pose instead of flailing --
    front_hip_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_hip_joint", "FL_hip_joint"])},
    )
    front_thigh_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh_joint", "FL_thigh_joint"])},
    )
    front_calf_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_CALF_JOINT_NAMES)},
    )

    # -- CoM-CoP balance (TumblerNet); stance = hind feet --
    # Weights relaxed from the naive paper-derived starting point (-0.3/-0.01/-0.5)
    # down to the reference's own actual validated config (outputs/random_dog/Imi/
    # test_reward/train_cfg_robot.py, paired with a real deployed rear-leg-biped
    # checkpoint): inv_pendulum -0.1, inv_pendulum_acc -0.0001, cart_table_len_xy
    # -0.1. Real single-hind-leg-support walking necessarily involves CoM-CoP
    # swing; the original weights over-penalized it relative to what actually
    # worked on hardware, biasing PPO toward a static, balanced freeze instead of
    # a dynamic gait (sandbox try1 confirmed this: near-max tilt_reward but the
    # robot only ever lifted one leg and froze).
    pendulum_angle = RewTerm(
        func=mdp.pendulum_angle_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        },
    )
    pendulum_instability = RewTerm(
        func=mdp.pendulum_instability_penalty,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        },
    )
    handle_length = RewTerm(
        func=mdp.handle_length_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        },
    )

    # -- other --
    # Reference: `penalize_contacts_on = ["thigh", "calf"]` -- covers *all four*
    # legs' thigh/calf (not just the front/swing legs): a hind (stance) leg's
    # thigh or calf touching the ground means the robot has collapsed/over-flexed
    # its support leg, which is just as undesirable as a front-leg collision.
    # Hip contact is handled separately below as a hard termination, not a
    # reward penalty (reference: `terminate_after_contacts_on = [..., "hip"]`).
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )

    # -- bipedal encouragement (TumblerNet r4): actually get the front feet off
    # the ground and walking, not just leaning -- neither is substituted by the
    # front_hip/thigh/calf_motion joint-deviation terms above (a different,
    # smoothness-group mechanism in the reference). Ported from sandbox try2/
    # try6 after finding these two terms entirely missing was why earlier
    # attempts either stayed flat (tilt_reward alone, no lift) or lifted one leg
    # and froze (tilt_reward increased, still no force/air-time incentive).
    front_contact_force = RewTerm(
        func=mdp.front_foot_contact_force,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FOOT_NAMES)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FOOT_NAMES),
            "threshold": 0.5,
        },
    )
    # Isaac Lab's RewardManager sums weighted terms with no clamp (unlike the
    # reference, which sets `only_positive_rewards=True` to clip each step's
    # total reward to >= 0). Without this, front_contact_force alone made
    # ending the episode immediately cheaper than enduring ~1000 steps of it --
    # confirmed empirically (sandbox try2/try4: every episode collapsed to ~5-8
    # steps within 100 iterations). This counteracts that regardless of which
    # term would otherwise make death attractive. Weight matches isaaclab_tasks'
    # own h1/g1/cassie rough_env_cfg.py (all -200.0, all sharing our step_dt).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Reference: `terminate_after_contacts_on = ["base", "trunk", "hip"]` -- any
    # hip (all four) touching the ground ends the episode immediately, it is not
    # just a soft reward penalty like thigh/calf contact above.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    # Catches a full fall/collapse. Unlike the quadruped velocity task, a plain
    # "bad_orientation" (tilt-limit) termination cannot be reused here -- the
    # target bipedal posture *is* a large tilt from flat, so it would trigger on
    # the desired end state instead of a genuine fall.
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 bipedal (2-leg stance) locomotion environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # See ACTUATOR_MAX_DELAY_STEPS: modelling the deploy-time actuation lag is what
        # makes these stances transfer. Applied here rather than on the shared
        # UNITREE_GO2_CFG, which every other Go2 task also uses.
        actuator = self.scene.robot.actuators["GO2HV"]
        actuator.min_delay = ACTUATOR_MIN_DELAY_STEPS
        actuator.max_delay = ACTUATOR_MAX_DELAY_STEPS

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
