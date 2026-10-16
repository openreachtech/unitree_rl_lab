"""Front-leg handstand for Go2w: up on the two front wheels, hind legs tucked, rolling there.

Ported from ``feat/multi-task``'s ``Go2-Multitask-Biped-Front`` (go2), as a standalone skill -- no
mixture-of-experts observation layout, no jump command. What changed for the wheeled robot:

*It rolls instead of stepping.* go2 walks on its front feet, and its reward set is built around a
two-foot stepping cadence (``feet_air_time_positive_biped``). Go2w's front feet are wheels whose
axles point sideways, so in a nose-down stance it can balance and travel fore-aft by driving them,
like a two-wheeled pendulum. The cadence term is dropped, and the velocity command is fore-aft and
yaw only (``lin_vel_y`` = 0): a wheel cannot roll sideways, and yaw comes from the wheels' speed
difference.

*The wheels are kept out of the joint penalties.* ``joint_vel``/``acc``/``torques``, ``energy`` and
the limit term are scoped to the 12 leg joints, as in the Go2w jump tasks: rolling is the wheels'
job, and a continuous joint has no position limit. The lifted pair's wheels get their own speed
penalty instead, so they do not spin in the air.

*Heights are re-derived for the longer leg.* go2's targets assume a 0.213 + 0.213 m leg ending in a
0.022 m foot; Go2w's calf ends in a wheel axle 0.2264 m down, plus a 0.084 m wheel radius. Go2w is
also 19.7 kg against go2's ~15, the extra mass in the wheels at the leg tips -- if the rise stalls at
the torque ceiling, an assist-force curriculum like the jump tasks' is the fallback.

Everything else -- the balance terms, contact pricing, termination penalty, actuation delay and
spawn-height sampling -- is go2's, whose reasons are recorded at length on ``feat/multi-task``.
"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2w.jump_env_cfg import (
    ACTUATORS,
    LEG_JOINT_NAMES,
    ActionsCfg,
    RobotSceneCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import EventCfg as VelocityEventCfg

# Front legs stand, hind legs lift. The ``*_foot`` bodies are the wheels.
STANCE_FEET = ["FR_foot", "FL_foot"]
LIFTED_FEET = ["RR_foot", "RL_foot"]
LIFTED_HIP_JOINTS = ["RR_hip_joint", "RL_hip_joint"]
LIFTED_THIGH_JOINTS = ["RR_thigh_joint", "RL_thigh_joint"]
LIFTED_CALF_JOINTS = ["RR_calf_joint", "RL_calf_joint"]
LIFTED_WHEEL_JOINTS = ["RR_foot_joint", "RL_foot_joint"]
STANCE_LEG_LINKS = ["FR_thigh", "FL_thigh", "FR_calf", "FL_calf"]
LIFTED_LEG_LINKS = ["RR_thigh", "RL_thigh", "RR_calf", "RL_calf"]
FRONT_HIPS = ["FR_hip", "FL_hip"]

FRONT_HIP_HEIGHT_TARGET = 0.35
"""Where the front hips (here the shoulders) should sit. go2's 0.30 m is 67% of its 0.448 m reach
(0.213 + 0.213 + 0.022 foot); Go2w reaches 0.213 + 0.2264 + 0.084 (wheel radius, measured as the
wheel centre's resting height) = 0.524 m, and 67% of that is 0.35 m. Without this term the trunk
hunches until the nose scrapes."""

BASE_HEIGHT_TARGET = 0.65
"""go2's 0.55 m scaled by the same reach ratio. Like go2's, not meant to be attainable -- only to
point up; ``FRONT_HIP_HEIGHT_TARGET`` is the reachable target."""

SPAWN_HEIGHT_DROP = 0.045
"""Spawn (0.45 m) minus measured standing height (0.405 m). Sampled over so that both the drop and
the settled stand the deploy FSM hands over are in distribution -- go2 learned a rise that used the
drop and failed from a settled start."""

# Actuation delay in physics steps (5 ms each), drawn per environment. go2's front stance holds the
# stance shoulders at the torque ceiling during the rise, and an 8-10 ms command delay on hardware
# turned that into a saturated limit cycle (43 degrees of lean against 72 in simulation); training
# with 0-30 ms fixed it at every delay from 0 to 40 ms.
ACTUATOR_MIN_DELAY_STEPS = 0
ACTUATOR_MAX_DELAY_STEPS = 6
POLICY_HISTORY_LENGTH = 3


@configclass
class BipedSceneCfg(RobotSceneCfg):
    robot = RobotSceneCfg().robot.replace(
        actuators={
            name: actuator.replace(min_delay=ACTUATOR_MIN_DELAY_STEPS, max_delay=ACTUATOR_MAX_DELAY_STEPS)
            for name, actuator in ACTUATORS.items()
        }
    )


@configclass
class BipedFrontCommandsCfg:
    # Fore-aft +-1.0 m/s (go2's validated width: at a slower pace a shuffle tracks well enough that
    # nothing pushes the policy to commit) and yaw +-1.0 rad/s. No lateral command.
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )
    handstand = mdp.HandstandCommandCfg(asset_name="robot", stance=mdp.STANCE_FRONT)


@configclass
class BipedObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        # Leg joints only: a wheel's absolute angle drifts without bound.
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = POLICY_HISTORY_LENGTH
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        root_height = ObsTerm(func=mdp.base_pos_z, clip=(-100, 100))
        com_cop = ObsTerm(
            func=mdp.com_cop_vector,
            clip=(-100, 100),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = POLICY_HISTORY_LENGTH

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BipedEventCfg(VelocityEventCfg):
    """The Go2w velocity task's randomisation, plus a spawn height spanning the drop."""

    resolve_nested_terms = EventTerm(func=mdp.resolve_nested_term_params, mode="startup")

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.reset_base.params["pose_range"] = {
            **self.reset_base.params["pose_range"],
            "z": (-SPAWN_HEIGHT_DROP, 0.0),
        }


def upright_params(term, **term_params) -> dict:
    return {"term": term, "term_params": term_params}


@configclass
class BipedFrontRewardsCfg:
    # -- smoothness and limits, leg joints only ------------------------------------------------
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-2e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    energy = RewTerm(
        func=mdp.energy, weight=-2e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    # The lifted wheels have nothing to roll on; spinning them in the air is wasted and, on the
    # robot, a hazard. Same weight as the jump tasks' idle-wheel penalty.
    lifted_wheel_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_WHEEL_JOINTS)}
    )

    # -- contacts: counts, not forces (a raw-force penalty made ending the episode cheaper) ------
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_LEG_LINKS)},
    )
    # A stance leg resting a thigh or calf on the ground is the support collapsing (kneeling).
    stance_leg_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_LEG_LINKS)},
    )
    # The term that lifted go2's stance from 0.374 to 0.530 m: resting the head on the floor means
    # never having to stand up, and a shared bounded count priced it at almost nothing.
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"])},
    )
    # Isaac Lab sums rewards with no clamp at zero; without this, the bounded contact penalties
    # made ending the episode at once the cheapest policy.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- posture --------------------------------------------------------------------------------
    stance_pitch = RewTerm(func=mdp.stance_pitch_reward, weight=0.8)
    stance_roll = RewTerm(func=mdp.stance_roll_penalty, weight=-0.5)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": BASE_HEIGHT_TARGET})
    front_hip_height = RewTerm(
        func=mdp.front_body_height_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FRONT_HIPS),
            "target_height": FRONT_HIP_HEIGHT_TARGET,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "standstill_boost": 5.0,
        },
    )

    # -- the lifted legs: off the ground and tucked ----------------------------------------------
    lifted_contact = RewTerm(
        func=mdp.lifted_foot_contact,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_FEET)},
    )
    lifted_hip_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.15, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_HIP_JOINTS)}
    )
    lifted_thigh_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_THIGH_JOINTS)}
    )
    lifted_calf_motion = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIFTED_CALF_JOINTS)}
    )

    # -- CoM-CoP balance (TumblerNet), on the stance wheels. The CoP is taken at the wheel centres,
    #    a wheel radius above the contact point -- a small, constant offset in the vertical. ------
    pendulum_angle = RewTerm(
        func=mdp.pendulum_angle_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )
    pendulum_instability = RewTerm(
        func=mdp.pendulum_instability_penalty,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )
    handle_length = RewTerm(
        func=mdp.handle_length_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
        },
    )

    # -- the task, paid only once the robot is up -------------------------------------------------
    # Yaw-frame tracking: the trunk is pitched 70-90 degrees, so its body xy plane is nowhere near
    # the horizontal plane the command lives in.
    track_lin_vel_xy = RewTerm(
        func=mdp.when_upright,
        weight=1.0,
        params=upright_params(mdp.track_lin_vel_xy_yaw_frame_exp, command_name="base_velocity", std=math.sqrt(0.25)),
    )
    track_ang_vel_z = RewTerm(
        func=mdp.when_upright,
        weight=0.5,
        params=upright_params(mdp.track_ang_vel_z_exp, command_name="base_velocity", std=math.sqrt(0.25)),
    )
    upright_balance = RewTerm(func=mdp.when_upright, weight=0.5, params=upright_params(mdp.upright_balance_reward))
    # Lean into the commanded direction -- for a wheeled pendulum as for a walking one, the mass
    # goes ahead of the support to accelerate.
    support_polygon = RewTerm(
        func=mdp.when_upright,
        weight=-0.1,
        params=upright_params(
            mdp.support_polygon_penalty,
            asset_cfg=SceneEntityCfg("robot", body_names=STANCE_FEET),
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=STANCE_FEET),
            command_name="base_velocity",
        ),
    )
    # go2's `feet_air_time` (two-foot stepping cadence) is deliberately absent: Go2w rolls.

    # What "done" means, as its own term so the log carries it directly.
    stance_held = RewTerm(func=mdp.handstand_success, weight=1.0)


@configclass
class BipedTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Any hip on the ground: in this stance the front hips are the lowest part of the trunk.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    # No orientation limit: the target posture is a 75-degree tilt.
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})


@configclass
class RobotEnvCfgBipedFront(ManagerBasedRLEnvCfg):
    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BipedObservationsCfg = BipedObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: BipedFrontCommandsCfg = BipedFrontCommandsCfg()
    rewards: BipedFrontRewardsCfg = BipedFrontRewardsCfg()
    terminations: BipedTerminationsCfg = BipedTerminationsCfg()
    events: BipedEventCfg = BipedEventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfgBipedFront(RobotEnvCfgBipedFront):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
