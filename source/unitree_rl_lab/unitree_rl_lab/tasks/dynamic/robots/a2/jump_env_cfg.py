"""A2 jump base: Go2's ``jump_env_cfg.py`` on a robot 2.49x the mass and 1.34x the height.

Everything here is Go2's dynamic-task base with three kinds of change, and nothing else:

  1. STRUCTURAL.  A2's trunk link is ``base_link``, not ``base``, and there are no
     ``Head_*`` links.  ``*_hip`` exists but carries no collision geometry at all, so a
     contact term named on it can never fire -- the undesired-contact set is
     ``[.*_thigh, .*_calf]``, which is what ``locomotion/robots/a2`` already uses.

  2. SCALED.  Three ratios, all taken from the dimension table in
     ``assets/robots/unitree.py`` rather than invented here:

         R_m = 40.071 / 16.087 = 2.49    total mass
         R_L = 0.275 / 0.213   = 1.29    thigh/calf link length
         R_t = sqrt(R_L)       = 1.14    time, from the pendulum scaling a leg obeys

     Anything with units of force scales by R_m, anything with units of time by R_t, and
     a gravity torque (m*g*L) by R_m * R_L = 3.21 -- the same factor ``unitree.py`` used
     to carry Go2's PD gains across.  Dimensionless quantities (tilt thresholds,
     randomisation fractions, action scale) are left exactly as Go2 has them.

  3. UNCHANGED BUT WORTH SAYING.  The actuator model is ``UNITREE_A2_CFG``'s own, not a
     jump-specific override.  Go2 needed ``GO2_CORRECTED_ACTUATOR_CFG`` because the stock
     config gave its calf the bare motor curve; A2's config was reconstructed correctly in
     the first place (peak torque, no-load speed, Fs/Fd and armature all read out of
     ``a2_description`` / ``a2.xml``), so there is nothing to correct.
"""

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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_A2_CFG
from unitree_rl_lab.tasks.dynamic import mdp

# --- scaling ratios, from the dimension table in assets/robots/unitree.py ---------------
MASS_RATIO = 40.071 / 16.087  # 2.491
LENGTH_RATIO = 0.275 / 0.213  # 1.291, thigh/calf link length
TIME_RATIO = LENGTH_RATIO**0.5  # 1.136

# Height every other height quantity is measured against: ``height_delta`` is
# ``root_pos_w[2] - NOMINAL_STANDING_HEIGHT``, and three separate things key off it --
# ``max_height``, ``motion_progress_standing``'s height term, and the ``landed`` gate
# (which needs |height_delta| < 0.10). Getting it wrong costs reward for standing
# correctly, which is exactly what happened on Go2 while this sat at the 0.40 default.
#
# MEASURED against a trained A2-Jump-Phase1 policy (300 iterations), holding its idle stand
# inside the A2-Jump-60 environment so the mass and PD-gain randomisation is active:
#
#     64 envs, 100 steps    mean 0.4174   median 0.4163   std 0.0137
#
# An earlier value of 0.36 -- A2's PASSIVE settle, released at its default joint angles and
# left to sag -- was wrong by +0.057 m, and a passive settle is the wrong thing to measure:
# a trained policy actively holds its commanded pose instead of sagging into it, landing
# essentially on the 0.428 m those joint angles describe geometrically. (Go2 differs here:
# its policy holds 0.287 against a 0.329 m geometric pose, i.e. it does NOT recover the sag.
# A2's PD gains are proportionally stiffer relative to its own weight, which is the likely
# reason -- so Go2's ratio cannot be borrowed, the number has to be measured per robot.)
#
# What the 0.057 m error cost, all three at once:
#     max_height                over-reported every jump by 0.057 m
#     motion_progress_standing  height_term = exp(-0.057^2/0.01) = 0.72, i.e. 28% of that
#                               reward forfeited for standing correctly
#     landed                    needs |height_delta| < 0.10, and standing already spent
#                               0.057 of it -- leaving 0.043 m of margin for a landing
#                               crouch, which is how `success` stays 0 and the assist
#                               curriculum never opens its 60% gate
NOMINAL_STANDING_HEIGHT = 0.416


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
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
    )
    robot: ArticulationCfg = UNITREE_A2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    jump = mdp.JumpCommandCfg(
        asset_name="robot",
        assist_body_names=["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
        resampling_time_range=(1.0e9, 1.0e9),
        auto_trigger=False,
        nominal_standing_height=NOMINAL_STANDING_HEIGHT,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    # 0.25 unchanged: this scales a joint-angle delta, which is dimensionless, and it is
    # what locomotion/robots/a2 already uses.
    #
    # Named for the action CLASS, not descriptively. ``export_deploy_cfg`` keys the exported
    # deploy YAML off this attribute name, and both it and the deploy-side C++ registry look
    # actions up as "JointPositionAction" -- Go2's dynamic tasks call this attribute
    # ``joint_position`` and consequently exported an offset of 0.0 under a key go2_ctrl does
    # not recognise. Every locomotion task in this repo already uses this spelling.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, params={"command_name": "jump"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, params={"command_name": "jump"})
        root_height = ObsTerm(func=mdp.root_height)
        root_roll_angle = ObsTerm(func=mdp.root_roll_angle)
        root_pitch_angle = ObsTerm(func=mdp.root_pitch_angle)
        maximum_jump_height = ObsTerm(func=mdp.maximum_jump_height, params={"command_name": "jump"})
        accumulated_root_pitch = ObsTerm(func=mdp.accumulated_root_pitch, params={"command_name": "jump"})
        accumulated_root_roll = ObsTerm(func=mdp.accumulated_root_roll, params={"command_name": "jump"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Spawn jitter, not a dynamic quantity -- left at Go2's values.
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
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
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class StandingRewardsCfg:
    upright = RewTerm(func=mdp.upright_reward, weight=2.0)
    standing_pose = RewTerm(func=mdp.standing_pose_reward, weight=1.0)
    stillness = RewTerm(func=mdp.stillness_reward, weight=1.0)
    # Joint velocity and acceleration both scale as inverse powers of R_t, so A2's values
    # for an equivalent motion are 1.14x and 1.29x SMALLER than Go2's. Squaring turns that
    # into a 1.3x / 1.7x weaker penalty at the same weight -- under a factor of two, and in
    # the direction that costs nothing, so both weights carry across untouched.
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # -2.0e-4 -> -2.0e-5. This one genuinely cannot carry across: standing torque is a
    # gravity torque, m*g*L, which is 2.49 * 1.29 = 3.21x larger on A2, and the term squares
    # it -- 10.3x. Dividing the weight by 10.3 keeps the penalty worth the same fraction of
    # the standing rewards above. (This is the same m*g*L factor `unitree.py` used to carry
    # Go2's PD gains over.) Note the jump task removes this term entirely; it only shapes
    # Phase 1.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            # No Head_* links on A2, and *_hip carries no collision geometry, so a term
            # named on it could never fire. Matches locomotion/robots/a2.
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_thigh", ".*_calf"],
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        # "base_link", not Go2's "base".
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    assist_force = None


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    # env_spacing 2.0 -> 3.0: A2's hip-to-hip span is 1.34x Go2's and a 1 m jump travels
    # further before it settles.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=3.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: StandingRewardsCfg = StandingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz control on a 200 Hz sim, as both Go2's jump task and A2's locomotion tasks
        # use. Control rate is not scaled with the robot: it is set by what the hardware
        # runs at, not by leg length.
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
