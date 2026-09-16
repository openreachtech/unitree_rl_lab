"""Shared A2 MDP: the Go2 velocity task one size up.

A near-verbatim port of ``robots/go2/velocity_env_cfg.py``. Everything structural is
the same -- the same observation groups, the same reward terms, the same curriculum --
and the differences are the ones A2's body forces. They fall into four groups, each
derived from one scale factor. The factors are quoted here and at every site that uses
one; they are deliberately not module constants, because nothing computes with them --
the scaled values are written out literally so a reader sees the number the sim gets:

  lengths   x1.29, the leg-length ratio (0.213 + 0.213 -> 0.275 + 0.275 m). Terrain
            heights, clearance targets, scan extents, ring radii.
  times     x1.14, sqrt of the length ratio. A leg is a pendulum, so its natural
            cadence goes as sqrt(L): gait periods, air-time thresholds, and the
            velocities that follow from them.
  torques   x3.22, the m*g*L ratio (16.087 x 0.213 -> 40.071 x 0.275). Only the reward
            weights that read torque directly -- squared torque is x10.4, and left at Go2's
            weight it would swamp every tracking term.
  masses    x2.49 total (16.087 -> 40.071 kg), x2.84 for the trunk alone (6.921 ->
            19.651 kg). Payload randomisation.

Three naming differences, all forced by ``a2_description``:
  * the trunk link is ``base_link``, not ``base``.
  * there are no ``Head_*`` links.
  * ``*_hip`` carries no collision geometry, so no contact term can fire on a hip.
    ``undesired_contacts`` therefore covers thighs and calves only.

The height-scan grid keeps Go2's 29 x 21 = 609 rays by scaling the resolution along
with the extent (0.05 -> 0.065 m over 1.4 x 1.0 -> 1.82 x 1.30 m), so the grid is the
same grid on a body 1.3x longer rather than a finer one -- same raycast cost at 4096
envs, and the same "one scanner, one resolution" rule the Go2 configs settled on.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_A2_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# A2's trunk is 0.73 m front to back against Go2's 0.38, so the tile has to be bigger to
# leave the same run-up either side of a scaled platform_width. 8.0 -> 10.0 m is the
# length scale rounded up; horizontal_scale and vertical_scale stay at Go2's values,
# which are mesh discretisation rather than difficulty, and being relatively finer here
# costs nothing.
A2_TERRAIN_TILE = (10.0, 10.0)

# The one height-scan grid, Go2's 609 rays over a 1.3x footprint (see module docstring).
# 1.82 / 0.065 = 28 and 1.30 / 0.065 = 20, so the pattern is 29 x 21 exactly.
HEIGHT_SCAN_RESOLUTION = 0.065
HEIGHT_SCAN_SIZE = (1.82, 1.30)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Ground terrain. A bare plane, and a placeholder: every registered A2 task overrides
    # it -- Phase 1 with its own plane, Phases 2-4 with the generator each phase needs.
    # It stays here only so the base scene is a complete, instantiable config.
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
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_A2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # Mounted on ``base_link``; A2's trunk link is not called ``base``. ordering="yx"
    # (idx = ix * num_y + iy) to match _height_scan_indices, exactly as on Go2.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=HEIGHT_SCAN_RESOLUTION, size=list(HEIGHT_SCAN_SIZE), ordering="yx"
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

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
        # Go2's (-1.0, 3.0) on a 6.921 kg trunk, i.e. -14% to +43%, applied to A2's
        # 19.651 kg trunk as the same fractions rather than the same kilograms.
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.8, 8.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        # x/y spread scaled x1.29 with the body, so a reset displaces the robot by the
        # same fraction of its own length as on Go2.
        params={
            "pose_range": {"x": (-0.65, 0.65), "y": (-0.65, 0.65), "yaw": (-3.14, 3.14)},
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
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        # x1.14: a push is a velocity, so it scales with the robot's own speed scale, not
        # with its mass. The disturbance stays equally hard to reject.
        params={"velocity_range": {"x": (-0.57, 0.57), "y": (-0.57, 0.57)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Linear velocity limits x1.14 (Froude); angular left at Go2's. Body-frame yaw rate
    # scales as 1/sqrt(L), i.e. x0.88, which would put the limit at 0.88 rad/s -- but a
    # 1.3x longer robot turning at a given yaw rate sweeps its feet 1.3x faster, and the
    # Go2 lineage never found +-1.0 to be the binding constraint. Left alone, and flagged
    # here as the one command range not derived from a scale factor.
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.15, 1.15), lin_vel_y=(-0.45, 0.45), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # scale 0.25 unchanged: the action is a joint-angle delta in radians, and A2's leg is
    # geometrically similar to Go2's, so the same angle means the same posture change.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Identical to Go2's, term for term and scale for scale: every entry is either an
    angle, an angular rate, a command, or a previous action, none of which change
    meaning with body size. The 45-dim policy vector is therefore the same 45 dims.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        # scale 0.01 / 3.4: A2's joints carry ~3.2x Go2's torque at the same posture, so
        # the same scale would hand the critic a term 3.4x louder than every neighbour.
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.003, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Weights are Go2's except where the quantity being penalised is itself larger on A2.
    Three are:

      joint_torques   sum of squared torque, x10.4 at the same posture -> weight / 11.8
      energy          sum of |torque| x |velocity|, x3.22 / 1.14 = x2.82 -> weight / 3.0
      joint_vel       sum of squared joint velocity, x(1/1.14)^2 = x0.77 -> weight x1.3

    ``joint_acc`` is left at Go2's -2.5e-7. Squared joint acceleration scales as
    (1/1.29)^2 = 0.6, so the faithful value is -4.2e-7, but the term contributes on the
    order of 1e-3 to the return either way and the existing weight is already the
    conservative direction. Flagged rather than changed, so the difference is a decision
    rather than an oversight.

    Everything else -- tracking, orientation, air time, slide, limits, action rate -- is
    in units that do not change with body size, or is already normalised by a target that
    does (see the clearance terms in velocity_env_cfg_blind.py).
    """

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0013)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.7e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-6.6e-6)

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            # x1.14: the threshold separates "walking" from "standing still", and A2's
            # walking speeds are Froude-scaled.
            "velocity_threshold": 0.34,
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            # x1.14: a swing is a pendulum half-period, so the target air time scales
            # with sqrt(leg length) like every other gait time here.
            "threshold": 0.57,
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

    # -- other
    # Thighs and calves only. Go2's list also has ``Head_.*`` and ``.*_hip``; A2 has no
    # head links at all, and its hip links carry no collision geometry, so a hip can
    # never register a contact and naming it would only risk the sensor failing to
    # resolve the body.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings. env_spacing x1.29 with the body; it only bites on Phase 1, whose
    # terrain is a bare plane rather than a tiled generator.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=3.2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings. 50 Hz control on a 200 Hz sim, same as Go2: the control rate
        # is set by the hardware, not by the robot's size.
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
