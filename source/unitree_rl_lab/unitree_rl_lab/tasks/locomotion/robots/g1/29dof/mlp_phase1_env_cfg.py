"""G1 MLP Phase1: default velocity task + sim-to-real and reward ideas from the literature.

Differences from ``velocity_env_cfg.RobotEnvCfg`` (everything else is inherited):

From humanoid-gym:
  1. Action delay / noise randomization (``DelayedNoisyJointPositionAction``).
     Sim only; actor output, observations and deploy.yaml layout are unchanged.
  2. Privileged critic observations: push force, foot friction, torso added mass,
     foot contact state. Critic only; the actor (deployed policy) is unchanged.

From van Marum et al., "Revisiting Reward Design and Evaluation for Robust Humanoid
Standing and Walking" (clock-free SaW):
  3. Gait rewards without a clock: the clock-locked ``gait`` reward (whose clock the
     policy never observed) is replaced by ``single_foot_contact`` + ``feet_air_time_cmd``.
  4. Pushes are sustained forces (N for a duration), not instantaneous velocity kicks.
  5. Commands are sampled by category (stand 10%, sagittal / lateral / turn / omni),
     resampled every 4-8 s. vx / vy follow the paper's ranges; yaw goes up to +-1.0 rad/s
     (paper: +-0.5), starting at +-0.5. All three use the level curriculum.

Left-right symmetry augmentation is configured in the runner cfg
(``SymmetricPPORunnerCfg``), not here.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

FEET_BODIES = ".*_ankle_roll_link"


@configclass
class MlpPhase1ActionsCfg(ActionsCfg):
    # keep the term name "JointPositionAction": export_deploy_cfg special-cases it for the offset
    JointPositionAction = mdp.DelayedNoisyJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        max_delay=0.5,
        noise_std=0.02,
    )


@configclass
class MlpPhase1CommandsCfg(CommandsCfg):
    base_velocity = mdp.CategoricalLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.0,  # standing comes from category_probs
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        category_probs=(0.1, 0.225, 0.225, 0.225, 0.225),
        ranges=mdp.CategoricalLevelVelocityCommandCfg.Ranges(
            # yaw starts at +-0.5: at +-0.1 rad/s ignoring the command costs only ~4% of the
            # tracking reward, so the policy never learned to turn
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.CategoricalLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 2.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class MlpPhase1EventCfg(EventCfg):
    # sustained force on the torso; the term runs every policy step (interval set in __post_init__)
    push_robot = EventTerm(
        func=mdp.push_by_force,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "body_name": "torso_link",
            "force_range": (20.0, 100.0),
            "duration_range": (0.2, 0.5),
            "push_interval_s": (4.0, 8.0),
        },
    )


@configclass
class MlpPhase1RewardsCfg(RewardsCfg):
    gait = None  # clock-locked; replaced by the two clock-free terms below
    single_foot_contact = RewTerm(
        func=mdp.single_foot_contact,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "command_name": "base_velocity",
            "grace_s": 0.2,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_cmd,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )


@configclass
class MlpPhase1CurriculumCfg(CurriculumCfg):
    # yaw tracking reward plateaus near 0.5 x weight (the default 0.8 was never reached)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels, params={"threshold": 0.4})


@configclass
class MlpPhase1ObservationsCfg(ObservationsCfg):
    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        push_force = ObsTerm(func=mdp.push_force)
        feet_friction = ObsTerm(
            func=mdp.body_static_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES)},
        )
        torso_added_mass = ObsTerm(
            func=mdp.body_added_mass,
            scale=1.0 / 3.0,  # randomized in (-1, 3) kg -> roughly unit scale
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
        feet_contact = ObsTerm(
            func=mdp.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES)},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class MlpPhase1EnvCfg(RobotEnvCfg):
    actions: MlpPhase1ActionsCfg = MlpPhase1ActionsCfg()
    commands: MlpPhase1CommandsCfg = MlpPhase1CommandsCfg()
    events: MlpPhase1EventCfg = MlpPhase1EventCfg()
    rewards: MlpPhase1RewardsCfg = MlpPhase1RewardsCfg()
    curriculum: MlpPhase1CurriculumCfg = MlpPhase1CurriculumCfg()
    observations: MlpPhase1ObservationsCfg = MlpPhase1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        step_dt = self.sim.dt * self.decimation
        self.events.push_robot.interval_range_s = (step_dt, step_dt)


@configclass
class MlpPhase1PlayEnvCfg(MlpPhase1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # mirror RobotPlayEnvCfg
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
