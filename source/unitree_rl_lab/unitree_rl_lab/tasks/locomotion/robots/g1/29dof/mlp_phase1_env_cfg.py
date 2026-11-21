"""G1 MLP Phase1: default velocity task + sim-to-real additions from humanoid-gym.

Differences from ``velocity_env_cfg.RobotEnvCfg`` (everything else is inherited):
  1. Action delay / noise randomization (``DelayedNoisyJointPositionAction``).
     Sim only; actor output, observations and deploy.yaml layout are unchanged.
  2. Privileged critic observations: push kick, foot friction, torso added mass,
     foot contact state. Critic only; the actor (deployed policy) is unchanged.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import ActionsCfg, EventCfg, ObservationsCfg, RobotEnvCfg

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
class MlpPhase1EventCfg(EventCfg):
    # same as the default push, but records the kick for the critic
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_logged,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class MlpPhase1ObservationsCfg(ObservationsCfg):
    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        push_vel = ObsTerm(func=mdp.push_velocity)
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
    events: MlpPhase1EventCfg = MlpPhase1EventCfg()
    observations: MlpPhase1ObservationsCfg = MlpPhase1ObservationsCfg()


@configclass
class MlpPhase1PlayEnvCfg(MlpPhase1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # mirror RobotPlayEnvCfg
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
