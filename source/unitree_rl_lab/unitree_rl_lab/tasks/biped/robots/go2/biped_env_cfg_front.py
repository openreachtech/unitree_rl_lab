"""Bipedal (2-leg stance) locomotion task for Go2 using the FRONT legs as the
stance/support legs (hind legs lifted/tucked) -- the mirror-image stance the paper
describes: "By changing the contact penalty for front legs to hind legs, the robot
can perform bipedal locomotion with front legs" (Xiao et al. 2025 / "TumblerNet").

Scene, domain randomization, commands, action space, terminations, and the
estimator/actor architecture are identical to the hind-leg-stance ``biped_env_cfg``
(reused via inheritance); only the stance feet and the tucked-leg motion penalties
swap sides.

Two pieces added after sandbox experiments (round: Try1-3, see
``sandbox/SUMMARY.md``) found the front-leg stance touched the ground with the
head (front of the body) at a standstill, and that lateral movement was
half-range compared to forward/backward for no particular reason:

- ``front_body_height`` targets ``FR_hip``/``FL_hip`` (always physically at
  the front of the body, regardless of which end is currently the stance leg)
  at 0.30 m -- Go2's own thigh+calf segment lengths (0.213 m each) cap the
  physical reach at ~0.426 m, so ``BASE_HEIGHT_TARGET`` (0.55 m, borrowed from
  an unrelated reference robot) was never reachable to begin with. Boosted
  5x specifically while the commanded velocity is near zero
  (``command_name``/``command_threshold``/``standstill_boost``), since
  ``rel_standing_envs`` is ~0.1 -- a standstill-specific shortfall would
  otherwise get diluted into a healthy-looking population average dominated
  by the 90% of training spent walking.
- ``lin_vel_y`` widened from the inherited +-0.5 m/s to +-1.0 m/s, matching
  ``lin_vel_x`` (no reason found for the half-width default; MuJoCo-confirmed
  working at full width).

This absorbed what would otherwise have been a separate forward-only "Phase 2"
(mirroring the hind-leg ``biped_env_cfg_phase2.py``) -- MuJoCo testing showed
the front-leg stance moving cleanly in all commanded directions without one,
so no `biped_env_cfg_front_phase2.py` exists.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import CommandsCfg, ObservationsCfg, RewardsCfg, RobotEnvCfg

# Hind legs swing/tuck (lifted); front legs are the stance/support legs -- mirror
# image of the hind-leg-stance biped_env_cfg's STANCE_FOOT_NAMES / *_motion penalties.
HIND_CALF_JOINT_NAMES = ["RR_calf_joint", "RL_calf_joint"]
STANCE_FOOT_NAMES = ["FR_foot", "FL_foot"]
HIND_FOOT_NAMES = ["RR_foot", "RL_foot"]
# Always physically at the front of the body regardless of which end is the
# current stance leg -- used as the "head clearance" proxy (see module docstring).
FRONT_HIP_BODY_NAMES = ["FR_hip", "FL_hip"]


@configclass
class ObservationsCfgFront(ObservationsCfg):
    """Same layout as ``ObservationsCfg``, but ``com_cop`` uses the front feet as stance."""

    @configclass
    class CriticCfgFront(ObservationsCfg.CriticCfg):
        com_cop = ObservationsCfg.CriticCfg().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
            }
        )

    critic: CriticCfgFront = CriticCfgFront()

    @configclass
    class EstimatorTargetCfgFront(ObservationsCfg.EstimatorTargetCfg):
        com_cop = ObservationsCfg.EstimatorTargetCfg().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
            }
        )

    estimator_target: EstimatorTargetCfgFront = EstimatorTargetCfgFront()


@configclass
class RewardsCfgFront(RewardsCfg):
    """Same reward terms as ``RewardsCfg``, with stance/swing legs swapped front<->hind."""

    # -- keep the swing (now hind) legs near their default pose instead of flailing --
    front_hip_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_hip_joint", "RL_hip_joint"])},
    )
    front_thigh_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"])},
    )
    front_calf_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIND_CALF_JOINT_NAMES)},
    )

    # -- CoM-CoP balance (TumblerNet); stance = front feet --
    pendulum_angle = RewardsCfg().pendulum_angle.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )
    pendulum_instability = RewardsCfg().pendulum_instability.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )
    handle_length = RewardsCfg().handle_length.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAMES),
        }
    )

    # `RewardsCfg.front_contact_force` targets FRONT_FOOT_NAMES (FR/FL) -- the
    # *stance* legs here, not the swing legs. Left un-overridden, the policy
    # would be penalized for putting its own support legs on the ground.
    # Retarget to the (now-swing) hind feet.
    front_contact_force = RewTerm(
        func=mdp.front_foot_contact_force,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FOOT_NAMES)},
    )

    # Keeps the head (front of the body) clear of the ground -- see module
    # docstring for why this is needed on top of base_height, why the target is
    # 0.30 m, and why the standstill boost.
    front_body_height = RewTerm(
        func=mdp.front_body_height_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FRONT_HIP_BODY_NAMES),
            "target_height": 0.30,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "standstill_boost": 5.0,
        },
    )


@configclass
class CommandsCfgFront(CommandsCfg):
    """Same as ``CommandsCfg``, but ``gait_mode`` is pinned to front-biped instead
    of hind-biped (inert either way -- this file's reward set never reads
    gait_mode -- but keeps the observation semantically correct), and
    ``lin_vel_y`` widened to +-1.0 m/s (see module docstring)."""

    gait_mode = mdp.PinnedGaitModeCommandCfg(asset_name="robot", pinned_mode=mdp.MODE_FRONT_BIPED)

    def __post_init__(self):
        self.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)


@configclass
class RobotEnvCfgFront(RobotEnvCfg):
    """Front-leg-stance variant of the Go2 bipedal env."""

    commands: CommandsCfgFront = CommandsCfgFront()
    observations: ObservationsCfgFront = ObservationsCfgFront()
    rewards: RewardsCfgFront = RewardsCfgFront()


@configclass
class RobotPlayEnvCfgFront(RobotEnvCfgFront):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
