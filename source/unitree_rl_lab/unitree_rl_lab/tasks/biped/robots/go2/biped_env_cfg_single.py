"""Go2-Biped-Single: single-leg (FR_foot alone) standing, escalating from the
2-leg ``Go2-Biped-Front`` stance.

Mechanically much harder than the 2-leg stance: a 2-leg stance still has a
line segment as its support base (weight can shift along it, matching the
"handle_length" CoM-CoP framing already in this codebase), but a single foot
is a *point* -- balance requires active correction in every direction at
once, closer to balancing a rod on a fingertip than to the fore-aft
weight-shift the 2-leg biped rewards were built around. No direct precedent
in the TumblerNet paper (it only covers 2-leg fore/hind stances) or elsewhere
in this codebase.

Picked FR as the stance foot (either front leg works, robot is laterally
symmetric). Changes from ``Go2-Biped-Front``:

- CoM-CoP rewards (``pendulum_angle``, ``pendulum_instability``,
  ``handle_length``) retargeted from both front feet to ``FR_foot`` alone --
  and the matching ``com_cop`` *observation* terms (critic + estimator
  target) too, so the ground-truth CoM-CoP the estimator is trained to
  predict matches what the reward actually uses.
- ``front_hip/thigh/calf_motion`` (pins RR/RL to their default pose) weights
  -> 0.0: with only one supporting leg, the other three (FL, RR, RL) need
  full freedom to act as counterbalancing "arms," not be held near a fixed
  default the way a genuine swing leg was in the 2-leg stance.
- ``front_contact_force`` retargeted from the 2 hind feet to all 3
  non-stance feet (``FL_foot``, ``RR_foot``, ``RL_foot``) -- guards against
  quietly falling back to a 2-leg or tripod stance. Weight raised from -0.6
  to -1.2 (sandbox try3): try2's own training log showed this term's
  improvement flattening hard over the back half of 10k iterations (-0.16 at
  resume down to only -0.08 by the end) while play.py/MuJoCo still showed
  the free foot repeatedly tapping down to stabilize -- since this task
  already trains 100% on standing commands (no dilution to fix, unlike the
  earlier Phase2/Front head-droop cases), the fix was a heavier weight, not
  more iterations.
- ``front_body_height`` retargeted from both front hips to ``FR_hip`` alone,
  same 0.30 m target (still the physical-reach-calibrated value from the
  2-leg work).
- ``base_velocity`` forced to always-standing (``rel_standing_envs=1.0``):
  single-leg *standing* only for now, no walking attempted yet.

MuJoCo-confirmed (sandbox try3, mid-training) balancing on one leg. Training
continues past this promotion to improve stability further -- see
``sandbox/try3.py``/``SUMMARY.md`` for the live experimental record; this
file always reflects the currently-promoted recipe, not necessarily the
exact checkpoint in ``logs/rsl_rl/go2_biped_single``.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg_front import (
    ObservationsCfgFront,
    RobotEnvCfgFront,
    RobotPlayEnvCfgFront,
)

STANCE_FOOT_NAME = ["FR_foot"]
FREE_FOOT_NAMES = ["FL_foot", "RR_foot", "RL_foot"]
STANCE_HIP_NAME = ["FR_hip"]


@configclass
class ObservationsCfgSingle(ObservationsCfgFront):
    """Same as ``ObservationsCfgFront``, but ``com_cop`` uses only ``FR_foot`` as stance."""

    @configclass
    class CriticCfgSingle(ObservationsCfgFront.CriticCfgFront):
        com_cop = ObservationsCfgFront.CriticCfgFront().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAME),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAME),
            }
        )

    critic: CriticCfgSingle = CriticCfgSingle()

    @configclass
    class EstimatorTargetCfgSingle(ObservationsCfgFront.EstimatorTargetCfgFront):
        com_cop = ObservationsCfgFront.EstimatorTargetCfgFront().com_cop.replace(
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAME),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAME),
            }
        )

    estimator_target: EstimatorTargetCfgSingle = EstimatorTargetCfgSingle()


def _apply_single_overrides(cfg):
    cfg.observations = ObservationsCfgSingle()

    cfg.rewards.pendulum_angle.params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAME),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAME),
    }
    cfg.rewards.pendulum_instability.params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAME),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAME),
    }
    cfg.rewards.handle_length.params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=STANCE_FOOT_NAME),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE_FOOT_NAME),
    }

    cfg.rewards.front_hip_motion.weight = 0.0
    cfg.rewards.front_thigh_motion.weight = 0.0
    cfg.rewards.front_calf_motion.weight = 0.0

    cfg.rewards.front_contact_force = RewTerm(
        func=mdp.front_foot_contact_force,
        weight=-1.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FREE_FOOT_NAMES)},
    )

    cfg.rewards.front_body_height.params["asset_cfg"] = SceneEntityCfg("robot", body_names=STANCE_HIP_NAME)

    cfg.commands.base_velocity.rel_standing_envs = 1.0


@configclass
class RobotEnvCfgSingle(RobotEnvCfgFront):
    """Single-leg-stance (FR_foot) variant of the Go2 bipedal env."""

    def __post_init__(self):
        super().__post_init__()
        _apply_single_overrides(self)


@configclass
class RobotPlayEnvCfgSingle(RobotPlayEnvCfgFront):
    def __post_init__(self):
        super().__post_init__()
        _apply_single_overrides(self)
