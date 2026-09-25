"""Sandbox Try-9: add mdp.foot_touchdown_height_gain (hypothesis B) on top of
Try-7's config -- NOT on top of Try-8, so A and B are tested in isolation
against the same Try-7 ancestor and don't confound each other.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-7 (feet_gait weight=0.2 retained, heading_drift_penalty
weight=-0.3 retained, heading_command off, Phase2 origin, demote_fraction=0.5,
step_width=0.3), plus a new foot_touchdown_height_gain reward term (weight=1.0).

Rationale (2026-07-09, same mujoco observation as Try-8): the robot gets
stuck when a front leg can't lift the body high enough for the rear feet to
reach the next step. Where Try-8 (body_height_gain) rewards the base's
continuous vertical velocity, this hypothesis rewards the discrete EVENT of
any foot -- most critically the rear foot that keeps failing to hook the
step -- touching down higher than its own best height so far this episode.
See rewards.py docstring for the full mechanism: tracks a per-foot running
best touchdown height (so bouncing to the same height repeatedly cannot farm
it, only genuine net progress per foot pays out), normalised by
target_gain=0.15m and capped at 1.0 per foot per step, gated on nonzero
command, uses the same episode_length_buf==1 reset-detection fix already
validated in heading_drift_penalty/Try-6.

``preserve_order=True`` is not load-bearing here (unlike Try-7's feet_gait,
which maps specific offsets to specific legs) since this term just sums
per-foot gains with no leg-identity-dependent parameter -- kept anyway for
consistency/safety so asset_cfg and sensor_cfg body indices are guaranteed
to refer to the same physical foot.

weight=1.0 is a first guess matched to forward_command_progress's weight=1.5
scale (per-event reward normalised into [0,1] per foot, same rough order of
magnitude) -- watch Episode_Reward/foot_touchdown_height_gain and mujoco
behavior to tune if needed.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import SandboxPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try7 import (
    RewardsCfgGo2Try7,
    RobotEnvCfgGo2Try7,
)

__all__ = ["RobotEnvCfgGo2Try9", "RobotPlayEnvCfgGo2Try9", "SandboxPPORunnerCfg"]


@configclass
class RewardsCfgGo2Try9(RewardsCfgGo2Try7):
    foot_touchdown_height_gain = RewTerm(
        func=mdp.foot_touchdown_height_gain,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "target_gain": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                preserve_order=True,
            ),
        },
    )


@configclass
class RobotEnvCfgGo2Try9(RobotEnvCfgGo2Try7):
    rewards: RewardsCfgGo2Try9 = RewardsCfgGo2Try9()


@configclass
class RobotPlayEnvCfgGo2Try9(RobotEnvCfgGo2Try9):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
