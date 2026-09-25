"""Sandbox Try-8: add mdp.body_height_gain (hypothesis A) to directly reward
the base for gaining absolute world height, on top of Try-7's config.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-7 (feet_gait weight=0.2 retained, heading_drift_penalty
weight=-0.3 retained, heading_command off, Phase2 origin, demote_fraction=0.5,
step_width=0.3), plus a new body_height_gain reward term (weight=1.0).

Rationale (2026-07-09, direct mujoco testing of Try-7): with front feet
staggered across two step heights (e.g. right front on step 1, left front on
step 2), the robot cannot generate enough lift through the higher-placed
front leg to raise its body on tall/narrow stairs (works fine on
shallow/wide stairs where less lift is needed) -- so the rear feet never
reach the step behind. feet_gait only rewards contact-TIMING and gave no
signal for this; forward_command_progress only rewards horizontal velocity,
so the exact moment the robot is straining to lift (zero horizontal progress
yet) is currently reward-invisible. body_height_gain rewards positive dz/dt
of the base directly, independent of horizontal motion, filling that gap.
See rewards.py docstring for the full mechanism (dz/dt scaled to be
comparable to forward_command_progress's velocity units, gated on nonzero
command, clamped as a safety cap, uses the same episode_length_buf==1
reset-detection fix already validated in heading_drift_penalty/Try-6).

weight=1.0 is a first guess matched to forward_command_progress's weight=1.5
scale (same m/s units) -- watch Episode_Reward/body_height_gain and mujoco
behavior to tune if needed. Being tested in isolation against hypothesis B
(velocity_env_cfg_try9.py, foot_touchdown_height_gain) -- both add on top of
Try-7 independently so neither confounds the other's result.
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

__all__ = ["RobotEnvCfgGo2Try8", "RobotPlayEnvCfgGo2Try8", "SandboxPPORunnerCfg"]


@configclass
class RewardsCfgGo2Try8(RewardsCfgGo2Try7):
    body_height_gain = RewTerm(
        func=mdp.body_height_gain,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )


@configclass
class RobotEnvCfgGo2Try8(RobotEnvCfgGo2Try7):
    rewards: RewardsCfgGo2Try8 = RewardsCfgGo2Try8()


@configclass
class RobotPlayEnvCfgGo2Try8(RobotEnvCfgGo2Try8):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
