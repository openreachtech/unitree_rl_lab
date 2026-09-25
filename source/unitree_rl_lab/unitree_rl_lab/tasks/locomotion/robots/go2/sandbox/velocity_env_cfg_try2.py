"""Sandbox Try-2: soften heading correction to reduce the fall-rate regression from Try-1.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-1 (same reward weights, same Phase2 origin, same
demote_fraction=0.5, same heading_command=True), except
CommandsCfg.base_velocity.heading_control_stiffness is lowered 1.0 -> 0.5.

Rationale (grounded in Try-1's own logged evidence, not a guess): Try-1 reached
terrain_levels~6.3 (goal >6.0 met) but Episode_Termination/bad_orientation rose
from ~0.0015 (old exploit-tainted regime) to a stable 0.13-0.15 -- roughly 100x.
Two logged signals point at heading correction, not the climbing reward terms
themselves, as the driver: Metrics/base_velocity/error_vel_yaw sat at ~0.47 rad
(~27 deg) for the whole plateau (the heading controller was fighting a large,
persistent error rather than converging), and Episode_Reward/base_angular_velocity
(ang_vel_xy_l2, roll/pitch angular velocity penalty, untouched weight=-0.05)
averaged -0.114, i.e. real roll/pitch wobble, not just noise. heading_command's
correction torque (ang_vel_z = clip(stiffness * heading_error, ...)) is applied
every step regardless of the robot's current climbing phase; halving the
stiffness makes the yaw-holding pressure gentler (still fully closes the
180-degree-reversal exploit -- any sustained wrong heading still draws a
continuous, if weaker, correction -- since a reversal was never about resisting
a brief nudge, it was about avoiding a sustained cost) while hopefully reducing
how much it fights the stairs' inherent pitch disturbance moment-to-moment.
Nothing else changes, so any shift in bad_orientation isolates this one lever.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import (
    RewardsCfgGo2Try1,
    SandboxPPORunnerCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RobotEnvCfgGo2

__all__ = ["RobotEnvCfgGo2Try2", "RobotPlayEnvCfgGo2Try2", "SandboxPPORunnerCfg"]


@configclass
class RobotEnvCfgGo2Try2(RobotEnvCfgGo2):
    rewards: RewardsCfgGo2Try1 = RewardsCfgGo2Try1()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.heading_control_stiffness = 0.5


@configclass
class RobotPlayEnvCfgGo2Try2(RobotEnvCfgGo2Try2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
        self.commands.base_velocity.heading_control_stiffness = 0.5
