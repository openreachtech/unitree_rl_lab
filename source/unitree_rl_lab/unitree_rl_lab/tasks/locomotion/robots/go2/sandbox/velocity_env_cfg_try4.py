"""Sandbox Try-4: restore genuine flat-zero ang_vel_z training exposure to fix
lateral movement / straight-line drift, by lowering rel_heading_envs.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-1 (same reward weights, same Phase2 origin,
demote_fraction=0.5, heading_command=True, step_width=0.3 -- deliberately NOT
combined with Try-3's narrower width, to isolate this one variable), except
CommandsCfg.base_velocity.rel_heading_envs is lowered 1.0 -> 0.3.

Rationale (grounded in direct mujoco testing feedback on Try-1 and Try-3,
2026-07-07): the user reports TWO related symptoms with both heading_command
exports so far: (1) lateral (lin_vel_y) movement is broken -- on flat ground
too, not just stairs, ruling out a stairs-specific reward interaction; (2)
while climbing, the robot sometimes drifts diagonally left instead of going
straight, even under a plain forward-only command.

With rel_heading_envs=1.0 (Try-1/Try-3's setting), heading_control continuously
recomputes ang_vel_z from the live heading error every step, for every env.
Even once heading has converged, small ongoing disturbances (stair pitch,
foot contact impulses, ...) keep re-perturbing the heading slightly, so
ang_vel_z is *almost never exactly, persistently zero* during training. But
mujoco's keyboard deploy sends a genuinely flat, constant ang_vel_z=0 for as
long as the user isn't pressing y/u -- a training-time-rare-to-nonexistent
input pattern. This plausibly explains both symptoms: the policy never
learned to hold a straight, non-drifting line under a truly static zero yaw
command, and never got much practice combining a sustained flat ang_vel_z
with an active lin_vel_y (pure strafing).

Lowering rel_heading_envs to 0.3 means 70% of envs keep the OLD, pre-fix
behavior (ang_vel_z freely and independently sampled, which is exactly the
distribution that used to produce working lateral movement/straight-line
walking), while the remaining 30% still hold a target heading and should
still provide some ongoing pressure against the reversal-climb exploit.
This is a real trade-off, not a full closure of the exploit -- if the
reversal comes back, dial rel_heading_envs back up and try
heading_control_stiffness (Try-2, interrupted, still worth revisiting) or a
different mitigation instead.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import (
    RewardsCfgGo2Try1,
    SandboxPPORunnerCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RobotEnvCfgGo2

__all__ = ["RobotEnvCfgGo2Try4", "RobotPlayEnvCfgGo2Try4", "SandboxPPORunnerCfg"]


@configclass
class RobotEnvCfgGo2Try4(RobotEnvCfgGo2):
    rewards: RewardsCfgGo2Try1 = RewardsCfgGo2Try1()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_heading_envs = 0.3


@configclass
class RobotPlayEnvCfgGo2Try4(RobotEnvCfgGo2Try4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
        self.commands.base_velocity.rel_heading_envs = 0.3
