"""Sandbox Try-5: reward-only replacement for heading_command.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-1 (same reward weights, same Phase2 origin,
demote_fraction=0.5, step_width=0.3), EXCEPT:
  1. heading_command is OFF (reverted globally in velocity_env_cfg.py --
     command generation is back to plain free ang_vel_z sampling, same as
     before this whole investigation started).
  2. A new reward term, mdp.heading_drift_penalty, weight=-0.3, is added.

Rationale (2026-07-07, user directive after testing both Try-1 and Try-4 in
mujoco): heading_command=True (Try-1, Try-3, Try-4) never fixed the reported
symptoms (lateral movement broken even on flat ground, straight-line drift
while climbing) and, per direct user testing of Try-4
(rel_heading_envs=0.3), made general walking quality worse than Try-1 despite
tensorboard metrics (terrain_levels, bad_orientation, entropy) looking fine
or even slightly better. That is a strong signal that continuously altering
the ang_vel_z COMMAND DISTRIBUTION itself (what heading_command does) has
side effects on overall policy quality that these particular training
metrics do not surface, and that layering more command-generation tweaks on
top of it (rel_heading_envs, heading_control_stiffness) was not converging on
a fix -- user explicitly flagged that stacking further patches onto a
not-clearly-working approach ("積み重ねる方式はやっぱりよくないかも") was
the wrong direction.

This switches strategy entirely: instead of changing what command the policy
is trained against, leave command generation exactly as it always was
(unchanged from before any of this session's heading work) and instead add a
targeted REWARD term (see mdp.heading_drift_penalty docstring in rewards.py
for the full mechanism) that penalises the robot's actual yaw diverging from
what the commanded ang_vel_z, integrated since spawn, predicts. This should
only cost the policy something when it does an UNREQUESTED rotation (like a
reversal-climb flip); it should never fight legitimate large commanded turns,
and critically it does not touch the command distribution the policy trains
against at all, so it should not have the same broad locomotion-quality
side effects heading_command did.

weight=-0.3 is a first, evidence-free guess at the right magnitude (unlike
the target_clearance/joint_pos etc. weights, which were tuned against logged
evidence) -- treat this trial's own bad_orientation/entropy/mujoco behavior
as the first evidence for whether -0.3 is too strong, too weak, or about
right, and adjust in a Try-6 if needed.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import (
    RewardsCfgGo2Try1,
    SandboxPPORunnerCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RobotEnvCfgGo2

__all__ = ["RobotEnvCfgGo2Try5", "RobotPlayEnvCfgGo2Try5", "SandboxPPORunnerCfg"]


@configclass
class RewardsCfgGo2Try5(RewardsCfgGo2Try1):
    heading_drift_penalty = RewTerm(
        func=mdp.heading_drift_penalty,
        weight=-0.3,
        params={"command_name": "base_velocity"},
    )


@configclass
class RobotEnvCfgGo2Try5(RobotEnvCfgGo2):
    rewards: RewardsCfgGo2Try5 = RewardsCfgGo2Try5()


@configclass
class RobotPlayEnvCfgGo2Try5(RobotEnvCfgGo2Try5):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
