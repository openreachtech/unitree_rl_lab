"""Sandbox Try-6: re-test of Try-5's design, after fixing a real bug in
mdp.heading_drift_penalty.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Identical config to Try-5 (same reward weights including heading_drift_penalty
weight=-0.3, heading_command off, Phase2 origin, demote_fraction=0.5,
step_width=0.3) -- this file just re-registers it under a new task id so the
run history stays clean.

What actually changed is in rewards.py, not here: heading_drift_penalty used
`env.episode_length_buf == 0` to detect "just reset, capture fresh spawn
yaw". But `episode_length_buf` is incremented BEFORE reward computation each
step and only reset to 0 AFTER reward computation (see
ManagerBasedRLEnv.step()), so it is NEVER 0 at the point this reward function
runs -- the first post-reset reward call sees it at 1. That condition never
fired, so the "expected yaw" tracker never reset across episode boundaries:
every fresh episode's actual (randomly re-spawned) yaw was compared against a
stale expected-yaw carried over from a completely unrelated previous episode,
producing a large, essentially random "error" almost constantly, not a real
drift signal. This explains Try-5's result (terrain_levels~4.5, bad_orientation
~47%, heading_drift_penalty averaging -0.69/step) -- the robot was being
punished by noise, not by anything it could learn to avoid. Fixed to check
`episode_length_buf == 1`. Try-6 is the first real test of whether the
underlying idea (penalize yaw diverging from the commanded-ang_vel_z integral,
to discourage unrequested reversal-climbing without touching command
generation) actually works.
"""

from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try5 import (
    RewardsCfgGo2Try5,
    RobotEnvCfgGo2Try5,
    RobotPlayEnvCfgGo2Try5,
    SandboxPPORunnerCfg,
)

__all__ = [
    "RewardsCfgGo2Try6",
    "RobotEnvCfgGo2Try6",
    "RobotPlayEnvCfgGo2Try6",
    "SandboxPPORunnerCfg",
]

RewardsCfgGo2Try6 = RewardsCfgGo2Try5
RobotEnvCfgGo2Try6 = RobotEnvCfgGo2Try5
RobotPlayEnvCfgGo2Try6 = RobotPlayEnvCfgGo2Try5
