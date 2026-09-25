"""Sandbox Try-7: add a fixed-clock "lateral sequence walk" gait-phase reward
to fix uncoordinated front/rear leg timing while climbing.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-6 (heading_drift_penalty weight=-0.3 retained,
heading_command off, Phase2 origin, demote_fraction=0.5, step_width=0.3),
plus a new reward term using the already-existing (but previously unused)
mdp.feet_gait function.

Rationale (2026-07-08, grounded in direct mujoco testing of Try-6 + user's
own biomechanical observation): user reports the front feet place solidly on
the next step, but the rear legs don't keep pace and end up scrambling
("バタバタ") to catch up -- an uncoordinated front/rear timing problem, not a
balance or heading problem. User asked whether thinking about how real dogs
climb stairs might suggest a fix.

It does: quadrupeds (dogs included) switch from a trot to a "lateral sequence
walk" on stairs/uncertain footing -- moving exactly ONE leg at a time in the
sequence RH-RF-LH-LF (or its mirror), with a high (~75%) stance duty factor,
so 3 feet are always planted and the base of support never shrinks to just a
diagonal pair. This is the standard, well-documented gait quadrupeds default
to specifically when stability matters more than speed (Hildebrand gait
diagrams). Nothing in the current reward set enforces any consistent
leg-timing relationship at all -- feet_air_time/air_time_variance_penalty
only look at each leg's OWN air-time statistics in isolation, never at
cross-leg phase coordination -- so the policy has had no signal pushing it
toward a coordinated sequence, consistent with the "front gets ahead, rear
scrambles" symptom.

mdp.feet_gait(period, offset, sensor_cfg, threshold, command_name) already
exists in rewards.py (apparently added for a different experiment/robot, but
never wired into RewardsCfgGo2) and does exactly this: for each foot it
computes a phase clock (global_phase + per-leg offset) and rewards +1 per
step when actual contact state matches the phase-predicted stance/swing
state. Configured here as:
  - body order RR, FR, RL, FL with offsets 0.0, 0.25, 0.5, 0.75 (lateral
    sequence walk: one leg swings at a time, in a full round-robin).
  - threshold=0.75 (75% stance / 25% swing duty factor per leg -- maximizes
    time with 3+ feet down, the actual stability property being borrowed
    from real quadruped gait, not the specific leg-order identity, which
    doesn't materially matter as long as swing windows never overlap).
  - period=1.0s: a first guess at a comfortable step-cycle duration for
    careful climbing, not derived from any logged evidence -- watch
    Episode_Reward/feet_gait and actual mujoco behavior to see if it's too
    fast/slow and adjust in a Try-8 if needed.
  - weight=0.2: feet_gait's raw range is [0, 4] per step (sum of 4 per-leg
    0/1 matches), notably larger scale than most other terms (e.g.
    track_lin_vel_xy_exp is bounded [0, 1] at weight=1.5); 0.2 keeps its
    plausible max contribution (~0.8) in the same ballpark as the existing
    dominant terms instead of swamping them.
  - ``preserve_order=True`` on the SceneEntityCfg is required -- it defaults
    to False, which would silently ignore the given body_names order and
    break the offset-to-leg mapping.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import SandboxPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try6 import (
    RewardsCfgGo2Try6,
    RobotEnvCfgGo2Try6,
)

__all__ = ["RobotEnvCfgGo2Try7", "RobotPlayEnvCfgGo2Try7", "SandboxPPORunnerCfg"]


@configclass
class RewardsCfgGo2Try7(RewardsCfgGo2Try6):
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,
        params={
            "period": 1.0,
            "offset": [0.0, 0.25, 0.5, 0.75],
            "threshold": 0.75,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["RR_foot", "FR_foot", "RL_foot", "FL_foot"],
                preserve_order=True,
            ),
        },
    )


@configclass
class RobotEnvCfgGo2Try7(RobotEnvCfgGo2Try6):
    rewards: RewardsCfgGo2Try7 = RewardsCfgGo2Try7()


@configclass
class RobotPlayEnvCfgGo2Try7(RobotEnvCfgGo2Try7):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
