"""Sandbox Try-1: re-baseline the accumulated reward design from a fixed Phase2
checkpoint, under the corrected (non-exploitable, non-softened) environment.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical reward weights to the old greedy-accumulation "BEST"
(target_clearance=0.15, joint_pos=-0.2, flat_orientation_l2=-0.5,
base_linear_velocity=-1.0, air_time_variance=-0.2, forward_command_progress=1.5),
frozen here so future edits to velocity_env_cfg_go2.py cannot silently change
this Try's definition after the fact. Two things differ from how "BEST" was
actually produced:
  1. demote_fraction is the stock 0.5 here, not the softened 0.25 used through
     2026-07-06 (reverted per user directive -- the curriculum's own judgment
     of progress must not be modified, only the reward function/weights).
  2. CommandsCfg.base_velocity.heading_command = True (shared fix in
     velocity_env_cfg.py) closes the backward-climbing exploit discovered via
     mujoco testing on 2026-07-06.
This run measures what the same reward design achieves without either crutch.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RewardsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RobotEnvCfgGo2


@configclass
class SandboxPPORunnerCfg(BasePPORunnerCfg):
    """All go2/sandbox Try-N tasks share this experiment_name so every trial's
    runs land in the same familiar `logs/rsl_rl/unitree_go2_velocity_v1/` root
    (and can `--load_run`/`--checkpoint` any checkpoint already there, e.g. the
    fixed Phase2 baseline), instead of BasePPORunnerCfg's default empty
    experiment_name, which auto-fills to a per-task-id folder and would hide
    Try-N's checkpoints in their own isolated experiment root."""

    experiment_name = "unitree_go2_velocity_v1"


@configclass
class RewardsCfgGo2Try1(RewardsCfg):
    foot_clearance_terrain_adaptive = RewTerm(
        func=mdp.foot_clearance_terrain_adaptive,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.15,
            "command_name": "base_velocity",
        },
    )
    forward_command_progress = RewTerm(
        func=mdp.forward_command_progress,
        weight=1.5,
        params={"command_name": "base_velocity"},
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh"]),
        },
    )


@configclass
class RobotEnvCfgGo2Try1(RobotEnvCfgGo2):
    rewards: RewardsCfgGo2Try1 = RewardsCfgGo2Try1()


@configclass
class RobotPlayEnvCfgGo2Try1(RobotEnvCfgGo2Try1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
