from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
    StandingRewardsCfg,
)


@configclass
class CommandsCfgPhase2(CommandsCfg):
    jump = CommandsCfg().jump.replace(
        auto_trigger=True,
        enable_jump=True,
        enable_backflip=False,
        enable_sideflip=False,
        trigger_time_range=(0.8, 1.2),
        target_height_range=(0.20, 0.20),
        target_pitch_turns_range=(0.0, 0.0),
        target_roll_turns_range=(0.0, 0.0),
        command_duration_s=0.50,
        assist_duration_s=0.10,
        total_assist_force=400.0,
        initial_assist_scale=1.0,
    )


@configclass
class JumpRewardsCfg(StandingRewardsCfg):
    upright = None
    standing_pose = None
    stillness = None

    pre_jump_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    jump_progress = RewTerm(
        func=mdp.jump_progress_reward,
        weight=1.0,
        params={"command_name": "jump", "scale": 0.01},
    )
    jump_progress_standing = RewTerm(
        func=mdp.jump_progress_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfgPhase2(CurriculumCfg):
    assist_force = CurrTerm(
        func=mdp.assist_force_decay,
        params={
            "command_name": "jump",
            "success_threshold": 0.60,
            "decay_step": 0.01,
            "minimum_episodes": 1024,
        },
    )


@configclass
class RobotEnvCfgPhase2(RobotEnvCfg):
    """Phase 2: command-triggered assisted jumping with EFGCL decay."""

    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    curriculum: CurriculumCfgPhase2 = CurriculumCfgPhase2()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3.0


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
