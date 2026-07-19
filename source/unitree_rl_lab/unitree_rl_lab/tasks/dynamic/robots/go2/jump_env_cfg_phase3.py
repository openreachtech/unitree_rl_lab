from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic import mdp
from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    RobotEnvCfg,
    StandingRewardsCfg,
    TerminationsCfg,
)

# Motion-selection flags. Enable both to sample one motion per environment.
TRAIN_BACKFLIP = True
TRAIN_SIDEFLIP = False


@configclass
class CommandsCfgPhase3(CommandsCfg):
    jump = CommandsCfg().jump.replace(
        auto_trigger=True,
        enable_jump=False,
        enable_backflip=TRAIN_BACKFLIP,
        enable_sideflip=TRAIN_SIDEFLIP,
        trigger_time_range=(0.8, 1.2),
        target_height_range=(0.0, 0.0),
        target_pitch_turns_range=(-1.0, -1.0),
        target_roll_turns_range=(-1.0, -1.0),
        assist_body_names=["FR_hip", "FL_hip", "RR_hip"],
        backflip_assist_body_names=("FR_hip", "FL_hip"),
        sideflip_assist_body_names=("FR_hip", "RR_hip"),
        command_duration_s=0.50,
        assist_duration_s=0.10,
        backflip_assist_force=350.0,
        sideflip_assist_force=600.0,
        initial_assist_scale=1.0,
        minimum_landing_time_s=0.80,
        state_file="logs/rsl_rl/unitree_go2_jump_phase3/jump_curriculum_state.json",
    )


@configclass
class FlipRewardsCfg(StandingRewardsCfg):
    upright = None
    standing_pose = None
    stillness = None

    pre_motion_standing = RewTerm(
        func=mdp.pre_jump_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress = RewTerm(
        func=mdp.motion_progress_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    motion_progress_standing = RewTerm(
        func=mdp.motion_progress_standing_reward,
        weight=1.0,
        params={"command_name": "jump"},
    )
    non_target_rotation = RewTerm(
        func=mdp.non_target_angular_velocity_penalty,
        weight=-0.05,
        params={
            "command_name": "jump",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FlipTerminationsCfg(TerminationsCfg):
    bad_orientation = None


@configclass
class CurriculumCfgPhase3(CurriculumCfg):
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
class RobotEnvCfgPhase3(RobotEnvCfg):
    """Phase 3: assisted backflip, optionally mixed with sideflip."""

    commands: CommandsCfgPhase3 = CommandsCfgPhase3()
    rewards: FlipRewardsCfg = FlipRewardsCfg()
    terminations: FlipTerminationsCfg = FlipTerminationsCfg()
    curriculum: CurriculumCfgPhase3 = CurriculumCfgPhase3()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4.0


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Play mode ignores the training curriculum's saved decay state and defaults to no
        # external assist -- edit initial_assist_scale below (0.0-1.0) to manually test with
        # partial/full assist force instead.
        self.commands.jump.state_file = None
        self.commands.jump.initial_assist_scale = 0.0
        self.observations.policy.enable_corruption = False
