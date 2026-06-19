import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import (
    CRITIC_HEIGHT_SCAN_CFG,
    CurriculumCfgGo2,
    apply_manual_curriculum_level,
    apply_play_velocity_ranges,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_train_cfg import MANUAL_CURRICULUM_LEVEL
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
)

# Terrain layout for manual curriculum (columns are assigned by proportion).
GO2_CURRICULUM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0,
        ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

POLICY_HISTORY_LENGTH = 3
CRITIC_HISTORY_LENGTH = 3


@configclass
class PolicyCfgGo2(ObservationsCfg.PolicyCfg):
    """Go2 policy: per-term observation history for temporal context."""

    joint_pos_rel = ObservationsCfg.PolicyCfg().joint_pos_rel.replace(history_length=POLICY_HISTORY_LENGTH)
    joint_vel_rel = ObservationsCfg.PolicyCfg().joint_vel_rel.replace(history_length=POLICY_HISTORY_LENGTH)
    last_action = ObservationsCfg.PolicyCfg().last_action.replace(history_length=POLICY_HISTORY_LENGTH)

    def __post_init__(self):
        super().__post_init__()


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic: privileged ``height_scan`` plus per-term observation history."""

    height_scan = CRITIC_HEIGHT_SCAN_CFG
    joint_pos_rel = ObservationsCfg.CriticCfg().joint_pos_rel.replace(history_length=CRITIC_HISTORY_LENGTH)
    joint_vel_rel = ObservationsCfg.CriticCfg().joint_vel_rel.replace(history_length=CRITIC_HISTORY_LENGTH)
    last_action = ObservationsCfg.CriticCfg().last_action.replace(history_length=CRITIC_HISTORY_LENGTH)


@configclass
class ObservationsCfgGo2(ObservationsCfg):
    """Go2 observations: policy history; extended critic for privileged training."""

    policy: PolicyCfgGo2 = PolicyCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class CommandsCfgGo2(CommandsCfg):
    """Go2 v1: reduce standing-only env fraction."""

    base_velocity = CommandsCfg().base_velocity.replace(rel_standing_envs=0.01)


@configclass
class RewardsCfgGo2(RewardsCfg):
    """Go2-specific reward tuning."""

    track_ang_vel_z = RewardsCfg().track_ang_vel_z.replace(weight=1.0)

    feet_height_body_stairs = RewTerm(
        func=mdp.feet_height_body_stairs,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            # Extra swing lift above nominal stance; ramps 0.05 m -> 0.23 m with terrain level.
            "step_height_range": (0.05, 0.23),
            "tanh_mult": 2.0,
        },
    )

    wild_foot_clearance = RewTerm(
        func=mdp.wild_foot_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],  # trot: FR+RL vs FL+RR
            "radius": 0.1,
        },
    )




@configclass
class RobotSceneCfgGo2(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_generator=GO2_CURRICULUM_TERRAIN_CFG,
    )


@configclass
class RobotEnvCfgGo2(RobotEnvCfg):
    """Go2 velocity env with manual terrain curriculum."""
    curriculum_level: int = MANUAL_CURRICULUM_LEVEL
    focus_spin_in_place: bool = False

    scene: RobotSceneCfgGo2 = RobotSceneCfgGo2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgGo2 = CommandsCfgGo2()
    observations: ObservationsCfgGo2 = ObservationsCfgGo2()
    rewards: RewardsCfgGo2 = RewardsCfgGo2()
    curriculum: CurriculumCfgGo2 = CurriculumCfgGo2()

    def __post_init__(self):
        super().__post_init__()
        apply_manual_curriculum_level(self)


@configclass
class RobotPlayEnvCfgGo2(RobotEnvCfgGo2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5
        apply_play_velocity_ranges(self)
