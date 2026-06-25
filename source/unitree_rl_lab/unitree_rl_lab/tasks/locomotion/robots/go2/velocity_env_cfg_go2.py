from dataclasses import replace

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import (
    CRITIC_HEIGHT_SCAN_CFG,
    PLAY_VEL_RANGES,
    CurriculumCfgGo2,
    apply_manual_curriculum_level,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg

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
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.05, 0.15),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic: base fields plus privileged ``height_scan`` (enabled per curriculum level)."""

    height_scan = CRITIC_HEIGHT_SCAN_CFG

@configclass
class PolicyCfgGo2(ObservationsCfg.PolicyCfg):
    height_scan = None  # populated by apply_phase_height_scanner per curriculum_level

@configclass
class ObservationsCfgGo2(ObservationsCfg):
    """Go2 observations: policy unchanged; extended critic for privileged training."""

    policy: PolicyCfgGo2 = PolicyCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class RewardsCfgGo2(RewardsCfg):
    foot_clearance_torso_relative = None  # disabled: replaced by terrain-adaptive version below

    # -- Fix 2 (revised): terrain-relative foot clearance, now a POSITIVE, capped,
    #    swing+command-gated reward. The previous exp(-penalty) form scored its maximum
    #    (1.0) for a fully-planted standing robot, giving no gradient to step and
    #    reinforcing the standing local optimum that stalled the curriculum. Now a
    #    planted robot earns 0 and a robot that lifts its swing feet over step edges
    #    earns up to ~1 per swing foot (capped). Weight lowered 1.0 -> 0.5 so a single
    #    statically-held leg cannot be farmed for more reward than walking.
    foot_clearance_terrain_adaptive = RewTerm(
        func=mdp.foot_clearance_terrain_adaptive,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.08,
            "command_name": "base_velocity",
        },
    )

    # -- Fix 5: positive, linear forward-progress signal toward the commanded
    #    heading. Bootstraps the robot onto the stairs and feeds the curriculum,
    #    which advances on displacement from spawn. Weight raised 1.0 -> 1.5 to widen
    #    the moving-vs-standing reward gap (up to par with track_lin_vel_xy) while
    #    staying clamped to the commanded speed so it cannot drive reckless lunging.
    forward_command_progress = RewTerm(
        func=mdp.forward_command_progress,
        weight=1.5,
        params={"command_name": "base_velocity"},
    )

    # -- Fix 6: relax the air-time variance penalty (-1.0 -> -0.2). Stairs are climbed
    #    one leg at a time, so per-leg air times are necessarily asymmetric; penalising
    #    that at -1.0 fought the natural stair gait directly. A standing robot has zero
    #    variance either way, so this does not strengthen the standing optimum; it only
    #    removes a penalty on asymmetric stepping. Mild -0.2 keeps flat-gait regularity.
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    # -- Fix 3: relax rewards that directly contradict stair climbing. Climbing
    #    requires body pitch, vertical motion and large joint excursions; the stock
    #    weights penalised all three so hard that "don't climb" was the local optimum.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)      # was -2.5
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)            # was -2.0
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.4,                                                              # was -0.7
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    # Drop ``.*_calf`` from the contact penalty: shins legitimately brush step edges
    # while climbing, and penalising that discouraged climbing.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh"]),
        },
    )


@configclass
class RobotSceneCfgGo2(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_generator=GO2_CURRICULUM_TERRAIN_CFG,
        max_init_terrain_level=0,
    )


@configclass
class RobotEnvCfgGo2(RobotEnvCfg):
    """Go2 velocity env with manual terrain curriculum."""
    curriculum_level: int = 3
    scene: RobotSceneCfgGo2 = RobotSceneCfgGo2(num_envs=4096, env_spacing=2.5)
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
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
