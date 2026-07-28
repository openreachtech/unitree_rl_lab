import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion import terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase3 import (
    RewardsCfgPhase3BalanceFloating,
    RobotEnvCfgPhase3BalanceFloating,
    RobotSceneCfgPhase3Balance,
)

# =============================================================================
# Phase 4: continual learning on top of Phase3-balance-floating -- dedicated
# terrain mix for stepping over short free-standing walls.
#
# Phase3-balance-floating already climbs stairs, including floating (open-riser)
# ones; that behavior is expected to carry over via the checkpoint (trained
# with --previous-task pointed at the promoted balance-floating checkpoint, not
# from scratch), not by keeping stairs in this phase's terrain mix. This phase
# mirrors the stairfocus/balance pattern of specializing the mix on one
# obstacle: 10% flat (keeps flat-ground gait honest) + solid/floating thin
# walls (mirrors the pyramid_stairs / floating_pyramid_stairs split in the
# stair terrain mixes).
# =============================================================================
PHASE4_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=20,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        # Short free-standing wall to step over. First attempt at a fixed 0.20 m
        # height went nowhere -- terrain_levels collapsed and never recovered
        # (robot never got past the wall) -- so height now scales with
        # difficulty too, 0.05 m (easy) -> 0.25 m (hard), same convention as
        # step_height_range on the stair terrains; thickness still narrows
        # 0.15 m (easy) -> 0.03 m (hard); walls spaced a fixed 0.60 m apart.
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.40,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
        # Same wall, but hollowed out: no solid body, just the thin tread
        # hovering at wall_height with an open gap underneath -- mirrors
        # floating_pyramid_stairs_inv's role in the stair terrain mixes.
        "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
            proportion=0.50,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)

@configclass
class RobotSceneCfgPhase4(RobotSceneCfgPhase3Balance):
    terrain = RobotSceneCfgPhase3Balance().terrain.replace(
        terrain_generator=PHASE4_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


# =============================================================================
# Gait fix, promoted from sandbox/try1.py after MuJoCo confirmed it: Phase4
# inherited RewardsCfgPhase3BalanceFloating unchanged, and MuJoCo testing
# found the robot walked flat ground (and crossed walls) with an unnecessary
# wide-legged, side-to-side hip-swinging gait instead of a natural stride.
#
#   - joint_pos restored to the Go2 default weight (-0.7, was -0.3 via
#     RewardsCfgPhase3Balance's stair-climbing relaxation -- more than half
#     the normal cost for any joint drifting from its default pose). Tried
#     alone first; didn't fix the swing (it's a blanket penalty across all 12
#     joints, so it can't isolate hip abduction from thigh/calf motion
#     climbing still needs), but is still correct to keep on its own terms.
#   - joint_deviation_hips (new): penalizes hip-joint deviation specifically,
#     gated to the open-loop swing phase (mdp.joint_deviation_swing_gated_l1)
#     rather than terrain flatness -- a terrain-flatness gate was tried first
#     and relaxes exactly during obstacle crossing, which is precisely where
#     the unwanted hip motion also showed up. Swing phase is common to both
#     flat-ground and obstacle-crossing lift, so it reaches both.
#   - calf_flexion_clearance (new): a positive reward for clearing an
#     obstacle via calf/knee flexion specifically (mdp.calf_flexion_clearance_reward),
#     scaled by the same obstacle-lookahead/roughness signal
#     adaptive_foot_clearance_reward uses. A penalty alone only says "don't
#     use hip," not "use calf instead" -- this gives the policy an explicit,
#     rewarded substitute so suppressing hip motion doesn't just remove
#     capability (an earlier ungated-penalty-only attempt traded away wall-
#     climbing entirely over continued training instead of finding this
#     substitute on its own).
#
# MuJoCo confirmed: lateral hip-swing gone, walls now cleared via calf/knee
# flexion ("bending the elbow") as intended.
#
# A further try (flat_orientation_l2/base_linear_velocity tightened toward
# Phase3's own values, for a still-more-natural flat gait) made things worse
# and was not promoted -- left at RewardsCfgPhase3BalanceFloating's values.
# =============================================================================


@configclass
class RewardsCfgPhase4(RewardsCfgPhase3BalanceFloating):
    joint_pos = RewardsCfgPhase3BalanceFloating().joint_pos.replace(weight=-0.7)

    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_swing_gated_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
            ),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
        },
    )

    calf_flexion_clearance = RewTerm(
        func=mdp.calf_flexion_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "calf_asset_cfg": SceneEntityCfg(
                "robot", joint_names=["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_flex": 0.05,
            "max_flex": 0.4,
            "max_obstacle_height": 0.25,
            "roughness_ref": 0.05,
        },
    )


@configclass
class RobotEnvCfgPhase4(RobotEnvCfgPhase3BalanceFloating):
    """Phase 4: Phase3-balance-floating's terminations/commands, unchanged --
    terrain mix changes (flat + thin_wall + floating_thin_wall) and rewards
    gain the gait fix above. See module docstring."""

    scene: RobotSceneCfgPhase4 = RobotSceneCfgPhase4(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase4 = RewardsCfgPhase4()


@configclass
class RobotPlayEnvCfgPhase4(RobotEnvCfgPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Play-only terrain: drop "flat" and use exactly 2 columns (one per
        # wall type) x 5 rows (5 difficulty levels) so each row shows one
        # thin_wall and one floating_thin_wall side by side at that
        # difficulty -- .replace() here builds a new TerrainGeneratorCfg
        # rather than mutating PHASE4_TERRAIN_CFG's sub_terrains dict in
        # place, which is a module-level object shared with the training cfg.
        self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.replace(
            num_rows=5,
            num_cols=2,
            sub_terrains={
                "thin_wall": terrains.MeshThinWallTerrainCfg(
                    proportion=0.5,
                    wall_height_range=(0.05, 0.25),
                    wall_thickness_range=(0.15, 0.05),
                    wall_spacing=0.60,
                    platform_width=2.0,
                    border_width=1.0,
                ),
                "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
                    proportion=0.5,
                    wall_height_range=(0.05, 0.25),
                    wall_thickness_range=(0.15, 0.05),
                    wall_spacing=0.60,
                    platform_width=2.0,
                    border_width=1.0,
                ),
            },
        )
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
