import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import TerminationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)

PHASE2_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.50,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase2(RobotSceneCfgPhase1):
    terrain = RobotSceneCfgPhase1().terrain.replace(
        terrain_generator=PHASE2_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class CommandsCfgPhase2(CommandsCfgPhase1):
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
        ),
    )


# =============================================================================
# Anti-stall/climbing reward push, promoted from sandbox/try1.py after MuJoCo
# confirmed no flat-idle flapping (built gated from the start, reusing the
# reward combination already validated in the Phase3 sandbox lineage --
# see velocity_env_cfg_phase3.py's Try-1/Try-3/Try-4 writeups and
# sandbox/SUMMARY.md for the original discovery). Target: push terrain_levels
# past the ~3.8-3.9 plateau the plain RewardsCfgGo2 defaults hit on Phase2's
# rough/box terrain (no active terrain-height reward term at all).
#
#   - wild_foot_clearance -> mdp.adaptive_foot_clearance_reward, gated
#     (command_name/min_cmd_norm=0.1): its swing gate is an open-loop clock,
#     ungated it pays for marching in place at rest.
#   - base_height_climb (new): gated the same way -- ungated it chases local
#     terrain height even standing still.
#   - forward_command_progress, stall_penalty, stair_commit: all three
#     self-gate inside the reward function (zero whenever |command| <= 0.1,
#     or -- stair_commit -- only fire during an actual front-planted/
#     hind-down straddle, which can't occur on flat ground), so no extra
#     gating params needed.
#   - bad_orientation limit_angle 0.8 -> 1.0 (TerminationsCfgPhase2 below):
#     a robot committing to climb a box edge pitches further than flat-ground
#     walking; without this the termination fires before a climb attempt can
#     resolve.
# =============================================================================


@configclass
class RewardsCfgPhase2(RewardsCfgGo2):
    wild_foot_clearance = RewardsCfgGo2().wild_foot_clearance.replace(
        func=mdp.adaptive_foot_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_clearance": 0.03,
            "max_clearance": 0.22,
            "roughness_ref": 0.04,
            "command_name": "base_velocity",
            "min_cmd_norm": 0.1,
        },
    )

    base_height_climb = RewTerm(
        func=mdp.base_height_climb_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "nominal_clearance": 0.35,
            "std": 0.15,
            "command_name": "base_velocity",
            "min_cmd_norm": 0.1,
        },
    )

    forward_command_progress = RewardsCfgGo2().forward_command_progress.replace(weight=0.8)

    stall_penalty = RewTerm(
        func=mdp.stall_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_scale": 0.1,
        },
    )

    stair_commit = RewTerm(
        func=mdp.stair_commit_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "height_gap_threshold": 0.08,
            "max_forward_speed": 1.0,
            "max_climb_speed": 0.5,
        },
    )


@configclass
class TerminationsCfgPhase2(TerminationsCfg):
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class RobotEnvCfgPhase2(RobotEnvCfgPhase1):
    """Phase 2: rough terrain and boxes."""

    scene: RobotSceneCfgPhase2 = RobotSceneCfgPhase2(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase2 = CommandsCfgPhase2()
    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()
    terminations: TerminationsCfgPhase2 = TerminationsCfgPhase2()


@configclass
class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 3
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Raw ray-hit markers (red) show true sensor geometry, never the noise applied
        # downstream to the policy's observation -- off so only the noisy overlay is visible.
        self.scene.height_scanner.debug_vis = False
        # Temporary: overlay a noisy-height sample (orange) so the configured height-scan
        # noise applied to the policy's observation is visible in the viewport.
        self.observations.policy.height_scan.params["debug_vis_noise"] = True
