import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import TerminationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase2 import (
    CommandsCfgPhase2,
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)

PHASE3_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.25),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.60,
            step_height_range=(0.05, 0.25),
            step_width=0.23,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# Tread narrows from 0.25 m (difficulty 0 / row 0) to 0.19 m (difficulty 1 /
# row 9) instead of being 0.19 m at every level -- the same difficulty-driven
# interpolation step_height_range already uses, applied to tread width too
# (see terrains.MeshPyramidStairsVariableWidthCfg). A robot promoted through
# terrain_levels sees an easy, wide-tread staircase before it ever reaches
# the narrow-tread rows. Used by Phase3-stairfocus below (sandbox Try-10
# result: terrain_levels 4.869, softer initial collapse on first stairs
# contact, faster recovery than the fixed-width PHASE3_TERRAIN_CFG).
PHASE3_TERRAIN_CFG_VARIABLE_WIDTH = terrain_gen.TerrainGeneratorCfg(
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
        # 1段ごとに登らせる想定（段差、踏み幅広め）
        "pyramid_stairs_wide": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 1段ごとに登らせる想定（段差、踏み幅広め）
        "pyramid_stairs_inv_wide": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 普通の階段
        "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.40,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase3(RobotSceneCfgPhase2):
    terrain = RobotSceneCfgPhase2().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class CommandsCfgPhase3(CommandsCfgPhase2):
    base_velocity = CommandsCfgPhase2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.2, 1.2)
        ),
    )


@configclass
class RewardsCfgPhase3(RewardsCfgGo2):
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-1.0)
    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-0.5)
    joint_torques = RewardsCfgGo2().joint_torques.replace(weight=-1e-4)
    action_rate = RewardsCfgGo2().action_rate.replace(weight=-0.05)
    feet_air_time = RewardsCfgGo2().feet_air_time.replace(
        weight=0.07,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.35,
        },
    )
    air_time_variance = RewardsCfgGo2().air_time_variance.replace(weight=-0.2)
    wild_foot_clearance = RewardsCfgGo2().wild_foot_clearance.replace(weight=0.4)
    # foot_clearance_terrain_adaptive = RewardsCfgGo2().foot_clearance_terrain_adaptive.replace(weight=0.4)
    # forward_command_progress = RewardsCfgGo2().forward_command_progress.replace(weight=1.0)


@configclass
class RobotEnvCfgPhase3(RobotEnvCfgPhase2):
    """Phase 3: inverted pyramid stairs."""

    scene: RobotSceneCfgPhase3 = RobotSceneCfgPhase3(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase3 = CommandsCfgPhase3()
    rewards: RewardsCfgPhase3 = RewardsCfgPhase3()


@configclass
class RobotPlayEnvCfgPhase3(RobotEnvCfgPhase3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 3
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges



# =============================================================================
# Phase3-balance: promoted from sandbox Try-13. Independent of StairFocus
# above (which now tracks Try-8) -- kept as its own flattened class so
# StairFocus's later reward changes can't leak into this validated recipe.
# Swaps the fixed-target foot-clearance reward for mdp.adaptive_foot_clearance_reward
# (obstacle-aware lookahead + terrain-roughness gate), so the clearance target
# collapses to a natural ~3 cm lift on flat ground and only scales up toward
# real riser height near a stair. Trades peak terrain_levels (~4.9, measured
# on the original easier terrain before it was hardened) for a natural
# flat-ground gait and much higher episode survivability (time_out 91.5% vs
# ~79-80%, bad_orientation 3.7% vs ~11-15%).
#
# Try-4 -> Try-11: feet_slide strengthened -0.1 -> -0.2 (stock isaaclab_tasks
# reward, previously inherited unchanged from the base RewardsCfg despite all
# the clearance-reward work) to discourage foot-dragging/scuffing, on top of
# adaptive_foot_clearance_reward's lift incentive.
#
# Try-11 -> Try-13: wild_foot_clearance's max_clearance raised 0.20 -> 0.23 m
# -- 0.20 m was capping the adaptive clearance target 1 cm *below* a 0.21 m
# stair riser, a plausible direct cause of toe-catching on the edge. MuJoCo
# confirmed improvement (fewer bad_orientation falls) with no flat-ground
# regression. Try-12 additionally added landing_stability_reward alongside
# this same clearance raise, but that caused idle foot-fidgeting and stair
# refusal (landing_stability is farmable by standing still and tapping a
# foot, with nothing in Balance to make standing still costly) -- dropped,
# not part of this promotion. Terrain is unchanged by this promotion -- still
# whatever RobotSceneCfgPhase3Balance below already uses, no floating-stairs
# terrain.
# =============================================================================


@configclass
class RewardsCfgPhase3Balance(RewardsCfgPhase3):
    """Preserves the exact Try-13 recipe (Try-1/2's relaxed penalties +
    Try-4's terrain-adaptive clearance + Try-11's strengthened feet_slide +
    Try-13's raised clearance cap) that produced its validated result --
    independent of RewardsCfgPhase3StairFocus (which now tracks Try-8)."""

    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.3)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.2)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.3)
    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)
    forward_command_progress = RewardsCfgPhase3().forward_command_progress.replace(weight=0.8)
    feet_air_time = RewardsCfgPhase3().feet_air_time.replace(weight=0.1)
    wild_foot_clearance = RewardsCfgPhase3().wild_foot_clearance.replace(
        func=mdp.adaptive_foot_clearance_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_clearance": 0.03,
            "max_clearance": 0.23,
            "roughness_ref": 0.05,
        },
    )
    # Strengthened from the base RewardsCfg's -0.1 (see promotion note above).
    feet_slide = RewardsCfgPhase3().feet_slide.replace(weight=-0.2)


@configclass
class RobotSceneCfgPhase3Balance(RobotSceneCfgPhase3):
    terrain = RobotSceneCfgPhase3().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_VARIABLE_WIDTH,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgPhase3Balance(RobotEnvCfgPhase3):
    """Phase 3 - balance: natural flat-ground gait + terrain_levels >= 4.5 (sandbox Try-13 result: 5.049).

    Extends RobotEnvCfgPhase3 directly. Uses the variable-width terrain
    (PHASE3_TERRAIN_CFG_VARIABLE_WIDTH) and the independent
    RewardsCfgPhase3Balance above.
    """

    scene: RobotSceneCfgPhase3Balance = RobotSceneCfgPhase3Balance(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase3Balance = RewardsCfgPhase3Balance()


@configclass
class RobotPlayEnvCfgPhase3Balance(RobotEnvCfgPhase3Balance):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# =============================================================================
# Phase3-stairfocus: promoted from sandbox Try-10 (best stair-climbing result
# to date, superseding the Try-8 and original Try-1/Try-2 promotions).
#
# Composition (each piece traces to the sandbox try that introduced it):
#   Try-1/2 : relaxed flat_orientation_l2/base_linear_velocity/joint_pos/
#             undesired_contacts, feet_air_time 0.2, forward_command_progress 0.8
#   Try-5   : wild_foot_clearance -> mdp.adaptive_foot_clearance_reward
#             (obstacle-aware lookahead + roughness gate, tuned for the taller
#             risers: max_clearance 0.22, roughness_ref 0.04)
#   Try-6   : + base_height_climb (rewards the base for actually rising onto
#             a step instead of leaving the hind legs behind)
#   Try-7   : + stall_penalty (penalizes freezing at an obstacle instead of
#             attempting it)
#   Try-8   : joint_torques/action_rate/energy relaxed further (room for a
#             committed, dynamic push) + stair_commit (sharply rewards
#             forward+upward progress specifically while straddling a step)
#   Try-10  : scene.terrain -> PHASE3_TERRAIN_CFG_VARIABLE_WIDTH (tread narrows
#             with difficulty instead of being 0.19 m at every level); rewards
#             unchanged from Try-8
#
# MuJoCo sim-to-sim testing (fixed 0.21 m step / 0.19 m tread) confirmed real
# behavioral progress through the sandbox lineage:
#   Try-1/2 (this class's original definition) : climbs smoothly on the
#       *old*, easier terrain, but with an exaggerated flat-ground gait
#       (see Phase3-balance) and gets stuck straddling a step once terrain
#       got harder.
#   Try-5/6/7 (harder terrain, various fixes)  : freezes at the very first
#       step rather than attempting to climb -- surviving by doing nothing,
#       not by succeeding (time_out 91-93% while terrain_levels stayed ~3.6-3.9).
#   Try-8 (fixed 0.19 m tread)                 : climbs to about the 4th
#       step before falling -- a genuine break from the freeze/stuck
#       plateau, at the cost of some survivability (time_out 84.9%,
#       base_contact 9.6%) from more committed, less risk-averse attempts.
#       terrain_levels reached ~4.75, still rising, not yet plateaued.
#   Try-10 (this config, variable-width tread) : same rewards as Try-8, but
#       reaches the 4th step with front legs while the hind legs have
#       already reached the 1st step (vs Try-8's hind legs left further
#       behind) -- confirmed improved. terrain_levels reached ~4.87, with a
#       softer initial collapse and faster recovery on first stairs contact
#       than the fixed-width terrain.
#
# Still open (see sandbox try11+ if pursued): front feet can still get
# several steps ahead of the hind feet before the hind legs catch up -- no
# current reward caps that stride span.
# =============================================================================


@configclass
class RewardsCfgPhase3StairFocus(RewardsCfgPhase3):
    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.3)
    base_linear_velocity = RewardsCfgPhase3().base_linear_velocity.replace(weight=-0.2)
    joint_pos = RewardsCfgPhase3().joint_pos.replace(weight=-0.3)
    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)
    feet_air_time = RewardsCfgPhase3().feet_air_time.replace(weight=0.2)
    forward_command_progress = RewardsCfgPhase3().forward_command_progress.replace(weight=0.8)

    # Relaxed further so the policy has room for a committed, dynamic push
    # onto a step rather than only the efficient, smooth gait these
    # penalties otherwise shape for flat walking.
    joint_torques = RewardsCfgPhase3().joint_torques.replace(weight=-3e-5)
    action_rate = RewardsCfgPhase3().action_rate.replace(weight=-0.02)
    energy = RewardsCfgPhase3().energy.replace(weight=-5e-6)

    # Terrain-adaptive foot clearance: obstacle-aware lookahead + roughness
    # gate instead of a fixed target, so flat sections don't get an
    # exaggerated marching lift; scales toward the real riser height (up to
    # 0.25 m) as a step approaches.
    wild_foot_clearance = RewardsCfgPhase3().wild_foot_clearance.replace(
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
        },
    )

    # Reward the base for sitting at nominal standing height above whatever
    # terrain is directly beneath it, so it actually rises onto a step
    # instead of leaving the hind legs behind (the "straddling" failure mode).
    base_height_climb = RewTerm(
        func=mdp.base_height_climb_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "nominal_clearance": 0.35,
            "std": 0.15,
        },
    )

    # Penalize near-zero body speed while a command is active, so freezing
    # at an obstacle isn't the locally safer strategy.
    stall_penalty = RewTerm(
        func=mdp.stall_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_scale": 0.1,
        },
    )

    # Sharply reward forward+upward body progress specifically while the
    # front feet are planted on terrain higher than the terrain under the
    # hind feet -- concentrates gradient on the exact moment a decisive
    # hind-leg push matters most.
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
class TerminationsCfgPhase3StairFocus(TerminationsCfg):
    # Stairs require more sustained body pitch than flat/rough terrain.
    bad_orientation = TerminationsCfg().bad_orientation.replace(params={"limit_angle": 1.0})


@configclass
class RobotSceneCfgPhase3StairFocus(RobotSceneCfgPhase3):
    terrain = RobotSceneCfgPhase3().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_VARIABLE_WIDTH,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgPhase3StairFocus(RobotEnvCfgPhase3):
    """Phase 3 - stair focus: best stair-climbing result to date (sandbox Try-10:
    terrain_levels ~4.87 on the variable-width terrain, reaches ~4th step with
    front legs / ~1st step with hind legs in MuJoCo)."""

    scene: RobotSceneCfgPhase3StairFocus = RobotSceneCfgPhase3StairFocus(num_envs=4096, env_spacing=2.5)
    rewards: RewardsCfgPhase3StairFocus = RewardsCfgPhase3StairFocus()
    terminations: TerminationsCfgPhase3StairFocus = TerminationsCfgPhase3StairFocus()


@configclass
class RobotPlayEnvCfgPhase3StairFocus(RobotEnvCfgPhase3StairFocus):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# Balance terrain + floating inverted pyramid stairs (thin treads / open risers).
# Same mix as PHASE3_TERRAIN_CFG_VARIABLE_WIDTH, with 0.30 of the solid
# inverted-stairs mass moved onto MeshFloatingInvertedPyramidStairsTerrainCfg.
PHASE3_TERRAIN_CFG_BALANCE_FLOATING = terrain_gen.TerrainGeneratorCfg(
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
        "pyramid_stairs_wide": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs": terrains.MeshPyramidStairsVariableWidthCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "floating_pyramid_stairs": terrains.MeshFloatingPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_wide": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.60, 0.30),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsVariableWidthCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "floating_pyramid_stairs_inv": terrains.MeshFloatingInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.25),
            step_width_range=(0.27, 0.23),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase3BalanceFloating(RobotSceneCfgPhase3Balance):
    terrain = RobotSceneCfgPhase3Balance().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_BALANCE_FLOATING,
        max_init_terrain_level=5,
    )


# =============================================================================
# Phase3-balance-floating: promoted from sandbox Try-1 through Try-7.
#
# Problem 1 (Try-1): on the plain balance-floating recipe (RewardsCfgPhase3Balance
# + default TerminationsCfg), the robot detects a stair, reaches the edge, then
# stops. RewardsCfgPhase3Balance's forward_command_progress rewards positive
# progress but gives zero gradient once a robot has already stopped, so on
# terrain hard enough that climbing attempts often fail (this mix is 30%
# floating_pyramid_stairs_inv, whose pit has no safety floor -- a missed
# footing falls further than on solid stairs), freezing at the obstacle
# becomes the reward-maximizing policy instead of committing to the climb.
#
# Fix 1: port the anti-stall terms already validated in RewardsCfgPhase3StairFocus
# (base_height_climb, stall_penalty, stair_commit) plus its matching
# bad_orientation relaxation, onto this task's own reward/termination classes
# (RewardsCfgPhase3Balance and plain Phase3-balance are untouched).
#
# Try-1 result (resumed from Unitree-Go2-Velocity-v2-Phase2, ~2900 iterations):
#   terrain_levels 5.514 (still rising, no plateau) vs 4.899 for plain balance
#   time_out 87.1% / base_contact 5.3% / bad_orientation 7.7%
#   (vs plain balance's time_out 91.5% / bad_orientation 3.7% -- expected
#   trade: a robot that commits to climbing instead of freezing fails more
#   often than one that stands still, but reaches much further)
#   stall_penalty ~ -0.055 (near zero -- little residual stalling)
#   stair_commit/base_height_climb both firing positive, confirming the robot
#   is actively pushing through the straddle rather than freezing there.
#
# Problem 2 (Try-2 through Try-7): MuJoCo deploy testing found the policy
# flattening/flapping its legs on flat ground at zero command. rel_standing_envs
# is 0.01 for all Go2 tasks (CommandsCfgGo2) -- only 1% of envs ever get a
# near-zero command -- so this checkpoint had almost never been trained on
# "stand still, zero command." Worse, two reward terms are unconditional on
# command: base_height_climb_reward always chases local terrain height (Try-2
# raising rel_standing_envs alone made this worse, not better, by giving 10x
# more zero-command episodes -- still spawned across the full terrain mix, not
# just flat ground -- to a term that was still always-on); and
# wild_foot_clearance's swing gate (_cpg_leg_phases_rad) is a pure open-loop
# clock, independent of command or motion, so it kept paying for a marching
# gait even standing still (Try-3/Try-4).
#
# Fix 2: rel_standing_envs 0.01 -> 0.1 (Try-2), gate base_height_climb (Try-3)
# and wild_foot_clearance (Try-4) to 0 whenever |command| <= 0.1. That fixed
# four-leg marching but left a small single-leg twitch at rest (nothing
# actively rewards *stillness*, only removes reward for motion). Try-5 added
# quiet_standing_reward (all feet planted + low joint velocity) to close that
# gap, but at weight 0.5 with only a command gate, it also rewarded "freeze
# here" for standing envs parked on/near a stair -- MuJoCo testing showed
# climbing then regressed (front legs would reach the 4th step, then fall,
# instead of driving the hind legs up). Try-6 added a terrain-flatness gate
# (reusing wild_foot_clearance's roughness-gate pattern) so quiet_standing only
# fires on a genuinely flat terrain cell, never on/near a stair -- structurally
# closing that interference path regardless of weight -- and cut weight to
# 0.15 as a second precaution, which fixed climbing but weakened the
# flat-terrain fix (0.15 was calibrated for firing ~10x more often, before the
# flatness gate made it rare). Try-7 raised the weight back to 0.5 now that the
# gate (not the weight) is what protects climbing.
#
# Try-7 result (resumed from Unitree-Go2-Velocity-v2-Phase2, 2x3000 iterations):
#   terrain_levels 5.26 (comparable to Try-1's 5.51 plateau)
#   base_contact 0.043 / bad_orientation ~0.09 / time_out 0.865
#   stair_commit 0.27 -- best of the whole Try-1..Try-7 lineage
#   quiet_standing near-zero in the population-wide average (expected: it only
#   fires for the rare standing-env-on-a-flat-cell subset) but MuJoCo testing
#   confirmed both the flat-ground twitch and the stair-climbing regression
#   were resolved together.
# =============================================================================


@configclass
class CommandsCfgPhase3BalanceFloating(CommandsCfgPhase3):
    # Restore a real standing-env fraction (Try-2): CommandsCfgGo2 drops this
    # to 0.01 for all Go2 tasks, which left this checkpoint almost never
    # trained on "stand still, zero command."
    base_velocity = CommandsCfgPhase3().base_velocity.replace(rel_standing_envs=0.1)


@configclass
class RewardsCfgPhase3BalanceFloating(RewardsCfgPhase3Balance):
    base_height_climb = RewTerm(
        func=mdp.base_height_climb_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "nominal_clearance": 0.35,
            "std": 0.15,
            # Try-3: gate to 0 near zero command, so it can't chase terrain
            # height while genuinely standing still.
            "command_name": "base_velocity",
            "min_cmd_norm": 0.1,
        },
    )

    # Try-4: same command gate as base_height_climb -- wild_foot_clearance's
    # swing detection is a pure open-loop clock (elapsed time only), so
    # without this it kept paying for a marching gait even at rest.
    wild_foot_clearance = RewardsCfgPhase3Balance().wild_foot_clearance.replace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "lookahead_distance": 0.15,
            "natural_clearance": 0.03,
            "max_clearance": 0.23,
            "roughness_ref": 0.05,
            "command_name": "base_velocity",
            "min_cmd_norm": 0.1,
        },
    )

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

    # Try-5/Try-6/Try-7: positive reward for literal stillness (all feet
    # planted + low joint velocity), gated to only the |command| <= 0.1 AND
    # terrain-roughness <= roughness_ref regime -- structurally incapable of
    # firing on/near a stair, so it can't compete with stall_penalty/
    # stair_commit's "don't freeze on the stairs" pressure. Weight 0.5 (Try-7)
    # after Try-6's 0.15 proved too weak once gated down to the rare
    # standing-on-flat-cell subset.
    quiet_standing = RewTerm(
        func=mdp.quiet_standing_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "command_name": "base_velocity",
            "max_cmd_norm": 0.1,
            "joint_vel_std": 1.0,
            "roughness_ref": 0.05,
        },
    )


@configclass
class TerminationsCfgPhase3BalanceFloating(TerminationsCfg):
    # Floating stairs require more sustained body pitch than solid stairs.
    bad_orientation = TerminationsCfg().bad_orientation.replace(params={"limit_angle": 1.0})


@configclass
class RobotEnvCfgPhase3BalanceFloating(RobotEnvCfgPhase3Balance):
    """Phase 3 - balance + floating inverted stairs terrain."""

    scene: RobotSceneCfgPhase3BalanceFloating = RobotSceneCfgPhase3BalanceFloating(
        num_envs=4096, env_spacing=2.5
    )
    commands: CommandsCfgPhase3BalanceFloating = CommandsCfgPhase3BalanceFloating()
    rewards: RewardsCfgPhase3BalanceFloating = RewardsCfgPhase3BalanceFloating()
    terminations: TerminationsCfgPhase3BalanceFloating = TerminationsCfgPhase3BalanceFloating()


@configclass
class RobotPlayEnvCfgPhase3BalanceFloating(RobotEnvCfgPhase3BalanceFloating):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
