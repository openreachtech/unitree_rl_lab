"""Go2-Crouch-Phase4: Phase2's crouch-height tracking, now on rough ground and walls to
step over. Skips a stair phase entirely (there is no Go2-Crouch-Phase3) -- Phase2 already
proved crouched walking survives varied, uneven terrain (boxes); Phase4 asks the same
question of a genuinely different obstacle shape (thin free-standing walls), not more of
the same kind of unevenness stairs would add.

Builds on Phase2 (inherits its base_velocity limits and everything about base_height
unchanged) with:

- Terrain: a generated grid, 4 columns x 10 rows. Column 1 is random rough ground, columns
  2-4 are thin_wall (a 1:3 rough:wall column ratio, i.e. proportion 0.25/0.75) -- see
  PHASE4_TERRAIN_CFG. Walls are lower/thinner than
  Unitree-Go2-Velocity-v1-Phase4's own thin_wall (0.03m->0.20m height, fixed 0.05m
  thickness, vs. that phase's 0.05m->0.25m / 0.15m->0.03m) -- a deliberately easier
  obstacle for this task's first attempt at combining crouch-height tracking with an
  obstacle to step over, not a claim those values are wrong in general. Same wall_spacing
  (0.60m). Not Go2W Phase5's wheeled-tuned wall (much taller/thicker, suited to a
  different platform).
- ``undesired_contacts`` split in two, mirroring Go2W Phase5's Try30 fix (see its
  sandbox/SUMMARY.md): Head_/hip contact stays penalized everywhere
  (``undesired_contacts``, weight -0.3), but thigh/calf contact (``undesired_contacts_legs``)
  is exempted specifically on thin_wall columns -- pushing a leg against the wall to step
  over it is the intended, load-bearing contact this task exists to produce, not an
  undesired one. Unlike Go2W Phase5, this doesn't use ``MixedGoalVelocityCommand``'s
  ``rough_env_mask`` -- Go2-Crouch keeps the plain ``UniformLevelVelocityCommand``
  Phase1/2 already validated (porting the full goal-directed command design to legged Go2
  was tried once already, judged not working -- see sandbox/velocity_env_cfg_phase5_try1.py's
  deletion note in git history). Instead, ``undesired_contacts_terrain_column_aware`` reads
  the terrain column directly from ``env.scene.terrain``, so it works with any command type.
- ``terrain_levels_climb_demote_on_fail`` replaces ``terrain_levels_vel``: demotes on
  base_contact/bad_orientation regardless of distance travelled, not just on low net
  displacement -- avoids the "promoted past its actual ability, stuck failing forever"
  dead zone plain distance-based demotion leaves on obstacle terrain (Go2W
  sandbox/SUMMARY.md's Try26 record).

``base_height`` command/curriculum/reward are untouched from Phase1 -- same crouch task,
now just walking over walls instead of flat/rough/box ground.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_crouch_phase1 import (
    RewardsCfgCrouchPhase1,
    RobotSceneCfgCrouchPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_crouch_phase2 import (
    CommandsCfgCrouchPhase2,
    RobotEnvCfgCrouchPhase2,
)

PHASE4_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=4,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # Lower/thinner than Unitree-Go2-Velocity-v1-Phase4's own thin_wall: height scales
        # 0.03m (easy) -> 0.20m (hard) with difficulty, thickness fixed at 0.05m (not
        # difficulty-scaled), walls spaced a fixed 0.60m apart.
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.75,
            wall_height_range=(0.03, 0.20),
            wall_thickness_range=(0.05, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


@configclass
class RobotSceneCfgCrouchPhase4(RobotSceneCfgCrouchPhase1):
    terrain = RobotSceneCfgCrouchPhase1().terrain.replace(
        terrain_type="generator",
        terrain_generator=PHASE4_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class RewardsCfgCrouchPhase4(RewardsCfgCrouchPhase1):
    # Split from the inherited flat undesired_contacts (Head_.*/.*_hip/.*_thigh/.*_calf,
    # weight -1, penalized everywhere): Head/hip stays strict on every column, thigh/calf
    # gets a dedicated, terrain-column-aware term below instead. Weights match Go2W
    # Phase5's own Try30-derived split.
    undesired_contacts = RewardsCfgCrouchPhase1().undesired_contacts.replace(
        weight=-0.3,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip"])},
    )
    undesired_contacts_legs = RewTerm(
        func=mdp.undesired_contacts_terrain_column_aware,
        weight=-0.3,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "exempt_terrain_names": ("thin_wall",),
        },
    )


@configclass
class CurriculumCfgCrouchPhase4(CurriculumCfg):
    """terrain_levels_climb_demote_on_fail instead of the inherited terrain_levels_vel --
    see module docstring -- plus crouch_depth_levels for base_height."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_climb_demote_on_fail)
    crouch_depth_levels = CurrTerm(func=mdp.crouch_depth_levels)


@configclass
class RobotEnvCfgCrouchPhase4(RobotEnvCfgCrouchPhase2):
    """Phase 4: rough terrain and walls to step over."""

    scene: RobotSceneCfgCrouchPhase4 = RobotSceneCfgCrouchPhase4(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgCrouchPhase2 = CommandsCfgCrouchPhase2()
    rewards: RewardsCfgCrouchPhase4 = RewardsCfgCrouchPhase4()
    curriculum: CurriculumCfgCrouchPhase4 = CurriculumCfgCrouchPhase4()


@configclass
class RobotPlayEnvCfgCrouchPhase4(RobotEnvCfgCrouchPhase4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_height.ranges = self.commands.base_height.limit_ranges
