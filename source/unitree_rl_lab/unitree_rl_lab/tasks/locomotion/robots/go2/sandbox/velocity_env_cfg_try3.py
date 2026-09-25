"""Sandbox Try-3: narrow the trained stair tread depth to match the real/mujoco target.

Always resume with:
  --load_run 2026-07-04_05-26-55 --checkpoint model_6998.pt

Strategy: identical to Try-1 (same reward weights, same Phase2 origin,
demote_fraction=0.5, heading_command=True), except the terrain generator's
stair ``step_width`` (tread depth, along the walking direction) is changed
from the fixed 0.3 to a fixed 0.2 for both ``pyramid_stairs`` and
``pyramid_stairs_inv``. ``step_height_range=(0.05, 0.23)`` is left untouched
(already randomized).

Rationale (grounded in direct mujoco testing feedback, 2026-07-07): after
Try-1, the user tested the exported policy in mujoco. General stair climbing
was much smoother than before (heading fix validated), but the height=20cm/
width=20cm staircase specifically still failed -- the robot gets caught/stuck
shortly after starting to climb, unable to move. This exactly reproduces the
original bug report that kicked off this whole investigation (colleague's
robot froze front-foot-first on 20cm/20cm stairs). The training terrain's
``step_width`` has always been a FIXED 0.3, never randomized or varied, unlike
step_height. A 20cm tread depth is therefore entirely out-of-distribution for
the policy's foot-placement timing, regardless of any reward tuning -- this
looks like a distribution-coverage gap, not a reward-shaping problem, so no
amount of reward-weight tuning was ever going to fix it.

This trial deliberately does NOT randomize step_width yet (a sensible next
step, but the allocation/proportion design needs more thought per user
request). It first tests the simpler question: does training at a single
narrower fixed width (0.2, matching the real/mujoco target) work, and if so,
does the resulting policy still generalize back up to the *wider* 0.3m tread
it no longer trains on? If narrow-only training also loses the wider case,
that's itself informative for how the eventual randomized range should be
chosen.

Also unresolved from the same mujoco session: lateral (lin_vel_y) movement
appears broken even on flat ground, not just stairs -- a separate, more
general regression not addressed by this trial (see Try-2, interrupted, and
consider revisiting rel_heading_envs as a targeted fix for that specifically).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.go2_curriculum import PLAY_VEL_RANGES
from unitree_rl_lab.tasks.locomotion.robots.go2.sandbox.velocity_env_cfg_try1 import (
    RewardsCfgGo2Try1,
    SandboxPPORunnerCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotSceneCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RobotEnvCfgGo2

__all__ = ["RobotEnvCfgGo2Try3", "RobotPlayEnvCfgGo2Try3", "SandboxPPORunnerCfg"]

# Identical to GO2_CURRICULUM_TERRAIN_CFG (velocity_env_cfg_go2.py) except
# step_width 0.3 -> 0.2 on both stair sub-terrains.
GO2_CURRICULUM_TERRAIN_CFG_TRY3 = terrain_gen.TerrainGeneratorCfg(
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
            step_width=0.2,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.2,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgGo2Try3(RobotSceneCfg):
    terrain = RobotSceneCfg().terrain.replace(
        terrain_generator=GO2_CURRICULUM_TERRAIN_CFG_TRY3,
        max_init_terrain_level=0,
    )


@configclass
class RobotEnvCfgGo2Try3(RobotEnvCfgGo2):
    rewards: RewardsCfgGo2Try1 = RewardsCfgGo2Try1()
    scene: RobotSceneCfgGo2Try3 = RobotSceneCfgGo2Try3(num_envs=4096, env_spacing=2.5)


@configclass
class RobotPlayEnvCfgGo2Try3(RobotEnvCfgGo2Try3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = PLAY_VEL_RANGES
