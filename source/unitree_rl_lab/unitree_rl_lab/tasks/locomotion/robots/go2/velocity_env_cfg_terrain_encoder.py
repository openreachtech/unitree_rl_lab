"""Environments for training the terrain encoder behind a frozen blind policy.

The walking policy here is finished and never learns again. It exists to drive the
robot across the terrain so the LiDAR fan sees something worth reconstructing, and
the encoder (``assets/models/terrain_encoder.py``) learns to turn the noisy grid
the fan produces into the terrain that is really there.

One environment per walking phase, each paired with the policy trained on that
phase's terrain. That pairing is the point: a policy asked to cross ground it was
never trained on falls, and every fall resets the encoder's recurrent state, which
is exactly the signal the encoder is trying to learn to use. Phase 1 is skipped --
it is flat ground, and there is nothing there to reconstruct.

    Phase 2   rough / boxes      go2_blind_gru_phase2 checkpoint
    Phase 3   stairs             go2_blind_gru_phase3 checkpoint
    Phase 4   thin walls         go2_blind_gru_phase4 checkpoint

Each inherits the config its ``Go2-Blind-GRU-PhaseN`` task registers, not the
similarly named base class -- Phase 3's task resolves to the "balance matched"
variant, whose terrain differs from ``RobotEnvCfgPhase3``'s, and the encoder has
to see the ground its driver was actually trained on.

The encoder carries its weights across phases the way the policy did.

Each config adds two things to the phase's ordinary training environment: the LiDAR
fan and its grid, and the clean top-down scan that supplies the targets. The scan
raycasts from 20 m up against static meshes only, so the robot is transparent to it
and the terrain *under the body* comes back correctly -- which matters, because
those cells are the ones the fan can never reach.

Noise is on at full strength from the first iteration. Miki et al. ramp theirs
because their student is learning to walk at the same time and a broken map early
on costs them the gait; here the gait is already finished and frozen, so there is
nothing to protect.
"""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import torch

from unitree_rl_lab.assets.models.terrain_encoder import TerrainEncoder
from unitree_rl_lab.assets.models.terrain_encoder_belief import BeliefTerrainEncoder
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.observations import _height_scan_indices
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    GO2_HEIGHT_SCAN_OFFSET,
    GO2_HEIGHT_SCAN_CENTER_X,
    GO2_HEIGHT_SCAN_CENTER_Y,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    GO2_LIDAR_BODY_HALF_EXTENT_X,
    GO2_LIDAR_BODY_HALF_EXTENT_Y,
    GO2_LIDAR_SCANNER_CFG,
    LidarMapObsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    apply_lidar_view,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase2 import (
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase3 import (
    PLAY_TERRAIN_CFG_PHASE3,
    RobotEnvCfgPhase3BalanceMatched,
    RobotSceneCfgPhase3Balance,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase4 import (
    PLAY_TERRAIN_CFG_PHASE4,
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    ObservationsCfgGo2,
)


# The blind phases leave RobotSceneCfg's height_scanner at its 10 cm default -- 17 x 11
# = 187 rays -- because nothing in that lineage reads a fine grid: the policy is blind
# and the critic was cut back to foot-local terms. The encoder's target has to line up
# with the LiDAR grid instead, cell for cell at 5 cm over 29 x 21.
#
# A second raycaster rather than a re-resolved height_scanner: several reward terms
# still index that sensor, and although rewards go unused while the policy is frozen,
# silently changing what they read is the kind of coupling that surfaces later as an
# unexplained difference between this environment and the phase it was copied from.
# 609 extra downward rays is cheap next to the fan's 1080.
GO2_TERRAIN_TARGET_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    # "yaw", like height_scanner: the grid stays gravity-aligned and keeps its cell
    # spacing as the body pitches, so a target cell means the same patch of ground
    # whatever the robot is doing.
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(
        resolution=HEIGHT_SCAN_RESOLUTION,
        size=list(HEIGHT_SCAN_SIZE),
        # idx = ix * num_y + iy, matching _height_scan_indices and the LiDAR term.
        ordering="yx",
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


@configclass
class TerrainTargetCfg(ObsGroup):
    """The reconstruction target: the true height grid, all 609 cells.

    Uncropped on purpose. The policy's grid drops the 221 cells under the body
    because no beam reaches them, but those are precisely the cells whose only
    possible source is the encoder's memory, so they are the most informative part
    of the target. Excluding them would discard the one measurement that separates
    a recurrent estimator from a single-frame filter.

    ``offset`` has to be the one the LiDAR term uses, not ``height_scan``'s 0.5
    default. Both compute ``base_z - hit_z - offset`` from the same base link, so
    matching offsets is what puts the target and the input on one zero -- level
    ground reads 0.047 in each. Leave the default in and the encoder would spend
    its capacity learning a 0.23 m constant.

    Clipped like every other height term so a robot that has fallen off a ledge
    cannot hand the regression a five-metre target.
    """

    height = ObsTerm(
        func=mdp.height_scan,
        params={
            "sensor_cfg": SceneEntityCfg("terrain_target_scanner"),
            "offset": GO2_HEIGHT_SCAN_OFFSET,
        },
        clip=(-1.0, 5.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TerrainEncoderObsCfg(ObservationsCfgGo2):
    """Blind policy and privileged critic untouched, plus the encoder's two grids.

    ``policy`` is the 45-dim proprioception the frozen network reads, and the
    encoder reads the same vector -- deliberately the same one, so nothing enters
    the encoder that the robot could not also hand the controller.
    """

    lidar_map: LidarMapObsCfg = LidarMapObsCfg()
    terrain_target: TerrainTargetCfg = TerrainTargetCfg()


def _finish(env_cfg) -> None:
    """Shared tail of every phase's ``__post_init__``.

    Two curricula come down from the phase's training config, and both of them are
    wrong here. They exist to protect a policy that is still learning to walk; this
    policy finished learning weeks ago, and every step they spend easing it in is a
    step the encoder spends looking at terrain it has already mastered.

    *Command velocity.* ``CommandsCfg`` opens at +-0.1 m/s and ``lin_vel_cmd_levels``
    widens it by 0.1 once per episode, so reaching the +-1.0 m/s limit takes about
    375 iterations of the robot creeping. Play configs already sidestep this by
    assigning ``limit_ranges`` outright; the same applies here, for the same reason.

    *Terrain level.* Environments spawn uniformly in rows ``0 .. max_init_terrain_level``
    -- 2 of 10 in Phase 2, 5 of 10 in Phases 3 and 4 -- and ``terrain_levels_vel``
    ratchets them up one step per episode. ``None`` spreads them over every row from
    the first step instead.

    Dropping the ``terrain_levels`` term also has a side effect worth naming, because
    it has bitten this codebase before: ``RobotEnvCfg.__post_init__`` reads that term
    to decide whether the generator lays its patches out by ascending difficulty or at
    random. Without the flag set back afterwards, "row" would stop meaning "difficulty"
    and the spread would be over nothing in particular.

    Frozen levels rather than a running curriculum because the encoder wants coverage,
    not competence. A ratchet settles wherever the policy's ability runs out -- around
    level 5 of 10 on these courses -- and the tallest walls would go unobserved. The
    encoder's job is to see a 25 cm wall correctly, not to get over it.
    """
    env_cfg.scene.lidar_scanner.update_period = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene.terrain_target_scanner.update_period = env_cfg.decimation * env_cfg.sim.dt
    # The fan-built grid is data here, not a picture; markers would cost frames for
    # nothing. ``play.py`` on the phase's own task is where to look at it.
    env_cfg.scene.height_scanner.debug_vis = False

    # Full command range immediately. Copied rather than aliased to limit_ranges, so a
    # later curriculum term cannot widen the limit it is clamping against.
    env_cfg.commands.base_velocity.ranges = copy.deepcopy(
        env_cfg.commands.base_velocity.limit_ranges
    )
    env_cfg.curriculum.lin_vel_cmd_levels = None

    # Every difficulty, held fixed, with the generator still ordering rows by it.
    env_cfg.curriculum.terrain_levels = None
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = None


@configclass
class RobotSceneCfgEncoderPhase2(RobotSceneCfgPhase2):
    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG
    terrain_target_scanner: RayCasterCfg = GO2_TERRAIN_TARGET_SCANNER_CFG


@configclass
class RobotEnvCfgEncoderPhase2(RobotEnvCfgPhase2):
    scene: RobotSceneCfgEncoderPhase2 = RobotSceneCfgEncoderPhase2(num_envs=4096, env_spacing=2.5)
    observations: TerrainEncoderObsCfg = TerrainEncoderObsCfg()

    def __post_init__(self):
        super().__post_init__()
        _finish(self)


@configclass
class RobotSceneCfgEncoderPhase3(RobotSceneCfgPhase3Balance):
    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG
    terrain_target_scanner: RayCasterCfg = GO2_TERRAIN_TARGET_SCANNER_CFG


@configclass
class RobotEnvCfgEncoderPhase3(RobotEnvCfgPhase3BalanceMatched):
    scene: RobotSceneCfgEncoderPhase3 = RobotSceneCfgEncoderPhase3(num_envs=4096, env_spacing=2.5)
    observations: TerrainEncoderObsCfg = TerrainEncoderObsCfg()

    def __post_init__(self):
        super().__post_init__()
        _finish(self)


@configclass
class RobotSceneCfgEncoderPhase4(RobotSceneCfgPhase4):
    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG
    terrain_target_scanner: RayCasterCfg = GO2_TERRAIN_TARGET_SCANNER_CFG


@configclass
class RobotEnvCfgEncoderPhase4(RobotEnvCfgPhase4):
    scene: RobotSceneCfgEncoderPhase4 = RobotSceneCfgEncoderPhase4(num_envs=4096, env_spacing=2.5)
    observations: TerrainEncoderObsCfg = TerrainEncoderObsCfg()

    def __post_init__(self):
        super().__post_init__()
        _finish(self)


# ===========================================================================
# Play. Small, watchable, and pointed at Phase 2's ground: 2 columns (rough | boxes)
# by 5 rows of ascending difficulty, one difficulty band per row.
#
# What goes on screen is the fan's grid in green and red -- green where a beam landed
# this step, red where the cell is holding an older reading -- and, drawn by
# ``scripts/rsl_rl/play_terrain_encoder.py``, the encoder's own estimate in blue. The
# true terrain is deliberately absent: the question this view answers is whether the
# blue follows the ground where the red says the sensor is blind, and a third colour
# lying underneath both would only make that harder to see.
#
#   python scripts/rsl_rl/play_terrain_encoder.py \
#       --policy_checkpoint logs/rsl_rl/go2_blind_gru_phase2/<run>/model_6497.pt \
#       --encoder_checkpoint logs/terrain_encoder/<run>/encoder_1000.pt
# ===========================================================================
@configclass
class RobotPlayEnvCfgEncoderPhase2(RobotEnvCfgEncoderPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 2
        # _finish already frees the spawn over every row (max_init_terrain_level=None)
        # and holds the levels there, so all five difficulties are on screen at once
        # and stay put while being watched.
        apply_lidar_view(self)


@configclass
class RobotPlayEnvCfgEncoderPhase3(RobotEnvCfgEncoderPhase3):
    """Phase 3's own viewing course: 2 columns (pyramid | inverted) x 3 step heights.

    The training terrain is 20 columns of five stair variants, which is the right mix to
    learn on and an unreadable thing to look at. This is the cut-down version the phase
    already keeps for play -- step heights centred on 5 / 10 / 15 cm.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        # ``.replace(curriculum=True)`` and not a plain assignment, for two reasons.
        # ``_finish`` sets the flag on the generator that was there *before* this line,
        # so a bare swap silently lands a generator whose ``curriculum`` is still at its
        # ``False`` default -- and that mode samples difficulty uniformly per tile
        # instead of taking it from the row, which turns the rows into noise. And
        # ``replace`` returns a new object rather than mutating the module-level config
        # the blind policy's own play task shares.
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG_PHASE3.replace(curriculum=True)
        apply_lidar_view(self)


# The blind policy's Phase 4 play course runs 10 / 15 / 20 / 25 cm, sized to bracket
# where that policy starts failing. The encoder is being judged on what it can see rather
# than on what the robot can climb, so this one starts a band lower and keeps the whole
# range crossable -- a wall the robot never gets over is a wall it only ever views from
# one side.
#
# Rows are difficulty bands, not exact heights: TerrainGenerator derives difficulty as
# (row + jitter) / num_rows with the jitter uniform on [0, 1), so the range is chosen to
# put the band *centres* where they are wanted. (0.025, 0.225) over 4 rows gives
# 2.5-7.5, 7.5-12.5, 12.5-17.5 and 17.5-22.5 cm.
# proportion has to be equalised as well. Columns are handed out by cumulative
# proportion -- column c takes the first type whose running total exceeds c/num_cols --
# so the training mix's 2:1 over two columns puts the solid wall in both and the floating
# tread in neither. Phase 3's play config equalises for the same reason; this one did not,
# and the floating column simply was not there.
PLAY_TERRAIN_CFG_ENCODER_PHASE4 = PLAY_TERRAIN_CFG_PHASE4.replace(
    sub_terrains={
        name: cfg.replace(proportion=1.0, wall_height_range=(0.025, 0.225))
        for name, cfg in PLAY_TERRAIN_CFG_PHASE4.sub_terrains.items()
    },
)
"""2 x 4: solid wall | floating tread, rows centred on 5 / 10 / 15 / 20 cm, 5 cm thick."""


@configclass
class RobotPlayEnvCfgEncoderPhase4(RobotEnvCfgEncoderPhase4):
    """Phase 4's viewing course: solid wall | floating wall, 4 heights, 5 cm thick.

    Wall heights centre on 5 / 10 / 15 / 20 cm. The floating column is the interesting
    one to watch here: the tread hangs with a gap beneath it, so the fan sees a surface
    where the ground is not, and what the estimate does underneath it is the question.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        # See the note on Phase 3: the swap has to carry curriculum=True with it.
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG_ENCODER_PHASE4.replace(
            curriculum=True
        )
        apply_lidar_view(self)


def build_terrain_encoder(
    device: torch.device,
    extero_channels: int = 16,
    hidden_channels: int = 16,
    arch: str = "belief",
    belief_latent: int = 96,
):
    """An encoder whose grid matches the LiDAR term's, cell for cell.

    ``arch`` picks the encoder. ``belief`` -- Miki et al.'s flat recurrence -- is the
    default: it matched the convolutional one on accuracy and trains 3.8x faster in a
    17.8th of the memory. ``convgru`` is kept because it does win on the wall terrain,
    which is the one that resembles the deployment site. Both take the same grid, return
    the same ``(mean, log_std, hidden)``, and differ only in the network between. The
    channel arguments apply to ``convgru`` only; ``belief`` takes its widths from
    ``config.yaml``, apart from ``belief_latent``. See ``sandbox/TERRAIN_ENCODER.md``.

    Lives here rather than in either script so the trainer and the viewer construct the
    same thing, and here rather than beside the model so ``assets.models`` keeps no
    dependency on the task configs.

    The crop has to be the LiDAR one (0.40 / 0.30), not the top-down tasks' -- the fan's
    blind cone is wider, so its grid keeps 388 cells where theirs keeps 492. Taking the
    indices from the same helper the observation term uses is what holds the flat
    observation vector and the 29 x 21 image in register.
    """
    keep_index, _ = _height_scan_indices(
        resolution=HEIGHT_SCAN_RESOLUTION,
        size_x=HEIGHT_SCAN_SIZE[0],
        size_y=HEIGHT_SCAN_SIZE[1],
        scanner_offset_x=GO2_HEIGHT_SCAN_CENTER_X,
        scanner_offset_y=GO2_HEIGHT_SCAN_CENTER_Y,
        exclude_half_extent_x=GO2_LIDAR_BODY_HALF_EXTENT_X,
        exclude_half_extent_y=GO2_LIDAR_BODY_HALF_EXTENT_Y,
        device=device,
    )
    num_x = round(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1
    num_y = round(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1
    # Level ground on zero, so the scaled input is signed terrain deviation.
    common = dict(grid_shape=(num_x, num_y), proprio_dim=45, height_offset=0.0, keep_index=keep_index)
    if arch == "convgru":
        return TerrainEncoder(
            extero_channels=extero_channels, hidden_channels=hidden_channels, **common
        ).to(device)
    if arch == "belief":
        return BeliefTerrainEncoder(
            extero_dim=int(keep_index.numel()), extero_latent=belief_latent, **common
        ).to(device)
    raise ValueError(f"arch must be 'convgru' or 'belief', got {arch!r}")
