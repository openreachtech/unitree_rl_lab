"""Perceptive Phase 5: one wall, and a goal on the far side of it.

Phase 4 asks the policy to cross walls but never tells it to. Its command is a random
velocity resampled every few seconds, so most of an episode is not a crossing attempt at
all, and its ``terrain_levels`` ratchet promotes only after 4 m of net displacement --
which on Phase 4's four concentric rings means clearing every one of them and walking to
the tile edge. Measured against a pinned wall, that ratchet understates the blind policy
by more than a factor of two: it parks at level 2.50 (11 cm by the terrain's own lerp)
while actually clearing a 20 cm wall in 99.6 % of attempts.

This phase changes three things and keeps everything else:

  1. **One wall instead of four** (``wall_spacing`` 0.60 -> 3.00). Crossing becomes a
     single unambiguous event, which is what both the goal and the ratchet need.
  2. **A goal beyond the wall** (``MixedGoalVelocityCommand``): one per episode, steered
     toward every step, command dropping to zero on arrival. Every episode is now an
     attempt.
  3. **A ratchet that can actually fire** (``terrain_levels_climb_demote_on_fail``),
     with ``promote_distance`` set from this terrain's geometry rather than half a tile.

Measured against pinned walls, the first version of this phase moved the wall the policy
can clear from 20 cm to 30 cm: the Phase 4 checkpoints cross a 25 cm wall in 0 % of
attempts, this phase's in 100 %, and a 30 cm wall -- which it had never seen -- in 88 %.
Wall height and thickness are ``PHASE5_WALL_HEIGHT_RANGE``/``..._THICKNESS_RANGE``,
raised from Phase 4's once that was measured.

``base_contact`` is relaxed from Phase 4's, and only in its direction test -- see
``TerminationsCfgPhase5``. Phase 4's flat 1 N rule was ending the episodes in which the
robot puts its weight on the wall's top edge, which is how it gets over a 30 cm wall
rather than a failure to avoid one.

The relaxed termination and ``wall_body_height`` were each trained and measured
separately before being folded together here; see ``RewardsCfgPhase5`` for the numbers.
Neither works alone -- the reward asks the robot to put its body on the wall, and the
strict rule reads that as a crash -- which is why the first comparison ranked them
backwards.

    python scripts/rsl_rl/train.py --task Go2-Perceptive-Mid360-Phase5 --headless \\
        --max_iterations 2000 --resume --previous-task Go2-Perceptive-Mid360-Phase2
"""

from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CurriculumCfg,
    TerminationsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase2 import (
    CommandsCfgPhase2,
    RobotSceneCfgPhase2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase4 import (
    RewardsCfgPhase4,
    RobotEnvCfgPhase4,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_mid360 import (
    _attach_perceptive,
)

# ---------------------------------------------------------------------------
# Terrain. Phase 4's tile and Phase 4's wall ranges, with the ring spacing opened up
# until only one ring fits:
#
#     num_walls = (size - 2*border_width - platform_width) // (2*wall_spacing) + 1
#               = (8.0 - 2.0 - 2.0) // (2*3.0) + 1 = 1
#
# and that ring sits at (size - 2*border_width)/2 - 0.5*wall_spacing = 1.50 m from
# spawn. The column split is Phase 4's 2:1, so two thirds of the grid is the solid wall
# and one third the floating tread, for the reason Phase 4 gives: this lineage's Phase 3
# is plain stairs, so the floating tread is the newer of the two and is dialled back.
# ---------------------------------------------------------------------------
PHASE5_TILE_SIZE = 8.0
PHASE5_TERRAIN_BORDER = 1.0
PHASE5_WALL_SPACING = 3.0
PHASE5_PLATFORM_WIDTH = 2.0

PHASE5_WALL_DISTANCE = (PHASE5_TILE_SIZE - 2 * PHASE5_TERRAIN_BORDER) / 2 - 0.5 * PHASE5_WALL_SPACING
"""1.50 m: the wall ring's distance from spawn, on the axes."""

PHASE5_WALL_CORNER = PHASE5_WALL_DISTANCE * math.sqrt(2)
"""2.12 m: the ring is a *square*, so its corner is this far out. A robot heading
diagonally reaches this radius while still inside -- see PHASE5_PROMOTE_DISTANCE."""

PHASE5_WALL_HEIGHT_RANGE = (0.10, 0.40)
"""Solid wall. Raised twice: from Phase 4's (0.05, 0.25) once the goal-directed policy
measured 100 % at 25 cm, then to 40 cm once it measured 98 % at 30 cm as well. The floor
came up with it -- 15 cm was crossed 100 % of the time on both wall types, so the bottom
rows had stopped teaching anything."""

PHASE5_FLOATING_HEIGHT_RANGE = (0.10, 0.30)
"""Floating wall, capped 10 cm below the solid one, because a tall enough floating tread
stops being a wall to climb and becomes a bar to walk under. The gap beneath it is
``height - tread_thickness`` = ``height - 0.04``, against a trunk whose underside sits at
about 26.5 cm at nominal stance:

    30 cm wall -> 26 cm gap, does not fit
    35 cm wall -> 31 cm gap, walks under standing

Ducking would still clear ``promote_distance``, so the curriculum would promote a policy
that never learned to climb -- the same "terrain_levels says yes, the robot says no"
failure this phase was built to get away from. ``wall_body_height_reward`` reads each
column's own range off the terrain, so the two caps do not have to be reconciled by
hand."""

PHASE5_WALL_THICKNESS_RANGE = (0.15, 0.05)
"""Unchanged. Height and thickness still move together, so the hardest row is both the
tallest and the thinnest."""

PHASE5_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(PHASE5_TILE_SIZE, PHASE5_TILE_SIZE),
    border_width=20.0,
    # 3 columns at 2:1 -- solid wall in columns 0-1, floating in column 2. Rows stay at
    # 10 so a terrain level maps to the same wall as it does in Phase 4.
    num_cols=3,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=2.0,
            wall_height_range=PHASE5_WALL_HEIGHT_RANGE,
            wall_thickness_range=PHASE5_WALL_THICKNESS_RANGE,
            wall_spacing=PHASE5_WALL_SPACING,
            platform_width=PHASE5_PLATFORM_WIDTH,
            border_width=PHASE5_TERRAIN_BORDER,
        ),
        # The same wall hollowed out: a tread hovering at wall_height with an open gap
        # underneath. Capped lower than the solid wall so the gap never opens up enough to
        # walk under -- see PHASE5_FLOATING_HEIGHT_RANGE.
        "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=PHASE5_FLOATING_HEIGHT_RANGE,
            wall_thickness_range=PHASE5_WALL_THICKNESS_RANGE,
            wall_spacing=PHASE5_WALL_SPACING,
            platform_width=PHASE5_PLATFORM_WIDTH,
            border_width=PHASE5_TERRAIN_BORDER,
        ),
    },
)

# ---------------------------------------------------------------------------
# Goal placement and the promotion rim, which are coupled from both sides.
#
# Lower bound: the ring is a square, so a radial promotion test has to clear its
# *corner*, not its face. The Go2W campaign this is ported from checks only the face
# (promote 1.60 m against a 1.25 m ring whose corner is at 1.77 m), which leaves a robot
# free to walk diagonally to the promotion radius without touching anything. The assert
# below closes that.
#
# Upper bound: the goal command zeroes at ``arrival_radius``, so a robot that correctly
# arrives at the nearest allowed goal stops at ``min(goal_radius) - arrival_radius``. Set
# the rim beyond that and a robot doing exactly as it is told never promotes.
# ---------------------------------------------------------------------------
PHASE5_GOAL_RADIUS_RANGE = (2.8, 3.0)
PHASE5_ARRIVAL_RADIUS = 0.4
PHASE5_PROMOTE_DISTANCE = 2.30

assert PHASE5_WALL_CORNER < PHASE5_PROMOTE_DISTANCE <= (
    PHASE5_GOAL_RADIUS_RANGE[0] - PHASE5_ARRIVAL_RADIUS
), "promote_distance must lie in (square-ring corner, min_goal_radius - arrival_radius]"


@configclass
class RobotSceneCfgPhase5(RobotSceneCfgPhase2):
    terrain = RobotSceneCfgPhase2().terrain.replace(
        terrain_generator=PHASE5_TERRAIN_CFG,
        max_init_terrain_level=2,
    )


@configclass
class CommandsCfgPhase5(CommandsCfgPhase2):
    """One goal per episode, placed beyond the wall, with the command zeroing on arrival.

    ``rough_terrain_names`` is empty because this terrain has no non-wall column: Phase 4
    trains on walls only and this phase keeps that mix, so every env is goal-directed.
    Two consequences worth knowing, both accepted:

      * ``lin_vel_y`` is never commanded again -- the goal branch synthesises forward
        speed and yaw rate only. Whatever strafing Phase 1/2 taught is not exercised here.
      * ``cfg.ranges`` is read by nobody, which makes ``lin_vel_cmd_levels`` inert. It is
        switched off in the curriculum below rather than left to log a dead metric.
    """

    base_velocity = mdp.MixedGoalVelocityCommandCfg(
        asset_name=CommandsCfgPhase2().base_velocity.asset_name,
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.01,
        debug_vis=CommandsCfgPhase2().base_velocity.debug_vis,
        ranges=CommandsCfgPhase2().base_velocity.ranges,
        limit_ranges=CommandsCfgPhase2().base_velocity.limit_ranges,
        rough_terrain_names=(),
        goal_radius_range=PHASE5_GOAL_RADIUS_RANGE,
        arrival_radius=PHASE5_ARRIVAL_RADIUS,
        max_lin_vel=1.0,
        max_ang_vel=1.0,
        heading_control_stiffness=1.0,
    )


@configclass
class TerminationsCfgPhase5(TerminationsCfg):
    """Phase 4's terminations with one exemption on ``base_contact``.

    The stock rule fires at 1 N of trunk contact, on magnitude alone. Measured on a 30 cm
    wall with the rule disabled: 86 % of robots put the trunk on the wall's top edge, and
    the force at that first touch has a median *horizontal* component of 0.0 N -- pure
    vertical support. Those episodes were being ended as collisions. Disabling the rule
    raised the measured crossing rate from 87.9 % to 99.6 % (solid) and 60.9 % to 78.5 %
    (floating), which is the cost of treating "rest weight on the wall" as a crash.

    ``illegal_contact_excluding_top`` exempts the upward-dominant case and leaves the
    magnitude test in place for everything else, so a head-on hit -- whose horizontal
    component stays large -- still terminates. The threshold stays at Phase 4's 1 N:
    the directional exemption is the whole change, and a second loosened constant would
    make it impossible to say which one mattered.
    """

    base_contact = DoneTerm(
        func=mdp.illegal_contact_excluding_top,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfgPhase5(CurriculumCfg):
    """Swaps the stock ratchet for one this terrain can actually satisfy.

    ``terrain_levels_vel`` promotes past ``size/2`` = 4 m and demotes below
    ``|cmd| * episode_s * 0.5``, which for any command over 0.4 m/s exceeds the promotion
    threshold outright -- so an env that does not promote is demoted, every reset. On
    Phase 4's four rings that made promotion nearly unreachable and demotion nearly
    certain, which is the shape of the terrain_levels collapse measured there (peak 4.32,
    final 2.50).

    ``fail_termination_names`` deliberately omits ``base_contact``. The default demotes on
    it, and with this lineage's 1 N trunk threshold the first attempts at lifting the body
    over a wall will graze it -- so demoting on that would punish the exact behaviour the
    phase is trying to produce, and the level would fall every time the policy tried to
    improve. A graze still ends the episode, as it does in Phase 4; it just no longer
    costs the env its row. ``bad_orientation`` still demotes: tipping over is a real
    failure at any stage.
    """

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_climb_demote_on_fail,
        params={
            "promote_distance": PHASE5_PROMOTE_DISTANCE,
            "fail_termination_names": ("bad_orientation",),
        },
    )
    lin_vel_cmd_levels = None


@configclass
class RewardsCfgPhase5(RewardsCfgPhase4):
    """Phase 4's rewards plus the goal-tracking set: ANYmal Parkour's Table S2 terms and
    Table S3's arrival term, ported from the Go2W Phase 5 campaign.

    Weights are that campaign's, which were tuned against its own reward set rather than
    this one -- ``goal_position_tracking`` at 10.0 sits next to this lineage's
    ``track_lin_vel_xy`` at 1.5. They are a starting point to be read off the first few
    hundred iterations' per-term contributions, not a transplant that is known to balance.

    ``goal_position_tracking``/``goal_heading_tracking`` only fire in a 1 s window at
    ``arrival_deadline_s``. The 8 s default comes from a wheeled robot; here the goal sits
    2.8-3.0 m out with a wall at 1.50 m, so 8 s asks for about 0.35 m/s of average
    progress. That should be comfortable, but it has not been confirmed on this gait, and
    a crossing that takes longer earns nothing from either term for the rest of the
    episode -- which is the failure ``goal_arrival`` exists to cover.

    ``wall_body_height`` is the only term here that reads the body's *height*; the five
    goal terms all measure horizontal progress, heading or speed, so without it a robot
    standing at the wall and a robot with its front feet on top of it score identically.
    It was trained and measured as a separate variant before being folded in. Against the
    same five goal terms alone, on a 30 cm wall at 5 cm thickness:

                        without          with
        solid wall        96.9 %       100.0 %
        floating wall     56.6 %        96.1 %

    The first attempt at that comparison ran under a flat 1 N ``base_contact`` and said
    the opposite (5.9 % with, 61.7 % without) -- the term rewards putting the body on the
    wall, which that rule read as a crash. See ``TerminationsCfgPhase5``; the two changes
    only work together.

    ``min_target_z`` is the nominal standing height. Without it the term asks the robot to
    *crouch* over the bottom of the curriculum: the target is ``wall_height + 0.15``,
    which is below a 32 cm stance for any wall under 17 cm -- levels 0-3 of 9 here. See
    the function's own docstring.

    ``gate_width_far`` is 1.5 m against 0.6 m on the near side, so the pull does not taper
    off the moment the front of the robot is past the wall -- the hindquarters still have
    to come over.
    """

    goal_move_in_direction = RewTerm(
        func=mdp.goal_move_in_direction_reward,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )
    goal_position_tracking = RewTerm(
        func=mdp.goal_position_tracking_reward,
        weight=10.0,
        params={"command_name": "base_velocity", "arrival_deadline_s": 8.0, "activation_window": 1.0},
    )
    goal_heading_tracking = RewTerm(
        func=mdp.goal_heading_tracking_reward,
        weight=5.0,
        params={"command_name": "base_velocity", "arrival_deadline_s": 8.0, "activation_window": 1.0},
    )
    goal_dont_wait = RewTerm(
        func=mdp.goal_dont_wait_penalty_3d,
        weight=-1.0,
        params={"command_name": "base_velocity", "speed_threshold": 0.2},
    )
    goal_arrival = RewTerm(
        func=mdp.goal_arrival_reward,
        weight=0.15,
        params={"command_name": "base_velocity"},
    )
    wall_body_height = RewTerm(
        func=mdp.wall_body_height_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            # wall_height_range left unset: the term reads each column's own range off
            # the terrain, which is the only way to serve a solid column capped at 40 cm
            # and a floating one capped at 30 cm from one term.
            "wall_distance": PHASE5_WALL_DISTANCE,
            "gate_width": 0.6,
            "gate_width_far": 1.5,
            "nominal_clearance": 0.15,
            "std": 0.15,
            "min_target_z": 0.32,  # GO2_NOMINAL_BASE_Z
        },
    )


@configclass
class _RobotEnvCfgPhase5(RobotEnvCfgPhase4):
    """Phase 4's MDP with this phase's terrain, command, curriculum and terminations."""

    scene: RobotSceneCfgPhase5 = RobotSceneCfgPhase5(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase5 = CommandsCfgPhase5()
    curriculum: CurriculumCfgPhase5 = CurriculumCfgPhase5()
    terminations: TerminationsCfgPhase5 = TerminationsCfgPhase5()


@configclass
class RobotEnvCfgPerceptiveMid360Phase5(_RobotEnvCfgPhase5):
    rewards: RewardsCfgPhase5 = RewardsCfgPhase5()

    def __post_init__(self):
        super().__post_init__()
        _attach_perceptive(self, debug_vis=False)


# ---------------------------------------------------------------------------
# Play: the same single ring at four pinned heights, one row each, ascending.
#
# The two columns are pinned to different bands, because they are capped differently in
# training: the solid wall runs 25 / 30 / 35 / 40 cm, the floating one 15 / 20 / 25 / 30.
# A row therefore does *not* show the same height on both sides -- it shows each column at
# the same fraction of its own range, which is what the curriculum means by a level.
#
# curriculum=True is set explicitly. RobotEnvCfg.__post_init__ sets that flag, but on
# whichever generator is attached when it runs -- a play config that swaps the generator
# afterwards gets TerrainGeneratorCfg's False default back, and difficulty is then
# sampled per tile with the rows meaning nothing.
#
# Rows are bands, not single heights: difficulty is (row + jitter)/num_rows with the
# jitter uniform on [0, 1), so an exact height cannot be requested. Over 4 rows the band
# centres sit at difficulty 0.125 / 0.375 / 0.625 / 0.875, and the ranges below put those
# centres on the heights above.
# ---------------------------------------------------------------------------
PLAY_TERRAIN_CFG_PHASE5 = PHASE5_TERRAIN_CFG.replace(
    num_rows=4,
    num_cols=2,
    curriculum=True,
    sub_terrains={
        # Equal proportions: columns are handed out by cumulative proportion, so the
        # training mix's 2:1 over two columns would put the solid wall in both and the
        # floating tread nowhere.
        "thin_wall": PHASE5_TERRAIN_CFG.sub_terrains["thin_wall"].replace(
            proportion=1.0,
            wall_height_range=(0.225, 0.425),  # centres on 25 / 30 / 35 / 40 cm
            wall_thickness_range=(0.05, 0.05),
        ),
        "floating_thin_wall": PHASE5_TERRAIN_CFG.sub_terrains["floating_thin_wall"].replace(
            proportion=1.0,
            wall_height_range=(0.125, 0.325),  # centres on 15 / 20 / 25 / 30 cm
            wall_thickness_range=(0.05, 0.05),
        ),
    },
)
"""2 x 4: solid wall 25 / 30 / 35 / 40 cm | floating tread 15 / 20 / 25 / 30 cm.

Thickness is pinned at 5 cm -- the training mix's thinnest -- so height is the only thing
that changes down a column, and every row is as hard as that height gets."""


def _play(train_cls):
    @configclass
    class _Cfg(train_cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 32
            self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG_PHASE5.copy()
            # Spread the spawn over all four rows; num_rows - 1.
            self.scene.terrain.max_init_terrain_level = 3
            _attach_perceptive(self, debug_vis=True)

    return _Cfg


RobotPlayEnvCfgPerceptiveMid360Phase5 = _play(RobotEnvCfgPerceptiveMid360Phase5)
