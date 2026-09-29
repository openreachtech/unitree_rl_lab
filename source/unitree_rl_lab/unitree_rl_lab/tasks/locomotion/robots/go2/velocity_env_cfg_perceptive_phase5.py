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

The terrain grew a third wall type on the way here. ``razor_wall`` is the same solid wall
thinned towards 1 cm, at the edge of what a 5 cm-cell map can resolve; it was added
because the two original columns left a hole at the *bottom*, not the top. The solid
column started at 21.5 cm and the floating one at 11 cm, so a low obstacle was something
the policy had never met: it crossed a 5.5 cm wall in 18.8 % of attempts while crossing a
14.5 cm one at 1 cm thick in 98.8 %. Thinness was never the difficulty; low height was.

Measured against pinned walls, 256 robots driven straight at one, each column on its own
height-and-thickness pairing (``model_4700``, 1000 iterations from Phase 2):

    solid       25      30      35      40      45      50   cm
                100 %   100 %   100 %   98.8 %  89.8 %  0.4 %
    floating    10      15      20      25      30
                100 %   100 %   100 %   98.0 %  99.2 %
    razor       20      25      30      35      40      45
                100 %   100 %   100 %   100 %   90.6 %  85.5 %

against Phase 4's checkpoints, which cross a 25 cm wall in 0 % of attempts.

Two results worth keeping. The razor column reaches 45 cm at 1 cm thick (85.5 %) against
the solid wall's 89.8 % at the same height, so thickness has stopped being most of the
difficulty -- and no crossing at any height passes the foot-inside-the-slab test, so these
are real steps rather than the solver letting a thin box through. And raising the razor
cap from 30 to 40 cm improved the columns it does not touch: solid 45 cm went 5.1 % ->
89.8 % and floating 30 cm went 73.4 % -> 99.2 % across that one change. Practising a wall
that cannot be seen until late appears to train the leg-lift itself. Single run, so it is
a lead rather than a finding.

``base_contact`` is relaxed from Phase 4's, and only in its direction test -- see
``TerminationsCfgPhase5``. Phase 4's flat 1 N rule was ending the episodes in which the
robot puts its weight on the wall's top edge, which is how it gets over the wall rather
than a failure to avoid one. That and ``wall_body_height`` were trained and measured
separately before being folded together; neither works alone, because the reward asks the
robot to put its body on the wall and the strict rule reads that as a crash.

One thing not to trust here: ``terrain_levels`` does not rank checkpoints, and the reason
is mechanical rather than statistical. ``terrain_levels_climb_demote_on_fail`` promotes
past ``promote_distance`` and demotes on ``distance < 0.5`` or a named failure
termination. The failure this task converges to when it goes wrong is *neither*: the
robot walks to about 1.1 m, stops in front of the wall, and times out. That clears the
0.5 m floor and is not a listed termination, so the env is neither promoted nor demoted
and parks on a row it cannot cross, for the rest of training.

The consequence is that the level can sit at its maximum while the policy crosses nothing.
One run ended at 6.96 on a checkpoint that cleared no wall at any height, while a
mid-run checkpoint from the same run cleared 50 cm 79 % of the time; another held 6.85 for
800 iterations straight through a collapse from 89.8 % to 0 % on a 40 cm wall. Nothing
else in the logs moves either -- reward, entropy, noise_std and termination rates are all
smooth across that cliff. Pick checkpoints by measured crossing rate, and measure several:
the useful one has repeatedly been a few hundred iterations before the end.

    python scripts/rsl_rl/train.py --task Go2-Perceptive-Mid360-Phase5 --headless \\
        --max_iterations 1000 --resume --previous-task Go2-Perceptive-Mid360-Phase2
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

PHASE5_WALL_HEIGHT_RANGE = (0.20, 0.45)
"""Solid wall. Raised each time the policy outgrew the last ceiling: Phase 4's
(0.05, 0.25) -> (0.10, 0.30) at 100 % on 25 cm -> (0.10, 0.40) at 98 % on 30 cm ->
(0.20, 0.50) after 94 % on 40 cm with no falls -- and then back down to 0.45.

50 cm turned out to be the wrong place to stop. It was crossed 79 % of the time, which
looked like a ceiling matched to the ability, but it is also the first height that
produces falls, and the razor run showed what that costs: 50 cm was the *first* thing
lost, and it went by the policy trying and falling (61.7 % falls at iteration 6300)
before it stopped approaching tall walls at all. A top row the policy can only reach by
failing is a top row that teaches it to stop trying. 45 cm keeps the hardest row inside
what the policy actually lands.

The floor stays at 20 cm. Ten rows have to stretch across the range: at a 10 cm floor the
bottom four rows are 12-24 cm, which this policy clears without noticing, and each
promotion is then a 4 cm jump where it matters. From 20 cm the step is 2.5 cm and every
row is doing work -- while row 0 is still 21.25 cm, crossed 100 % of the time, so an env
that demotes all the way down lands somewhere it can recover from."""

PHASE5_RAZOR_HEIGHT_RANGE = (0.05, 0.40)
PHASE5_RAZOR_THICKNESS_RANGE = (0.05, 0.01)
"""A wall too thin to reliably see. Started at (0.05, 0.15), where the height was
deliberately trivial so the only difficulty was perceiving the wall; the cap rose to 0.30
and then 0.40 as each was measured solved. Rows now run 6.8 cm at 4.8 cm thick to 38.2 cm
at 1.2 cm, so the column asks for both: notice a wall the map can barely resolve, *and*
clear a real height once it is noticed.

One thing to know before moving this cap again: thickness lerps across the same ten rows,
so stretching the height axis carries every thickness onto a taller wall rather than
simply adding a harder row on top. Going 30 -> 40 moved 1.2 cm from a 28.7 cm wall to a
38.2 cm one, and 28 cm went from 1.2 cm thick to 2.4 cm. The thin-at-moderate-height case
is replaced, not kept alongside, so a skill measured at the old cap has to be re-measured
rather than assumed.

The map bins into 5 cm cells, so a 1 cm wall occupies a fifth of the cell it falls in and
is only registered when a ray happens to land on that fifth. With the sensor putting about
175 returns into the grid's 609 cells each step, that works out at a 5.7 % chance per step
of the wall showing up in its own cell, against 29 % for a 5 cm one:

    thickness   seen in its cell, per step   95 % seen after
       5 cm              29 %                  0.18 s
       3 cm              17 %                  0.32 s
       1 cm               5.7 %                1.01 s

A wall 1.5 m out enters the grid's 0.7 m reach about 0.8 s before contact at 1 m/s, so at
1 cm the map is still deciding whether the wall exists when the robot arrives. That is the
point: the hold makes it stick once seen, and some approaches will not see it in time.

Row 0 stays at 6.8 cm and 4.8 cm thick, which is the low obstacle nothing else in the
terrain provides -- the gap that cost this lineage an 18.8 % crossing rate on a 5.5 cm
wall before this column existed."""

PHASE5_FLOATING_HEIGHT_RANGE = (0.05, 0.25)
"""Floating wall, capped well below the solid one, because a tall enough floating tread
stops being a wall to climb and becomes a bar to walk under. The gap beneath it is
``height - tread_thickness`` = ``height - 0.04``, against a trunk whose underside sits at
about 26.5 cm at nominal stance:

    25 cm wall -> 21 cm gap, does not fit, with margin
    30 cm wall -> 26 cm gap, does not fit, but only just
    35 cm wall -> 31 cm gap, walks under standing

Ducking would still clear ``promote_distance``, so the curriculum would promote a policy
that never learned to climb -- the same "terrain_levels says yes, the robot says no"
failure this phase was built to get away from. The cap was 30 cm, which is inside that
limit only by 0.5 cm, and that margin is measured against a *nominal* stance -- a robot
that crouches at all turns the hardest row of this column into a walk-under. 25 cm buys
5 cm of margin, and the height it gives up is covered by the razor column, which now
reaches 38.2 cm.

The floor deliberately does *not* rise with the solid column's. The solid one starts at
20 cm because ten rows have to reach 45 and anything below that is wasted on a policy
this far along. This column runs 5-25 cm at 2 cm a step, against 20-45 cm at 2.5 cm a
step there and 5-40 cm at 3.5 cm in the razor column, so between the three the curriculum
spans the whole range at a useful resolution everywhere -- and two of the three now start
low enough to keep the low-obstacle case in the mix, which is the gap that cost this
lineage a 18.8 % crossing rate on a 5.5 cm wall before the razor column existed.
``wall_body_height_reward`` reads each column's own range off the terrain, so the three
floors and three caps need no reconciling by hand."""

PHASE5_WALL_THICKNESS_RANGE = (0.15, 0.05)
"""Unchanged. Height and thickness still move together, so the hardest row is both the
tallest and the thinnest."""

PHASE5_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(PHASE5_TILE_SIZE, PHASE5_TILE_SIZE),
    border_width=20.0,
    # 6 columns at 1:1:1 -- solid wall in 0-1, floating in 2-3, razor in 4-5. Was 3:2:1,
    # which gave the razor column a sixth of the envs on the grounds that it was a
    # perception exercise rather than a locomotion one; it is now a full wall in its own
    # right (up to 38.2 cm) and gets an equal share.
    #
    # Columns are handed out by *cumulative* proportion (column i takes the first
    # sub-terrain whose running total passes i / num_cols), so the column count and the
    # proportions have to be chosen together: 4 columns at 1:1:1 would give the last
    # sub-terrain one column and the first two more. 6 divides by 3 exactly. The assert
    # below is what keeps that honest. Rows stay at 10.
    num_cols=6,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=1.0,
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
        # Same solid wall, thinned until the map can barely resolve it.
        "razor_wall": terrains.MeshThinWallTerrainCfg(
            proportion=1.0,
            wall_height_range=PHASE5_RAZOR_HEIGHT_RANGE,
            wall_thickness_range=PHASE5_RAZOR_THICKNESS_RANGE,
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

def _assigned_columns(cfg) -> dict[str, int]:
    """How many columns each sub-terrain actually gets, by TerrainGenerator's own rule.

    Column ``i`` takes the first sub-terrain whose running proportion total passes
    ``i / num_cols``, so a sub-terrain can be listed, given a proportion, and still be
    generated zero times -- silently. That has happened twice in this lineage: Phase 4's
    play course put the solid wall in both of its columns and the floating tread in
    neither, and 4 columns at the 1:1:1 used here would hand the razor wall one column
    while the other two take three between them.
    """
    names = list(cfg.sub_terrains)
    total = sum(c.proportion for c in cfg.sub_terrains.values())
    cumsum, running = [], 0.0
    for c in cfg.sub_terrains.values():
        running += c.proportion / total
        cumsum.append(running)
    counts = {n: 0 for n in names}
    for col in range(cfg.num_cols):
        idx = next(i for i, t in enumerate(cumsum) if col / cfg.num_cols + 0.001 < t)
        counts[names[idx]] += 1
    return counts


_PHASE5_COLUMNS = _assigned_columns(PHASE5_TERRAIN_CFG)
assert all(n > 0 for n in _PHASE5_COLUMNS.values()), (
    f"a sub-terrain is never generated: {_PHASE5_COLUMNS}"
)
assert len(set(_PHASE5_COLUMNS.values())) == 1, (
    f"the three wall types are meant to get equal shares, got {_PHASE5_COLUMNS}"
)


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

    Weights are that campaign's, tuned against its own reward set rather than this one --
    ``goal_position_tracking`` at 10.0 sits next to this lineage's ``track_lin_vel_xy`` at
    1.5. They were never retuned, and the phase reached its ceiling anyway, so they are
    left alone.

    What the per-term contributions actually say, though, is that the arrival half of this
    set does almost nothing here. Averaged over a run: ``goal_move_in_direction`` +0.38 and
    ``wall_body_height`` +0.24 carry it, while ``goal_arrival`` sits at ~0.00 from the
    first iteration to the last and ``Metrics/goal_distance`` never falls below ~3.0 m --
    the goal is placed 2.8-3.0 m out, so the robots are crossing the wall without going on
    to arrive. ``goal_position_tracking``/``goal_heading_tracking`` only fire in a 1 s
    window at ``arrival_deadline_s`` (8 s, from a wheeled robot), so they are mostly
    inert for the same reason.

    That is worth knowing before tuning any of them: what makes this phase work is being
    *pointed* at something past the wall and being paid to lift the body over it, not the
    arrival bonus. Dropping the arrival terms has not been tried -- they are cheap, and
    they are what stops ``wall_body_height`` paying to rear up against the wall forever.

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
            # the terrain, which is the only way to serve three columns capped at 45, 25
            # and 40 cm and floored at 20, 5 and 5 from one term.
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
# The three columns are pinned to different bands, because they are capped differently in
# training: the solid wall runs 30 / 35 / 40 / 45 cm, the floating tread 10 / 15 / 20 /
# 25, the razor wall 25 / 30 / 35 / 40 -- which is also the split the training terrain
# uses.
# A row therefore does *not* show the same height across the row -- it shows each column at
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
    num_cols=3,
    curriculum=True,
    sub_terrains={
        # One column each: three equal proportions over three columns, so the cumulative
        # rule hands out exactly one apiece. num_cols has to move with the number of wall
        # types -- at 2 columns the razor wall would simply not be generated.
        "thin_wall": PHASE5_TERRAIN_CFG.sub_terrains["thin_wall"].replace(
            proportion=1.0,
            wall_height_range=(0.275, 0.475),  # centres on 30 / 35 / 40 / 45 cm
            wall_thickness_range=(0.05, 0.05),
        ),
        "floating_thin_wall": PHASE5_TERRAIN_CFG.sub_terrains["floating_thin_wall"].replace(
            proportion=1.0,
            wall_height_range=(0.075, 0.275),  # centres on 10 / 15 / 20 / 25 cm
            wall_thickness_range=(0.05, 0.05),
        ),
        "razor_wall": PHASE5_TERRAIN_CFG.sub_terrains["razor_wall"].replace(
            proportion=1.0,
            wall_height_range=(0.225, 0.425),  # centres on 25 / 30 / 35 / 40 cm
            wall_thickness_range=(0.01, 0.01),
        ),
    },
)
"""3 x 4: solid 30 / 35 / 40 / 45 cm | floating tread 10 / 15 / 20 / 25 | razor 25 / 30 /
35 / 40 at 1 cm thick.

Thickness is pinned at each column's thinnest -- 5 cm for the first two, 1 cm for the
razor -- so height is the only thing that changes down a column, and every row is as hard
as that height gets."""


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
