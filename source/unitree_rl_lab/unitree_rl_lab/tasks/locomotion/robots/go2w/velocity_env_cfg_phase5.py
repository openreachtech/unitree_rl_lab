import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion import terrains
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import (
    CurriculumCfg,
    TerminationsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase2 import CommandsCfgPhase2
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase3 import CommandsCfgPhase3, RewardsCfgPhase3
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# =============================================================================
# Phase 5 -- extreme obstacle crossing.
#
# --- Terrain/command/reward redesign (2026-08-18) --------------------------------------
#
# Folded in from the Go2w-v1-Phase5-Try15 sandbox experiment (2026-08-17/18), itself a
# port of Go2w-v2-Teacher-Phase5-Try1 (deleted 2026-08-19 after folding; see sandbox/
# SUMMARY.md for the original reasoning) to this v1/GRU line. Confirmed in MuJoCo:
# controls correctly, no runaway under a zero command, crosses 0.40 m. This replaces the
# pyramid_stairs terrain, UniformTerrainGatedVelocityCommand, and climb_progress/
# motion_without_cmd-style reward set the two campaigns below (Try 1-14) had converged on
# -- those were never able to fully solve "stop when told to" on this task even with a
# GRU (a Student built on top of the TCN/V2 equivalent of that design was later found
# still driving off under a zero command in MuJoCo, prompting this whole redesign; see
# sandbox/SUMMARY.md's "2026-08-18: superseded by the thin_wall / goal-directed redesign"
# section for that investigation).
#
# What changed:
#   * Terrain: pyramid_stairs/pyramid_stairs_inv -> a single free-standing thin_wall ring
#     per tile (terrains.MeshThinWallTerrainCfg) on flat ground -- no pit, so Lesson 1
#     below (inverted-pyramid pit depth) no longer applies at all; matches the actual
#     MuJoCo test scene (a row of vertical walls) far more directly than a stepped
#     pyramid ever did.
#   * Command: UniformTerrainGatedVelocityCommand -> MixedGoalVelocityCommand -- "rough"
#     columns keep a full omnidirectional command (reusing CommandsCfgPhase2's own
#     ranges), "thin_wall" columns get a goal placed just beyond the wall with the
#     command dropping to exactly zero on arrival, giving the genuine "arrived -> stop"
#     exposure Lesson 4/5 (both below) were only ever able to approximate.
#   * Reward: forward_command_progress/forward_stall_penalty/climb_progress/
#     motion_without_cmd/wheel_motion_without_cmd all removed, replaced by direct ports
#     of ANYmal Parkour's (Hoeller/Rudin et al. 2023) Table S2 goal-tracking terms
#     (mdp.goal_move_in_direction_reward/goal_position_tracking_reward/
#     goal_heading_tracking_reward/goal_dont_wait_penalty) -- see those functions'
#     docstrings in mdp/rewards.py for the paper mapping and the departures from a
#     literal port.
#
# What's unchanged from the Try 1-14 campaigns and still load-bearing:
#   * TerminationsCfgPhase5/CurriculumCfgPhase5 below -- Try15 never touched either.
#   * Lesson 3 (relaxing terminations has diminishing returns; removing base_contact
#     outright backfires) -- re-confirmed independently in the sandbox again in 2026-08:
#     a later Try that disabled base_contact for training saw terrain_levels *regress*
#     rather than improve, the same failure mode Try4 hit originally.
#   * Lesson 6 (the policy network mattered more than any reward/command tweak; GRU
#     fixed a MuJoCo command-drift an MLP couldn't, at identical reward/terrain) -- this
#     phase still trains a GruPPORunnerCfg policy (go2w/__init__.py), not
#     BasePPORunnerCfg.
#
# --- What the first campaign established (Try 1-9, 2026-08-02..08-11) -- historical,
#     terrain-shape-specific lessons that no longer apply now that the terrain itself has
#     changed, kept for the general principles they illustrate -----------------------------
#
# 1. Terrain *shape* dominated everything else. Three separate runs sat at ~0.36 m no
#    matter what termination thresholds were relaxed. The cause was geometric, not
#    behavioural: in an inverted pyramid the robot spawns on the pit floor at
#    -(num_steps + 1) * step_height, and num_steps comes from *platform_width*, not from
#    step_height (isaaclab mesh_terrains.py:179-183). At the old 8.0 m tile with
#    platform_width=2.0 that was num_steps=3 -- a pit four steps deep on a 1.2 m square
#    floor, so "0.80 m steps" really meant escaping a 3.13 m well, and raising the height
#    ceiling made the task harder along an axis nobody intended. General principle that
#    outlives the pyramid terrain itself: check the resulting geometry a size/width/
#    platform change produces, not just the nominal height/step parameter.
#
# 2. Curriculum resolution matters. Widening step_height_range without adding rows makes
#    each promotion a bigger jump, and custom_terrain_levels_climb is a one-way ratchet
#    (it demotes only below 0.5 m of progress), so a robot promoted past its ability parks
#    on a row it cannot clear and keeps crashing there -- see the terrain block below for
#    the current num_rows/height range (reduced from this era's 20 on 2026-08-18 for
#    memory; the per-row step is correspondingly coarser now, see that block's own note).
#
# 3. Relaxing terminations has sharply diminishing returns. base_contact went
#    30 -> 80 -> 150 -> 400 N over four runs chasing a plateau that turned out to be
#    terrain shape; Try4's attempt to remove it *entirely* backfired outright (the policy
#    crashed into walls more recklessly rather than climbing). Treat a high termination
#    rate as a symptom to diagnose, not a threshold to raise. Still directly load-bearing
#    -- see the 2026-08-18 note above.
#
# --- What the second campaign established (Try 10-14, 2026-08-12..08-13) -- the
#     command/reward-gating mechanics below are specific to the terrain this section
#     replaces, but the underlying "check terrain-column-specific task validity" and
#     "check MuJoCo, not just Isaac Lab" principles carry forward -------------------------
#
# 4. lin_vel_y and lin_vel_x's reverse half were both pinned to a single global range
#    (mostly zero), terrain-blind, so Phase1-3's strafing was forgotten entirely and
#    reverse got almost no training exposure. Terrain-*gating* both -- full range on
#    "rough" columns, forward-only/no-strafe on the stair columns where lateral motion has
#    no task value -- restored both without touching the climbing setup at all (see
#    mdp/commands/velocity_command.py's UniformTerrainGatedVelocityCommand -- superseded
#    below by MixedGoalVelocityCommand, which keeps the same "rough" columns'
#    omnidirectional command but replaces the stair-column mechanism entirely).
#
# 5. climb_progress_reward had no command gate, unlike its siblings (forward_command_
#    progress, forward_stall_penalty) -- it paid out in full during standing envs too, at
#    this reward set's largest weight, which let "moving is rewarded" beat the much
#    sparser "stop when told to" signal. Gating it, doubling motion_without_cmd's weight,
#    and adding a wheel-joint-velocity-specific penalty (motion_without_cmd_penalty only
#    ever saw the base's *resultant* velocity, never the wheel actuator command itself)
#    together fixed the worst of a MuJoCo-observed zero-command drift, but not all of it
#    -- see the 2026-08-18 redesign above for what replaced this whole approach.
#
# 6. **The policy network mattered more than any of the above.** Try 13 (MLP, every fix
#    above applied) still drove forward with no command in MuJoCo and failed to climb even
#    0.20 m there, despite reaching the *highest* terrain_levels of the whole campaign in
#    Isaac Lab. Try 14, changing nothing but the network (MLP -> GRU), fixed the
#    command-drift and climbed 0.40 m on the identical environment. terrain_levels (or any
#    Isaac-Lab-only metric) is not sufficient evidence a change helped -- check MuJoCo.
#    Still load-bearing: this phase trains GruPPORunnerCfg (go2w/__init__.py). A policy
#    trained here cannot resume from an MLP checkpoint or a checkpoint from a different
#    network family (RSL-RL's checkpoint load is a strict state_dict load).
# =============================================================================

# Free-standing thin walls on flat ground -- no pit, unlike the pyramid_stairs terrain
# this replaces (Lesson 1 above no longer applies). At this tile's geometry (size=5.5,
# border_width=1.0, platform_width=2.0, wall_spacing=1.00) exactly one wall ring forms
# per side -- one wall to cross, matching the single-wall MuJoCo lanes this design was
# validated against. wall_thickness_range is held ~fixed (not scaled thin-at-high-
# difficulty) at roughly MuJoCo's actual wall thickness; only wall_height_range carries
# the difficulty curriculum. "rough" at 25% also gets its own full omnidirectional
# command (see CommandsCfgPhase5 below), so it is no longer just a mild balance exercise
# between wall crossings -- worth a real share of the column grid on its own terms. Tile
# size still sets the curriculum's promotion distance (custom_terrain_levels_climb
# promotes past size * 0.35 = 1.925 m from spawn).
#
# Grid shrunk 20x20 -> 4x10 (2026-08-18, memory) with rough/thin_wall reproportioned
# 30/70 -> 25/75 -- ported from the Go2w-v1-Phase5-Try16 sandbox experiment (terrain only;
# Try16's own front_leg_push_reward/base_contact changes were not folded in). Halving
# num_rows halves the wall-height curriculum's resolution (per-row step ~0.028 -> ~0.056
# m) and, since max_init_terrain_level below is an absolute row index unaffected by this
# change, roughly doubles the *fraction* of the height range a resumed/fresh run starts
# at on average (level 7 was ~35% of a 0-19 row range, now ~70% of a 0-9 one) -- not
# reduced to compensate; flag if that starts too hard.
PHASE5_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(5.5, 5.5),
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
            noise_range=(0.0, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.75,
            wall_height_range=(0.10, 0.60),
            wall_thickness_range=(0.40, 0.40),
            wall_spacing=1.00,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)


@configclass
class RobotSceneCfgPhase5(RobotSceneCfgPhase4):
    # max_init_terrain_level=7 predates the terrain redesign above but is kept -- it
    # matters on every resume, not just a cold start: the terrain_levels curriculum state
    # is not stored in the checkpoint, so each resume re-climbs from this value and a
    # lower setting simply burns iterations recovering ground already covered.
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=PHASE5_TERRAIN_CFG,
        max_init_terrain_level=7,
    )


@configclass
class CommandsCfgPhase5(CommandsCfgPhase3):
    """base_velocity: "rough" columns get CommandsCfgPhase2's own full omnidirectional
    ranges/limit_ranges (continuous with the checkpoint this phase bootstraps from, since
    Phase2 is the last common ancestor before this phase's own terrain diverges); every
    other column (the wall rings) gets goal-directed steering via
    MixedGoalVelocityCommand -- one random goal per episode placed just beyond the wall
    (goal_radius_range clears both the wall ring and the curriculum's own promotion rim
    at tile_size*0.35=1.925 m), synthesized lin_vel_x/ang_vel_z toward it, and the command
    dropping to exactly zero on arrival.

    This replaces UniformTerrainGatedVelocityCommand (Lesson 4 above), which forced
    lin_vel_x >= 0.4 on every non-"rough" column -- the robot almost never received a
    genuine near-zero command while actually on the obstacle, since rel_standing_envs was
    the only other source of a zero command and a standing episode never reaches the wall
    in the first place. MixedGoalVelocityCommand's per-episode goal gives direct
    "arrived -> stop and hold" exposure on the terrain that needs it instead.

    resampling_time_range=(20.0, 20.0) applies to "rough" envs too now, not just wall
    envs -- one command draw per 20 s episode, vs. Phase1/2's (10.0, 10.0) (two draws per
    episode). This is a real, if minor, deviation from "identical to Phase1/2": it halves
    the number of distinct command segments a "rough" env trains against per episode. The
    alternative (a second, branch-specific resampling cadence) is not exposed by
    CommandTerm's resample scheduling (a single per-term cfg value, drawn per-env-when-
    due, not something _resample_command can override per-branch).
    """

    base_velocity = mdp.MixedGoalVelocityCommandCfg(
        asset_name=CommandsCfgPhase3().base_velocity.asset_name,
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.01,
        debug_vis=CommandsCfgPhase3().base_velocity.debug_vis,
        ranges=CommandsCfgPhase2().base_velocity.ranges,
        limit_ranges=CommandsCfgPhase2().base_velocity.limit_ranges,
        rough_terrain_names=("rough",),
        goal_radius_range=(1.75, 2.5),
        arrival_radius=0.5,
        max_lin_vel=1.0,
        max_ang_vel=1.0,
        heading_control_stiffness=1.0,
    )


@configclass
class RewardsCfgPhase5(RewardsCfgPhase3):
    """On top of Phase3: direct ports of ANYmal Parkour's (Hoeller/Rudin et al. 2023)
    Table S2 goal-tracking terms, replacing this phase's earlier climb_progress/
    motion_without_cmd-style reward set entirely (Lesson 5 above) -- see each function's
    own docstring in mdp/rewards.py for the paper mapping and the departures from a
    literal port. goal_move_in_direction/goal_position_tracking/goal_heading_tracking
    are gated off on "rough" columns (no goal exists there); goal_position_tracking/
    goal_heading_tracking are additionally off once arrived (so they don't fight "stop
    and hold").

    undesired_contacts relaxed -1 -> -0.3 and flat_orientation_l2 softened -1.0 -> -0.5,
    both predating this redesign and kept unchanged: climbing something near standing
    height needs room to touch the obstacle on the way, and a tall wall requires the base
    to pitch, which flat_orientation_l2 otherwise penalises continuously even on a
    successful attempt.

    goal_arrival added 2026-08-24 (folded from Go2w-v1-Phase5-Try24), on top of the four
    terms above, none of which removed -- see mdp/rewards.py's own "Table S2 vs Table S3"
    module docstring for the full reasoning. goal_position_tracking/goal_heading_tracking
    only ever fire in a single 1 s window (arrival_deadline_s=8.0,
    activation_window=1.0) applied to this task's single, episode-long global goal --
    a mismatch with the paper's own intent for those terms (a *local* target reissued
    every ~0.2 s by a separate navigation module), which meant a wall crossing taking
    longer than 8 s got zero credit from either term for the rest of the episode,
    including while correctly holding position at the goal afterward. goal_arrival is
    the paper's own fix for exactly this (Table S3's "Position tracking (Navigation)"):
    it only checks the actual episode-end outcome, once, so a slow-but-successful
    crossing is no longer scored the same as never trying. A sibling sandbox Try
    (Try25) additionally replaced goal_position_tracking itself with a continuous
    potential-based term (a since-deleted ``goal_progress_reward`` -- abandoned per
    direct instruction despite a measured survivability improvement, base_contact
    termination 74% -> 1.4%; see sandbox/SUMMARY.md for the full record) -- not
    folded in here.

    undesired_contacts split 2026-08-26 (folded from Go2w-v1-Phase5-Try30) into two
    terms: Head/hip contact stays penalised on every column at the same -0.3 weight as
    before (unchanged consequence -- a robot has no legitimate reason to hit a wall
    with its head or hips even while climbing), while thigh/calf contact
    (``undesired_contacts_legs``, ``mdp.undesired_contacts_column_aware``) is now
    exempted specifically on "wall" columns, at 1 N threshold, thigh/calf were
    penalised for the exact load-bearing contact climbing a wall near standing height
    requires -- pushing against and resting on it on the way up -- which plausibly
    incentivised avoiding the wall altogether. Measured (Go2w-v1-Phase2 ->
    2000 iterations): terrain_levels reached 6.53 (this project's highest yet) with a
    healthy, non-degenerate termination distribution (base_contact 37.8%, time_out
    57.5% -- genuinely mixed outcomes, not the near-zero-base_contact/near-all-time_out
    pattern that turned out to mean "avoiding the wall entirely" in Try27's case).
    Checked in MuJoCo: reaches a front-leg foothold at 0.60 m, the highest climb
    attempt confirmed so far -- but still shares the same not-yet-resolved "trembles
    and creeps forward while meant to be holding position" issue first flagged on
    Try26's checkpoint (see the terrain_levels docstring below).

    wheel_vel_without_cmd (Try31) briefly folded in here 2026-08-28, then reverted
    the same day: penalising wheel velocity under a zero command
    (``mdp.wheel_vel_without_cmd_penalty``) fixed the trembling/creeping issue as a
    short refinement on top of Try30's own already-converged checkpoint (weight
    -0.001, confirmed in MuJoCo), but a fresh, continuous 3000-iteration run of the
    default with all three folds present *from Phase2* told a different story:
    terrain_levels peaked around iteration 3628 (~5.0) then declined steadily to 2.8
    by the end, with bad_orientation climbing from ~1% to ~20.6% over that same
    stretch. A 3-way ablation (Go2w-v1-Phase5-Try32/33/34, ~2000 iterations each from
    Phase2) isolated the cause: Try32 (old-default reproduction, no folds) and Try33
    (+Try26 only) and Try34 (+Try26+Try30, still no wheel_vel_without_cmd) all showed
    a healthy, monotonically-improving terrain_levels trajectory with no decline --
    only the combination *including* wheel_vel_without_cmd, trained continuously from
    scratch over a long run, showed the decline. Conclusion: the term is safe and
    effective as a short, targeted polish on an already-competent checkpoint, but not
    as part of the base Phase2->Phase5 training recipe itself. Moved to a separate,
    permanent adjustment task -- ``Go2w-v1-Phase5-Adjust``
    (velocity_env_cfg_phase5_adjust.py), meant to be run for ~1000 iterations against
    this default's own latest checkpoint whenever "trembles/creeps under a zero
    command" needs addressing, rather than baked into this class. See sandbox/
    SUMMARY.md for the full ablation record.

    wall_body_height added 2026-09-04 (folded from Go2w-v1-Phase5-Try39):
    ``mdp.wall_body_height_reward``, rewarding the base for reaching wall-top-plus-
    clearance height while near the wall and before arriving. Nothing else in this
    reward set ever sees base *height* -- ``goal_position_tracking``/
    ``track_lin_vel_xy_exp`` both only measure horizontal progress -- so a robot with
    its front feet on the wall and its torso still low had no local gradient toward
    "lift the torso", only "keep going forward" ones that can't tell climbing from
    being stuck. See that function's own docstring in mdp/rewards.py for why it reads
    the wall's known geometry (terrain_levels -> wall_height_range lerp) instead of a
    height-scan sensor.

    Three variants of this term were trained from this default's own checkpoint
    (Try36 unconditional / Try37 one-time bonus / Try38 progress-gated / Try39
    extended far-side gate). Try36's own MuJoCo check was the first real progress on
    this problem in the whole campaign -- the robot began rearing up and propping its
    front legs on the wall's top edge -- but it held that leaning pose indefinitely,
    since a continuous per-step height reward pays as well for holding it forever as
    for passing through it. Try39's ``gate_width_far=1.5`` (vs. the original symmetric
    0.6 m) keeps the pull active up to 1.5 m *past* the wall, so it does not taper off
    right where the hindquarters still need to come over; that is the version folded
    here. Play-checked at pinned 0.60 m: still mostly unable to cross at this point,
    but this is the reward baseline Try40/41/42's termination fixes build on, and with
    Try41's relaxed base_contact on top roughly 30 % of individuals clear 0.60 m in
    Play -- so the reward is load-bearing for that result even though it does not get
    there alone.

    Caution when reading terrain_levels around this fold: Try39 measures *higher*
    terrain_levels than Try41/42 (6.77 vs ~6.35 at matched iterations) while being
    *worse* at actually crossing 0.60 m in Play. The curriculum promotes on net
    displacement from spawn (> tile_size*0.35 = 1.925 m) and demotes on base_contact/
    bad_orientation, so relaxing base_contact removes a demotion trigger but also
    enables a "leaning on the wall at ~1.25 m until time_out" state that neither
    promotes nor demotes -- the ratchet stalls (visible as Try41's plateau at ~6.2-6.3,
    alongside its higher mean goal_distance, 1.93 vs 1.60, and longer episodes,
    904 vs 703). The whole 6.77-vs-6.35 gap is ~2 cm of wall height either way. This
    is the same "inflated plateau" pattern already recorded for the pre-Try26 default
    (terrain_levels 5.01 at 74 % base_contact) -- see sandbox/SUMMARY.md.
    """

    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(
        weight=-0.3, params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip"])}
    )
    undesired_contacts_legs = RewTerm(
        func=mdp.undesired_contacts_column_aware,
        weight=-0.3,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "command_name": "base_velocity",
        },
    )
    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.5)

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
        func=mdp.goal_dont_wait_penalty,
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
            "wall_height_range": (0.10, 0.60),
            "wall_distance": 1.25,
            "gate_width": 0.6,
            "gate_width_far": 1.5,
            "nominal_clearance": 0.15,
            "std": 0.15,
        },
    )


@configclass
class TerminationsCfgPhase5(TerminationsCfg):
    """Unchanged by the 2026-08-18 terrain/command/reward redesign above. Every threshold
    here is an estimate tuned against observed termination rates and play-mode checks,
    not derived from a hardware force spec.

    bad_orientation's limit_angle is 2.0 rad (~114.6 deg), up from the base config's much
    tighter value. The angle is acos(-projected_gravity_b[:, 2]), the base's tilt away from
    level, where pi/2 (1.571 rad) is already fully on its side or stood vertical -- so this
    permits a pronounced rear-back and only fires once the robot is tipped past horizontal
    toward inverted. Note that this termination is not what suppresses rearing in practice
    (it accounted for ~1.3 % of terminations); flat_orientation_l2 is, which is why that
    weight was halved rather than this angle raised further.

    base_contact and foot_impact both use mdp.illegal_contact_excluding_top rather than
    plain illegal_contact. That variant splits contacts by direction: resting or pushing
    down on a step's flat top reads as vertical-dominant and is exempt, while slamming
    into a vertical riser reads as horizontal-dominant and still terminates -- so genuine
    climbing technique is not punished but reckless charging is. Introduced after an
    earlier try showed that removing base_contact outright made the policy *more*
    reckless, not less (Lesson 3 above; re-confirmed independently in 2026-08 against the
    new thin_wall terrain too).

    Thresholds are 400 N for the base and 2500 N for the wheels, against a ~191.5 N total
    body weight. Both are far above where they started (30 N and 1500 N) and the escalation
    was mostly wasted: base_contact climbed 30 -> 80 -> 150 -> 400 N across four runs while
    the plateau it was chasing turned out to be terrain geometry (Lesson 1/3 above). They
    are kept at the values the best run used, but the lesson is the one recorded there --
    diagnose a high termination rate before raising the number. Note also that
    illegal_contact_excluding_top tests the contact history window's *maximum*, so a
    brief spike during a forceful but controlled push can trip a threshold well under body
    weight.
    """

    bad_orientation = TerminationsCfg().bad_orientation.replace(params={"limit_angle": 2.0})
    base_contact = DoneTerm(
        func=mdp.illegal_contact_excluding_top,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 400.0},
    )
    foot_impact = DoneTerm(
        func=mdp.illegal_contact_excluding_top,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 2500.0},
    )


@configclass
class CurriculumCfgPhase5(CurriculumCfg):
    """Unchanged by the 2026-08-18 terrain/command/reward redesign above. Swaps the stock
    terrain_levels_vel for custom_terrain_levels_climb. The stock ratchet's move_up needs
    a 4 m net displacement from spawn, unreachable at throttled speed on hard terrain, and
    its move_down floor scales with commanded speed so it fires on almost every reset --
    Phase3 showed levels collapsing to ~0.06 as a result. custom_terrain_levels_climb
    promotes on a reachable fraction of the tile (35 % = 1.925 m here) and demotes only on
    real failure (< 0.5 m moved), so partial progress keeps its level and keeps
    practising.

    The flip side, worth remembering when reading terrain_levels: because it never demotes
    for partial progress, the equilibrium is every robot parked at the hardest row it can
    still make *some* progress on. The mean is therefore "ability plus a bit", and a
    persistent non-zero termination rate is the normal steady state, not necessarily a bug.
    Also worth remembering (Lesson 6 above): terrain_levels measures Isaac Lab training
    progress only -- it is not a substitute for checking the policy in MuJoCo.

    lin_vel_cmd_levels replaced 2026-08-24 with mdp.lin_vel_cmd_levels_column_aware
    (folded directly in, not via a sandbox Try -- judged low-risk since it only touches
    "rough"-column envs' own velocity-range curriculum). MixedGoalVelocityCommand's
    "wall" envs never draw from the ``cfg.ranges`` object this term widens (their
    command is synthesized from ``max_lin_vel``/``max_ang_vel`` instead), so folding a
    wall env's track_lin_vel_xy reward into the average that decides whether to widen
    ``cfg.ranges`` was pure noise on a decision that only actually concerns "rough"
    envs -- see that function's own docstring in mdp/curriculums.py. Unrelated to
    terrain_levels (a separate curriculum term entirely).

    terrain_levels replaced 2026-08-25 with mdp.terrain_levels_climb_demote_on_fail
    (folded from Go2w-v1-Phase5-Try26). custom_terrain_levels_climb's move_down only
    fires below 0.5 m of net displacement, so an env promoted past its real ability
    could crash (base_contact/bad_orientation) after already covering more than that,
    neither promoted nor demoted -- stuck at a level it was genuinely failing, with
    nothing pulling it back down. The replacement additionally demotes on those two
    termination causes regardless of distance; see its own docstring in
    mdp/curriculums.py for the full reasoning. Measured (Go2w-v1-Phase2 ->
    2500 iterations): terrain_levels reached 6.16 (vs. the old function's own ~5.4-5.5
    over a comparable budget) with a lower base_contact rate (43% vs. ~74-75%) --
    and, checked in MuJoCo, this is this project's **first confirmed 0.50 m wall
    crossing**, though command-following was reported as sluggish/hard to control at
    that checkpoint -- not yet resolved, worth investigating further.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_climb_demote_on_fail)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels_column_aware)


@configclass
class RobotEnvCfgPhase5(RobotEnvCfgPhase4):
    """Phase 5: thin_wall crossing at 0.10-0.60 m, goal-directed commands on the wall
    columns (full omnidirectional on "rough"), direct goal-tracking rewards (ANYmal
    Parkour Table S2 ports), and top-surface-exempt contact terminations. Trained with a
    GRU policy (go2w/__init__.py's GruPPORunnerCfg) -- an MLP was not sufficient to
    reliably learn "stop when told to" on this task (see the module docstring's Lesson 6)."""

    scene: RobotSceneCfgPhase5 = RobotSceneCfgPhase5(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgPhase5 = CommandsCfgPhase5()
    rewards: RewardsCfgPhase5 = RewardsCfgPhase5()
    terminations: TerminationsCfgPhase5 = TerminationsCfgPhase5()
    curriculum: CurriculumCfgPhase5 = CurriculumCfgPhase5()

    def __post_init__(self):
        super().__post_init__()
        # Critic-only height_scan, widened on the lower bound. mdp.height_scan returns
        # base_z - hit_z - 0.5 (the RayCasterCfg's 20 m offset applies to ray_starts, not
        # to data.pos_w -- ray_caster.py:224 vs :241-248), so a step of height h ahead
        # reads -0.05 - h at Go2W's 0.45 m nominal base height: -0.815 at this range's top
        # row, close enough to the old -1.0 floor that it saturated whenever the robot
        # crouched. The policy observation group has no height_scan at all and stays
        # proprioception-only, so this affects value estimation only -- no tensor shape
        # changes, and checkpoints stay loadable across this edit.
        self.observations.critic.height_scan.clip = (-2.0, 5.0)


@configclass
class RobotPlayEnvCfgPhase5(RobotEnvCfgPhase5):
    """Inspection layout: wall only (no "rough" -- play mode is for looking at the climb;
    off-obstacle behaviour is what a deploy sim is for), 4 columns x 1 row pinned to
    exact 30/40/50/60 cm (60 cm is PHASE5_TERRAIN_CFG's own wall_height_range ceiling).

    Exact per-height pinning requires one column per height (a degenerate (h, h)
    wall_height_range) -- a single column's height varies continuously-with-jitter across
    rows instead (terrain_generator.py: difficulty = (row + eta)/num_rows, eta ~ U(0,1)),
    so "1 column, N rows, one exact height per row" isn't directly constructible.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.sub_terrains = {
            f"wall_{round(h * 100)}cm": terrains.MeshThinWallTerrainCfg(
                proportion=1.0,
                wall_height_range=(h, h),
                wall_thickness_range=(0.40, 0.40),
                wall_spacing=1.00,
                platform_width=2.0,
                border_width=1.0,
            )
            for h in (0.30, 0.40, 0.50, 0.60)
        }
