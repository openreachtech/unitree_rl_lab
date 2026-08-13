import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import (
    CurriculumCfg,
    TerminationsCfg,
    WHEEL_JOINT_NAMES,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase3 import CommandsCfgPhase3, RewardsCfgPhase3
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# =============================================================================
# Phase 5 -- extreme obstacle crossing.
#
# This file is the consolidated result of the Go2W Phase5 sandbox, two campaigns:
# Try 1 - Try 9 (2026-08-02 .. 2026-08-11) and Try 10 - Try 14 (2026-08-12 .. 2026-08-13).
# Every try has been folded in and deleted; see sandbox/SUMMARY.md for the reasoning that
# would otherwise be lost with them -- especially Lesson 7, which is why this phase's
# rsl_rl_cfg_entry_point is GruPPORunnerCfg rather than BasePPORunnerCfg (go2w/__init__.py).
#
# --- What the first campaign established (Try 1-9) ------------------------------------
#
# 1. Terrain *shape* dominated everything else. Three separate runs sat at ~0.36 m no
#    matter what termination thresholds were relaxed. The cause was geometric, not
#    behavioural: in an inverted pyramid the robot spawns on the pit floor at
#    -(num_steps + 1) * step_height, and num_steps comes from *platform_width*, not from
#    step_height (isaaclab mesh_terrains.py:179-183). At the old 8.0 m tile with
#    platform_width=2.0 that was num_steps=3 -- a pit four steps deep on a 1.2 m square
#    floor, so "0.80 m steps" really meant escaping a 3.13 m well, and raising the height
#    ceiling made the task harder along an axis nobody intended. Reshaping to num_steps=1
#    (below) cut foot_impact from 28 % to 3 % and broke the 0.36 m plateau immediately.
#    Anything that changes size / step_width / platform_width has to be checked against
#    the resulting pit depth and floor size, not just the nominal step height.
#
# 2. Curriculum resolution matters. Widening step_height_range without adding rows makes
#    each promotion a bigger jump, and custom_terrain_levels_climb is a one-way ratchet
#    (it demotes only below 0.5 m of progress), so a robot promoted past its ability parks
#    on a row it cannot clear and keeps crashing there. num_rows=20 keeps the per-row step
#    fine enough to matter (see the terrain block below for the current range).
#
# 3. Relaxing terminations has sharply diminishing returns. base_contact went
#    30 -> 80 -> 150 -> 400 N over four runs chasing a plateau that turned out to be
#    terrain shape; Try4's attempt to remove it *entirely* backfired outright (the policy
#    crashed into walls more recklessly rather than climbing). Treat a high termination
#    rate as a symptom to diagnose, not a threshold to raise.
#
# --- What the second campaign established (Try 10-14) ---------------------------------
#
# 4. lin_vel_y and lin_vel_x's reverse half were both pinned to a single global range
#    (mostly zero), terrain-blind, so Phase1-3's strafing was forgotten entirely and
#    reverse got almost no training exposure. Terrain-*gating* both -- full range on
#    "rough" columns, forward-only/no-strafe on the stair columns where lateral motion has
#    no task value -- restores both without touching the climbing setup at all. See
#    mdp/commands/velocity_command.py's UniformTerrainGatedVelocityCommand.
#
# 5. climb_progress_reward had no command gate, unlike its siblings (forward_command_
#    progress, forward_stall_penalty) -- it paid out in full during standing envs too, at
#    this reward set's largest weight, which let "moving is rewarded" beat the much
#    sparser "stop when told to" signal. Gating it, doubling motion_without_cmd's weight,
#    and adding a wheel-joint-velocity-specific penalty (motion_without_cmd_penalty only
#    ever saw the base's *resultant* velocity, never the wheel actuator command itself)
#    together fixed the worst of a MuJoCo-observed zero-command drift.
#
# 6. **The policy network mattered more than any of the above.** Try 13 (MLP, every fix
#    above applied) still drove forward with no command in MuJoCo and failed to climb even
#    0.20 m there, despite reaching the *highest* terrain_levels of the whole campaign in
#    Isaac Lab. Try 14, changing nothing but the network (MLP -> GRU), fixed the
#    command-drift and climbed 0.40 m on the identical environment. terrain_levels (or any
#    Isaac-Lab-only metric) is not sufficient evidence a change helped -- check MuJoCo.
#    This is why Go2W-v1-Phase1 through Phase5 all point at GruPPORunnerCfg now, not just
#    this phase, and why a policy trained here cannot resume from an older MLP checkpoint
#    (RSL-RL's checkpoint load is a strict state_dict load; ActorCriticRecurrent's
#    parameters don't match ActorCritic's -- retrain through Phase1 -> Phase2 -> ... again).
# =============================================================================

# Mix is rough 10 % / pyramid_stairs 20 % / pyramid_stairs_inv 70 % (they sum to 1.0, so
# the proportions read directly as percentages of the 20-column grid).
#
# pyramid_stairs_inv spawns the robot on a pit floor and is therefore the *climbing*
# case; pyramid_stairs spawns it on top of a pyramid and is the *descending* one --
# from the origins the two mesh generators return, +(num_steps+1)*step_height vs
# -(num_steps+1)*step_height (isaaclab mesh_terrains.py:146 and :246). The split is
# deliberately climb-heavy.
#
# Geometry, with size=5.5 / border_width=1.0 / platform_width=2.0 / step_width=1.00:
# num_steps = (5.5 - 2*1.0 - 2.0) // (2*1.00) + 1 = 1, giving a 1.5 m square floor, a
# 1.00 m tread, then grade -- two steps of step_height, pit depth 2*step_height. Note
# platform_width must stay above (inner_size - 2*step_width) or num_steps rises and the
# centre platform collapses to zero (or negative) width; that degenerate case was hit for
# real during the sandbox and reads in play mode as the robot spawning "buried".
#
# Tile size also sets the curriculum's promotion distance: custom_terrain_levels_climb
# promotes past size * 0.35 = 1.925 m, and the rim sits 1.75 m from the spawn point, so
# promotion now requires fully escaping the pit. On the old 8.0 m tile the rim was at
# 3.0 m and the 2.80 m threshold could be met while still standing on the final step.
#
# step_height_range narrowed (0.10, 0.80) -> (0.10, 0.60) in the second campaign (Try 11):
# the 0.80 m ceiling was larger than the policy could use (at the ~0.44 m equilibrium the
# first campaign reached, levels ~13-19 were never reached), so narrowing puts the working
# point nearer the top of the range and tightens resolution from 0.035 to 0.025 m/level
# over the span that's actually used. num_rows stays at 20.
PHASE5_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(5.5, 5.5),
    border_width=20.0,
    num_cols=20,
    num_rows=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Rough ground, 10 % (2026-08-11). Other phases spend this slot on a flat column,
        # but flat is not actually missing from a stairs-only mix: escaping an inverted
        # pyramid puts the robot on the tile's level rim and border, and the grid's own
        # 20 m outer border is flat, so level ground is already encountered every time a
        # climb succeeds. Unstructured ground is the thing genuinely absent, and it is
        # also closer to what an obstacle course looks like off the obstacle. Kept mild
        # (+/-3 cm, well under the wheel radius) so it exercises balance without competing
        # with the stairs for difficulty. Also the only column type where lin_vel_y and
        # reverse lin_vel_x are active -- see CommandsCfgPhase5 below.
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.0, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.60),
            step_width=1.00,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 0.20 / 0.70 alongside rough's 0.10 -- the three sum to exactly 1.0, so the
        # proportions read directly as percentages of the column grid.
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.70,
            step_height_range=(0.10, 0.60),
            step_width=1.00,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase5(RobotSceneCfgPhase4):
    # max_init_terrain_level=7 is now ~0.29 m under this narrowed 20-row range (was
    # ~0.36 m under the old 0.10-0.80 m range) -- a lower, more conservative starting
    # height, not a change made to compensate for anything. It matters on every resume,
    # not just a cold start: the terrain_levels curriculum state is not stored in the
    # checkpoint, so each resume re-climbs from this value and a lower setting simply
    # burns iterations recovering ground already covered.
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=PHASE5_TERRAIN_CFG,
        max_init_terrain_level=7,
    )


@configclass
class CommandsCfgPhase5(CommandsCfgPhase3):
    """base_velocity: forward-biased on stairs, full range (including strafe and reverse)
    on "rough" columns only.

    lin_vel_y and lin_vel_x's reverse half are both terrain-gated via
    ``UniformTerrainGatedVelocityCommandCfg`` (mdp/commands/velocity_command.py): full
    range on "rough" columns, pinned to zero (lin_vel_y) or clamped to the forward floor
    (lin_vel_x) everywhere else. The terrain's rings surround the spawn platform
    symmetrically, so on stairs "forward" always means driving straight at whichever face
    is ahead -- strafing and reversing have no task value there, and lateral motion is the
    hardest mode for a wheeled quadruped (the legs have to step sideways), so training them
    on stairs would cost the most dilution for the least gain. But that reasoning is
    terrain-*specific*: it doesn't hold on the "rough" columns, where strafing/reversing are
    exactly as valid as they were in Phase1-3, and pinning them globally to zero (as this
    phase originally did) meant the robot had a full phase of training with zero exposure
    to either -- it forgot both (sandbox Try10/Try13). ang_vel_z is untouched everywhere --
    turning to square up with a face is still legitimate, and turning is what covers
    repositioning.

    lin_vel_y: nonzero on "rough" at Phase3's old pre-Phase5 band, ranges (-0.1, 0.1) /
    limit (-0.7, 0.7) -- the range the robot actually trained against before this phase
    zeroed it, not an arbitrary new choice.

    lin_vel_x runs (-0.4, 1.2) *on rough*; reverse is capped well under the forward
    ceiling everywhere else via ``restricted_lin_vel_x_min=0.4`` because it is for backing
    out, not for driving, and stairs give reverse nothing to do (backing away from a
    pyramid_stairs_inv pit just returns to the rim it started from). ``ranges`` starts at
    (0.4, 0.4) and lin_vel_cmd_levels widens it by +/-0.1 per promotion (both bounds move
    together, each clamped independently against its own limit), so training begins
    forward-only and earns the full range gradually. limit_ranges.lin_vel_x is (-1.2, 1.2)
    -- widened from an earlier (-0.4, 1.2) once reverse was moved to "rough"-only and no
    longer needed to be timid about stair obstacles it can no longer be commanded into.

    lin_vel_x's floor is 0.4 m/s, in ranges *and* limit_ranges (lin_vel_cmd_levels clamps
    against limit_ranges, so the floor has to be in both to hold, and
    UniformTerrainGatedVelocityCommand's own floor-clamp on non-"rough" columns uses the
    same 0.4 via restricted_lin_vel_x_min). This came out of the terrain_levels plateau:
    UniformVelocityCommand._resample_command draws lin_vel_x uniformly across the current
    range every resampling_time_range (10 s, twice per 20 s episode), so a floor of 0.0
    meant a large share of episode-segments were commanded at near-zero speed. That is not
    merely "slower" -- forward_command_progress caps its reward at cmd_norm and
    track_lin_vel_xy_exp penalises exceeding the command, so a low draw removes the
    incentive to move on the obstacle at all, while the termination risk of attempting a
    climb is unchanged. Raising the floor lifted the mean commanded speed from 0.6 to
    0.8 m/s.

    rel_standing_envs raised 0.01 -> 0.1 to pay for that floor. With lin_vel_x unable to
    be drawn below 0.4 on stair columns, the *only* way those envs ever see a zero command
    is a standing env, and at 1 % that is far too rare to learn from: a MuJoCo check of the
    resulting policy had it driving off the moment it entered the RL state with nothing
    commanded, while a Phase1 policy under the same controller stopped correctly. Phase1's
    lin_vel_x spans (-2.0, 2.0), so roughly a tenth of its draws land near zero and
    standing is thoroughly trained; here zero is out of distribution on stairs and the
    policy defaults to what it always did, which is drive forward. 0.1 restores about the
    same zero-command exposure Phase1 gets naturally, at the cost of ~10 % of
    episode-segments no longer practising the climb. The alternative -- dropping the floor
    back to 0 -- would give that exposure too but re-open the plateau the floor was
    introduced to fix. Note this is a training-side fix: it only takes effect on a run
    that starts from here, and no controller-side change can substitute for it, because a
    policy that has never been told to stop will not stop when the command finally reaches
    zero -- see the climb_progress/motion_without_cmd gating in RewardsCfgPhase5, which
    closes the rest of that gap; the command-distribution fix alone was not sufficient.
    """

    base_velocity = mdp.UniformTerrainGatedVelocityCommandCfg(
        asset_name=CommandsCfgPhase3().base_velocity.asset_name,
        resampling_time_range=CommandsCfgPhase3().base_velocity.resampling_time_range,
        rel_standing_envs=0.1,
        debug_vis=CommandsCfgPhase3().base_velocity.debug_vis,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.4), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-1.0, 1.0)
        ),
        lateral_terrain_names=("rough",),
        restricted_lin_vel_x_min=0.4,
    )


@configclass
class RewardsCfgPhase5(RewardsCfgPhase3):
    """On top of Phase3: making a climb worth attempting, and making "stop when told to"
    actually hold on a wheeled base.

    forward_command_progress (+0.8) and forward_stall_penalty (-1.0) replace the inherited
    reward set's reliance on track_lin_vel_xy_exp, whose exponential kernel has almost no
    gradient far from the commanded speed -- so a robot stopped at an obstacle got no push
    to start moving, and "attempted and failed" scored the same as "never tried", making
    freezing a locally safe strategy. Both functions already existed in mdp/rewards.py for
    Go2's climbing configs; Go2W's chain had simply never included them.

    forward_stall_penalty rather than plain stall_penalty: stall_penalty gates on raw
    planar speed, which is direction-agnostic, so the centre-of-mass wobble from rotating
    in place near an obstacle softened it without any real progress -- and track_ang_vel_z
    independently rewarded the turning. Play-mode confirmed the resulting behaviour
    (spinning in place instead of committing). forward_stall_penalty gates on the same
    vel_b . cmd_dir projection forward_command_progress uses, so only genuine forward
    progress counts.

    climb_progress (+2.0) rewards vertical speed directly, gated on command
    (``command_name="base_velocity"``). Nothing else in the set recognises "you are
    climbing" as valuable in itself -- crossing an obstacle was only rewarded indirectly,
    through the net XY displacement it eventually produces. It is magnitude-based so it
    covers both sub-terrains without conditioning on which one an env is on, and rate-based
    so it cannot be farmed by reaching a height and camping there. The command gate was
    added after this term (unconditional at the time) turned out to be the main cause of a
    MuJoCo-observed zero-command drift: any vertical bob (rough-terrain noise, a step-edge
    wobble) paid out in full during standing envs too, at this reward set's largest weight,
    which let "moving is rewarded" beat the much sparser motion_without_cmd signal.
    Residual risk, not yet observed but worth watching: a robot bouncing vertically in
    place could in principle collect it without crossing anything (while commanded);
    forward_stall_penalty is the counter-pressure there.

    flat_orientation_l2 softened -1.0 -> -0.5. It is sum(projected_gravity_b[:, :2]**2),
    i.e. 1.0 at a full 90 deg tilt, so at -1.0 a robot holding a reared-back posture paid
    -1.0 every step it held it -- a continuous cost against exactly the posture a tall step
    requires, paid even when the attempt succeeds. Halved rather than removed: this term is
    the main thing keeping the wheeled base level in ordinary driving. Watch for the base
    riding nose-up on flat ground; if that appears, this is the term to restore.

    motion_without_cmd (-2.0) makes "stop when told to stop" an explicit objective, on the
    base's resultant linear/yaw velocity. A MuJoCo check found the policy driving off on
    its own under a zero command; probing the exported network with a stationary, level,
    default-pose observation reproduced it offline, with wheel outputs of 20-30 rad/s
    whether the command was zero, forward or reverse. The reward set had nothing that
    actually penalised this on a wheeled base: ``feet_contact_without_cmd`` rewards feet
    *in contact*, which is a sound proxy for standing on a legged robot but is collected in
    full by a Go2W rolling at speed, so the only remaining pressure was
    track_lin_vel_xy_exp's kernel, which is already saturated near zero by the time the
    robot is visibly moving. Raising rel_standing_envs to 0.1 (see CommandsCfgPhase5) gave
    the policy the *exposure* to zero commands it was missing; this gives it a gradient to
    learn from once exposed. Weight doubled from an initial -1.0 once climb_progress's
    unconditional payout (above) was identified as outweighing it during that exposure.

    wheel_motion_without_cmd (-0.03) closes a gap motion_without_cmd cannot: that term only
    sees the base's *resultant* velocity, never the wheel actuator command itself. A policy
    can satisfy "base didn't move much" under Isaac Sim's friction/load model while still
    outputting a nontrivial wheel command, and a different contact model (MuJoCo) can let
    that same command actually roll the robot -- confirmed by probing the exported network
    exactly as above. Scoped to WHEEL_JOINT_NAMES, gated the same way
    (``cmd_norm < cmd_threshold``), computed on wheel joint velocity directly.

    Even with all of the above, an MLP alone was not sufficient to fully solve
    "stop when told to" on this task -- see go2w/__init__.py's GruPPORunnerCfg and
    sandbox/SUMMARY.md's Lesson 7. These reward terms remain necessary (they measurably
    helped) but were not, on their own, sufficient.

    undesired_contacts relaxed -1 -> -0.3 for the same reason base_contact's termination is
    loosened below: climbing something near standing height needs room to touch the
    obstacle on the way.
    """

    undesired_contacts = RewardsCfgPhase3().undesired_contacts.replace(weight=-0.3)
    flat_orientation_l2 = RewardsCfgPhase3().flat_orientation_l2.replace(weight=-0.5)

    forward_command_progress = RewTerm(
        func=mdp.forward_command_progress,
        weight=0.8,
        params={"command_name": "base_velocity"},
    )
    forward_stall_penalty = RewTerm(
        func=mdp.forward_stall_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_scale": 0.1,
        },
    )
    climb_progress = RewTerm(
        func=mdp.climb_progress_reward,
        weight=2.0,
        params={"max_climb_speed": 0.5, "command_name": "base_velocity"},
    )
    motion_without_cmd = RewTerm(
        func=mdp.motion_without_cmd_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_threshold": 0.1,
        },
    )
    wheel_motion_without_cmd = RewTerm(
        func=mdp.wheel_motion_without_cmd_penalty,
        weight=-0.03,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "cmd_threshold": 0.1,
        },
    )


@configclass
class TerminationsCfgPhase5(TerminationsCfg):
    """Every threshold here is an estimate tuned against observed termination rates and
    play-mode checks, not derived from a hardware force spec.

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
    reckless, not less.

    Thresholds are 400 N for the base and 2500 N for the wheels, against a ~191.5 N total
    body weight. Both are far above where they started (30 N and 1500 N) and the escalation
    was mostly wasted: base_contact climbed 30 -> 80 -> 150 -> 400 N across four runs while
    the plateau it was chasing turned out to be terrain geometry (see the module docstring's
    Lesson 1/3). They are kept at the values the best run used, but the lesson is the one
    recorded there -- diagnose a high termination rate before raising the number. Note also
    that illegal_contact_excluding_top tests the contact history window's *maximum*, so a
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
    """Swaps the stock terrain_levels_vel for custom_terrain_levels_climb. The stock
    ratchet's move_up needs a 4 m net displacement from spawn, unreachable at throttled
    speed on hard terrain, and its move_down floor scales with commanded speed so it fires
    on almost every reset -- Phase3 showed levels collapsing to ~0.06 as a result.
    custom_terrain_levels_climb promotes on a reachable fraction of the tile
    (35 % = 1.925 m here) and demotes only on real failure (< 0.5 m moved), so partial
    progress keeps its level and keeps practising.

    The flip side, worth remembering when reading terrain_levels: because it never demotes
    for partial progress, the equilibrium is every robot parked at the hardest row it can
    still make *some* progress on. The mean is therefore "ability plus a bit", and a
    persistent non-zero termination rate is the normal steady state, not necessarily a bug.
    Also worth remembering (sandbox Lesson 7): terrain_levels measures Isaac Lab training
    progress only -- it is not a substitute for checking the policy in MuJoCo.
    """

    terrain_levels = CurrTerm(func=mdp.custom_terrain_levels_climb)


@configclass
class RobotEnvCfgPhase5(RobotEnvCfgPhase4):
    """Phase 5: stair-crossing at 0.10-0.60 m steps, terrain-gated commands (full range on
    "rough", forward-only/no-strafe on stairs), direct climb reward, and top-surface-exempt
    contact terminations. Trained with a GRU policy (see go2w/__init__.py's
    GruPPORunnerCfg) -- an MLP was not sufficient to reliably learn "stop when told to" on
    this task, see sandbox/SUMMARY.md's Lesson 7."""

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
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # Inspection layout: the two stair types only (rough is dropped -- play mode is for
        # looking at the climb; off-obstacle behaviour, including the lin_vel_y/reverse
        # gate that "rough" columns exercise, is what a deploy sim is for), each at a fixed
        # step, one per column. Heights are the top three 0.10 m steps below the trained
        # (0.10, 0.60) ceiling -- 0.40/0.50/0.60, rescaled down from the first campaign's
        # 0.60/0.70/0.80 when the training range itself was narrowed (see the module
        # docstring's Lesson 4/second-campaign notes).
        #
        # The heights are pinned with a degenerate (h, h) step_height_range rather than by
        # difficulty: the generator varies difficulty over *rows* and picks the sub_terrain
        # by *column* (terrain_generator.py:244-263), so with num_rows=1 the difficulty is a
        # random U(0,1) per tile and cannot pin a height, while a (h, h) range returns h
        # whatever difficulty comes out. Six equal proportions over six columns put exactly
        # one sub_terrain in each. Geometry matches training (step_width / platform_width /
        # border_width unchanged), so num_steps stays 1 and each tile is two steps of h,
        # i.e. a tower or pit 2h tall.
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.terrain.terrain_generator.sub_terrains = {
            f"{label}_{round(h * 100)}cm": cls(
                proportion=1.0,
                step_height_range=(h, h),
                step_width=1.00,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            )
            for label, cls in (
                ("pyramid", terrain_gen.MeshPyramidStairsTerrainCfg),
                ("inv", terrain_gen.MeshInvertedPyramidStairsTerrainCfg),
            )
            for h in (0.40, 0.50, 0.60)
        }
