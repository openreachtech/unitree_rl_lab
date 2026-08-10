import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import CurriculumCfg, TerminationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase3 import CommandsCfgPhase3, RewardsCfgPhase3
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# =============================================================================
# Phase 5 -- extreme obstacle crossing.
#
# This file is the consolidated result of the Go2W Phase5 sandbox (Try 1 - Try 8,
# 2026-08-02 .. 2026-08-10, since removed). Everything below that differs from Phase4 was
# validated by a training run; the reasoning for each piece is kept inline so the next
# change does not have to re-derive it.
#
# Best result reached: terrain_levels equilibrated at 9.2 / 19, i.e. **0.44 m steps**, with
# foot_impact at 3.0 %, base_contact at 4.4 %, 91 % of episodes surviving to time_out, and
# the velocity-command curriculum saturated at its 1.2 m/s ceiling. Play-mode checks of
# that policy cleared 0.60 m with the better individuals and failed at 0.70 m.
#
# --- What the sandbox actually established -------------------------------------------
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
#    on a row it cannot clear and keeps crashing there. num_rows=20 over 0.10-0.80 m keeps
#    a 0.035 m step, finer than the 0.04 m the earlier 0.10-0.50 m / 10-row setup had.
#
# 3. Relaxing terminations has sharply diminishing returns. base_contact went
#    30 -> 80 -> 150 -> 400 N over four runs chasing a plateau that turned out to be
#    terrain shape; Try4's attempt to remove it *entirely* backfired outright (the policy
#    crashed into walls more recklessly rather than climbing). Treat a high termination
#    rate as a symptom to diagnose, not a threshold to raise.
#
# --- Known open items ----------------------------------------------------------------
#
# * step_height_range's 0.80 m ceiling is now larger than the policy can use: at the 0.44 m
#   equilibrium, levels ~13-19 are never reached. Narrowing to (0.10, 0.60) would put the
#   0.44 m working point near level 13 and tighten resolution to 0.025 m/level. Left at
#   0.80 because that is the range the best run actually used -- the narrowing is an
#   untested improvement, not an established one.
# * the EFGCL wall-bump assist was removed on 2026-08-10 -- disabled since Try1 and never
#   re-enabled, so it contributed nothing to any result above. See CommandsCfgPhase5 for
#   what it did and what re-adding it would take.
# =============================================================================

# pyramid_stairs_inv (80 %) spawns the robot on a pit floor and is therefore the *climbing*
# case; pyramid_stairs (20 %) spawns it on top of a pyramid and is the *descending* one --
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.80),
            step_width=1.00,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.80,
            step_height_range=(0.10, 0.80),
            step_width=1.00,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfgPhase5(RobotSceneCfgPhase4):
    # max_init_terrain_level=7 is 0.363 m under this 20-row range -- roughly where the
    # policy actually operates. It matters on every resume, not just a cold start: the
    # terrain_levels curriculum state is not stored in the checkpoint, so each resume
    # re-climbs from this value and a lower setting simply burns iterations recovering
    # ground already covered.
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=PHASE5_TERRAIN_CFG,
        max_init_terrain_level=7,
    )


@configclass
class CommandsCfgPhase5(CommandsCfgPhase3):
    """base_velocity restricted to forward-only, with a non-zero floor.

    No strafing or reverse: lin_vel_y is pinned to exactly zero and lin_vel_x is one-sided.
    The terrain's rings surround the spawn platform symmetrically, so "forward" always
    means driving straight at whichever face is currently ahead; commanding the robot to
    approach sideways or backwards is wasted exploration. ang_vel_z is untouched -- turning
    to square up with a face is still legitimate.

    lin_vel_x's floor is 0.4 m/s, in ranges *and* limit_ranges (lin_vel_cmd_levels clamps
    against limit_ranges, so the floor has to be in both to hold). This came out of the
    terrain_levels plateau: UniformVelocityCommand._resample_command draws lin_vel_x
    uniformly across the current range every resampling_time_range (10 s, twice per 20 s
    episode), so a floor of 0.0 meant a large share of episode-segments were commanded at
    near-zero speed. That is not merely "slower" -- forward_command_progress caps its
    reward at cmd_norm and track_lin_vel_xy_exp penalises exceeding the command, so a low
    draw removes the incentive to move on the obstacle at all, while the termination risk
    of attempting a climb is unchanged. Raising the floor lifted the mean commanded speed
    from 0.6 to 0.8 m/s.

    rel_standing_envs raised 0.01 -> 0.1 to pay for that floor. With lin_vel_x unable to
    be drawn below 0.4, the *only* way this policy ever sees a zero command is a standing
    env, and at 1 % that is far too rare to learn from: a 2026-08-10 MuJoCo check of the
    resulting policy had it driving off the moment it entered the RL state with nothing
    commanded, while a Phase1 policy under the same controller stopped correctly. Phase1's
    lin_vel_x spans (-2.0, 2.0), so roughly a tenth of its draws land near zero and
    standing is thoroughly trained; here zero is out of distribution and the policy
    defaults to what it always did, which is drive forward. 0.1 restores about the same
    zero-command exposure Phase1 gets naturally, at the cost of ~10 % of episode-segments
    no longer practising the climb. The alternative -- dropping the floor back to 0 --
    would give that exposure too but re-open the plateau the floor was introduced to fix.
    Note this is a training-side fix: it only takes effect on a run that starts from here,
    and no controller-side change can substitute for it, because a policy that has never
    been told to stop will not stop when the command finally reaches zero.

    Removed 2026-08-10: the EFGCL wall-bump assist (WallBumpAssistCommand plus its
    wall_bump_assist_decay curriculum, ported from feat/jump --
    doc/papers/EFGCL_...md). It was switched off at Try1 (assist_force_per_leg=0.0) so the
    reward-design work would be attributable and was never turned back on, so every result
    this file records was produced without it and it was pure dead weight in the command
    manager. A 2026-08-04 ablation had suggested it genuinely accelerated climbing skill
    rather than inflating terrain_levels, so if assisted exploration is wanted again the
    idea is worth revisiting -- but it would need re-implementing against the current
    terrain, which is stairs rather than the thin walls it was written for.
    """

    base_velocity = CommandsCfgPhase3().base_velocity.replace(
        rel_standing_envs=0.1,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class RewardsCfgPhase5(RewardsCfgPhase3):
    """Four changes on top of Phase3, all of them about making a climb worth attempting.

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

    climb_progress (+2.0) rewards vertical speed directly. Nothing else in the set
    recognises "you are climbing" as valuable in itself -- crossing an obstacle was only
    rewarded indirectly, through the net XY displacement it eventually produces. It is
    magnitude-based so it covers both sub-terrains without conditioning on which one an env
    is on, and rate-based so it cannot be farmed by reaching a height and camping there.
    Residual risk, not yet observed but worth watching: a robot bouncing vertically in
    place could in principle collect it without crossing anything; forward_stall_penalty is
    the counter-pressure.

    flat_orientation_l2 softened -1.0 -> -0.5. It is sum(projected_gravity_b[:, :2]**2),
    i.e. 1.0 at a full 90 deg tilt, so at -1.0 a robot holding a reared-back posture paid
    -1.0 every step it held it -- a continuous cost against exactly the posture a tall step
    requires, paid even when the attempt succeeds. Halved rather than removed: this term is
    the main thing keeping the wheeled base level in ordinary driving. Watch for the base
    riding nose-up on flat ground; if that appears, this is the term to restore.

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
        params={"max_climb_speed": 0.5},
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
    climbing technique is not punished but reckless charging is. Introduced after Try4
    showed that removing base_contact outright made the policy *more* reckless, not less.

    Thresholds are 400 N for the base and 2500 N for the wheels, against a ~191.5 N total
    body weight. Both are far above where they started (30 N and 1500 N) and the escalation
    was mostly wasted: base_contact climbed 30 -> 80 -> 150 -> 400 N across four runs while
    the plateau it was chasing turned out to be terrain geometry. They are kept at the
    values the best run used, but the lesson is the one recorded at the top of this file --
    diagnose a high termination rate before raising the number. Note also that
    illegal_contact_excluding_top tests the contact history window's *maximum*, so a brief
    spike during a forceful but controlled push can trip a threshold well under body weight.
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
    """

    terrain_levels = CurrTerm(func=mdp.custom_terrain_levels_climb)


@configclass
class RobotEnvCfgPhase5(RobotEnvCfgPhase4):
    """Phase 5: stair-crossing at 0.10-0.80 m steps, forward-only commands with a 0.4 m/s
    floor, direct climb reward, and top-surface-exempt contact terminations."""

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
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 2
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
