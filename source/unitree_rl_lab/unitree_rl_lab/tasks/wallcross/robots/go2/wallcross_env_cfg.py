"""Environment configurations for the hind-stance wall crossing task.

``Go2-Multitask-Biped-Hind`` with four things changed and nothing else:

* the terrain is the concentric-wall generator, with a row curriculum;
* the stance is released, and ``jump_command`` fired, when the robot reaches its wall
  (``WallCrossCommand``);
* the crossing is scored and ends the episode (rewards and terminations below);
* ``base_contact`` exempts contact on the wall.

The observation is the unified multi-task layout, unchanged, so a ``Biped-Hind`` checkpoint loads as
it is and the result drops into the mixture without widening anything.
"""

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped.robots.go2.biped_front_env_cfg_multitask import BipedTerminationsCfg
from unitree_rl_lab.tasks.biped.robots.go2.biped_hind_env_cfg_multitask import (
    LIFTED_LEG_LINKS,
    BipedHindCommandsCfg,
    BipedHindRewardsCfg,
    RobotEnvCfgBipedHind,
    RobotPlayEnvCfgBipedHind,
)
from unitree_rl_lab.tasks.wallcross import mdp
from unitree_rl_lab.tasks.wallcross.terrains import WALL_TERRAIN_CFG, WALL_TERRAIN_THIN_FIRST_CFG

SPAWN_XY_RANGE = (-0.2, 0.2)
"""Spawn offset from the tile centre. The biped task's +-0.5 m would put a robot half a metre from a
wall whose inner face is 1.0 m out, before it has even risen."""

SPAWN_YAW_RANGE = (-0.17, 0.17)
"""Facing a wall square-on, +-10 deg. The ring is square, so an arbitrary heading meets the wall at up
to 45 deg or runs into a corner; that comes later, once square-on crossings work. (The stance drifts
and turns on its own -- 0.9 m and about -0.7 rad/s at a zero command -- so the heading at the jump
is less tidy than this.)"""

APPROACH_SPEED = (0.5, 0.8)
"""Forward command toward the wall. The hind stance delivers about 60% of its command (0.38 m/s at
0.6). With (0.3, 0.6) and walls out to 2.5 m, 66% of the first run's jumps were forced by the time
limit, on average 0.9 m short of the wall; this, walls out to 2.0 m and an 8 s limit leave the
slowest case (1.7 m at 0.3 m/s) inside it."""


def _place_on_walls(cfg, terrain=WALL_TERRAIN_CFG):
    cfg.scene.terrain.terrain_generator = terrain.replace()
    cfg.scene.terrain.terrain_generator.curriculum = True
    pose = cfg.events.reset_base.params["pose_range"]
    cfg.events.reset_base.params["pose_range"] = {
        **pose, "x": SPAWN_XY_RANGE, "y": SPAWN_XY_RANGE, "yaw": SPAWN_YAW_RANGE,
    }


@configclass
class WallCrossCommandsCfg(BipedHindCommandsCfg):
    handstand = mdp.WallHandstandCommandCfg(asset_name="robot", stance=mdp.STANCE_HIND, pinned=True)
    wallcross = mdp.WallCrossCommandCfg()

    def __post_init__(self):
        super().__post_init__()
        for ranges in (self.base_velocity.ranges, self.base_velocity.limit_ranges):
            ranges.lin_vel_x = APPROACH_SPEED
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
        self.base_velocity.rel_standing_envs = 0.0


@configclass
class WallCrossRewardsCfg(BipedHindRewardsCfg):
    """The hind stance's rewards, plus the crossing.

    The stance terms retire on their own: they are gated on the stance command, which the jump
    releases (0.25 s crossfade). What is added is sized so that, per episode, crossing > holding
    still at the wall > falling:

    * before the jump the stance nets about +2.7/s (measured on ``model_1499``: +3.4 stance and
      tracking, -0.71 always-on penalties);
    * after it, ``wall_alive`` (+0.7/s) cancels those penalties, so holding still is worth ~0;
    * ``wall_progress`` pays up to 1 point for reaching the wall, 4 for the base rising to 0.15 m over
      its top (a jump, not a tip-over from the stance height), 4 for getting base and feet over and 3
      for standing up past it (base 0.24 m, level within 15 deg -- the success pose);
    * ``wall_success`` pays 20 once;
    * a fall costs 4 (the inherited -200 x 0.02 s), a success or a timeout nothing.
    """

    undesired_contacts = RewTerm(
        func=mdp.before_jump,
        weight=-1.0,
        params={
            "term": mdp.undesired_contacts,
            "term_params": {
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LIFTED_LEG_LINKS),
            },
        },
    )
    termination_penalty = RewTerm(
        func=mdp.terminated_by, weight=-200.0, params={"term_keys": ["base_contact", "collapsed"]}
    )
    wall_progress = RewTerm(func=mdp.wall_progress, weight=1.0)
    wall_alive = RewTerm(func=mdp.wall_alive, weight=0.7)
    wall_success = RewTerm(func=mdp.wall_success, weight=20.0)


@configclass
class WallCrossTerminationsCfg(BipedTerminationsCfg):
    base_contact = DoneTerm(
        func=mdp.illegal_contact_off_wall,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    wall_success_done = DoneTerm(func=mdp.wall_success_done)
    wall_timeout = DoneTerm(func=mdp.wall_timeout)


@configclass
class WallCrossCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.wall_terrain_levels)


@configclass
class RobotEnvCfgWallCross(RobotEnvCfgBipedHind):
    commands: WallCrossCommandsCfg = WallCrossCommandsCfg()
    rewards: WallCrossRewardsCfg = WallCrossRewardsCfg()
    terminations: WallCrossTerminationsCfg = WallCrossTerminationsCfg()
    curriculum: WallCrossCurriculumCfg = WallCrossCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _place_on_walls(self)
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class RobotPlayEnvCfgWallCross(RobotEnvCfgWallCross):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None


@configclass
class RobotEnvCfgWallCrossThinFirst(RobotEnvCfgWallCross):
    """Thin walls before high ones: stages A0, A, C' (thinned at 0.30-0.40 m), B' (raised thin)."""

    def __post_init__(self):
        super().__post_init__()
        _place_on_walls(self, WALL_TERRAIN_THIN_FIRST_CFG)


@configclass
class RobotPlayEnvCfgWallCrossThinFirst(RobotPlayEnvCfgWallCross):
    def __post_init__(self):
        super().__post_init__()
        _place_on_walls(self, WALL_TERRAIN_THIN_FIRST_CFG)


@configclass
class WallCrossPacedRewardsCfg(WallCrossRewardsCfg):
    """The crossing's rewards, plus holding the commanded pace until the jump (see
    ``mdp.approach_overspeed``). -10 costs about 11 points per second at run6's 1.9 m/s against 0.65."""

    approach_overspeed = RewTerm(func=mdp.approach_overspeed, weight=-10.0)


@configclass
class RobotEnvCfgWallCrossThinFirstPaced(RobotEnvCfgWallCrossThinFirst):
    """Thin-first walls, approach at the commanded pace."""

    rewards: WallCrossPacedRewardsCfg = WallCrossPacedRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstPaced(RobotPlayEnvCfgWallCrossThinFirst):
    rewards: WallCrossPacedRewardsCfg = WallCrossPacedRewardsCfg()


@configclass
class WallCrossPacedStrictRewardsCfg(WallCrossRewardsCfg):
    """Paced, tighter: margin 0.2 -> 0.05 m/s, weight -10 -> -30 (run8). run7 slowed the approach from
    1.9 to 1.2 m/s but still parked at 1.1 m/s and fell 0.2 s later in every no-jump episode (40718 of
    40909, model_8998), while the hind-stance expert parks at 0.4 m/s and stands. Parked, the command
    is zero, so this term also charges moving on after the stop (-27 points per second at 1 m/s)."""

    approach_overspeed = RewTerm(func=mdp.approach_overspeed, weight=-30.0, params={"margin": 0.05})


@configclass
class RobotEnvCfgWallCrossThinFirstPacedStrict(RobotEnvCfgWallCrossThinFirst):
    """Thin-first walls, approach held close to the commanded pace."""

    rewards: WallCrossPacedStrictRewardsCfg = WallCrossPacedStrictRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstPacedStrict(RobotPlayEnvCfgWallCrossThinFirst):
    rewards: WallCrossPacedStrictRewardsCfg = WallCrossPacedStrictRewardsCfg()


@configclass
class WallCrossPacedStrictFallRewardsCfg(WallCrossPacedStrictRewardsCfg):
    """PacedStrict, and a fall before the jump costs 20 instead of 4 (run9): the inherited -4 plus
    -800 x 0.02 s here. run8 (PacedStrict alone) collapsed within ~30 iterations into falling 0.2 s
    after the start, since -4 was the cheapest way out -- already for the hind-stance expert a fall at
    the wall (+4.39) paid more than standing there (+1.68). A fall after the jump stays at 4, so that
    jumping is not made riskier."""

    pre_jump_fall = RewTerm(
        func=mdp.before_jump,
        weight=-800.0,
        params={"term": mdp.terminated_by, "term_params": {"term_keys": ["base_contact", "collapsed"]}},
    )


@configclass
class RobotEnvCfgWallCrossThinFirstPacedStrictFall(RobotEnvCfgWallCrossThinFirst):
    """Thin-first walls, strict pace, a fall before the jump priced at 20."""

    rewards: WallCrossPacedStrictFallRewardsCfg = WallCrossPacedStrictFallRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstPacedStrictFall(RobotPlayEnvCfgWallCrossThinFirst):
    rewards: WallCrossPacedStrictFallRewardsCfg = WallCrossPacedStrictFallRewardsCfg()


@configclass
class WallCrossPacedStrictFallFloorRewardsCfg(WallCrossPacedStrictFallRewardsCfg):
    """PacedStrictFall, and a floor under the stance's base height until the jump (run10; see
    ``mdp.stance_height_floor``). -50 costs 6 points per second at run9's 0.33 m, 3.5 at run7's 0.38 m,
    nothing from 0.45 m (run4-run6 approached at 0.43-0.49 m and jumped from there)."""

    stance_height_floor = RewTerm(func=mdp.stance_height_floor, weight=-50.0, params={"min_height": 0.45})


@configclass
class RobotEnvCfgWallCrossThinFirstPacedStrictFallFloor(RobotEnvCfgWallCrossThinFirst):
    """Thin-first walls, strict pace, a fall before the jump priced at 20, base kept above 0.45 m."""

    rewards: WallCrossPacedStrictFallFloorRewardsCfg = WallCrossPacedStrictFallFloorRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstPacedStrictFallFloor(RobotPlayEnvCfgWallCrossThinFirst):
    rewards: WallCrossPacedStrictFallFloorRewardsCfg = WallCrossPacedStrictFallFloorRewardsCfg()


@configclass
class RobotEnvCfgWallTerrainCheck(RobotPlayEnvCfgBipedHind):
    """The hind-stance play environment on the wall terrain, spread over every row."""

    def __post_init__(self):
        super().__post_init__()
        _place_on_walls(self)
        # Spread the environments over all rows so every stage is on screen, and generate the rows
        # in difficulty order. `apply_multitask_post_init` turns the generator's curriculum off when
        # there is no terrain_levels term, which would scatter difficulties across the grid.
        self.scene.terrain.max_init_terrain_level = None


WALKER_STATE_BANK = os.environ.get(
    "WALLCROSS_WALKER_BANK",
    os.path.join(os.path.dirname(__file__), "../../../../../../../reproduce_logs/wallcross_env/walker_bank/walker_states_gallop2500.pt"),
)
"""States of the walking expert (Gallop-Phase2 model_2500) at 0.5-0.8 m/s, from
``scripts/rsl_rl/record_walker_states.py``. Local file, not in git."""


def _start_from_walk(cfg):
    """run11: episodes start mid-walk, every episode crosses.

    The stance is only asked for when the robot is to go over a wall: it walks on four legs (walking
    expert) and the crossing expert takes over when the stance comes on (tanaka, 2026-09-30). Walking
    past a wall without crossing is the walking expert's job, so the no-jump episodes -- which
    run7-run10 could not make the stance survive -- are dropped, and the episode starts from a
    recorded walking state instead of a standstill."""
    cfg.events.reset_from_walk = EventTerm(
        func=mdp.reset_from_state_bank, mode="reset", params={"path": os.path.abspath(WALKER_STATE_BANK)}
    )
    cfg.commands.wallcross.jump_episode_probability = 1.0


@configclass
class WallCrossFromWalkRewardsCfg(WallCrossPacedRewardsCfg):
    """Paced (run7), and a fall before the jump costs 20 instead of 4, as in run9. Without it the first
    run11 start collapsed within ~40 iterations into falling 0.2 s after the reset (episode length 9
    steps): rising from a walk under the training noise fell often enough that -4 was the cheapest
    way out, the run8 collapse again. Measured on run7 model_8998 from the walk: a fall before the
    jump -5.6 points against +14.8 for a crossing."""

    pre_jump_fall = RewTerm(
        func=mdp.before_jump,
        weight=-800.0,
        params={"term": mdp.terminated_by, "term_params": {"term_keys": ["base_contact", "collapsed"]}},
    )


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalk(RobotEnvCfgWallCrossThinFirst):
    """run7's task (thin-first walls, paced approach), started from the walk, crossing every time."""

    rewards: WallCrossFromWalkRewardsCfg = WallCrossFromWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _start_from_walk(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalk(RobotPlayEnvCfgWallCrossThinFirst):
    rewards: WallCrossFromWalkRewardsCfg = WallCrossFromWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _start_from_walk(self)


@configclass
class WallCrossLandRewardsCfg(WallCrossFromWalkRewardsCfg):
    """FromWalk (run11), and a fall after the success judgement costs 20 instead of 4 (run12). The
    walking expert takes over after the landing, so the robot has to stay on its feet: in run11's
    evaluation from the walk only 5137 of 15814 "crossed" episodes were still up at the end of the
    2 s after success, and in the full walk-rise-cross-walk sequence the largest share of falls
    (163-187 of 512) came in that window, before the hand-back."""

    post_success_fall = RewTerm(
        func=mdp.after_success,
        weight=-800.0,
        params={"term": mdp.terminated_by, "term_params": {"term_keys": ["base_contact", "collapsed"]}},
    )


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLand(RobotEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandRewardsCfg = WallCrossLandRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLand(RobotPlayEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandRewardsCfg = WallCrossLandRewardsCfg()


LANDING_PITCH_RATE_WEIGHT = float(os.environ.get("WALLCROSS_LANDING_PITCH_W", "-1.0"))


@configclass
class WallCrossLandPitchRewardsCfg(WallCrossLandRewardsCfg):
    """FromWalkLand (run12), and the pitch rate charged on the far side of the wall until success
    (run13, ``mdp.landing_pitch_rate``)."""

    landing_pitch_rate = RewTerm(func=mdp.landing_pitch_rate, weight=LANDING_PITCH_RATE_WEIGHT)


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandPitch(RobotEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandPitchRewardsCfg = WallCrossLandPitchRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandPitch(RobotPlayEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandPitchRewardsCfg = WallCrossLandPitchRewardsCfg()


@configclass
class WallCrossLandAllRewardsCfg(WallCrossLandRewardsCfg):
    """FromWalkLand (run12), and a fall between the jump and success costs 20 instead of 4 too
    (run13), so every fall now costs 20. run12 on thin walls from the walk: 15-23% of the jumps still
    fell before success at every height from 0.3 to 0.8 m, 69% of them already past the wall
    (nose-first landings). Budget on run12: crossed +10.2, fell after the jump -4.5, held at the wall
    until the timeout -19.2, fell before the jump -27.6 -- so -20 here puts a fall at the wall level
    with standing there, not below it.

    Checked and not used: charging the pitch rate on the far side (``mdp.landing_pitch_rate``). On
    run12 it cost the crossings more (-7.3 points at weight -1) than the falls (-4.7)."""

    crossing_fall = RewTerm(
        func=mdp.during_crossing,
        weight=-800.0,
        params={"term": mdp.terminated_by, "term_params": {"term_keys": ["base_contact", "collapsed"]}},
    )


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAll(RobotEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandAllRewardsCfg = WallCrossLandAllRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll(RobotPlayEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandAllRewardsCfg = WallCrossLandAllRewardsCfg()


@configclass
class WallCrossLandAllTimeoutRewardsCfg(WallCrossLandAllRewardsCfg):
    """FromWalkLandAll (run13), and ending at the wall by the timeout costs 10 more (run14). run13's
    budget had standing at the wall until the timeout (-11.5) above falling after the jump (-15.6),
    and run13 learned that: timeouts at the wall 111 -> 269 (thin-first walls from the walk), and in
    the walk-rise-cross-walk sequence 78 of 512 jumped and then stood 0.46 m (p50) short of the wall
    (run12: 23). With -10 an attempt that falls is better than not trying."""

    wall_timeout_penalty = RewTerm(func=mdp.terminated_by, weight=-500.0, params={"term_keys": ["wall_timeout"]})


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllTimeout(RobotEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandAllTimeoutRewardsCfg = WallCrossLandAllTimeoutRewardsCfg()


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllTimeout(RobotPlayEnvCfgWallCrossThinFirstFromWalk):
    rewards: WallCrossLandAllTimeoutRewardsCfg = WallCrossLandAllTimeoutRewardsCfg()


def _add_heightmap(cfg):
    """Height-map variant: the wall ahead (``mdp.wall_heightmap``, 155 columns) appended to the actor's
    and the critic's observation, after the unified layout. The blind checkpoint carries over with
    zero weights on the new columns (``scripts/rsl_rl/widen_wallcross_heightmap.py``)."""
    cfg.observations.policy.wall_heightmap = ObsTerm(
        func=mdp.wall_heightmap, clip=(-1.0, 2.0), noise=Unoise(n_min=-0.02, n_max=0.02)
    )
    cfg.observations.critic.wall_heightmap = ObsTerm(func=mdp.wall_heightmap, clip=(-1.0, 2.0))
    cfg.events.assert_layout = EventTerm(func=mdp.assert_observation_layout_with_heightmap, mode="startup")


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM(RobotEnvCfgWallCrossThinFirstFromWalkLandAll):
    """run13's task with the height map (HM1)."""

    def __post_init__(self):
        super().__post_init__()
        _add_heightmap(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll):
    def __post_init__(self):
        super().__post_init__()
        _add_heightmap(self)


def _add_speed_limit_dr(cfg):
    """Randomize PhysX's joint speed limits per episode (``mdp.randomize_joint_velocity_limit``)."""
    cfg.events.joint_speed_limit = EventTerm(func=mdp.randomize_joint_velocity_limit, mode="reset")


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllVL(RobotEnvCfgWallCrossThinFirstFromWalkLandAll):
    """run13's task with the joint speed limits randomized (blind, run15)."""

    def __post_init__(self):
        super().__post_init__()
        _add_speed_limit_dr(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllVL(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll):
    def __post_init__(self):
        super().__post_init__()
        _add_speed_limit_dr(self)


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMVL(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    """The height-map task with the joint speed limits randomized (HM2)."""

    def __post_init__(self):
        super().__post_init__()
        _add_speed_limit_dr(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMVL(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    def __post_init__(self):
        super().__post_init__()
        _add_speed_limit_dr(self)


def _add_takeoff_kick(cfg):
    """Landing practice: a random base velocity at the first all-feet-off step after the cue
    (``mdp.takeoff_kick``, every control step)."""
    cfg.events.takeoff_kick = EventTerm(func=mdp.takeoff_kick, mode="interval", interval_range_s=(0.02, 0.02))


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMKick(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    """The height-map task with landing practice (HM4)."""

    def __post_init__(self):
        super().__post_init__()
        _add_takeoff_kick(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMKick(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    def __post_init__(self):
        super().__post_init__()
        _add_takeoff_kick(self)


def _add_descent_kick(cfg):
    """Landing practice after the apex (``mdp.descent_kick``, every control step)."""
    cfg.events.descent_kick = EventTerm(func=mdp.descent_kick, mode="interval", interval_range_s=(0.02, 0.02))


@configclass
class RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMDrop(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    """The height-map task with a random kick on the way down (HM5)."""

    def __post_init__(self):
        super().__post_init__()
        _add_descent_kick(self)


@configclass
class RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMDrop(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM):
    def __post_init__(self):
        super().__post_init__()
        _add_descent_kick(self)


def _add_no_cap(cfg):
    """Lift PhysX's joint speed caps to calf 35 / hip-thigh 45 rad/s every episode: Isaac's take-off then
    matches MuJoCo's within 5% (2026-10-04 open-loop replay), so the policy trains in MuJoCo-like
    push-off physics."""
    cfg.events.joint_speed_limit = EventTerm(
        func=mdp.randomize_joint_velocity_limit, mode="reset",
        params={"calf_range": (35.0, 35.0), "hip_thigh_range": (45.0, 45.0)},
    )


def _make(base, *mods):
    class _Cfg(base):
        def __post_init__(self):
            super().__post_init__()
            for m in mods:
                m(self)
    return configclass(_Cfg)


RobotEnvCfgWallCrossHMNoCap = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_no_cap)
RobotPlayEnvCfgWallCrossHMNoCap = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_no_cap)
RobotEnvCfgWallCrossHMDropNoCap = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_descent_kick, _add_no_cap)
RobotPlayEnvCfgWallCrossHMDropNoCap = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_descent_kick, _add_no_cap)
RobotEnvCfgWallCrossHMDropVL = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_descent_kick, _add_speed_limit_dr)
RobotPlayEnvCfgWallCrossHMDropVL = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_descent_kick, _add_speed_limit_dr)


def _add_approach_dr(cfg):
    """Randomize the state the robot arrives at the wall in (HM8, 2026-10-06).

    The MuJoCo check found the remaining sim2sim gap in the last ~0.7 s of the hind-leg approach: from
    the same Isaac state MuJoCo arrives at the cue farther (gap 0.29 vs 0.20 m) and slower (0.96 vs
    1.17 m/s) with a different pitch rate, and the fine-tuned policies (HM4-HM6) needed the precise
    Isaac arrival. Widen the cue distance (0.3-0.7 -> 0.15-0.8 m) and the approach command
    (0.5-0.8 -> 0.4-1.0 m/s), and push the base every 1.5-3 s (+-0.3 m/s) instead of every 5-10 s."""
    cfg.commands.wallcross.trigger_gap_range = (0.15, 0.8)
    for r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        r.lin_vel_x = (0.4, 1.0)
    cfg.events.push_robot.interval_range_s = (1.5, 3.0)
    cfg.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


RobotEnvCfgWallCrossHMRobust = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_approach_dr)
RobotPlayEnvCfgWallCrossHMRobust = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_approach_dr)
RobotEnvCfgWallCrossHMVLRobust = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_speed_limit_dr, _add_approach_dr)
RobotPlayEnvCfgWallCrossHMVLRobust = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM, _add_speed_limit_dr, _add_approach_dr)


def _add_mid360(cfg):
    """MID-360 variant (2026-10-07): the actor reads the height map the robot itself can build.

    ``wallcross.mid360``: the real MID-360 scan sequence cast from the real chin mount, binned into the
    29 x 21 grid of 5 cm the robot-side ``heightmap_generator`` publishes, with ort/feat/mid360's noise
    and hold. The analytic ``wall_heightmap`` (what no sensor can measure: the true wall, also behind
    it) moves to the critic only. Started from run13 widened by zero columns
    (``scripts/rsl_rl/widen_wallcross_mid360.py``)."""
    from unitree_rl_lab.tasks.wallcross.mid360 import mid360_map_term, mid360_scanner_cfg

    cfg.scene.mid360_scanner = mid360_scanner_cfg()
    cfg.observations.policy.mid360_map = mid360_map_term(noise=True)
    cfg.observations.critic.wall_heightmap = ObsTerm(func=mdp.wall_heightmap, clip=(-1.0, 2.0))
    cfg.events.assert_layout = EventTerm(func=mdp.assert_observation_layout_with_mid360, mode="startup")


RobotEnvCfgWallCrossMid360 = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAll, _add_mid360)
RobotPlayEnvCfgWallCrossMid360 = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll, _add_mid360)


def _blank_mid360(cfg):
    """Evaluation only: the actor's MID-360 map replaced by flat ground (the map's fill value) on every
    cell, to measure how much of the Mid360 policy's result comes from the map itself."""
    from unitree_rl_lab.tasks.wallcross.mid360 import MAP_FLAT_FILL, MID360_MAP_DIM

    cfg.observations.policy.mid360_map = ObsTerm(
        func=mdp.constant_columns, params={"value": MAP_FLAT_FILL, "dim": MID360_MAP_DIM}
    )


RobotEnvCfgWallCrossMid360Blank = _make(RobotEnvCfgWallCrossMid360, _blank_mid360)
RobotPlayEnvCfgWallCrossMid360Blank = _make(RobotPlayEnvCfgWallCrossMid360, _blank_mid360)


def _add_wall_body_contact(cfg):
    """Mid360-Clear (2026-10-08): the body must not touch the wall on the way over.

    ``base_contact`` exempts contact on the wall, so the Mid360 policy learned to clear it with
    ~0.07 m to spare and lean the body on the top edge. In MuJoCo that contact tips it forward: it
    rose only 0.36 m after the cue (HM3: 0.47, needed for 0.15 m over a 0.65 m wall: 0.43) and fell
    on landing in 32 of 50 trials (stood 2/50, HM3 19/50). A steel plate will not behave like either
    simulator's contact. So base, hips and thighs touching anything from the jump until success cost
    0.6 points per body per step; the feet (and calves) may still use the wall."""
    cfg.rewards.wall_body_contact = RewTerm(
        func=mdp.during_crossing,
        weight=-30.0,
        params={
            "term": mdp.undesired_contacts,
            "term_params": {
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]),
            },
        },
    )


RobotEnvCfgWallCrossMid360Clear = _make(RobotEnvCfgWallCrossMid360, _add_wall_body_contact)
RobotPlayEnvCfgWallCrossMid360Clear = _make(RobotPlayEnvCfgWallCrossMid360, _add_wall_body_contact)


def _gate_walls(cfg):
    """Mid360-Clear-Gate (2026-10-11): spend the tiles on the gate's walls.

    The gate is MuJoCo, thin walls of 0.60, 0.70 and 0.80 m, each crossed and stood on in 70% of fresh
    seeds. Clear model_21200 gives 66/65/44%: at 0.80 m it rises 0.50 m (0.585 needed) and hangs on the
    top. In B' a frontier tile is drawn from the 4 cm under the frontier and half the tiles go to easy
    walls, so the 0.60-0.76 m thin walls appear only as easy draws. Draw frontier tiles from the 25 cm
    under the frontier (0.55-0.80 m at the top row) and give easy walls 20% instead of 50%."""
    walls = cfg.scene.terrain.terrain_generator.sub_terrains["walls"]
    walls.easy_fraction = 0.2
    walls.frontier_band_height = 0.25


RobotEnvCfgWallCrossMid360ClearGate = _make(RobotEnvCfgWallCrossMid360Clear, _gate_walls)
RobotPlayEnvCfgWallCrossMid360ClearGate = _make(RobotPlayEnvCfgWallCrossMid360Clear, _gate_walls)


def _start_spread(cfg):
    """Fine-tuning a finished policy: start the envs on every row, not on row 0. A resume rebuilds the
    curriculum, so from row 0 the first ~700 iterations go back over the low walls (Clear: level 0 -> 52)."""
    cfg.scene.terrain.max_init_terrain_level = None


# Mid360-Robust (2026-10-11): fine-tune Mid360 model_20994 for MuJoCo with a rigid wall
# (`--isaac_wall --wall_solref 0.005 1`), where it crosses and walks on 75/78/67% at 0.60/0.70/0.80 m.
# What is left is the hind-stance approach (no-fall hangs at the wall ~17% at every height, falls during
# the rise ~5%) and landing at 0.80 m; reproduce_logs/h080_diag found the approach state and the joint
# speed caps as the remaining Isaac/MuJoCo differences once the wall is rigid.
RobotEnvCfgWallCrossMid360Robust = _make(RobotEnvCfgWallCrossMid360, _add_approach_dr, _add_speed_limit_dr, _start_spread)
RobotPlayEnvCfgWallCrossMid360Robust = _make(RobotPlayEnvCfgWallCrossMid360, _add_approach_dr, _add_speed_limit_dr)


def _add_safe_landing(cfg):
    """Mid360-SafeLand (2026-10-11): a landing the real robot survives.

    Mid360 model_20994 (MuJoCo, rigid wall, deploy hand-over: crossed and walked on 91/92/80% at
    0.60/0.70/0.80 m) lands with the front-right leg thrown forward at its limits and the shin on the
    floor (Isaac 151/159 landings, MuJoCo 1100-1700 N), the hips at their +-1.047 rad stops in almost
    every landing (MuJoCo goes 0.1-0.27 rad past them), and the chin MID-360 down to 5-9 cm above the
    floor (reproduce_logs/landing_safety). From the jump on:
    * thighs, calves and head touching anything but the wall: -8 per body per second;
    * the joint-limit term (soft limits at 0.9 of the range) again at -10, on top of the -10 everywhere;
    * the MID-360 below 0.15 m: -100 per metre per second.
    Smoke test from model_20994 at -15/-20: the first two cost a crossing about 24 points, more than the
    crossing earns (22), so they were halved (about 12) to keep crossing worth more than holding still."""
    cfg.rewards.landing_floor_contact = RewTerm(
        func=mdp.floor_contact_off_wall,
        weight=-8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", "Head_.*"])},
    )
    cfg.rewards.landing_dof_pos_limits = RewTerm(
        func=mdp.after_jump, weight=-10.0, params={"term": mdp.joint_pos_limits},
    )
    cfg.rewards.landing_sensor_height = RewTerm(func=mdp.sensor_height_deficit, weight=-100.0)


RobotEnvCfgWallCrossMid360SafeLand = _make(RobotEnvCfgWallCrossMid360, _add_safe_landing, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafeLand = _make(RobotPlayEnvCfgWallCrossMid360, _add_safe_landing)


LANDING_LEG_BODIES = [".*_thigh", ".*_calf", "Head_.*"]


def _safe_landing_fall(cfg):
    """SafeLand arm A (2026-10-11): a thigh, calf or head on the floor after the landing costs what a fall
    costs (16, once per episode, ``mdp.floor_contact_once``), without ending the episode.

    safeland_run1 charged it per second at -8 and it was a few percent of a crossing's return: the
    landing barely changed (MuJoCo legs on the floor 96-98% -> 62-98%). The fall price is what moved the
    landings before (run12/run13: -20 per fall). Tried first as a termination: 97% of model_20994's
    episodes ended on it before the success, the crossing reward vanished. Joint limits after the jump:
    -16 more; MID-360 below 0.15 m: -120 per metre per second."""
    cfg.rewards.landing_floor_once = RewTerm(
        func=mdp.floor_contact_once, weight=-16.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LANDING_LEG_BODIES)},
    )
    cfg.rewards.landing_dof_pos_limits = RewTerm(func=mdp.after_jump, weight=-16.0, params={"term": mdp.joint_pos_limits})
    cfg.rewards.landing_sensor_height = RewTerm(func=mdp.sensor_height_deficit, weight=-120.0)
    # 20-iteration smoke run from model_20994: with the crossing at 20 these cost 1.8x what it paid.
    cfg.rewards.wall_success.weight = 40.0


def _safe_landing_strong(cfg):
    """SafeLand arm B (2026-10-11): safeland_run1's per-second terms 2-2.4x (floor contact -16 per body,
    joint limits -24 more, MID-360 -120) and the crossing paid 40 instead of 20. Sized on a 20-iteration
    smoke run from model_20994: at 5x (-40/-60/-300) the three cost 2.8x what the crossing paid, enough
    to make not jumping the better deal; at these weights they cost about what the crossing pays, so
    a landing that avoids them is worth as much again as the crossing."""
    cfg.rewards.landing_floor_contact = RewTerm(
        func=mdp.floor_contact_off_wall, weight=-16.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LANDING_LEG_BODIES), "past_wall_only": True},
    )
    cfg.rewards.landing_dof_pos_limits = RewTerm(func=mdp.after_jump, weight=-24.0, params={"term": mdp.joint_pos_limits})
    cfg.rewards.landing_sensor_height = RewTerm(func=mdp.sensor_height_deficit, weight=-120.0)
    cfg.rewards.wall_success.weight = 40.0


def _scale_safety(factor):
    def mod(cfg):
        for key in ("landing_floor_contact", "landing_floor_once", "landing_dof_pos_limits", "landing_sensor_height"):
            if getattr(cfg.rewards, key, None) is not None:
                getattr(cfg.rewards, key).weight *= factor
    return mod


RobotEnvCfgWallCrossMid360SafeFall = _make(RobotEnvCfgWallCrossMid360, _safe_landing_fall, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafeFall = _make(RobotPlayEnvCfgWallCrossMid360, _safe_landing_fall)
RobotEnvCfgWallCrossMid360SafeStrong = _make(RobotEnvCfgWallCrossMid360, _safe_landing_strong, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafeStrong = _make(RobotPlayEnvCfgWallCrossMid360, _safe_landing_strong)
# Stage 2 of either arm (only if stage 1 misses the safety targets): the per-second terms doubled.
RobotEnvCfgWallCrossMid360SafeFall2 = _make(RobotEnvCfgWallCrossMid360, _safe_landing_fall, _scale_safety(2.0), _start_spread)
RobotEnvCfgWallCrossMid360SafeStrong2 = _make(RobotEnvCfgWallCrossMid360, _safe_landing_strong, _scale_safety(2.0), _start_spread)


def _safe_landing_pose(cfg):
    """SafeLand arm C (2026-10-12): pay for the standing pose after the landing (``mdp.landing_pose``,
    +10 per second at the default pose) instead of charging the bad one hard. Kept from safeland_run1
    (which kept its crossings): legs on the floor past the wall -4 per body per second, MID-360 below
    0.15 m -100. No extra joint-limit term: in arms A/B it was the largest of the charges (-0.3 to -0.6
    a crossing in the log, against 0.1 for the floor contact)."""
    # sigma 4: at sigma 1 the 20-iteration smoke run paid 0.02 (exp(-5) at model_20994's landings) -- no gradient.
    cfg.rewards.landing_pose = RewTerm(func=mdp.landing_pose, weight=10.0, params={"sigma": 4.0})
    cfg.rewards.landing_floor_contact = RewTerm(
        func=mdp.floor_contact_off_wall, weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LANDING_LEG_BODIES), "past_wall_only": True},
    )
    cfg.rewards.landing_sensor_height = RewTerm(func=mdp.sensor_height_deficit, weight=-100.0)


RobotEnvCfgWallCrossMid360SafePose = _make(RobotEnvCfgWallCrossMid360, _safe_landing_pose, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafePose = _make(RobotPlayEnvCfgWallCrossMid360, _safe_landing_pose)


def _safe_landing_pose_limits(cfg):
    """SafeLand arm D (2026-10-13): arm C, the pose reward 1.5x (+15) and the joint-limit term again from
    the jump on (-10). Arm C model_22400 kept its crossings (MuJoCo 98/94/83%) and took the legs off the
    floor in part (98 -> 57-76%), but every landing still drove a joint past its limit: the left hips
    (FL/RL, 0.16-0.19 rad past) and all four calves fully stretched (0.11-0.18 rad past). Started from
    22400, where crossing is already paid by the pose reward, so the limit term cannot be dodged by
    not crossing as it was in arms A/B."""
    cfg.rewards.landing_pose.weight = 15.0
    cfg.rewards.landing_dof_pos_limits = RewTerm(func=mdp.after_jump, weight=-10.0, params={"term": mdp.joint_pos_limits})


RobotEnvCfgWallCrossMid360SafePoseLim = _make(RobotEnvCfgWallCrossMid360, _safe_landing_pose, _safe_landing_pose_limits, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafePoseLim = _make(RobotPlayEnvCfgWallCrossMid360, _safe_landing_pose, _safe_landing_pose_limits)


# Blind SafePose (2026-10-13): arm C's landing-pose reward on the blind policy. Started from run15's
# model_20100 (VL task: joint speed limits randomized), the blind checkpoint that crosses and walks on best in
# MuJoCo with the rigid wall and the deploy hand-over (93/73% at 0.60/0.80 m, 30 trials; run13's final 23-27%).
RobotEnvCfgWallCrossBlindSafePose = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllVL, _safe_landing_pose, _start_spread)
RobotPlayEnvCfgWallCrossBlindSafePose = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll, _safe_landing_pose)


def _soft_landing_force(cfg):
    """Blind SafeSoft (2026-10-13): charge how hard the legs strike the floor, not whether they touch.
    Blind SafePose model_22000 still grazes the floor in 53 of 60 landings and the strikes are hard
    (1172 N at 0.60 m, 2115 N at 0.80 m in MuJoCo, 8-14 body weights), where the Mid360 SafePose landing
    got its strikes down to 228 N. The per-body count is replaced by the off-wall contact force of the
    thighs, calves and head in body weights, past the wall: -10 per body weight per second (a 0.06 s
    strike of 8 body weights costs about 5, of 14 about 8, against 20 for the crossing)."""
    cfg.rewards.landing_floor_contact = RewTerm(
        func=mdp.floor_contact_off_wall, weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LANDING_LEG_BODIES),
                "past_wall_only": True, "body_weights": True},
    )


RobotEnvCfgWallCrossBlindSafeSoft = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllVL, _safe_landing_pose, _soft_landing_force, _start_spread)
RobotPlayEnvCfgWallCrossBlindSafeSoft = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll, _safe_landing_pose, _soft_landing_force)


def _true_mount(cfg):
    """Mount fix (2026-10-14): ``sensor_height_deficit`` defaulted to (0.295, 0, -0.075), the point in the
    MuJoCo go2.xml, but the MID-360 sits at the real-robot mount the map is built from (``MID360_MOUNT_POS``,
    (0.287, 0, -0.137), h.mori's live launch) -- 6 cm lower. Re-measured at the mount, Mid360 SafePose
    model_22400 brings the sensor to p10 3.8 cm above the floor at 0.80 m (20% of landings below 5 cm),
    not the 7.0 cm reported. Same term and weight, charged at the real mount."""
    from unitree_rl_lab.tasks.wallcross.mid360 import MID360_MOUNT_POS

    cfg.rewards.landing_sensor_height.params["offset"] = tuple(MID360_MOUNT_POS)


RobotEnvCfgWallCrossMid360SafeMount = _make(RobotEnvCfgWallCrossMid360, _safe_landing_pose, _true_mount, _start_spread)
RobotPlayEnvCfgWallCrossMid360SafeMount = _make(RobotPlayEnvCfgWallCrossMid360, _safe_landing_pose, _true_mount)
RobotEnvCfgWallCrossBlindSafeMount = _make(RobotEnvCfgWallCrossThinFirstFromWalkLandAllVL, _safe_landing_pose, _soft_landing_force, _true_mount, _start_spread)
RobotPlayEnvCfgWallCrossBlindSafeMount = _make(RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll, _safe_landing_pose, _soft_landing_force, _true_mount)
