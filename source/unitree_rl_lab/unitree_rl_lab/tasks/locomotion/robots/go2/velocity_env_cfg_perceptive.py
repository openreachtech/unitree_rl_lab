"""Four ways to give the walking policy a height map, and one that gives it none.

The Blind-GRU lineage learned to cross rough ground, stairs and walls on
proprioception alone. This asks what a height map buys on top of that, and how much of
the answer depends on the map being clean. Four arms, identical in every respect but
the actor's exteroceptive input:

    blind     proprioception only                       the baseline
    belief    the LiDAR grid, denoised by the frozen
              terrain encoder                           the point of building it
    noisy     the LiDAR grid, raw                       what the sensor actually gives
    clean     the top-down scan                         an upper bound, not deployable

``belief`` and ``noisy`` see *the same observation*: 45 proprioceptive values and a
388-cell grid from the fan. They differ only in whether the encoder has already run on
it. That makes their gap a clean measurement of what the encoder adds, with the policy
free in both cases to do its own denoising through its GRU -- the harder and fairer
version of the question than giving the raw arm no memory to work with.

Where the exteroception enters
------------------------------
Through the GRU, not around it: the map joins proprioception in the policy observation
and ``ActorCriticRecurrent`` feeds the pair to the recurrence. That is Miki et al.'s
arrangement, it needs no network code at all, and it is what keeps the ``noisy`` arm
honest -- routed past the recurrence it would have no way to integrate a grid that is
42% unmeasured at any instant, and would lose by construction rather than on merit.

Observations are normalised (running mean and variance, as the paper does). The height
grid arrives in metres, where terrain varies by hundredths while the proprioceptive
terms have been scaled to order one; without normalisation the map would enter the GRU
roughly twenty times quieter than everything beside it.

The critic
----------
All four arms get the clean 388-cell scan as privileged input, on the LiDAR's crop so
the cells line up with what the actor sees. That is a change from the blind lineage,
whose critic had only foot-local terrain -- which is why ``blind`` is retrained here
rather than compared against the existing numbers. A difference in ``terrain_levels``
has to come from the actor's input, and it cannot if the critics differ.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    GO2_HEIGHT_SCAN_CENTER_X,
    GO2_HEIGHT_SCAN_CENTER_Y,
    GO2_HEIGHT_SCAN_OFFSET,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    GO2_LIDAR_BODY_HALF_EXTENT_X,
    GO2_LIDAR_BODY_HALF_EXTENT_Y,
    GO2_LIDAR_SCANNER_CFG,
    LIDAR_HEIGHT_SCAN_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind import (
    CriticCfgGo2,
    ObservationsCfgGo2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase1 import (
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase2 import (
    RobotEnvCfgPhase2,
    RobotSceneCfgPhase2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase3 import (
    PHASE3_TERRAIN_CFG_FLOATING,
    RobotEnvCfgPhase3BalanceMatched,
    RobotSceneCfgPhase3Balance,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_blind_phase4 import (
    PHASE4_TERRAIN_CFG,
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# The clean grid, cropped the way the LiDAR's is (0.40 / 0.30 rather than the top-down
# tasks' 0.30 / 0.20) so all four arms address the same 388 cells. The fan's blind cone
# is what sets that crop; matching it is what makes "same grid, different provenance"
# true rather than approximately true.
_CLEAN_HEIGHT_SCAN_PARAMS = {
    "sensor_cfg": SceneEntityCfg("height_scanner"),
    "offset": GO2_HEIGHT_SCAN_OFFSET,
    "resolution": HEIGHT_SCAN_RESOLUTION,
    "size": HEIGHT_SCAN_SIZE,
    "scanner_offset_xy": (GO2_HEIGHT_SCAN_CENTER_X, GO2_HEIGHT_SCAN_CENTER_Y),
    "exclude_half_extent_x": GO2_LIDAR_BODY_HALF_EXTENT_X,
    "exclude_half_extent_y": GO2_LIDAR_BODY_HALF_EXTENT_Y,
}

CLEAN_HEIGHT_SCAN_CFG = ObsTerm(
    func=mdp.height_scan_excluding_body,
    params=_CLEAN_HEIGHT_SCAN_PARAMS,
    clip=(-1.0, 5.0),
)
"""388 cells of true terrain. Privileged: no sensor produces this."""


@configclass
class PerceptiveCriticCfg(CriticCfgGo2):
    """The blind critic plus the clean grid, shared by all four arms.

    The blind lineage dropped the body-centred scan on the argument that a value
    function only has to explain why the last few steps went the way they did, and the
    per-foot rings cover that. It goes back in here for a different reason: with the
    actor's input under test, the critic has to be the one thing that is definitely
    identical across arms.
    """

    height_scan = CLEAN_HEIGHT_SCAN_CFG


@configclass
class _PerceptivePolicyBase(ObservationsCfgGo2.PolicyCfg):
    """Proprioception, unchanged from the blind policy. Arms add a grid after it.

    Order matters: the frozen encoder in the ``belief`` arm reconstructs these 45
    values itself, and ``mdp/belief_height_map.py`` mirrors this list. Change one and
    change the other.
    """


@configclass
class BlindObsCfg(ObservationsCfgGo2):
    critic: PerceptiveCriticCfg = PerceptiveCriticCfg()


@configclass
class NoisyObsCfg(ObservationsCfgGo2):
    @configclass
    class PolicyCfg(_PerceptivePolicyBase):
        height_scan = LIDAR_HEIGHT_SCAN_CFG

    policy: PolicyCfg = PolicyCfg()
    critic: PerceptiveCriticCfg = PerceptiveCriticCfg()


@configclass
class CleanObsCfg(ObservationsCfgGo2):
    @configclass
    class PolicyCfg(_PerceptivePolicyBase):
        height_scan = CLEAN_HEIGHT_SCAN_CFG

    policy: PolicyCfg = PolicyCfg()
    critic: PerceptiveCriticCfg = PerceptiveCriticCfg()


BELIEF_ENCODER_CHECKPOINT = (
    "logs/terrain_encoder/2026-08-29_22-24-19/encoder_4000.pt"
)
"""The adopted terrain encoder: belief architecture, Phase 2 -> 3 -> 4 then extended,
4000 iterations. Chosen over the convolutional one because accuracy was comparable
while training was 3.8x faster in a 17.8th of the memory -- see
``sandbox/TERRAIN_ENCODER.md``. Frozen here."""


@configclass
class BeliefObsCfg(ObservationsCfgGo2):
    @configclass
    class PolicyCfg(_PerceptivePolicyBase):
        height_scan = ObsTerm(
            func=mdp.BeliefHeightMap,
            params={
                "noisy_map": LIDAR_HEIGHT_SCAN_CFG,
                "checkpoint": BELIEF_ENCODER_CHECKPOINT,
            },
            clip=(-1.0, 5.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: PerceptiveCriticCfg = PerceptiveCriticCfg()


# ===========================================================================
# Scenes. The two arms fed by the fan need it in the scene; the other two do not, and
# leaving it out of those keeps 1080 rays per environment off the bill.
# ===========================================================================
# Phase 3 here runs the floating mix rather than the blind lineage's 20-column
# fixed-width one: 5 columns, 1 : 2 : 2 floating / inverted / floating-inverted. An
# open riser is the case proprioception cannot resolve ahead of contact -- nothing under
# the tread to catch a leg against -- so it is where a height map should pay for itself
# if it pays anywhere.
#
# Substituted at class level, not in __post_init__. RobotEnvCfg.__post_init__ reads the
# terrain_levels curriculum term to decide whether the generator lays its patches out by
# ascending difficulty, and sets the flag on whatever generator is present when it runs;
# swapping afterwards leaves a generator at its curriculum=False default and turns the
# rows into noise. That has already cost this repo one round of play configs.
@configclass
class FloatingSceneCfgPhase3(RobotSceneCfgPhase3Balance):
    terrain = RobotSceneCfgPhase3Balance().terrain.replace(
        terrain_generator=PHASE3_TERRAIN_CFG_FLOATING
    )


@configclass
class _LidarScene:
    lidar_scanner: RayCasterCfg = GO2_LIDAR_SCANNER_CFG


@configclass
class LidarSceneCfgPhase1(RobotSceneCfgPhase1, _LidarScene):
    pass


@configclass
class LidarSceneCfgPhase2(RobotSceneCfgPhase2, _LidarScene):
    pass


@configclass
class LidarSceneCfgPhase3(FloatingSceneCfgPhase3, _LidarScene):
    pass


@configclass
class LidarSceneCfgPhase4(RobotSceneCfgPhase4, _LidarScene):
    pass


def _finish(env_cfg, lidar: bool) -> None:
    """Shared tail: tick the fan, and stop the top-down scanner drawing itself."""
    if lidar:
        env_cfg.scene.lidar_scanner.update_period = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene.height_scanner.debug_vis = False


# ===========================================================================
# Sixteen environments: four arms across four phases. Written out rather than
# generated, because entry points assembled from f-strings have twice hidden live code
# from this repo's dead-code scans, and the classes are three lines each.
#
# Each inherits its phase's blind config, so terrain, rewards, curriculum, events and
# terminations are whatever that phase settled on. Only the observations change -- and
# for two of the arms, the presence of the fan in the scene.
# ===========================================================================
def _arm(base, scene_cls, obs_cls, lidar: bool):
    """Build one arm-phase config. Returns a class, not an instance."""

    @configclass
    class _Cfg(base):
        scene = scene_cls(num_envs=4096, env_spacing=2.5)
        observations = obs_cls()

        def __post_init__(self):
            super().__post_init__()
            _finish(self, lidar)

    return _Cfg


# -- blind: the baseline, retrained with the shared critic -------------------
BlindEnvCfgPhase1 = _arm(RobotEnvCfgPhase1, RobotSceneCfgPhase1, BlindObsCfg, False)
BlindEnvCfgPhase2 = _arm(RobotEnvCfgPhase2, RobotSceneCfgPhase2, BlindObsCfg, False)
BlindEnvCfgPhase3 = _arm(RobotEnvCfgPhase3BalanceMatched, FloatingSceneCfgPhase3, BlindObsCfg, False)
BlindEnvCfgPhase4 = _arm(RobotEnvCfgPhase4, RobotSceneCfgPhase4, BlindObsCfg, False)

# -- belief: the fan's grid, denoised by the frozen encoder ------------------
BeliefEnvCfgPhase1 = _arm(RobotEnvCfgPhase1, LidarSceneCfgPhase1, BeliefObsCfg, True)
BeliefEnvCfgPhase2 = _arm(RobotEnvCfgPhase2, LidarSceneCfgPhase2, BeliefObsCfg, True)
BeliefEnvCfgPhase3 = _arm(RobotEnvCfgPhase3BalanceMatched, LidarSceneCfgPhase3, BeliefObsCfg, True)
BeliefEnvCfgPhase4 = _arm(RobotEnvCfgPhase4, LidarSceneCfgPhase4, BeliefObsCfg, True)

# -- noisy: the fan's grid, raw ---------------------------------------------
NoisyEnvCfgPhase1 = _arm(RobotEnvCfgPhase1, LidarSceneCfgPhase1, NoisyObsCfg, True)
NoisyEnvCfgPhase2 = _arm(RobotEnvCfgPhase2, LidarSceneCfgPhase2, NoisyObsCfg, True)
NoisyEnvCfgPhase3 = _arm(RobotEnvCfgPhase3BalanceMatched, LidarSceneCfgPhase3, NoisyObsCfg, True)
NoisyEnvCfgPhase4 = _arm(RobotEnvCfgPhase4, LidarSceneCfgPhase4, NoisyObsCfg, True)

# -- clean: the true terrain, an upper bound --------------------------------
CleanEnvCfgPhase1 = _arm(RobotEnvCfgPhase1, RobotSceneCfgPhase1, CleanObsCfg, False)
CleanEnvCfgPhase2 = _arm(RobotEnvCfgPhase2, RobotSceneCfgPhase2, CleanObsCfg, False)
CleanEnvCfgPhase3 = _arm(RobotEnvCfgPhase3BalanceMatched, FloatingSceneCfgPhase3, CleanObsCfg, False)
CleanEnvCfgPhase4 = _arm(RobotEnvCfgPhase4, RobotSceneCfgPhase4, CleanObsCfg, False)


# ===========================================================================
# Play. Each phase's course cut down to something watchable, with the difficulty
# spread over every row and frozen there so nothing shifts while it is being looked at,
# and the command range opened to its limit from the first step.
#
# Phase 3 gets one column per sub-terrain: proportions are equalised because columns are
# handed out by cumulative proportion, and 20/40/40 over three columns would drop the
# floating ascent entirely. Phase 4 keeps its solid | floating pair, walls centred on
# 5 / 10 / 15 / 20 cm.
#
# For the two arms fed by the fan, its markers are switched on -- green where a beam
# landed this step, red where the cell is holding an older reading. Note what that does
# and does not show: it is the *input*. The belief arm's policy reads the encoder's
# reconstruction of it, which is not drawn here; ``play_terrain_encoder.py`` is the
# script that draws that, in blue.
# ===========================================================================
PLAY_TERRAIN_CFG_P3_FLOATING = PHASE3_TERRAIN_CFG_FLOATING.replace(
    num_rows=3,
    num_cols=3,
    sub_terrains={
        name: cfg.replace(proportion=1.0, step_height_range=(0.025, 0.175))
        for name, cfg in PHASE3_TERRAIN_CFG_FLOATING.sub_terrains.items()
    },
)
"""3 x 3: floating | inverted | floating inverted, rows centred on 5 / 10 / 15 cm."""

PLAY_TERRAIN_CFG_P4 = PHASE4_TERRAIN_CFG.replace(
    num_rows=4,
    num_cols=2,
    sub_terrains={
        name: cfg.replace(
            proportion=1.0,
            wall_height_range=(0.025, 0.225),
            wall_thickness_range=(0.05, 0.05),
        )
        for name, cfg in PHASE4_TERRAIN_CFG.sub_terrains.items()
    },
)
"""2 x 4: solid wall | floating tread, rows centred on 5 / 10 / 15 / 20 cm, 5 cm thick."""


def _play(train_cls, terrain_cfg, lidar: bool):
    """Turn one training config into its viewing counterpart."""

    @configclass
    class _Cfg(train_cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 24
            # Carry curriculum=True through the swap: __post_init__ has already set it on
            # the generator that was there, and TerrainGeneratorCfg defaults it to False,
            # which would sample difficulty per tile and make the rows meaningless.
            self.scene.terrain.terrain_generator = terrain_cfg.replace(curriculum=True)
            self.scene.terrain.max_init_terrain_level = None
            self.curriculum.terrain_levels = None
            self.curriculum.lin_vel_cmd_levels = None
            self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
            if lidar:
                self.observations.policy.height_scan = _visualising(
                    self.observations.policy.height_scan
                )

    return _Cfg


def _visualising(term: ObsTerm) -> ObsTerm:
    """Same term, with the fan drawing itself.

    The values are unchanged -- only ``debug_vis`` moves -- so the policy sees exactly
    what it sees in training. The belief arm wraps the fan rather than being it, so the
    swap goes one level in, on the term it was handed.
    """
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
        PLAY_LIDAR_HEIGHT_SCAN_CFG,
    )

    if term.func is mdp.BeliefHeightMap:
        params = dict(term.params)
        params["noisy_map"] = PLAY_LIDAR_HEIGHT_SCAN_CFG
        return term.replace(params=params)
    return PLAY_LIDAR_HEIGHT_SCAN_CFG


BlindPlayCfgPhase3 = _play(BlindEnvCfgPhase3, PLAY_TERRAIN_CFG_P3_FLOATING, False)
BeliefPlayCfgPhase3 = _play(BeliefEnvCfgPhase3, PLAY_TERRAIN_CFG_P3_FLOATING, True)
NoisyPlayCfgPhase3 = _play(NoisyEnvCfgPhase3, PLAY_TERRAIN_CFG_P3_FLOATING, True)
CleanPlayCfgPhase3 = _play(CleanEnvCfgPhase3, PLAY_TERRAIN_CFG_P3_FLOATING, False)

BlindPlayCfgPhase4 = _play(BlindEnvCfgPhase4, PLAY_TERRAIN_CFG_P4, False)
BeliefPlayCfgPhase4 = _play(BeliefEnvCfgPhase4, PLAY_TERRAIN_CFG_P4, True)
NoisyPlayCfgPhase4 = _play(NoisyEnvCfgPhase4, PLAY_TERRAIN_CFG_P4, True)
CleanPlayCfgPhase4 = _play(CleanEnvCfgPhase4, PLAY_TERRAIN_CFG_P4, False)
