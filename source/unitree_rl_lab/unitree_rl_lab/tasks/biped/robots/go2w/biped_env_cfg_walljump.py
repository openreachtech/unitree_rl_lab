"""Jump a wall from the hind stance: approach on the hind wheels, jump at the wall, land on four.

``feat/wallcross``'s hind-stance wall crossing (Go2-Multitask-WallCross, ThinFirst-FromWalkLandAll
-- the run its MuJoCo evaluation defaults to), on Go2w and this package's circular walls. Started from
``Go2w-Biped-Switch`` (stance on command, and back down onto four wheels when it is released), its
inputs widened for the two jump terms. Started from ``Go2w-Biped`` instead -- never on four wheels --
the policy got over the wall (4.2 of base + 4 wheels past it at iteration 300) and then went into the
front stance on the far side, the one other posture it knew: no success in 300 iterations.

* **Episode**: the hind stance from the first step, at a forward-only command. When the base is
  0.3-0.7 m from the first ring's inner face (or at 8 s), the stance is released and
  ``jump_command`` fires, ``target_height`` = wall height - 0.25 m (base apex 0.15 m over the top).
  Every episode jumps. Success: base 0.2 m and all four wheels past the outer face, at least three
  wheels down, base above 0.30 m, within 15 degrees of level, for 0.3 s.
* **Observation**: ``Go2w-Biped``'s plus ``jump_command`` (4) and ``jump_time`` (1), appended last in
  both groups, so a ``Go2w-Biped`` checkpoint widened by 15 columns (history 3) loads as it is.
* **Rewards**: the stance's (``Go2w-Biped``'s), faded out at the release (``in_stance``), plus the
  crossing, sized as feat/wallcross sized it so that crossing > holding still > falling: a
  potential worth 1 (approach) + 4 (the base rising to 0.15 m over the top) + 4 (base and wheels
  over) + 3 (standing up past it); 20 for the success, once; ``wall_alive`` cancelling the always-on
  penalties after the jump; 20 for any fall, before or after the success (its "LandAll"); 10 for
  running out the 8 s after the jump.
* **Terminations**: a trunk or hip contact ends the episode unless it is on the wall; the base
  under 0.15 m; 2 s after a success; 8 s after the jump without one.
* **Terrain**: one circular ring, three columns 5 / 15 / 25 cm thick, 5 -> 100 cm over 50 rows;
  a success moves a robot up a row, a failure down.

    python scripts/rsl_rl/widen_policy_inputs.py logs/rsl_rl/go2w_biped_switch/<run>/model_2000.pt \\
        logs/rsl_rl/go2w_biped_walljump/0001_init_from_switch/model_0.pt --extra 15
    python scripts/rsl_rl/train.py --task Go2w-Biped-WallJump --headless \\
        --resume --load_run 0001_init_from_switch --checkpoint model_0.pt
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg import (
    BipedCommandsCfg,
    BipedRewardsCfg,
    BipedUnifiedObservationsCfg,
    RobotEnvCfgBiped,
)
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_front import BipedSceneCfg
from unitree_rl_lab.tasks.biped.robots.go2w.biped_env_cfg_switch import STANCE_ONLY_TERMS
from unitree_rl_lab.tasks.biped.terrains import CIRCULAR_WALLS, WALL_TERRAIN_CFG
from unitree_rl_lab.tasks.dynamic import mdp as jump_mdp

APPROACH_SPEED = (0.5, 0.8)
"""feat/wallcross's: the hind stance delivers about 60 % of its command."""

EPISODE_LENGTH_S = 14.0
"""Up to 8 s to the trigger, 8 s after it at most -- in practice the jump comes at 2-4 s."""

WALL_THICKNESSES = (0.05, 0.15, 0.25)
WALL_HEIGHT_RANGE = (0.05, 1.00)
"""To 1 m (first run: 0.70, which model_800 already cleared 100 % of the time without noise)."""
NUM_ROWS = 50
NUM_RINGS = 1
"""One wall per tile: the episode ends at the first crossing, so the second ring was never reached."""

WALLJUMP_TERRAIN_CFG = WALL_TERRAIN_CFG.replace(
    num_rows=NUM_ROWS,
    num_cols=len(WALL_THICKNESSES),
    sub_terrains={
        f"wall_{round(t * 100)}cm": CIRCULAR_WALLS.replace(
            wall_thickness=t, wall_height_range=WALL_HEIGHT_RANGE, num_rings=NUM_RINGS
        )
        for t in WALL_THICKNESSES
    },
)

PLAY_WALL_HEIGHTS = (0.50, 0.75, 1.00)

WALLJUMP_PLAY_TERRAIN_CFG = WALLJUMP_TERRAIN_CFG.replace(
    num_rows=len(PLAY_WALL_HEIGHTS),
    sub_terrains={
        f"wall_{round(t * 100)}cm": CIRCULAR_WALLS.replace(
            wall_thickness=t, row_heights=PLAY_WALL_HEIGHTS, num_rings=NUM_RINGS
        )
        for t in WALL_THICKNESSES
    },
)


@configclass
class WallJumpSceneCfg(BipedSceneCfg):
    terrain = BipedSceneCfg().terrain.replace(
        terrain_type="generator", terrain_generator=WALLJUMP_TERRAIN_CFG, max_init_terrain_level=5
    )


@configclass
class WallJumpCommandsCfg(BipedCommandsCfg):
    jump = jump_mdp.JumpCommandCfg(
        asset_name="robot",
        assist_body_names=["base"],
        auto_trigger=False,
        initial_assist_scale=0.0,
        resampling_time_range=(1.0e9, 1.0e9),
        command_duration_s=0.5,
    )
    walljump = mdp.WallJumpCommandCfg()

    def __post_init__(self):
        super().__post_init__()
        # The hind stance, pinned on until the jump releases it.
        self.handstand.stance_front_probability = None
        self.handstand.balance_stance_population = False
        self.handstand.stance = mdp.STANCE_HIND
        for ranges in (self.base_velocity.ranges, self.base_velocity.limit_ranges):
            ranges.lin_vel_x = APPROACH_SPEED
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
        self.base_velocity.rel_standing_envs = 0.0
        self.base_velocity.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)


@configclass
class WallJumpObservationsCfg(BipedUnifiedObservationsCfg):
    """``Go2w-Biped``'s, with the jump cue appended last."""

    @configclass
    class PolicyCfg(BipedUnifiedObservationsCfg.PolicyCfg):
        jump_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "jump"})
        jump_time = ObsTerm(func=jump_mdp.jump_time_encoding, params={"command_name": "jump"})

    @configclass
    class CriticCfg(BipedUnifiedObservationsCfg.CriticCfg):
        jump_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "jump"})
        jump_time = ObsTerm(func=jump_mdp.jump_time_encoding, params={"command_name": "jump"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


FALL_TERMS = ["base_contact", "collapsed"]


@configclass
class WallJumpRewardsCfg(BipedRewardsCfg):
    # The stance's own terms (Go2w-Biped's) fade out at the release; see __post_init__.
    # The crossing.
    wall_progress = RewTerm(func=mdp.wall_progress, weight=1.0)
    wall_alive = RewTerm(func=mdp.wall_alive, weight=0.7)
    wall_success = RewTerm(func=mdp.wall_success, weight=20.0)
    # Any fall costs 20 (x 0.02 s), before or after the success: a robot down after "crossing" is
    # not over the wall in any use of the word. Replaces Go2w-Biped's is_terminated penalty, which
    # would also charge the success and timeout endings.
    termination_penalty = RewTerm(func=mdp.terminated_by, weight=-1000.0, params={"term_keys": FALL_TERMS})
    # Running out the clock after the jump costs 10, so trying and falling beats not trying.
    timeout_penalty = RewTerm(func=mdp.terminated_by, weight=-500.0, params={"term_keys": ["wall_timeout"]})

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Posture terms of the stance, and its contact rules (a crossing puts the front legs and the
        # head on the wall), only while the stance is commanded.
        for name in (*STANCE_ONLY_TERMS, "stance_leg_contact", "head_contact"):
            term = getattr(self, name)
            setattr(
                self,
                name,
                RewTerm(func=mdp.in_stance, weight=term.weight, params={"term": term.func, "term_params": term.params}),
            )


@configclass
class WallJumpTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact_off_wall,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip"]), "threshold": 1.0},
    )
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})
    wall_success_done = DoneTerm(func=mdp.wall_success_done)
    wall_timeout = DoneTerm(func=mdp.wall_timeout)


@configclass
class WallJumpCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.wall_jump_levels)


@configclass
class RobotEnvCfgBipedWallJump(RobotEnvCfgBiped):
    scene: WallJumpSceneCfg = WallJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: WallJumpObservationsCfg = WallJumpObservationsCfg()
    commands: WallJumpCommandsCfg = WallJumpCommandsCfg()
    rewards: WallJumpRewardsCfg = WallJumpRewardsCfg()
    terminations: WallJumpTerminationsCfg = WallJumpTerminationsCfg()
    curriculum: WallJumpCurriculumCfg = WallJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S


@configclass
class RobotPlayEnvCfgBipedWallJump(RobotEnvCfgBipedWallJump):
    """3 rows (50 / 75 / 100 cm) x 3 columns (5 / 15 / 25 cm thick), robots spread over the rows."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 36
        self.scene.terrain.terrain_generator = WALLJUMP_PLAY_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.spread_over_rows = EventTerm(func=mdp.spread_over_rows, mode="startup")
