from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CORRECTED_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
    RobotSceneCfgPhase1,
)

# Canonical gait phase offsets (theta_FR, theta_RL, theta_RR; FL fixed at 0), from cheetah/quadruped
# gait literature -- see Margolis & Agrawal, "Walk These Ways" (2022), Appendix. Handy for pinning
# GaitCommand to a specific style at play/eval time instead of the training-time random range.
GAIT_PRONK = (0.0, 0.0, 0.0)
GAIT_TROT = (0.5, 0.5, 0.0)
GAIT_PACE = (0.5, 0.0, 0.5)
GAIT_BOUND = (0.0, 0.5, 0.5)
GAIT_GALLOP_ROTARY = (0.1, 0.6, 0.5)
GAIT_GALLOP_TRANSVERSE = (0.4, 0.6, 0.1)

_GAIT_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class CurriculumCfgGo2Gallop(CurriculumCfg):
    tow_assist = CurrTerm(
        func=mdp.tow_assist_decay,
        params={
            "tow_command_name": "tow_assist",
            "velocity_command_name": "base_velocity",
            "error_threshold": 0.4,
            "success_threshold": 0.6,
            "decay_step": 0.01,
            "minimum_episodes": 1024,
        },
    )


@configclass
class CommandsCfgGo2GallopPhase1(CommandsCfgPhase1):
    """Promoted from sandbox Try 6. Inherits directly from ``CommandsCfgPhase1``, so there is no
    ``gait_command`` term at all (unlike the original gait-commandable Go2-Run, replaced below):
    ``RewardsCfgGo2GallopPhase1``'s self-referential ``paired_gait`` reward (see
    ``mdp.paired_gait_reward``) stands in for ``gait_tracking_reward`` instead, needing no
    external absolute-phase reference to grade against.
    """

    # A gallop has no sideways or backward variant in nature -- it's a straight-ahead gait --
    # so drop lateral motion entirely and restrict lin_vel_x to forward-only. Starting range
    # begins at ~1.0 m/s rather than Phase1's ~0.1 m/s -- no need to spend training time
    # crawling through the trivial-speed regime the tow-assist below is meant to skip past.
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=CommandsCfgPhase1().base_velocity.ranges.replace(lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0)),
        limit_ranges=CommandsCfgPhase1().base_velocity.limit_ranges.replace(
            lin_vel_x=(0.0, 3.5), lin_vel_y=(0.0, 0.0)
        ),
    )

    # EFGCL-style (Yoneda et al. 2026) physical guidance, adapted from feat/jump's one-shot
    # launch assist to a continuous forward tow: pulls the robot up toward its commanded
    # speed -- like being towed on a leash -- so it physically experiences running at speed
    # (and can learn correct footfall timing there) before it can generate that speed itself.
    # Decays via tow_assist_decay (see CurriculumCfgGo2Gallop) once velocity-tracking
    # success is consistently good, same success-rate-gated schedule as the jump assist.
    tow_assist = mdp.TowAssistCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        asset_name="robot",
        body_names=["base"],
        velocity_command_name="base_velocity",
        gain=40.0,
        max_force=150.0,
        initial_assist_scale=1.0,
        state_file="logs/rsl_rl/go2_gallop_phase1/tow_assist_state.json",
    )


@configclass
class RewardsCfgGo2GallopPhase1(RewardsCfgGo2):
    """Loosened further than ``RewardsCfgGo2``'s walk-tuned values, and feet_air_time weighted up --
    playtesting found the gallop looked more like a fast shuffle than a hop; these penalties were
    the main things left stamping out vertical bounce/pitch and discouraging air time.
    ``paired_gait`` (self-referential, see ``mdp.paired_gait_reward``) stands in for
    ``gait_tracking_reward``, and ``air_time_variance`` is disabled outright rather than replaced
    by a term that tracks a nonexistent command -- ``paired_gait`` takes over its role of shaping
    *structure* (front-pair/hind-pair relative timing, front-vs-hind alternation)
    self-referentially instead of against an external target.
    """

    air_time_variance = RewardsCfgGo2().air_time_variance.replace(weight=0.0)
    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-0.75)
    feet_air_time = RewardsCfgGo2().feet_air_time.replace(weight=0.2)

    paired_gait = RewTerm(
        func=mdp.paired_gait_reward,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_GAIT_FEET, preserve_order=True)},
    )


@configclass
class RobotEnvCfgGo2GallopPhase1(RobotEnvCfgPhase1):
    """Forward-only gallop-style running -- promoted from sandbox Try 6 (see
    ``CommandsCfgGo2GallopPhase1``/``RewardsCfgGo2GallopPhase1``). Observations are plain
    ``ObservationsCfgGo2`` (via ``RobotEnvCfgPhase1``) -- no gait-style inputs. Terrain is
    Phase1's, unchanged. Go2-Run below is this config minus the gallop shaping, plus the
    mujoco-matched actuator model -- and it measured faster; see its note.
    """

    commands: CommandsCfgGo2GallopPhase1 = CommandsCfgGo2GallopPhase1()
    curriculum: CurriculumCfgGo2Gallop = CurriculumCfgGo2Gallop()
    rewards: RewardsCfgGo2GallopPhase1 = RewardsCfgGo2GallopPhase1()


@configclass
class RobotPlayEnvCfgGo2GallopPhase1(RobotEnvCfgGo2GallopPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0


# Forward-speed threshold at/above which paired_gait is graded -- below it (which covers every
# backward/lateral command, since those have lin_vel_x <= 0), the term is inactive and the
# policy is free to pick its own natural footfall instead of being pushed toward a gallop-style
# structure that only makes sense running forward at speed.
_GALLOP_SPEED_THRESHOLD = 1.0


@configclass
class CommandsCfgGo2GallopPhase2(CommandsCfgPhase1):
    """Generalizes Go2-Gallop-Phase1 to omnidirectional commands (adds backward/lateral) --
    promoted from sandbox Try 7. Gates ``paired_gait`` on forward speed (see
    ``RewardsCfgGo2GallopPhase2``) so the gallop-style structure is only rewarded at/above
    ~1.0 m/s forward -- below that, or moving backward/lateral, the policy is free to choose
    its own footfall. Trained resuming from Go2-Gallop-Phase1
    (``--previous-task Go2-Gallop-Phase1 --resume``): neither phase has a gait observation, so
    their observation widths match exactly -- no checkpoint-surgery needed.
    """

    # Forward range/ceiling unchanged from Go2-Gallop-Phase1 (up to 3.5 m/s); adds backward
    # (down to -1.0 m/s start / -1.0 m/s ceiling) and lateral (+-0.5 m/s start / +-1.0 m/s
    # ceiling). lin_vel_cmd_levels (inherited via CurriculumCfg) widens both dims from these
    # starting ranges toward limit_ranges as tracking improves, same as every other velocity
    # task. ang_vel_z is untouched (Phase1's own (-0.5, 0.5) -> (-1.2, 1.2)).
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        ranges=CommandsCfgPhase1().base_velocity.ranges.replace(lin_vel_x=(-0.5, 1.5), lin_vel_y=(-0.5, 0.5)),
        limit_ranges=CommandsCfgPhase1().base_velocity.limit_ranges.replace(
            lin_vel_x=(-1.0, 3.5), lin_vel_y=(-1.0, 1.0)
        ),
    )

    # Same EFGCL tow-assist as Go2-Gallop-Phase1, own state_file. The force itself is
    # forward-only regardless (see TowAssistCommand._update_command) so it's a no-op for
    # backward/lateral commands anyway; it only ever helps the forward-fast regime this phase
    # already inherits a trained head start on via --resume.
    tow_assist = CommandsCfgGo2GallopPhase1().tow_assist.replace(
        state_file="logs/rsl_rl/go2_gallop_phase2/tow_assist_state.json"
    )


@configclass
class RewardsCfgGo2GallopPhase2(RewardsCfgGo2):
    """Same loosened-for-bounce tuning as Go2-Gallop-Phase1, with ``paired_gait`` gated on
    forward speed (see ``mdp.paired_gait_reward``'s ``velocity_command_name``/
    ``speed_threshold``) instead of unconditionally active -- backward/lateral commands always
    have ``lin_vel_x <= 0``, so they fall below the gate automatically.
    """

    air_time_variance = RewardsCfgGo2().air_time_variance.replace(weight=0.0)
    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-0.75)
    feet_air_time = RewardsCfgGo2().feet_air_time.replace(weight=0.2)

    paired_gait = RewTerm(
        func=mdp.paired_gait_reward,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_GAIT_FEET, preserve_order=True),
            "velocity_command_name": "base_velocity",
            "speed_threshold": _GALLOP_SPEED_THRESHOLD,
        },
    )


@configclass
class RobotEnvCfgGo2GallopPhase2(RobotEnvCfgPhase1):
    """Adds backward/lateral commands on top of Go2-Gallop-Phase1's forward-only gallop, with
    ``paired_gait`` gated on forward speed (see
    ``CommandsCfgGo2GallopPhase2``/``RewardsCfgGo2GallopPhase2``). Terrain is unchanged from
    Go2-Gallop-Phase1 (both ultimately inherit ``RobotEnvCfgPhase1``'s).
    """

    commands: CommandsCfgGo2GallopPhase2 = CommandsCfgGo2GallopPhase2()
    curriculum: CurriculumCfgGo2Gallop = CurriculumCfgGo2Gallop()
    rewards: RewardsCfgGo2GallopPhase2 = RewardsCfgGo2GallopPhase2()


@configclass
class RobotPlayEnvCfgGo2GallopPhase2(RobotEnvCfgGo2GallopPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0


# =============================================================================================
# Go2-Run -- promoted from sandbox Try 9 (``Go2-Speed-Free-Try-9``), the fastest Go2 policy
# measured so far: 3.72 m/s unaided, clean trot, tracking to 3.5 m/s with 0.08 m/s error.
#
# It replaces the previous Go2-Run, which was a different idea entirely (gait style exposed as a
# Walk-These-Ways-style command, with ``gait_command``/``gait_clock`` observations and
# ``gait_tracking_reward``). That went unused; the gait-command machinery itself is still in
# ``mdp`` and the canonical offsets above are still here, so it can be rebuilt if wanted. See
# git history for the old config.
#
# WHAT MAKES IT FAST -- three things, in order of measured importance:
#   1. the corrected actuator model (``UNITREE_GO2_CORRECTED_CFG``): the stock model gives the
#      calf less than half its real torque, and running fast is knee-extension-dominated
#   2. the tow assist (inherited from Go2-Gallop-Phase1): it physically drags the robot up to
#      speed early so it can learn footfall timing there before it can generate that speed --
#      then decays itself to exactly 0 by iteration ~1200, so the final policy is unaided
#   3. NO gait prescription. Try 8 was this same config with ``paired_gait`` at 0.2 (gallop
#      shaping) and lost at every commanded speed, collapsing to 0.71 m/s at a commanded 7.0
#      where this one still held 2.63. Go2's rigid trunk gets none of the spinal extension that
#      pays for a gallop in animals; told nothing, PPO picks a trot and the trot is faster.
#
# TWO DELIBERATE DEVIATIONS FROM TRY 9 AS RUN. Both are protective, and both come from Try 9's
# own post-mortem (see ``sandbox/try8.py``):
#
#   limit_ranges.lin_vel_x  8.0 -> 3.8 m/s
#       Try 9 used 8.0 to find out where the ratchet would stall on its own. That answered the
#       question and then destroyed the policy: resumed to iteration 5299, the commanded range
#       reached 4.6 against a robot topping out near 3.7, and the policy gave up and stood still
#       (3.72 -> 1.94 m/s, unrecoverable). Four later sandbox runs collapsed the same way, at a
#       commanded 4.6 / 4.1 / 3.9 / 4.4 -- so 3.8 sits just under the earliest observed collapse.
#       A low ceiling is a SAFETY DEVICE here, not a cap on ambition: it clamps the command
#       before it can outrun the reward. It also sets the deploy-time keyboard maximum
#       (0.8 x 3.8 = 3.04 m/s), which is about what the robot can actually hold.
#       Do not raise this to explore higher speeds -- that experiment belongs in sandbox, with
#       the curriculum/reward fixes it needs.
#
#   lin_vel_cmd_levels      one-way -> two-way, and persisted across --resume
#       ``decrease_threshold`` lets the commanded range step back down when tracking degrades
#       (dead band 0.6-0.8 x the reward weight), instead of only ever ratcheting up; Try 9's
#       one-way ratchet is why its collapse could not recover. ``state_file`` keeps the range
#       across a restart -- rsl_rl checkpoints store only network weights, so without it a
#       --resume re-climbs from 1.0 m/s (Try 9's resume wasted ~1300 iterations that way).
#
# Train from scratch (~1300 iterations reaches 3.72 m/s; the ratchet only fires once per
# episode cycle, ~42 iterations, so the command level cannot climb faster than that):
#
#     python scripts/rsl_rl/train_and_aggregate.py --task Go2-Run --max_iterations 1500
#
# To pick up Try 9's existing best weights instead:
#     ... --previous-task Go2-Speed-Free-Try-9
#
# Measure what it actually does (assist off, per-commanded-speed achieved speed, gait, torque):
#     python scripts/rsl_rl/measure_run_speed.py --task Go2-Run --headless \
#         --speeds 1,2,3,3.5,4 --checkpoint logs/rsl_rl/go2_run/<run>/model_1499.pt
# Repeated sweeps of the SAME checkpoint vary by ~0.2 m/s -- treat smaller differences as noise.
#
# NOT omnidirectional: forward + yaw only (``lin_vel_y`` is pinned to 0 in both ranges and
# limits, because ``lin_vel_cmd_levels`` widens x and y by the same step and a non-zero y limit
# would silently introduce lateral commands as x climbed). Go2-Gallop-Phase2 is the
# omnidirectional variant.
# =============================================================================================

RUN_SPEED_CEILING = 3.8
"""Forward command ceiling, m/s. See the note above before changing it."""

_RUN_LOG_DIR = "logs/rsl_rl/go2_run"


@configclass
class RobotSceneCfgGo2Run(RobotSceneCfgPhase1):
    """Phase1's flat terrain, with the mujoco-matched actuator model swapped in for the robot."""

    robot: ArticulationCfg = UNITREE_GO2_CORRECTED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class CommandsCfgGo2Run(CommandsCfgGo2GallopPhase1):
    """Go2-Gallop-Phase1's forward-only commands + tow assist, with the ceiling at
    ``RUN_SPEED_CEILING``. Starting range stays at Phase1's (0.0, 1.0) and climbs from there."""

    base_velocity = CommandsCfgGo2GallopPhase1().base_velocity.replace(
        limit_ranges=CommandsCfgPhase1().base_velocity.limit_ranges.replace(
            lin_vel_x=(0.0, RUN_SPEED_CEILING), lin_vel_y=(0.0, 0.0)
        ),
    )

    tow_assist = CommandsCfgGo2GallopPhase1().tow_assist.replace(
        state_file=f"{_RUN_LOG_DIR}/tow_assist_state.json"
    )


@configclass
class CurriculumCfgGo2Run(CurriculumCfgGo2Gallop):
    """Tow-assist decay (inherited) plus a two-way, persisted velocity ratchet."""

    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "decrease_threshold": 0.6,
            "state_file": f"{_RUN_LOG_DIR}/lin_vel_cmd_state.json",
        },
    )


@configclass
class RewardsCfgGo2Run(RewardsCfgGo2GallopPhase1):
    """Go2-Gallop-Phase1's rewards with the gallop shaping switched off -- the one substantive
    difference between Try 8 and Try 9, and the reason this task exists.

    ``air_time_variance`` stays at 0.0 too (inherited): it penalises unequal air/contact time
    across the four feet, which is a *trot* prescription. Leaving it off is what makes this
    prescription-free rather than a swap of one imposed gait for another.
    """

    paired_gait = RewardsCfgGo2GallopPhase1().paired_gait.replace(weight=0.0)


@configclass
class RobotEnvCfgGo2Run(RobotEnvCfgGo2GallopPhase1):
    """Forward-only running, no footfall prescription, mujoco-matched actuators."""

    scene: RobotSceneCfgGo2Run = RobotSceneCfgGo2Run(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgGo2Run = CommandsCfgGo2Run()
    curriculum: CurriculumCfgGo2Run = CurriculumCfgGo2Run()
    rewards: RewardsCfgGo2Run = RewardsCfgGo2Run()


@configclass
class RobotPlayEnvCfgGo2Run(RobotEnvCfgGo2Run):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Assist off: what the robot can do unaided is the whole point of this task.
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
        # And the velocity ratchet's state file off, for the same reason. The curriculum manager
        # still runs at play time, and the line above has just set the command range to
        # limit_ranges -- so a single play or measure_run_speed session would rewrite the training
        # state file with the ceiling, and the next --resume would start training with the command
        # already past what the robot can do. Observed on the sandbox tries (try13's file held 7.9
        # against a training value of 3.9).
        self.curriculum.lin_vel_cmd_levels.params["state_file"] = None
