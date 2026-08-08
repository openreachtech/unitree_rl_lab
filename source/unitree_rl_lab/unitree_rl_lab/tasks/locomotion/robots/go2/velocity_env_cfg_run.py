from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    CriticCfgGo2,
    ObservationsCfgGo2,
    PolicyCfgGo2,
    RewardsCfgGo2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
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
class CommandsCfgGo2Run(CommandsCfgPhase1):
    base_velocity = CommandsCfgPhase1().base_velocity.replace(
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.5, 3.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )

    # Gait style is itself a command (Walk These Ways): frequency/phase-offsets/duty cycle are
    # sampled per env and re-rolled on the same cadence as base_velocity, so a single policy
    # learns to produce whatever pattern it's told -- trot, pace, bound, or a gallop -- rather
    # than the network overfitting to one gait. Ranges are wide open-cycle-fraction coverage
    # (0-1) so the canonical gaits above are all reachable in training, not just at their
    # exact corner values.
    gait_command = mdp.GaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.GaitCommandCfg.Ranges(
            frequency=(1.5, 4.0),
            theta_fr=(0.0, 1.0),
            theta_rl=(0.0, 1.0),
            theta_rr=(0.0, 1.0),
            duty_cycle=(0.3, 0.7),
        ),
    )


@configclass
class PolicyCfgGo2Run(PolicyCfgGo2):
    gait_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "gait_command"})
    gait_clock = ObsTerm(func=mdp.gait_clock_obs, clip=(-100, 100), params={"command_name": "gait_command"})


@configclass
class CriticCfgGo2Run(CriticCfgGo2):
    gait_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "gait_command"})
    gait_clock = ObsTerm(func=mdp.gait_clock_obs, clip=(-100, 100), params={"command_name": "gait_command"})


@configclass
class ObservationsCfgGo2Run(ObservationsCfgGo2):
    policy: PolicyCfgGo2Run = PolicyCfgGo2Run()
    critic: CriticCfgGo2Run = CriticCfgGo2Run()


@configclass
class RewardsCfgGo2Run(RewardsCfgGo2):
    """Go2-Run reward tuning: swap the trot-symmetric shaping for a commandable gait-tracking
    term so the same policy can be driven into a bound/gallop-style pattern instead of only
    ever trotting faster.
    """

    # air_time_variance enforces equal air/contact time across all four feet, which directly
    # fights a bound/gallop pattern where front and hind pairs move on different schedules.
    # gait_tracking (below) replaces it with a term that tracks whatever pattern is commanded.
    air_time_variance = RewardsCfgGo2().air_time_variance.replace(weight=0.0)

    # Loosened so the natural vertical bounce / pitch of a bound-style flight phase isn't
    # stamped flat by penalty weights tuned for a slow, level walk.
    base_linear_velocity = RewardsCfgGo2().base_linear_velocity.replace(weight=-1.0)
    flat_orientation_l2 = RewardsCfgGo2().flat_orientation_l2.replace(weight=-1.25)

    gait_tracking = RewTerm(
        func=mdp.gait_tracking_reward,
        weight=-0.3,
        params={
            # preserve_order=True is required here: SceneEntityCfg otherwise resolves body_names
            # in the articulation's own body order, not the [FL, FR, RL, RR] order GaitCommand
            # assumes, silently misaligning feet with the wrong commanded phase.
            "asset_cfg": SceneEntityCfg("robot", body_names=_GAIT_FEET, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_GAIT_FEET, preserve_order=True),
            "command_name": "gait_command",
            "kappa": 6.0,
            "force_scale": 0.01,
        },
    )


@configclass
class RobotEnvCfgGo2Run(RobotEnvCfgPhase1):
    """Flat terrain, same as Phase1, with the x-velocity ceiling raised to 3.5 m/s for running,
    and gait style (trot/pace/bound/gallop/...) exposed as a command so the same policy can be
    driven into a cheetah-style gallop instead of a faster trot."""

    commands: CommandsCfgGo2Run = CommandsCfgGo2Run()
    observations: ObservationsCfgGo2Run = ObservationsCfgGo2Run()
    rewards: RewardsCfgGo2Run = RewardsCfgGo2Run()


@configclass
class RobotPlayEnvCfgGo2Run(RobotEnvCfgGo2Run):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Default the play/eval task to showcase the rotary gallop rather than a random style.
        theta_fr, theta_rl, theta_rr = GAIT_GALLOP_ROTARY
        self.commands.gait_command.ranges = mdp.GaitCommandCfg.Ranges(
            frequency=(3.0, 3.0),
            theta_fr=(theta_fr, theta_fr),
            theta_rl=(theta_rl, theta_rl),
            theta_rr=(theta_rr, theta_rr),
            duty_cycle=(0.3, 0.3),
        )


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
    """Promoted from sandbox Try 6. Inherits directly from ``CommandsCfgPhase1`` (not
    ``CommandsCfgGo2Run``), so there's no ``gait_command`` term at all:
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
    """Loosened further than Go2-Run's already-loosened values, and feet_air_time weighted up --
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
    ``ObservationsCfgGo2`` (via ``RobotEnvCfgPhase1``) -- no gait-style inputs, unlike Go2-Run,
    which observes ``gait_commands``/``gait_clock``. Terrain is Phase1's, unchanged.
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
