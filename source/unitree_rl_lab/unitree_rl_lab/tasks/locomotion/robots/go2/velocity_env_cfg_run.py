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
class CommandsCfgGo2GallopRotary(CommandsCfgGo2Run):
    """Same as Go2-Run's commands, but gait_command is narrowed to a small jitter band
    around the rotary gallop (theta_FR~0.1, theta_RL~0.6, theta_RR~0.5), with a lower duty
    cycle and higher frequency range typical of a real sprinting gallop, instead of covering
    the full trot/pace/bound/gallop space. Trades Go2-Run's "any gait on command"
    generality for a policy that spends its whole training budget specializing on one gait.
    """

    gait_command = CommandsCfgGo2Run().gait_command.replace(
        ranges=mdp.GaitCommandCfg.Ranges(
            frequency=(2.5, 4.0),
            theta_fr=(GAIT_GALLOP_ROTARY[0] - 0.05, GAIT_GALLOP_ROTARY[0] + 0.05),
            theta_rl=(GAIT_GALLOP_ROTARY[1] - 0.05, GAIT_GALLOP_ROTARY[1] + 0.05),
            theta_rr=(GAIT_GALLOP_ROTARY[2] - 0.05, GAIT_GALLOP_ROTARY[2] + 0.05),
            # Lowered from (0.25, 0.4): duty cycle is literally "fraction of the cycle each
            # foot spends planted" -- a lower value commands a longer relative flight phase
            # per stride, i.e. more hop/bounce, less shuffle.
            duty_cycle=(0.15, 0.3),
        ),
    )

    # A gallop has no sideways or backward variant in nature -- it's a straight-ahead gait --
    # so drop lateral motion entirely and restrict lin_vel_x to forward-only, instead of
    # inheriting Go2-Run's full omnidirectional range. Yaw (turning) is left alone. Starting
    # range begins at ~1.0 m/s rather than Phase1's ~0.1 m/s -- no need to spend training time
    # crawling through the trivial-speed regime the tow-assist below is meant to skip past.
    #
    # Note: this is a plain module-level helper call, not a class attribute -- configclass
    # turns *every* class-body assignment into a dataclass field (even underscore-prefixed
    # ones), and CommandManager would then try to instantiate it as a second, redundant
    # velocity command term. Keep any such intermediate value out of the class body.
    base_velocity = CommandsCfgGo2Run().base_velocity.replace(
        ranges=CommandsCfgGo2Run().base_velocity.ranges.replace(lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0)),
        limit_ranges=CommandsCfgGo2Run().base_velocity.limit_ranges.replace(
            lin_vel_x=(0.0, 3.5), lin_vel_y=(0.0, 0.0)
        ),
    )

    # EFGCL-style (Yoneda et al. 2026) physical guidance, adapted from feat/jump's one-shot
    # launch assist to a continuous forward tow: pulls the robot up toward its commanded
    # speed -- like being towed on a leash -- so it physically experiences running at speed
    # (and can learn correct footfall timing there) before it can generate that speed itself.
    # Decays via tow_assist_decay (see CurriculumCfgGo2GallopRotary) once velocity-tracking
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
        state_file="logs/rsl_rl/go2_gallop/tow_assist_state.json",
    )


@configclass
class CurriculumCfgGo2GallopRotary(CurriculumCfg):
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
class RewardsCfgGo2GallopRotary(RewardsCfgGo2Run):
    """Loosened further than Go2-Run's already-loosened values, and feet_air_time weighted up
    -- playtesting found the gallop looked more like a fast shuffle than a hop. These penalties
    were the main things left stamping out vertical bounce/pitch and discouraging air time."""

    base_linear_velocity = RewardsCfgGo2Run().base_linear_velocity.replace(weight=-0.5)
    flat_orientation_l2 = RewardsCfgGo2Run().flat_orientation_l2.replace(weight=-0.75)
    feet_air_time = RewardsCfgGo2Run().feet_air_time.replace(weight=0.2)


@configclass
class RobotEnvCfgGo2GallopRotary(RobotEnvCfgGo2Run):
    """Go2-Run with gait_command narrowed to rotary-gallop only -- see
    ``CommandsCfgGo2GallopRotary``. Terrain unchanged; velocity range is forward-only with a
    tow-assist curriculum, and rewards are tuned for a more hop-like flight phase -- see
    ``CommandsCfgGo2GallopRotary``/``RewardsCfgGo2GallopRotary``."""

    commands: CommandsCfgGo2GallopRotary = CommandsCfgGo2GallopRotary()
    curriculum: CurriculumCfgGo2GallopRotary = CurriculumCfgGo2GallopRotary()
    rewards: RewardsCfgGo2GallopRotary = RewardsCfgGo2GallopRotary()


@configclass
class RobotPlayEnvCfgGo2GallopRotary(RobotEnvCfgGo2GallopRotary):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        theta_fr, theta_rl, theta_rr = GAIT_GALLOP_ROTARY
        self.commands.gait_command.ranges = mdp.GaitCommandCfg.Ranges(
            frequency=(3.0, 3.0),
            theta_fr=(theta_fr, theta_fr),
            theta_rl=(theta_rl, theta_rl),
            theta_rr=(theta_rr, theta_rr),
            duty_cycle=(0.3, 0.3),
        )
        # Play mode ignores the training curriculum's saved decay state and defaults to no
        # tow assist, so what you see is the policy's own unassisted capability -- edit
        # initial_assist_scale below (0.0-1.0) to manually test with partial/full assist.
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
