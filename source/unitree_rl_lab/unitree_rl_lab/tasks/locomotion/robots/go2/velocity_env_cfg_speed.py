"""Go2 top-speed tasks, promoted from sandbox Try 8 (Phase 1) and Try 20 (Phase 2).

A two-phase recipe, because a single run cannot do both jobs. Phase 1 acquires an asymmetric
footfall pattern from scratch, with the gait shaped and the speed modest. Phase 2 throws the gait
shaping away and spends everything on speed, keeping whatever gait the policy decides is worth
keeping. Measured end to end: **5.31 m/s**, tracking a commanded 5.5 m/s within 0.5.

    Phase 1  Go2-Speed-Phase1   from scratch, ~1300 iterations   -> asymmetric gait, ~4.1 m/s
    Phase 2  Go2-Speed-Phase2   resume from Phase 1, ~2000 iter  -> 5.31 m/s

WHY TWO PHASES -- THE ONE THING FOUR FAILED RUNS TAUGHT
------------------------------------------------------
Shaping a gait and chasing speed at the same time does not work. Sandbox Try 8, 13, 14 and 18 all
tried it; the three that resumed from a walking policy (13, 14, 18) could not move the footfall
pattern at all, and the one that succeeded (Try 8) did so only because it trained FROM SCRATCH and
never had a trot to escape from. ``paired_gait_reward`` grades each leg pair through a Gaussian, so
a policy that already trots sits in a region where the reward is ~1e-20 with a derivative to match.
From random initialisation the pairs land inside the live region by chance and get pulled in.

So Phase 1's job is to be a from-scratch run with the gait reward on and the speed ambition low.
Phase 2's job is to take that footfall pattern as a starting condition and never mention gait again.

WHAT PHASE 2 CHANGES, AND WHY EACH ONE IS THERE
----------------------------------------------
Each item below was measured in isolation in the sandbox; the reward magnitudes quoted are the
logged ``weight x mean(term)`` values, which are directly comparable to each other.

  paired_gait      0.2 -> 0.0    No gait prescription at all. The gait is then free to decay back
                                 toward a trot if a trot is better -- and it does not: it stays
                                 0.83 of a cycle away from canonical trot, which is how we know the
                                 asymmetric pattern is earning its keep rather than being held in
                                 place by a reward term.
  forward_command_progress
                   0.0 -> 0.8    THE change that unblocked this whole line (sandbox Try 15). The
                                 exponential tracking kernel is nearly flat far from its target, so
                                 at a commanded 4.3 m/s against a 3.5 m/s robot it paid ~0.12 for
                                 running while the speed-dependent penalties charged ~0.8 -- making
                                 "stand still" the reward-maximising answer, which six runs duly
                                 found. This term is monotone in achieved speed and clamped at the
                                 command, so every 0.1 m/s pays and there is no flat region to give
                                 up in.
  joint_pos        -0.7 -> -0.3  An L2 penalty on deviation from the standing pose is a tax on
                                 stride amplitude. Cutting it took stride from 0.89 to 0.98 m/cycle
                                 (sandbox Try 16). ``stand_still_scale`` still multiplies it by 5
                                 when no motion is commanded, so standing posture stays anchored.
  joint_vel        -0.001 -> 0   The four effort taxes, worth 0.723 of penalty at speed and rising
  joint_acc      -2.5e-7 -> 0    quadratically with joint velocity -- i.e. steepest exactly where
  joint_torques    -2e-4 -> 0    the policy is trying to go. Removing them raised mean torque
  energy           -2e-5 -> 0    utilisation from 23% to ~32% (sandbox Try 19).
  limit lin_vel_x  4.0 -> 6.0    Phase 1's ceiling would clamp the curriculum long before the
                                 policy runs out of ability.

SPRINT ONLY. Read this before promoting anything derived from Phase 2 into a general-purpose task:
neither Isaac Lab nor unitree_mujoco models motor heating or battery sag, and Phase 2 deliberately
removes every penalty that stood between the policy and continuous peak torque. The resulting gait
is a legitimate top-speed number in simulation and is NOT a safe long-duration hardware policy.
Restore the four effort terms first.

WHAT THE END RESULT LOOKS LIKE (Try 20's model_3298, 128 envs, assist off, two sweeps agreeing)
-----------------------------------------------------------------------------------------------
    cmd   achieved   Hz   stride  flight%  torque%   >X1    gait
    4.0     4.08    4.0   1.02     21%      21%     11%    asymmetric, 0.50 from canonical pace
    5.0     4.98    4.3   1.15     29%      28%     25%
    5.5     5.20    4.5   1.17     35%      30%     28%
    6.0     5.31    4.5   1.18     38%      32%     29%

Compare the best trot-gaited policy from the same reward set (sandbox Try 19, 5.15-5.19 m/s): the
asymmetric gait gets there with a 7% longer stride at a LOWER stride frequency, so 28% of its
thigh/calf joint-steps sit past the actuator's torque-speed knee against the trot's 33%. It buys the
same speed with more headroom on the axis that is actually running out.

TRAINING
--------
    python scripts/rsl_rl/train_and_aggregate.py --task Go2-Speed-Phase1 --max_iterations 1300
    python scripts/rsl_rl/train_and_aggregate.py --task Go2-Speed-Phase2 \\
        --previous-task Go2-Speed-Phase1 --max_iterations 2000

Phase 1 takes no ``--previous-task``: from scratch is not incidental, it is the mechanism.
``--max_iterations`` is additive on a resume, so Phase 2's 2000 ends at ~3300.

Measure with ``scripts/rsl_rl/measure_run_speed.py`` (sweep to 6.0, and note the tool resets the
environments between commanded speeds -- results from before that fix read systematically low).

DEVIATIONS FROM THE SANDBOX RUNS THEY WERE PROMOTED FROM
--------------------------------------------------------
  Phase 1 ceiling 8.0 -> 4.0   Try 8 used 8.0 but only ever reached a commanded 3.5 in its 1300
                               iterations. A ceiling above what the phase can use is a trap for
                               longer runs, and Phase 1 has no progress term to protect it: an
                               earlier sandbox run left the ceiling at 8.0, let the commanded range
                               reach 4.6 against a robot topping out near 3.7, and the policy gave
                               up and stood still -- 3.72 -> 1.94 m/s, unrecoverable, because the
                               exponential tracking kernel pays almost nothing that far from its
                               target while every penalty still applies.
  both phases                  the velocity curriculum judges on the held-out unassisted
                               environments and can step DOWN as well as up, and it accumulates
                               ~1000 episodes per decision. Try 8 ran with the original one-way
                               ratchet, which decided on a single episode and could stop firing
                               altogether (see ``mdp.lin_vel_cmd_levels``).
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CORRECTED_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotSceneCfgPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_run import (
    CommandsCfgGo2GallopPhase1,
    CurriculumCfgGo2Gallop,
    RewardsCfgGo2GallopPhase1,
    RobotEnvCfgGo2GallopPhase1,
)

PHASE1_LOG_DIR = "logs/rsl_rl/go2_speed_phase1"
PHASE2_LOG_DIR = "logs/rsl_rl/go2_speed_phase2"

PHASE1_SPEED_CEILING = 4.0
PHASE2_SPEED_CEILING = 6.0

# A quarter of the fleet never receives the tow assist, so the velocity curriculum always has an
# honest sample of unaided ability to judge from. Without it the ratchet cannot tell "the robot
# runs this fast" from "the robot is towed this fast".
EVAL_ENV_FRACTION = 0.25

# Raise the commanded range above 0.8 x the tracking weight, lower it below 0.6 x. The down-step is
# what makes an out-of-reach ceiling survivable; the original term could only ever add.
INCREASE_THRESHOLD = 0.8
DECREASE_THRESHOLD = 0.6


@configclass
class RobotSceneCfgSpeed(RobotSceneCfgPhase1):
    """Phase1's flat terrain with the mujoco-matched actuator model (corrected calf torque, joint
    friction, armature). Running fast is knee-extension-dominated, so the stock model -- which
    gives the calf less than half its real torque -- would make any top-speed number a property of
    the model rather than of Go2."""

    robot: ArticulationCfg = UNITREE_GO2_CORRECTED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class CommandsCfgSpeedPhase1(CommandsCfgGo2GallopPhase1):
    """Forward-only commands (lateral pinned to zero in both ranges and limits, since
    ``lin_vel_cmd_levels`` widens x and y by the same step) plus the self-decaying tow assist."""

    base_velocity = CommandsCfgGo2GallopPhase1().base_velocity.replace(
        limit_ranges=CommandsCfgPhase1().base_velocity.limit_ranges.replace(
            lin_vel_x=(0.0, PHASE1_SPEED_CEILING), lin_vel_y=(0.0, 0.0)
        ),
    )

    tow_assist = CommandsCfgGo2GallopPhase1().tow_assist.replace(
        eval_env_fraction=EVAL_ENV_FRACTION,
        state_file=f"{PHASE1_LOG_DIR}/tow_assist_state.json",
    )


@configclass
class CurriculumCfgSpeedPhase1(CurriculumCfgGo2Gallop):
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "assist_free_only": True,
            "tow_command_name": "tow_assist",
            "increase_threshold": INCREASE_THRESHOLD,
            "decrease_threshold": DECREASE_THRESHOLD,
            "state_file": f"{PHASE1_LOG_DIR}/lin_vel_cmd_state.json",
        },
    )


@configclass
class RobotEnvCfgSpeedPhase1(RobotEnvCfgGo2GallopPhase1):
    """Gait acquisition. ``paired_gait`` stays at Go2-Gallop-Phase1's 0.2, ungated, exactly as in
    sandbox Try 8 -- that is what produces the asymmetric footfall, and it only works from a
    from-scratch start. Rewards are otherwise untouched, so this phase has no progress term and the
    full amplitude and effort taxes: it is not trying to be fast."""

    scene: RobotSceneCfgSpeed = RobotSceneCfgSpeed(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfgSpeedPhase1 = CommandsCfgSpeedPhase1()
    curriculum: CurriculumCfgSpeedPhase1 = CurriculumCfgSpeedPhase1()


@configclass
class RobotPlayEnvCfgSpeedPhase1(RobotEnvCfgSpeedPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Assist off, and both curriculum state files detached: the curriculum manager still runs at
        # play time, and with the command range set to limit_ranges above it would otherwise
        # overwrite the training state with the ceiling.
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
        self.curriculum.lin_vel_cmd_levels.params["state_file"] = None


@configclass
class CommandsCfgSpeedPhase2(CommandsCfgSpeedPhase1):
    """Phase 1's commands with the ceiling raised. The starting range stays at (0.0, 1.0) so the
    curriculum's down-step floor stays low; to resume without re-climbing, seed
    ``lin_vel_cmd_state.json`` instead of raising this."""

    base_velocity = CommandsCfgSpeedPhase1().base_velocity.replace(
        limit_ranges=CommandsCfgSpeedPhase1().base_velocity.limit_ranges.replace(
            lin_vel_x=(0.0, PHASE2_SPEED_CEILING)
        ),
    )

    tow_assist = CommandsCfgSpeedPhase1().tow_assist.replace(
        state_file=f"{PHASE2_LOG_DIR}/tow_assist_state.json"
    )


@configclass
class CurriculumCfgSpeedPhase2(CurriculumCfgSpeedPhase1):
    lin_vel_cmd_levels = CurriculumCfgSpeedPhase1().lin_vel_cmd_levels.replace(
        params={
            "assist_free_only": True,
            "tow_command_name": "tow_assist",
            "increase_threshold": INCREASE_THRESHOLD,
            "decrease_threshold": DECREASE_THRESHOLD,
            "state_file": f"{PHASE2_LOG_DIR}/lin_vel_cmd_state.json",
        }
    )


@configclass
class RewardsCfgSpeedPhase2(RewardsCfgGo2GallopPhase1):
    """The speed reward set. See the module docstring for what each line buys and what it measured.

    SPRINT ONLY -- the four zeroed terms are the ones that keep motor effort in check, and nothing
    in either simulator models heating.
    """

    # No footfall prescription. The gait inherited from Phase 1 is on its own from here.
    paired_gait = RewardsCfgGo2GallopPhase1().paired_gait.replace(weight=0.0)

    # Monotone, clamped at the commanded speed: every 0.1 m/s of real progress pays.
    forward_command_progress = RewardsCfgGo2GallopPhase1().forward_command_progress.replace(weight=0.8)

    # Stride amplitude stops being taxed as heavily.
    joint_pos = RewardsCfgGo2GallopPhase1().joint_pos.replace(weight=-0.3)

    # Effort taxes off.
    joint_vel = RewardsCfgGo2GallopPhase1().joint_vel.replace(weight=0.0)
    joint_acc = RewardsCfgGo2GallopPhase1().joint_acc.replace(weight=0.0)
    joint_torques = RewardsCfgGo2GallopPhase1().joint_torques.replace(weight=0.0)
    energy = RewardsCfgGo2GallopPhase1().energy.replace(weight=0.0)


@configclass
class RobotEnvCfgSpeedPhase2(RobotEnvCfgSpeedPhase1):
    """Speed. Resume this from Phase 1's checkpoint -- from scratch it has no gait to keep."""

    commands: CommandsCfgSpeedPhase2 = CommandsCfgSpeedPhase2()
    curriculum: CurriculumCfgSpeedPhase2 = CurriculumCfgSpeedPhase2()
    rewards: RewardsCfgSpeedPhase2 = RewardsCfgSpeedPhase2()


@configclass
class RobotPlayEnvCfgSpeedPhase2(RobotEnvCfgSpeedPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.tow_assist.state_file = None
        self.commands.tow_assist.initial_assist_scale = 0.0
        self.curriculum.lin_vel_cmd_levels.params["state_file"] = None
