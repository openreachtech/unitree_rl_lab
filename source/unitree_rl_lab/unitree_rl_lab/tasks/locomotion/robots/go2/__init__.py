import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Unitree-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2-v3-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2-v3-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2-v3-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Promoted from sandbox Try-4: terrain-adaptive foot clearance for a natural
# flat-ground gait, terrain_levels >= 4.5 (reached 4.899).
gym.register(
    id="Go2-v3-Phase3-balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3Balance",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3Balance",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Promoted from sandbox Try-1 + Try-2: maximizes terrain_levels (~5.3-5.4),
# exaggerated flat-ground gait as a tradeoff.
gym.register(
    id="Go2-v3-Phase3-stairfocus",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3StairFocus",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3StairFocus",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Balance rewards + terrain mix that includes floating inverted pyramid stairs.
# Promoted from sandbox Try-1 through Try-7:
#   Try-1: anti-stall rewards (base_height_climb, stall_penalty, stair_commit)
#     + relaxed bad_orientation, fixing the robot freezing at the stair edge
#     instead of climbing. terrain_levels 5.514 (vs 4.899 without the fix).
#   Try-2/3/4: fixed MuJoCo deploy testing showing the policy flattening/
#     flapping its legs on flat ground at zero command -- rel_standing_envs
#     0.01 -> 0.1, plus command-gating base_height_climb and
#     wild_foot_clearance (both otherwise unconditional on command).
#   Try-5/6/7: added quiet_standing_reward (positive reward for literal
#     stillness), gated by both command and terrain flatness so it can't
#     compete with stair-climbing behavior; weight settled at 0.5 (Try-7).
# See velocity_env_cfg_phase3.py for full per-try results and reasoning.
gym.register(
    id="Go2-v3-Phase3-balance-floating",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3BalanceFloating",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3BalanceFloating",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Continual learning on top of Phase3-balance-floating: dedicated terrain mix
# for stepping over short free-standing walls (10% flat, 90% thin_wall --
# height 0.05 -> 0.25 m and thickness 0.15 -> 0.03 m, both narrowing/rising
# with difficulty). Stair-climbing is expected to carry over from the
# checkpoint, not from keeping stairs in this phase's mix.
# Rewards/terminations/commands are unchanged from Phase3-balance-floating.
# Train with --previous-task Unitree-Go2-Velocity-v2-Phase3-balance-floating.
gym.register(
    id="Go2-v3-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:TeacherPPORunnerCfg",
    },
)

# Belief-encoder student distillation, split the way the paper schedules it
# (see velocity_env_cfg_student.py): flat terrain and a clean height scan first,
# then the deployment terrain mix with the height-scan noise ramping in.
# Phase1 distills the Phase4 teacher; Phase2 continues from the Phase1 student:
#   --task Go2-v3-Student-Phase1 --resume --previous-task Go2-v3-Phase4
#   --task Go2-v3-Student-Phase2 --resume --previous-task Go2-v3-Student-Phase1
gym.register(
    id="Go2-v3-Student-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_student:RobotEnvCfgStudentPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_student:RobotPlayEnvCfgStudentPhase1",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:StudentDistillationRunnerCfg",
    },
)

gym.register(
    id="Go2-v3-Student-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_student:RobotEnvCfgStudentPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_student:RobotPlayEnvCfgStudentPhase2",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:StudentDistillationRunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Blind Go2 with a recurrent policy -- the traversal half of the two-stage plan for
# the LiDAR height map. The end goal is an encoder/decoder that recovers true terrain
# from a noisy fan-built map, trained behind a *frozen* controller; a blind policy is
# the right one to freeze, since it cannot lean on the exteroceptive input the encoder
# is still learning to produce, so the two stages stay separable.
#
# Two ingredients, both taken from existing work rather than invented here:
#   * the Unitree-Go2-Velocity-v1 configs from feat/go2-curriculum, whose actor is
#     proprioception-only and whose critic keeps the privileged height scan -- ported
#     into velocity_env_cfg_blind*.py in this package;
#   * GruPPORunnerCfg, the GRU actor-critic used by Go2W-v1-Phase* on feat/go2w.
#
# The curriculum is v1's, unchanged: flat, then rough ground and boxes, then stairs.
#   --task Go2-Blind-GRU-Phase1
#   --task Go2-Blind-GRU-Phase2 --resume --previous-task Go2-Blind-GRU-Phase1
#   --task Go2-Blind-GRU-Phase3 --resume --previous-task Go2-Blind-GRU-Phase2
# ---------------------------------------------------------------------------

_RUNNER = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg"
# Relative to this package. The launcher filters tasks on env_cfg_entry_point
# starting with "locomotion.", and the walk that discovers these packages makes
# __name__ resolve to exactly that prefix; a hardcoded absolute path is silently
# skipped instead.
_CFG = __name__

# Phase 1: flat ground. Establishes a gait before any terrain is introduced.
gym.register(
    id="Go2-Blind-GRU-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 2: flat 10% / random rough 40% / boxes 50%.
gym.register(
    id="Go2-Blind-GRU-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Phase 3: stairs. The hardest thing a blind policy is asked to cross here, and the
# one that most needs the GRU -- a step edge is only observable through contact.
gym.register(
    id="Go2-Blind-GRU-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotEnvCfgPhase3BalanceMatched",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# Play-only: Phase 2 with the LiDAR noise pinned to one condition instead of drawn
# 60/30/10, so each can be inspected on its own. The policy is unchanged and never sees
# the map; only the drawn grid differs. See velocity_env_cfg_blind_phase2.py.
#   --task Go2-Blind-GRU-Phase2-Noise-Weak --checkpoint <phase2 checkpoint>
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase2-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase2:RobotPlayEnvCfgPhase2Noise{_level}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )

# Play-only: the Phase 3 default on a 2 x 3 stepped terrain (pyramid | inverted pyramid,
# rows centred on 5 / 10 / 15 cm), with the LiDAR noise pinned to one condition instead of
# drawn 60/30/10. The policy is unchanged and never sees the map; only the drawn grid
# differs. See velocity_env_cfg_blind_phase3.py.
#   --task Go2-Blind-GRU-Phase3-Noise-Weak --checkpoint <phase3 checkpoint>
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase3-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase3:RobotPlayEnvCfgPhase3Noise{_level}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )

# Phase 4: stepping over short free-standing walls, after Go2-v3-Phase4. Continual
# learning on top of Phase 3 -- stairs are a surface to climb, these are isolated walls
# to clear, invisible to a blind policy until a foot hits them. See
# velocity_env_cfg_blind_phase4.py.
#   --task Go2-Blind-GRU-Phase4 --resume --previous-task Go2-Blind-GRU-Phase3
gym.register(
    id="Go2-Blind-GRU-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

# The same three pinned-noise play variants Phase 2 and 3 have.
for _level in ("Weak", "Nominal", "Strong"):
    gym.register(
        id=f"Go2-Blind-GRU-Phase4-Noise-{_level}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4Noise{_level}",
            "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4Noise{_level}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )

# The same Phase 4 environment, registered a second time so a run that skips the stair
# phase gets its own log directory (experiment_name follows the task id). Nothing about
# the env differs -- only which checkpoint the run resumes from:
#   Go2-Blind-GRU-Phase4          <- Phase 3 (flat -> rough/boxes -> stairs -> walls)
#   Go2-Blind-GRU-Phase4-NoStairs <- Phase 2 (flat -> rough/boxes -> walls)
# The question is whether the stair phase helps wall crossing or costs it; earlier work
# on this lineage saw stair training erode the wall-crossing ability it was meant to
# build on. Note the two are not matched on total experience -- the stairs path arrives
# with roughly 3500 more iterations behind it -- so read the gap with that in mind.
#   --task Go2-Blind-GRU-Phase4-NoStairs --resume --load_run <phase2 run> --checkpoint ...
gym.register(
    id="Go2-Blind-GRU-Phase4-NoStairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{_CFG}.velocity_env_cfg_blind_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)
