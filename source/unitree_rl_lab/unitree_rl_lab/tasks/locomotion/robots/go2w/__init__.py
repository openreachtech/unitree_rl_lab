import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Go2W-v1-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v1-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v1-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v1-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg",
    },
)

# 2026-08-18: folded in from the Go2W-v1-Phase5-Try15 sandbox experiment -- thin_wall
# terrain + goal-directed command (MixedGoalVelocityCommand) + ANYmal-Parkour-style
# goal-tracking rewards, replacing the pyramid_stairs/UniformTerrainGatedVelocityCommand/
# climb_progress-style design the Try1-14 campaigns had converged on. Confirmed in
# MuJoCo: controls correctly, no runaway under a zero command, crosses 0.40 m. See
# velocity_env_cfg_phase5.py's module docstring for the full history (including which
# lessons from the old campaigns still apply and which are now superseded). Go2W-v2-*
# (velocity_env_cfg_v2.py's RobotEnvCfgV2Phase5, inheriting this unchanged) picked up
# the same redesign as a result -- a deliberate choice, not an oversight, since the old
# V2 Teacher/Student Phase5 pipeline was already known to have the same "won't stop"
# problem this redesign fixes. Follow-up sandbox tries against this phase (thin_wall
# thickness variants, a still-unsolved 0.50 m stall) are recorded in sandbox/SUMMARY.md --
# none are currently registered (cleared 2026-08-19, see sandbox/__init__.py).
gym.register(
    id="Go2W-v1-Phase5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase5:RobotEnvCfgPhase5",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase5:RobotPlayEnvCfgPhase5",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GruPPORunnerCfg",
    },
)

# v2: same phase curricula as v1, with a teacher-style privileged critic (xt encoder
# concat ot). Actor is still TCN-100; checkpoints are not compatible with v1.
gym.register(
    id="Go2W-v2-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TcnTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TcnTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase3",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TcnTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase4",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TcnTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Phase5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase5",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase5",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TcnTeacherPPORunnerCfg",
    },
)

# v2 Teacher: same env as Go2W-v2-Phase* (privileged xt, single-frame ot). Actor is
# the paper's teacher MLP (xt encoder concat ot → actions), not the TCN student.
gym.register(
    id="Go2W-v2-Teacher-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Teacher-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Teacher-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase3",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Teacher-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase4",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Go2W-v2-Teacher-Phase5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase5",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase5",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

# TCN-100 student distilled from a teacher checkpoint (Lee et al. 2020 Eq. 1). The
# teacher's trunk is transplanted into the student once, at Phase1 (see
# TcnStudentTeacher.load_state_dict) -- it doesn't change afterwards. Phase1/Phase2/Phase5
# instead ramp the student's OWN terrain curriculum (flat -> rough+box -> +wall), same
# terrain progression as the Teacher/PPO phases, each stage resuming the previous stage's
# own distillation checkpoint.
#
# 2026-08-16: trunk source switched to Go2W-v2-Teacher-Phase5-Try1 (confirmed in Isaac Lab
# play mode to stop correctly, including over a 60 cm wall) instead of the old
# Go2W-v2-Teacher-Phase5. Because the transplanted trunk itself changes, Phase1 and Phase2
# have to be redistilled from scratch -- a Student-Phase1/Phase2 checkpoint trained against
# the old teacher's trunk is not a valid resume target for a run whose trunk now comes from
# a different teacher. The old Go2W-v2-Student-Phase5 (thin_wall/goal-directed's Student
# counterpart) is replaced by Go2W-v2-Student-Phase5-Try1 (sandbox/__init__.py), reusing
# Teacher-Phase5-Try1's own env cfg (env cfg is otherwise identical between Teacher and
# Student -- only rsl_rl_cfg_entry_point differs):
#   --task Go2W-v2-Student-Phase1 --resume --previous-task Go2W-v2-Teacher-Phase5-Try1
#   --task Go2W-v2-Student-Phase2 --resume --previous-task Go2W-v2-Student-Phase1
#   --task Go2W-v2-Student-Phase5-Try1 --resume --previous-task Go2W-v2-Student-Phase2
gym.register(
    id="Go2W-v2-Student-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase1",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distillation_cfg:StudentDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2W-v2-Student-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase2",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distillation_cfg:StudentDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2W-v2-Student-Phase5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotEnvCfgV2Phase5",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_v2:RobotPlayEnvCfgV2Phase5",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distillation_cfg:StudentDistillationRunnerCfg"
        ),
    },
)
