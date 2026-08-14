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

# TCN-100 student distilled from the fully-trained Go2W-v2-Teacher-Phase5 checkpoint
# (Lee et al. 2020 Eq. 1). The teacher's trunk is transplanted into the student once,
# at Phase1 (see TcnStudentTeacher.load_state_dict) -- it doesn't change afterwards.
# Phase1/Phase2/Phase5 instead ramp the student's OWN terrain curriculum (flat ->
# rough+box -> +wall), same terrain progression as the Teacher/PPO phases, each
# stage resuming the previous stage's own distillation checkpoint:
#   --task Go2W-v2-Student-Phase1 --resume --previous-task Go2W-v2-Teacher-Phase5
#   --task Go2W-v2-Student-Phase2 --resume --previous-task Go2W-v2-Student-Phase1
#   --task Go2W-v2-Student-Phase5 --resume --previous-task Go2W-v2-Student-Phase2
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
