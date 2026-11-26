import gymnasium as gym

# Aliengo + Z1 arm. Build the robot asset and the arm goal set first:
#   python scripts/tools/build_quadruped_z1_urdf.py --base aliengo
#   python scripts/tools/sample_z1_workspace.py --headless
#
# Locomotion pre-training with a scripted moving arm (MLP policy, legs only).
# Phase 1: flat plane. Phase 2: random rough / boxes / slopes, initialised from Phase 1.
_RUNNER = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

for _phase in (1, 2):
    gym.register(
        id=f"Aliengoz1-Locomotion-Phase{_phase}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotEnvCfgPhase{_phase}",
            "play_env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotPlayEnvCfgPhase{_phase}",
            "rsl_rl_cfg_entry_point": _RUNNER,
        },
    )

# Whole-body end-effector pose tracking (MLP, legs + arm). Needs the initial states recorded from
# Aliengoz1-Locomotion-Phase2:  python scripts/tools/collect_init_states.py --headless
gym.register(
    id="Aliengoz1-Pose-Tracking-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_tracking_env_cfg:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.pose_tracking_env_cfg:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PoseTrackingPPORunnerCfg",
    },
)

# Whole-body end-effector twist tracking (MLP actor, one critic per reward group), from scratch.
# Phase 1: flat plane. Phase 2: random rough / boxes / slopes, initialised from Phase 1.
for _phase in (1, 2):
    gym.register(
        id=f"Aliengoz1-Twist-Tracking-Phase{_phase}",
        entry_point="unitree_rl_lab.envs.multi_critic_env:MultiCriticRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.twist_tracking_env_cfg:RobotEnvCfgPhase{_phase}",
            "play_env_cfg_entry_point": f"{__name__}.twist_tracking_env_cfg:RobotPlayEnvCfgPhase{_phase}",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TwistTrackingPPORunnerCfg",
        },
    )

# Arm only (fixed base, legs at default): the arm part of the twist tracking task in isolation.
gym.register(
    id="Aliengoz1-Twist-Tracking-ArmOnly",
    entry_point="unitree_rl_lab.envs.multi_critic_env:MultiCriticRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twist_tracking_env_cfg:RobotEnvCfgArmOnly",
        "play_env_cfg_entry_point": f"{__name__}.twist_tracking_env_cfg:RobotPlayEnvCfgArmOnly",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TwistTrackingArmOnlyPPORunnerCfg",
    },
)
