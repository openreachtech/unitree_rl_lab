import gymnasium as gym

from . import sandbox  # noqa: F401

gym.register(
    id="Unitree-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v1-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotEnvCfgPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase1:RobotPlayEnvCfgPhase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v1-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase2:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Promoted from sandbox Try-4: terrain-adaptive foot clearance for a natural
# flat-ground gait, terrain_levels >= 4.5 (reached 4.899).
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-balance",
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
    id="Unitree-Go2-Velocity-v1-Phase3-stairfocus",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3StairFocus",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3StairFocus",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Balance rewards + terrain mix that includes floating inverted pyramid stairs.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-balance-floating",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotEnvCfgPhase3BalanceFloating",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase3:RobotPlayEnvCfgPhase3BalanceFloating",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
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
    id="Unitree-Go2-Velocity-v1-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotEnvCfgPhase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_phase4:RobotPlayEnvCfgPhase4",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Go2-Crouch-Phase1 (renamed from Go2-Crouch on 2026-09-02, when Phase2 was introduced):
# based on Phase1 (same small velocity-command ranges, flat ground) plus a new base_height
# command term: walk while tracking a commanded base height, curriculum-widened from
# standing height down to a 25cm crouch -- see velocity_env_cfg_crouch_phase1.py for why
# it's a command rather than a reward constant, and for the reward-tuning history.
gym.register(
    id="Go2-Crouch-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase1:RobotEnvCfgCrouchPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase1:RobotPlayEnvCfgCrouchPhase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Phase 2: rough terrain and boxes (2 columns x 10 rows), base_velocity's lin_vel_x/lin_vel_y
# limits both widened to +-1.0 m/s. base_height command/curriculum/reward unchanged from
# Phase1 -- see velocity_env_cfg_crouch_phase2.py.
gym.register(
    id="Go2-Crouch-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase2:RobotEnvCfgCrouchPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase2:RobotPlayEnvCfgCrouchPhase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Phase 4 (no Phase3 -- see module docstring): rough terrain and thin walls to step over
# (4 columns x 10 rows, 1 rough : 3 wall). undesired_contacts split terrain-column-aware
# for the wall columns, terrain_levels_climb_demote_on_fail curriculum -- see
# velocity_env_cfg_crouch_phase4.py.
gym.register(
    id="Go2-Crouch-Phase4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase4:RobotEnvCfgCrouchPhase4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_crouch_phase4:RobotPlayEnvCfgCrouchPhase4",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

