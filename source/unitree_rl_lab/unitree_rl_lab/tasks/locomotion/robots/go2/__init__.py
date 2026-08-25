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

# Promoted from sandbox Try 9: the fastest Go2 policy measured so far (3.72 m/s unaided, clean
# trot). Forward-only running with NO footfall prescription -- the matched Try 8, which shaped a
# gallop, was slower at every commanded speed -- on the mujoco-matched actuator model
# (UNITREE_GO2_CORRECTED_CFG), bootstrapped by the self-decaying tow assist. Two protective
# deviations from Try 9 as run (3.8 m/s ceiling, two-way persisted velocity ratchet) are
# explained in velocity_env_cfg_run.py. This replaces the earlier Go2-Run, which exposed gait
# style as a command (gait_command/gait_clock observations + gait_tracking_reward) and went
# unused; recover it from git history if wanted.
gym.register(
    id="Go2-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotEnvCfgGo2Run",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotPlayEnvCfgGo2Run",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Forward-only running specialized on a gallop-style footfall. paired_gait_reward is a
# self-referential reward computed purely from the robot's own current + recent contact-sensor
# history (front-pair/hind-pair relative timing lag, front-vs-hind alternation), needing no
# external clock or gait observation -- promoted from sandbox Try-5 + Try-6, where rewarding
# the front feet landing a target ~0.1-cycle apart (matching GAIT_GALLOP_ROTARY's structure)
# looked like a clean gallop in Play.
gym.register(
    id="Go2-Gallop-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotEnvCfgGo2GallopPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotPlayEnvCfgGo2GallopPhase1",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Promoted from sandbox Try-7: generalizes Go2-Gallop-Phase1 to omnidirectional commands (adds
# backward/lateral). paired_gait is gated on forward commanded speed (>= ~1.0 m/s) rather than
# unconditionally active -- backward/lateral commands always have lin_vel_x <= 0, so they fall
# below the gate automatically and the policy is free to choose its own footfall there, instead
# of being pushed toward a gallop-style structure that only makes sense running forward at
# speed. Verified in MuJoCo. Train with --previous-task Go2-Gallop-Phase1 --resume.
gym.register(
    id="Go2-Gallop-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotEnvCfgGo2GallopPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_run:RobotPlayEnvCfgGo2GallopPhase2",
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

# =============================================================================================
# Go2-Speed-Phase1 / Phase2 -- the top-speed pair, promoted from sandbox Try 8 and Try 20.
# Measured end to end: 5.31 m/s, tracking a commanded 5.5 m/s within 0.5 m/s.
#
# Two phases because one run cannot do both jobs. Phase 1 trains FROM SCRATCH with the
# gait reward on and a modest 4.0 m/s ceiling, and comes out with an asymmetric footfall
# pattern (front pair 0.15 of a cycle apart, hind pair 0.06). From-scratch is the
# mechanism, not a detail: paired_gait_reward grades pairs through a Gaussian, so a
# policy that already trots sits where that reward is ~1e-20 with a matching derivative,
# which is why the three sandbox runs that resumed from a walker (Try 13, 14, 18) could
# not move the gait at all.
#
# Phase 2 resumes from Phase 1, switches the gait reward OFF entirely, and spends the
# whole budget on speed: forward_command_progress 0.8, joint_pos -0.3, the four effort
# taxes (joint_vel, joint_acc, joint_torques, energy) at zero, ceiling 6.0. The gait is
# then free to revert to a trot and does not -- it stays 0.83 of a cycle from canonical
# trot, reaching 5.31 m/s with a 1.18 m stride at 4.5 Hz, against the best trot-gaited
# policy's 5.15-5.19 m/s at a 1.09 m stride and 4.8 Hz.
#
# SPRINT ONLY: Phase 2 removes every penalty that limits motor effort, and no thermal or
# battery model exists in Isaac Lab or unitree_mujoco. Restore joint_vel / joint_acc /
# joint_torques / energy before deriving any general-purpose task from it.
#
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Speed-Phase1 --max_iterations 1300
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Speed-Phase2 \
#       --previous-task Go2-Speed-Phase1 --max_iterations 2000
# =============================================================================================

gym.register(
    id="Go2-Speed-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_speed:RobotEnvCfgSpeedPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_speed:RobotPlayEnvCfgSpeedPhase1",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Speed-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_speed:RobotEnvCfgSpeedPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_speed:RobotPlayEnvCfgSpeedPhase2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# =============================================================================================
# Go2-Multitask-Gallop-Phase1 / Phase2 -- the same two-stage gallop curriculum as
# Go2-Gallop-Phase1 / Phase2, re-expressed on the unified 122/330-column observation shared with
# the acrobatics side (unitree_rl_lab.tasks.multitask.obs_spec). Task definition is otherwise
# identical: same commands, rewards, terminations, events, terrain and both curricula. The jump
# command is present but never triggered, which is what fills the five columns the locomotion
# observation lacked; see velocity_env_cfg_multitask.py for why that cannot disturb the physics.
#
# Together with Go2-Multitask-Jump-Phase1 / Phase2 these produce two experts on one observation
# layout, so the multi-task mixture-of-experts policy can load both with a plain load_state_dict.
#
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Gallop-Phase1 \
#       --max_iterations 1300
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Gallop-Phase2 \
#       --previous-task Go2-Multitask-Gallop-Phase1
#
# Trained from scratch rather than widened from Go2-Gallop-Phase2: no checkpoint for that task
# exists in this working tree (logs/ is untracked). If one turns up elsewhere, prefer
#   python scripts/rsl_rl/widen_checkpoint.py --previous-task Go2-Gallop-Phase2 \
#       --task Go2-Multitask-Gallop-Phase2
# which carries the trained policy over exactly instead of paying for it again.
# =============================================================================================

gym.register(
    id="Go2-Multitask-Gallop-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_multitask:RobotEnvCfgMultitaskGallopPhase1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_multitask:RobotPlayEnvCfgMultitaskGallopPhase1",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Multitask-Gallop-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_multitask:RobotEnvCfgMultitaskGallopPhase2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_multitask:RobotPlayEnvCfgMultitaskGallopPhase2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
