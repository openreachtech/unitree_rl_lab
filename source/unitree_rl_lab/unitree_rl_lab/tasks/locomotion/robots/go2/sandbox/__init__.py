import gymnasium as gym

# =============================================================================
# Try 1 — Stair Climbing Reward Tuning (extends Unitree-Go2-Velocity-v1-Phase3)
#
# Problem: Phase 3 robot stalls and fails to climb pyramid stairs.
#
# Strategy:
#   Relax penalties that conflict with stair-climbing mechanics, raise foot
#   clearance targets to match actual step height, and enable the
#   forward_command_progress term to provide a gradient when the standard
#   velocity-tracking reward saturates near a stalled robot.
#
# Changes vs Phase 3:
#   flat_orientation_l2   : -1.0  → -0.3   body must pitch on slopes/stairs
#   base_linear_velocity  : -0.5  → -0.2   vertical body motion is necessary
#   joint_pos             : -0.7  → -0.3   larger joint excursions to step up
#   undesired_contacts    : -1.0  → -0.3   calves naturally brush stair edges
#   feet_air_time         : 0.07  → 0.15   longer swing phase to clear steps
#   wild_foot_clearance   : w=0.4, target=0.05 m
#                         → w=0.6, target=0.12 m  (steps up to 0.23 m)
#   forward_command_progress : 0.0 → 0.5   linear progress signal to bootstrap
#                                           climbing when vel-tracking saturates
# =============================================================================

gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.try1:RobotEnvCfgPhase3StairClimb",
        "play_env_cfg_entry_point": f"{__name__}.try1:RobotPlayEnvCfgPhase3StairClimb",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
