import gymnasium as gym

# =================================================================================================
# Go2-Multitask-WallCross-TerrainCheck -- the wall terrain under the hind-stance expert, unchanged.
#
# Not a training task. It exists to look at the concentric-wall terrain before any reward is written
# on top of it: that the tiles come out in the intended shape and order, and that a 3 cm wall stops
# a foot rather than letting it through. Everything except the terrain and the spawn is
# `Go2-Multitask-Biped-Hind`, so its checkpoints run here as they are.
#
#   python scripts/rsl_rl/check_wall_terrain.py --task Go2-Multitask-WallCross-TerrainCheck \
#       --checkpoint logs/rsl_rl/go2_multitask_biped_hind/<run>/model_1499.pt --headless
# =================================================================================================
gym.register(
    id="Go2-Multitask-WallCross-TerrainCheck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallTerrainCheck",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallTerrainCheck",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# =================================================================================================
# Go2-Multitask-WallCross -- rise onto the hind legs, walk to the wall, get over it, land on four.
#
# An expert for the mixture, on the unified observation unchanged (actor 124 / critic 335), so it is
# started from the hind-stance expert without widening anything:
#
#   python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-WallCross \
#       --previous-task Go2-Multitask-Biped-Hind --load_run <run> --checkpoint model_1499.pt
#
# The cue is the one every move already uses: at the wall the stance is released and
# `jump_command` fires, its `target_height` saying how far the base must rise for this wall. See
# `tasks/wallcross/mdp/commands.py`.
# =================================================================================================
gym.register(
    id="Go2-Multitask-WallCross",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCross",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCross",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst -- the same task with the wall thinned (at 0.30-0.40 m) before it
# is raised, so the high walls are met thin. See `order` in `tasks/wallcross/terrains.py`.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirst",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirst",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-Paced -- thin-first, with the approach held to the commanded pace
# until the jump (`approach_overspeed`). run6 ran at the wall at 1.9 m/s and could not stop when no
# jump was given.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-Paced",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstPaced",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstPaced",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-PacedStrict -- run8: `approach_overspeed` with margin 0.05 and weight
# -30. run7 (Paced) still parked at 1.1 m/s and fell in every no-jump episode.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-PacedStrict",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstPacedStrict",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstPacedStrict",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-PacedStrictFall -- run9: PacedStrict plus a fall before the jump
# priced at 20 instead of 4. run8 (PacedStrict) collapsed into falling 0.2 s after the start.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-PacedStrictFall",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstPacedStrictFall",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstPacedStrictFall",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-PacedStrictFallFloor -- run10: PacedStrictFall plus a floor under the
# stance's base height (0.45 m) until the jump. run9 approached crouched at 0.33 m, and parked, sank
# until a rear hip touched the floor 0.2 s later in every no-jump episode.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-PacedStrictFallFloor",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstPacedStrictFallFloor",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstPacedStrictFallFloor",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalk -- run11: the stance is only called for to cross a wall
# (tanaka, 2026-09-30). Episodes start mid-walk from recorded walking-expert states (0.5-0.8 m/s) with
# the stance already on, and every episode crosses; walking past a wall is the walking expert's job.
# Rewards are Paced (run7) unchanged.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalk",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalk",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLand -- run12: FromWalk plus a fall after the success
# judgement costs 20 (was 4), so the robot is still on its feet when the walking expert takes over.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLand",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLand",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandPitch -- run13: FromWalkLand plus the pitch rate charged
# on the far side of the wall until success (the falls after the jump were nose-first landings).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandPitch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandPitch",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandPitch",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll -- run13: FromWalkLand plus a fall between the jump
# and success costs 20 as well, so every fall costs 20.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAll",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAll",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAllTimeout -- run14: FromWalkLandAll plus -10 for
# ending at the wall by the timeout, so that trying and falling beats not trying.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAllTimeout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllTimeout",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllTimeout",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM -- run13's task plus a height map of the wall
# ahead on the actor and the critic (155 columns after the unified layout). Started from run13 with
# the new columns at zero weight: scripts/rsl_rl/widen_wallcross_heightmap.py.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllHM",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHM",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-VL -- PhysX joint speed limits drawn per episode (calf 15.7-35, hip/thigh 30.1-45 rad/s): the
# MuJoCo gap (2026-10-04) came from Isaac's hard speed cap draining the push-off.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-VL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllVL",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllVL",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-VL -- PhysX joint speed limits drawn per episode (calf 15.7-35, hip/thigh 30.1-45 rad/s): the
# MuJoCo gap (2026-10-04) came from Isaac's hard speed cap draining the push-off.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-VL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMVL",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMVL",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Kick -- HM4: the height-map task plus a random
# base velocity at take-off (up 0-0.9 m/s, along the heading -0.4..0.1, angular +-1 rad/s, 75% of
# jumps), to practise the higher, shorter jumps and harder landings seen in MuJoCo.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Kick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMKick",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMKick",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop -- HM5: the height-map task plus a random
# base velocity on the way down after the apex (down 0-1.2 m/s, along the heading -0.5..0.1, angular
# +-2 rad/s, 75% of jumps; the first setting, -0.8/-0.3/+-1, left HM3's falls unchanged): landing practice that a weaker push-off cannot cancel (HM4 backfired).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossThinFirstFromWalkLandAllHMDrop",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossThinFirstFromWalkLandAllHMDrop",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-NoCap -- height map, joint speed caps lifted (calf 35, hip/thigh 45 rad/s): MuJoCo-like push-off.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-NoCap",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossHMNoCap",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossHMNoCap",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop-NoCap -- height map, descent kick, joint speed caps lifted.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop-NoCap",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossHMDropNoCap",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossHMDropNoCap",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop-VL -- height map, descent kick, joint speed caps randomized.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Drop-VL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossHMDropVL",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossHMDropVL",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Robust -- height map, approach state randomized (cue gap 0.15-0.8 m, approach 0.4-1.0 m/s, pushes every 1.5-3 s).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossHMRobust",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossHMRobust",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-VL-Robust -- height map, joint speed caps randomized, approach state randomized.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-HM-VL-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossHMVLRobust",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossHMVLRobust",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360 -- the actor reads the MID-360 height map (29 x 21, real chin mount, noise); critic keeps the analytic wall map.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Blank -- evaluation only: the Mid360 task with the map held at flat ground.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Blank",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360Blank",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360Blank",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Clear -- Mid360, and base/hip/thigh contact during the crossing penalized.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Clear",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360Clear",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360Clear",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Clear-Gate -- Mid360-Clear, tiles concentrated on 0.55-0.80 m thin walls.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Clear-Gate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360ClearGate",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360ClearGate",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Robust -- Mid360 + approach-state and joint-speed-cap randomization, envs start on every row.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360Robust",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360Robust",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeLand -- Mid360 + landing safety (shin/thigh/head off the floor, joint limits, MID-360 height), envs start on every row.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeLand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeLand",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeLand",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeFall -- thigh/calf/head on the floor after the landing costs a fall (once), joint limits -16, MID-360 height -120.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeFall",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeFall",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeFall",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeStrong -- SafeLand per-second terms about 2x, crossing paid 40.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeStrong",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeStrong",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeStrong",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeFall2 -- SafeFall with the per-second terms doubled (stage 2).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeFall2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeFall2",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeFall",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeStrong2 -- SafeStrong with the per-second terms doubled (stage 2).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeStrong2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeStrong2",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeStrong",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafePose -- Mid360 + a reward for the standing pose after the landing (plus light floor-contact/MID-360 terms).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafePose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafePose",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafePose",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafePoseLim -- SafePose, pose reward 1.5x and the joint-limit term from the jump on.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafePoseLim",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafePoseLim",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafePoseLim",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafePose -- blind (run15's VL task) + the landing-pose reward of Mid360-SafePose.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafePose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossBlindSafePose",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossBlindSafePose",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafeSoft -- Blind-SafePose with the leg-strike force (body weights) charged instead of the contact count.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafeSoft",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossBlindSafeSoft",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossBlindSafeSoft",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeMount -- Mid360-SafePose with the MID-360 height term at the real mount (0.287, 0, -0.137).
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360-SafeMount",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossMid360SafeMount",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossMid360SafeMount",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafeMount -- Blind-SafeSoft with the MID-360 height term at the real mount.
gym.register(
    id="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Blind-SafeMount",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotEnvCfgWallCrossBlindSafeMount",
        "play_env_cfg_entry_point": f"{__name__}.wallcross_env_cfg:RobotPlayEnvCfgWallCrossBlindSafeMount",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
