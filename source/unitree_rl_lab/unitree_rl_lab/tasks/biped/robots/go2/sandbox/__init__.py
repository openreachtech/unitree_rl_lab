import gymnasium as gym

# NOTE: entry points must be lazy string references ("module:Class"), resolved by
# gym only once Isaac Sim's app is already running. Do not `from . import
# tryN` here -- that would eagerly import isaaclab.utils (needs `pxr`, only
# available after AppLauncher starts) at plain package-import time, e.g. when
# list_envs.py enumerates task IDs before launching the app.
#
# Try1-16 (the experimental history behind Go2-Biped-Phase1/Phase2, see
# ../biped_env_cfg.py and ../biped_env_cfg_phase2.py), a second round
# (mode-switching-in-one-step attempts, discarded), and a third round (Try1
# pinned-gait_mode Phase1, Try2 real switching attempt, Try3..7
# reward-shaping experiments -- lower base_height, free front legs, a
# stand_still penalty) have all had their tryN.py files deleted after
# promotion/rejection. Try6's recipe was briefly promoted into
# ../biped_env_cfg.py, then reverted -- Phase1's original recipe won a
# side-by-side metric comparison. See SUMMARY.md for the retained record of
# what was tried and why.
#
# No sandbox tasks are currently registered -- add new ones here as new tryN.py
# files are written.
