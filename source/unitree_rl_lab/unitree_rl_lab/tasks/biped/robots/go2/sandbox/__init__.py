import gymnasium as gym  # noqa: F401

# NOTE: entry points must be lazy string references ("module:Class"), resolved by
# gym only once Isaac Sim's app is already running. Do not `from . import
# tryN` here -- that would eagerly import isaaclab.utils (needs `pxr`, only
# available after AppLauncher starts) at plain package-import time, e.g. when
# list_envs.py enumerates task IDs before launching the app.
#
# Try1-16 (the experimental history behind Go2-Biped-Phase1/Phase2, see
# ../biped_env_cfg.py and ../biped_env_cfg_phase2.py) have been cleared after
# promotion. See SUMMARY.md for the retained record of what was tried and why.
