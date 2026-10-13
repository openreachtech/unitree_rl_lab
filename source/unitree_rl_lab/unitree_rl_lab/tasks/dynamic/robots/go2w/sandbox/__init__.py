"""Go2w-Jump-Phase2 sandbox -- one-change-at-a-time experiments against ``Go2w-Jump-Phase2``.

Currently empty in the repo: ``sandbox/try*.py`` are kept local (``.gitignore``) and every
experiment that worked has been folded into the default tasks.

Convention for adding a try:
  * inherit from the default ``Go2w-Jump-Phase2`` (``jump_env_cfg_phase2.py``), not from
    another try, so each run is a change against a known baseline;
  * register it locally as ``Go2w-Jump-Phase2-TryN`` with a fresh N -- reusing a number
    reuses its ``logs/rsl_rl/go2w_jump_phase2_tryN/`` directory (and its curriculum state
    file); Try1 (wheel_vel) and Try2 (actuator model) are taken;
  * once it works, fold the change into the default task and delete the try.
"""
