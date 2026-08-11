"""Go2W Phase5 sandbox -- one-change-at-a-time experiments against the default Phase5.

Currently empty: every experiment run so far has been folded into
``velocity_env_cfg_phase5.py``. See SUMMARY.md in this directory for what was tried, what
each attempt established, and which conclusions are load-bearing.

Convention for adding a try:
  * inherit from the *default* Phase5, not from another try, so each run is a single
    isolated change against a baseline whose numbers are known;
  * register it here as ``Go2W-v1-Phase5-TryN`` with a fresh N -- reusing a number reuses
    its ``logs/rsl_rl/go2w_v1_phase5_tryN/`` directory, and mixing two configs' runs in
    one directory has caused resume-from-the-wrong-checkpoint mistakes before;
  * record the outcome in SUMMARY.md, then fold the change into Phase5 and delete the try.
"""
