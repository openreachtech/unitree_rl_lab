"""The rungs of the A2 jump height ladder, in centimetres.

Kept in its own module, deliberately free of imports: ``__init__.py`` needs this tuple to
register one task per rung, and it is imported by ``scripts/rsl_rl/train.py`` and
``scripts/list_envs.py`` *before* the simulator app is launched, at which point pulling in
anything from ``isaaclab`` fails (``ModuleNotFoundError: pxr``). ``jump_env_cfg_jump.py``
reads the same tuple to generate the matching config classes.

Adding a rung means editing this line and nothing else -- the config classes and the gym
registrations are both generated from it.

Only 0.60 m is listed because only 0.60 m has been trained and verified end to end
(0.592 m unaided, 256/256 success). The rungs above it are NOT a matter of adding the
number: ``jump_assist_mass`` in ``jump_env_cfg_jump.py`` is calibrated for a 0.60 m
target, and a rung run on a mis-sized assist does not train -- it sits at zero success
with the assist stuck at 1.0, which is exactly what happened here before the calibration
sweep. Re-run that sweep for the new target height first.
"""

TARGET_HEIGHTS_CM = (60,)
