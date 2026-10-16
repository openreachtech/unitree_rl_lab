"""Go2w biped sandbox -- one-change-at-a-time experiments against the default biped tasks.

``sandbox/try*.py`` are kept local (``.gitignore``) and loaded below; every experiment that works
is folded into the default task and its try deleted. Try1-3 (Go2w-Biped: start from the hind
policy, balance the stance population, front-only ``front_hip_height``) have been folded.

Convention for adding a try:
  * build on a default task (``Go2w-Biped`` / ``-Front`` / ``-Hind``), not on another try, so each
    run is a change against a known baseline;
  * register it as ``<task>-TryN`` with a fresh N -- reusing a number reuses its log directory;
  * this file imports each ``try*.py`` before the simulator starts, so a try must not import
    isaaclab at module level -- register a function that builds the config instead
    (``env_cfg_entry_point=f"{__name__}:env_cfg"``, importing inside ``env_cfg``).
"""

import importlib
import pkgutil

# Load whatever local try*.py files exist, so their gym.register calls run. The task loader only
# imports packages, not plain modules, and the try files themselves are not committed.
for _module in pkgutil.iter_modules(__path__):
    if _module.name.startswith("try"):
        importlib.import_module(f"{__name__}.{_module.name}")
