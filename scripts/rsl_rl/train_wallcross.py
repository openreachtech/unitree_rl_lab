"""Run ``train.py`` with the wall crossing tasks visible to it.

Local helper. ``train.py`` validates ``--task`` against the families ``scripts/list_envs.py``
imports, a hard-coded list that does not include ``wallcross``. Rather than edit that shared list,
this registers the wallcross family the same way ``list_envs`` registers the others and then runs
``train.py`` unchanged, with the same arguments.
"""

import importlib
import pathlib
import runpy
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent.parent / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "tasks"))
importlib.import_module("wallcross.robots.go2")
importlib.import_module("wallcross.robots.anaguma")
sys.path.pop(0)

sys.argv[0] = str(here / "train.py")
sys.path.insert(0, str(here))
runpy.run_path(str(here / "train.py"), run_name="__main__")
