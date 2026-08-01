"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `unitree_rl_lab` extension. Their task
IDs are recognised by the prefixes in `TASK_ID_PREFIXES`.
"""

"""Launch Isaac Sim Simulator first."""


import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    for info in pkgutil.iter_modules(path, prefix):

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}/source/unitree_rl_lab/unitree_rl_lab/tasks/")
    for package in ["locomotion.robots", "mimic.robots"]:
        package = importlib.import_module(package)
        for _ in _walk_packages(package.__path__, package.__name__ + "."):
            pass
    sys.path.pop(0)


import_packages()

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

# The gym registry also holds gymnasium's own environments and the isaaclab_tasks
# ones (several of which contain "Unitree" in their name), and our task IDs share
# no single common prefix, so the accepted prefixes are enumerated here. train.py
# imports get_task_ids() so that renaming a task only needs a change in one place.
TASK_ID_PREFIXES = ("Unitree-", "Go2W-")


def get_task_ids() -> list[str]:
    """Task IDs registered by this repo, in registration order."""
    return [spec.id for spec in gym.registry.values() if spec.id.startswith(TASK_ID_PREFIXES)]


def main():
    """Print all environments registered in `unitree_rl_lab` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Unitree RL Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    for index, task_id in enumerate(get_task_ids()):
        task_spec = gym.registry[task_id]
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
