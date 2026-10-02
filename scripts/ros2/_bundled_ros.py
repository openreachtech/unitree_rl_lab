"""Reach Isaac's bundled ROS 2 (Humble, py3.11) without starting Isaac Sim.

The ``isaacsim.ros2.bridge`` extension ships a complete Humble for py3.11 -- the python
packages under ``humble/rclpy`` and the shared objects under ``humble/lib``. Isaac
normally exposes them via ``enable_extension()``, which needs the simulator running, but
nothing about ``import rclpy`` does: the python path and the dynamic loader path are the
whole dependency. A node that only wants to talk ROS (the VLM scorer, say) can therefore
skip booting a simulator it has no use for.

The catch is the loader. ``LD_LIBRARY_PATH`` is read once by ld.so at process start, so
setting ``os.environ`` later is invisible to it -- the only way in is to prepare the
environment and exec ourselves. Hence::

    from _bundled_ros import ensure_bundled_ros
    ensure_bundled_ros()          # may not return: re-execs this process
    import rclpy                  # now resolves to the bundled Humble

Call it before importing rclpy and before anything expensive, since everything above the
call runs twice.

Do NOT source a system ROS into the same shell. The bridge's libraries are chosen exactly
when no system ROS is on the environment, and mixing a py3.10 Humble with this py3.11 one
fails in ways that look like missing symbols.

``scripts/ros2/play_ros2.py`` does not use this: it needs the simulator anyway, so it
takes Isaac's own ``enable_extension`` route after the app is up. Both end at the same
libraries.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def bundled_ros_paths() -> tuple[str, str]:
    """(lib dir, python package dir) of the bundled Humble."""
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "isaacsim is not importable -- run this from the env_isaaclab venv, which is"
            " where the bundled ROS 2 lives."
        )
    ext = os.path.join(os.path.dirname(spec.origin), "exts", "isaacsim.ros2.bridge", "humble")
    lib, pkgs = os.path.join(ext, "lib"), os.path.join(ext, "rclpy")
    if not os.path.isdir(lib) or not os.path.isdir(pkgs):
        raise RuntimeError(f"bundled ROS 2 not found under {ext}")
    return lib, pkgs


def ensure_bundled_ros() -> None:
    """Put the bundled Humble on the loader path, re-execing if it is not there yet."""
    lib, pkgs = bundled_ros_paths()
    if lib in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if pkgs not in sys.path:
            sys.path.insert(0, pkgs)
        return
    env = dict(os.environ)
    env["ROS_DISTRO"] = "humble"
    env["LD_LIBRARY_PATH"] = (lib + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
    env["PYTHONPATH"] = (pkgs + ":" + env.get("PYTHONPATH", "")).rstrip(":")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
