# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a sequence of RL tasks by calling ``train.py`` for each phase.

Example::

    python scripts/rsl_rl/train_sequence.py \\
        --tasks Unitree-Go2-Velocity-v1-Phase1 Unitree-Go2-Velocity-v1-Phase2 Unitree-Go2-Velocity-v1-Phase3 \\
        --max_iterations 5000 10000 15000 \\
        -- --num_envs 4096 --headless --deploy-keyboard-commands

Everything after ``--`` is forwarded to ``train.py`` (except ``--task``, ``--max_iterations``, ``--resume``,
``--load_run``, and ``--checkpoint``).
Logs use ``train.py`` defaults: ``logs/rsl_rl/<task_name>/`` per task.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    return argv, []


def _reject_reserved(extra_argv: list[str]) -> None:
    for flag in ("--resume", "--load_run", "--checkpoint"):
        if flag in extra_argv:
            raise SystemExit(
                f"[ERROR] {flag} is not allowed. Checkpoints are chained automatically between tasks."
            )


def _log_root(task_name: str) -> str:
    experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    return os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))


def _latest_run_dir(log_root: str) -> str:
    runs = [
        os.path.join(log_root, name)
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name)) and not os.path.islink(os.path.join(log_root, name))
    ]
    if not runs:
        raise FileNotFoundError(f"No run directories found in: {log_root}")
    return max(runs, key=os.path.getmtime)


def _latest_checkpoint_name(run_dir: str) -> str:
    model_files = [name for name in os.listdir(run_dir) if re.match(r"model_.*\.pt", name)]
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found in: {run_dir}")
    model_files.sort(key=lambda name: f"{name:0>15}")
    return model_files[-1]


def _link_previous_run(prev_log_root: str, load_run: str, log_root: str) -> None:
    """Symlink the previous run into the next task log root for ``train.py --resume``."""
    os.makedirs(log_root, exist_ok=True)
    link_path = os.path.join(log_root, load_run)
    src_path = os.path.join(prev_log_root, load_run)
    if os.path.lexists(link_path):
        return
    os.symlink(src_path, link_path, target_is_directory=True)


def main() -> None:
    seq_argv, extra_argv = _split_argv(sys.argv[1:])
    _reject_reserved(extra_argv)

    parser = argparse.ArgumentParser(description="Train a sequence of RL tasks with checkpoint chaining.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Gym task IDs to train in order.")
    parser.add_argument("--max_iterations", nargs="+", type=int, required=True, help="Iterations per task.")
    seq_args = parser.parse_args(seq_argv)

    if len(seq_args.tasks) != len(seq_args.max_iterations):
        raise SystemExit("[ERROR] --tasks and --max_iterations must have the same length.")

    prev_log_root: str | None = None
    load_run: str | None = None
    checkpoint: str | None = None

    for index, (task_name, max_iterations) in enumerate(zip(seq_args.tasks, seq_args.max_iterations)):
        log_root = _log_root(task_name)
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--task", task_name, "--max_iterations", str(max_iterations)]
        cmd.extend(extra_argv)
        if load_run is not None:
            _link_previous_run(prev_log_root, load_run, log_root)
            cmd.extend(["--resume", "--load_run", load_run, "--checkpoint", checkpoint])

        print(f"\n[INFO] ===== Step {index + 1}/{len(seq_args.tasks)}: {task_name} =====")
        print(f"[INFO] Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        run_dir = _latest_run_dir(log_root)
        prev_log_root = log_root
        load_run = os.path.basename(run_dir)
        checkpoint = _latest_checkpoint_name(run_dir)
        print(f"[INFO] Latest checkpoint for next task: {log_root}/{load_run}/{checkpoint}")


if __name__ == "__main__":
    main()
