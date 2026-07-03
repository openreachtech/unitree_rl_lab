# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a sequence of RL tasks by calling ``train.py`` for each phase.

Example::

    python scripts/rsl_rl/train_phases.py \\
        --tasks Unitree-Go2-Velocity-v1-Phase1 Unitree-Go2-Velocity-v1-Phase2 Unitree-Go2-Velocity-v1-Phase3 \\
        --max_iterations 5000 10000 15000 \\
        -- --num_envs 4096 --headless --deploy-keyboard-commands

Everything after ``--`` is forwarded to ``train.py`` (except ``--task``, ``--max_iterations``, ``--resume``,
``--load_run``, ``--checkpoint``, and ``--previous-task``).
Logs use ``train.py`` defaults: ``logs/rsl_rl/<task_name>/`` per task.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from checkpoint_utils import latest_checkpoint_name, latest_run_dir, log_root_from_task

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    return argv, []


def _reject_reserved(extra_argv: list[str]) -> None:
    for flag in ("--resume", "--load_run", "--checkpoint", "--previous-task"):
        if flag in extra_argv:
            raise SystemExit(
                f"[ERROR] {flag} is not allowed. Checkpoints are chained automatically between tasks."
            )


def main() -> None:
    seq_argv, extra_argv = _split_argv(sys.argv[1:])
    _reject_reserved(extra_argv)

    parser = argparse.ArgumentParser(description="Train a sequence of RL tasks with checkpoint chaining.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Gym task IDs to train in order.")
    parser.add_argument("--max_iterations", nargs="+", type=int, required=True, help="Iterations per task.")
    seq_args = parser.parse_args(seq_argv)

    if len(seq_args.tasks) != len(seq_args.max_iterations):
        raise SystemExit("[ERROR] --tasks and --max_iterations must have the same length.")

    previous_task: str | None = None

    for index, (task_name, max_iterations) in enumerate(zip(seq_args.tasks, seq_args.max_iterations)):
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--task", task_name, "--max_iterations", str(max_iterations)]
        cmd.extend(extra_argv)
        if previous_task is not None:
            cmd.extend(["--previous-task", previous_task, "--resume"])

        print(f"\n[INFO] ===== Step {index + 1}/{len(seq_args.tasks)}: {task_name} =====")
        print(f"[INFO] Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        run_dir = latest_run_dir(log_root_from_task(task_name))
        checkpoint = latest_checkpoint_name(run_dir)
        previous_task = task_name
        print(f"[INFO] Latest checkpoint for next task: {run_dir}/{checkpoint}")


if __name__ == "__main__":
    main()
