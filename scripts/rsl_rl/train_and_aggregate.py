#!/usr/bin/env python3
"""Run training then aggregate TensorBoard logs with quiet subprocess output."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from checkpoint_utils import latest_run_dir, log_root_from_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and aggregate logs in one command.")
    parser.add_argument("--task", required=True, help="Current task ID passed to train.py")
    parser.add_argument(
        "--previous-task",
        default=None,
        help="Previous task ID. If set, train.py resumes from this task's latest checkpoint.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3000,
        help="Training iterations for train.py (default: 3000).",
    )
    parser.add_argument(
        "--aggregate-interval",
        type=int,
        default=100,
        help="Iteration interval for aggregate_tensorboard_logs.py.py (default: 100).",
    )
    return parser.parse_args()


def run_quiet(command: list[str], label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    raise RuntimeError(
        f"{label} failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )


def latest_summary_file(run_dir: Path) -> Path:
    summaries = sorted(run_dir.glob("events_log_summary_*.md"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"No aggregate markdown found in: {run_dir}")
    return summaries[-1]


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    aggregate_script = script_dir / "aggregate_tensorboard_logs.py.py"

    train_cmd = [
        sys.executable,
        str(train_script),
        "--task",
        args.task,
        "--num_envs",
        "4096",
        "--headless",
        "--max_iterations",
        str(args.max_iterations),
    ]
    if args.previous_task:
        train_cmd.extend(["--resume", "--previous-task", args.previous_task])

    run_quiet(train_cmd, "train.py")

    task_log_root = Path(log_root_from_task(args.task))
    aggregate_cmd = [
        sys.executable,
        str(aggregate_script),
        "--logdir",
        str(task_log_root),
        "--interval",
        str(args.aggregate_interval),
        "--overwrite",
    ]
    run_quiet(aggregate_cmd, "aggregate_tensorboard_logs.py.py")

    run_dir = Path(latest_run_dir(str(task_log_root)))
    summary_path = latest_summary_file(run_dir)
    print(f"[Summary File] {summary_path}")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
