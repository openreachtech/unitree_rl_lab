# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for chaining checkpoints between training tasks."""

from __future__ import annotations

import os
import re


def experiment_name_from_task(task_name: str) -> str:
    return task_name.lower().replace("-", "_").removesuffix("_play")


def log_root_from_task(task_name: str) -> str:
    return os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name_from_task(task_name)))


def latest_run_dir(log_root: str) -> str:
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root not found: {log_root}")

    runs = [
        os.path.join(log_root, name)
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name)) and not os.path.islink(os.path.join(log_root, name))
    ]
    if not runs:
        raise FileNotFoundError(f"No run directories found in: {log_root}")
    return max(runs, key=os.path.getmtime)


def latest_checkpoint_name(run_dir: str) -> str:
    model_files = [name for name in os.listdir(run_dir) if re.match(r"model_.*\.pt", name)]
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found in: {run_dir}")
    model_files.sort(key=lambda name: f"{name:0>15}")
    return model_files[-1]


def link_previous_run(prev_log_root: str, load_run: str, log_root: str) -> None:
    """Symlink a previous run into the current task log root for ``train.py --resume``."""
    os.makedirs(log_root, exist_ok=True)
    link_path = os.path.join(log_root, load_run)
    src_path = os.path.join(prev_log_root, load_run)
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"Previous run directory not found: {src_path}")
    if os.path.lexists(link_path):
        return
    os.symlink(src_path, link_path, target_is_directory=True)


def resolve_previous_task_checkpoint(
    previous_task: str,
    current_task: str,
    load_run: str | None = None,
    checkpoint: str | None = None,
) -> tuple[str, str, str, str]:
    """Resolve and link a checkpoint from ``previous_task`` for training ``current_task``.

    Returns:
        Tuple of ``(prev_log_root, current_log_root, load_run, checkpoint)``.
    """
    if previous_task == current_task:
        raise ValueError("--previous-task must differ from --task")

    prev_log_root = log_root_from_task(previous_task)
    current_log_root = log_root_from_task(current_task)

    if load_run is None:
        load_run = os.path.basename(latest_run_dir(prev_log_root))
    if checkpoint is None:
        checkpoint = latest_checkpoint_name(os.path.join(prev_log_root, load_run))

    link_previous_run(prev_log_root, load_run, current_log_root)
    return prev_log_root, current_log_root, load_run, checkpoint
