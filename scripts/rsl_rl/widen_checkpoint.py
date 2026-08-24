# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-express a single-task checkpoint on the unified multi-task observation.

``Go2-Gallop-Phase2`` and ``Go2-Jump-Phase2`` were trained on 117/319- and 47/56-column
observations; the multi-task family serves a 122/330-column superset. This widens a checkpoint's
input layers onto that superset by placing the original weights at the columns their inputs moved to
and zeroing the rest. Since a linear layer ignores inputs whose weights are zero, the widened
network computes **exactly** the original function -- the skill is carried over intact and the new
columns simply start inert.

Use it to start a unified-observation run from an existing policy instead of from scratch::

    python scripts/rsl_rl/widen_checkpoint.py --previous-task Go2-Jump-Phase2 \\
        --task Go2-Multitask-Jump-Phase2
    python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Jump-Phase2 --resume

``train.py --previous-task`` cannot be used for this hop: it symlinks the old run, and the old
weights no longer fit. The widened checkpoint is written as a real run directory under the new
task's log root so a plain ``--resume`` finds it.

Runs on any interpreter with PyTorch -- it deliberately loads ``obs_spec``/``weight_surgery`` by
path rather than importing ``unitree_rl_lab.tasks``, which would drag in Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import torch
from datetime import datetime

from checkpoint_utils import experiment_name_from_task, latest_checkpoint_name, latest_run_dir, log_root_from_task

_TASKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab", "tasks")
)


def _load_module(name: str, relative_path: str):
    """Import a module by file path, without executing its package's ``__init__``."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_TASKS_DIR, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


obs_spec = _load_module("_multitask_obs_spec", "multitask/obs_spec.py")
surgery = _load_module("_multitask_weight_surgery", "multitask/modules/weight_surgery.py")


def _resolve_mapping(source_dim: int, prefix: str):
    """Pick the column mapping for a checkpoint head, from the width it was trained at."""
    table = {
        "actor": {
            obs_spec.layout_dim(obs_spec.POLICY_LOCOMOTION): ("locomotion", obs_spec.POLICY_MAP_LOCOMOTION),
            obs_spec.layout_dim(obs_spec.POLICY_JUMP): ("acrobatics", obs_spec.POLICY_MAP_JUMP),
        },
        "critic": {
            obs_spec.layout_dim(obs_spec.CRITIC_LOCOMOTION): ("locomotion", obs_spec.CRITIC_MAP_LOCOMOTION),
            obs_spec.layout_dim(obs_spec.CRITIC_JUMP): ("acrobatics", obs_spec.CRITIC_MAP_JUMP),
        },
    }[prefix]
    if source_dim not in table:
        raise SystemExit(
            f"[error] The checkpoint's {prefix} input width is {source_dim}, which matches no known"
            f" single-task layout {sorted(table)}. Widen only checkpoints from this task family."
        )
    return table[source_dim]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--previous-task", required=True, help="Task ID whose checkpoint should be widened.")
    parser.add_argument("--task", required=True, help="Task ID to write the widened checkpoint for.")
    parser.add_argument("--load_run", default=None, help="Run directory to read. Defaults to the most recent.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file to read. Defaults to the latest.")
    parser.add_argument(
        "--keep-iter",
        action="store_true",
        help=(
            "Keep the source checkpoint's iteration counter. By default it is reset to 0, so that"
            " --max_iterations on the follow-up run means what it says."
        ),
    )
    args = parser.parse_args()

    source_root = log_root_from_task(args.previous_task)
    run_dir = os.path.join(source_root, args.load_run) if args.load_run else latest_run_dir(source_root)
    checkpoint_name = args.checkpoint or latest_checkpoint_name(run_dir)
    source_path = os.path.join(run_dir, checkpoint_name)

    print(f"[info] reading  {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"[error] {source_path} is not an rsl_rl checkpoint (no 'model_state_dict').")
    state_dict = checkpoint["model_state_dict"]

    widened: dict[str, torch.Tensor] = {
        key: value for key, value in state_dict.items() if not key.startswith(("actor.", "critic."))
    }
    for prefix, target_dim in (("actor", obs_spec.POLICY_DIM), ("critic", obs_spec.CRITIC_DIM)):
        source_dim = state_dict[surgery._first_linear_key(state_dict, prefix)].shape[1]
        if source_dim == target_dim:
            print(f"[info] {prefix}: already {target_dim} columns, copied unchanged")
            widened.update({f"{prefix}.{k}": v for k, v in _head(state_dict, prefix).items()})
            continue
        family, mapping = _resolve_mapping(source_dim, prefix)
        params = surgery.expand_state_dict(state_dict, target_dim, mapping, prefix)
        widened.update({f"{prefix}.{k}": v for k, v in params.items()})
        print(f"[info] {prefix}: {source_dim} -> {target_dim} columns ({family} layout)")

    # Adam's moments are per-column and no longer fit. Clearing the state -- while keeping the group
    # structure the loader checks -- lets the optimizer re-initialise them lazily on the first step.
    # One iteration of re-adaptation is a fair price for not having to reason about moment layout.
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer_state = {"state": {}, "param_groups": optimizer_state["param_groups"]}

    output = {
        "model_state_dict": widened,
        "optimizer_state_dict": optimizer_state,
        "iter": checkpoint.get("iter", 0) if args.keep_iter else 0,
        "infos": checkpoint.get("infos"),
    }

    target_root = log_root_from_task(args.task)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = os.path.join(target_root, f"{stamp}_widened_from_{experiment_name_from_task(args.previous_task)}")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"model_{output['iter']}.pt")
    torch.save(output, target_path)

    print(f"[info] wrote    {target_path}")
    print(f"[info] optimizer moments cleared; iteration counter {'kept' if args.keep_iter else 'reset to 0'}")
    print(f"\nNext: python scripts/rsl_rl/train_and_aggregate.py --task {args.task} --resume")


def _head(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) + 1 :]: value for key, value in state_dict.items() if key.startswith(f"{prefix}.")}


if __name__ == "__main__":
    main()
