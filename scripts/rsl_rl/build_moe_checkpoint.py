# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble the multi-task policy's starting checkpoint from two single-task runs.

Expert 0 is loaded from the locomotion run and expert 1 from the acrobatics run; expert 2 and the
gates stay randomly initialised. Both experts are expected to have been trained on the unified
122/330-column observation (``Go2-Multitask-Gallop-Phase2`` / ``Go2-Multitask-Jump-Phase2``), in
which case the weights load directly; a narrower checkpoint is widened on the way in, which leaves
the network computing exactly the same function.

The action-noise parameter is carried over too, and that is not cosmetic. rsl_rl defaults
``init_noise_std`` to 1.0 while these policies converged near 0.20 and 0.59; starting a fine-tune at
1.0 would sample actions with several times the noise they operate under and wreck both behaviours
on the first step -- exactly what initialising from them was meant to prevent.

The result is written as a run directory under the multi-task log root, so training picks it up with
a plain ``--resume``::

    python scripts/rsl_rl/build_moe_checkpoint.py \\
        --locomotion-task Go2-Multitask-Gallop-Phase2 \\
        --acrobatics-task Go2-Multitask-Jump-Phase2 \\
        --task Go2-Multitask-Phase1
    python scripts/rsl_rl/train_and_aggregate.py --task Go2-Multitask-Phase1 --resume

Runs on any interpreter with PyTorch and rsl_rl; it does not import ``unitree_rl_lab.tasks``, which
would drag in Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
from datetime import datetime

from checkpoint_utils import latest_checkpoint_name, latest_run_dir, log_root_from_task

_PACKAGE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab")
)
sys.path.insert(0, os.path.join(_PACKAGE, "tasks"))
sys.path.insert(0, os.path.join(_PACKAGE, "assets"))

from models import MoEActorCritic, MoEPPO, initialize_experts  # noqa: E402
from multitask import obs_spec  # noqa: E402


def resolve_checkpoint(task: str, load_run: str | None, checkpoint: str | None) -> str:
    root = log_root_from_task(task)
    run_dir = os.path.join(root, load_run) if load_run else latest_run_dir(root)
    return os.path.join(run_dir, checkpoint or latest_checkpoint_name(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Multi-task task ID to write the checkpoint for.")
    parser.add_argument("--locomotion-task", default=None, help="Task ID supplying expert 0.")
    parser.add_argument("--acrobatics-task", default=None, help="Task ID supplying expert 1.")
    parser.add_argument("--locomotion-run", default=None)
    parser.add_argument("--locomotion-checkpoint", default=None)
    parser.add_argument("--acrobatics-run", default=None)
    parser.add_argument("--acrobatics-checkpoint", default=None)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--gating-hidden-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument(
        "--noise-std-mode",
        default="min",
        choices=["min", "locomotion", "acrobatics", "keep"],
        help=(
            "How to combine the two checkpoints' action-noise parameters into the single shared one."
            " 'min' (default) takes the elementwise minimum, keeping sampled actions closest to what"
            " both were tuned for; the acrobatic skill is the more fragile under added noise."
        ),
    )
    args = parser.parse_args()

    if not args.locomotion_task and not args.acrobatics_task:
        raise SystemExit("[error] Give at least one of --locomotion-task / --acrobatics-task.")

    locomotion = (
        resolve_checkpoint(args.locomotion_task, args.locomotion_run, args.locomotion_checkpoint)
        if args.locomotion_task
        else None
    )
    acrobatics = (
        resolve_checkpoint(args.acrobatics_task, args.acrobatics_run, args.acrobatics_checkpoint)
        if args.acrobatics_task
        else None
    )

    obs = torch.zeros(2, obs_spec.POLICY_DIM), torch.zeros(2, obs_spec.CRITIC_DIM)
    from tensordict import TensorDict

    policy = MoEActorCritic(
        TensorDict({"policy": obs[0], "critic": obs[1]}, batch_size=[2]),
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=12,
        num_experts=args.num_experts,
        gating_hidden_dims=tuple(args.gating_hidden_dims),
        gating_prior_scale=0.0,
        actor_hidden_dims=args.hidden_dims,
        critic_hidden_dims=args.hidden_dims,
        activation="elu",
    )

    print()
    # The widening maps live with the observation layout, on the task side; the network takes
    # them as arguments so it does not have to know about any particular task's observation.
    report = initialize_experts(
        policy,
        locomotion,
        acrobatics,
        noise_std_mode=args.noise_std_mode,
        policy_sources={
            obs_spec.layout_dim(obs_spec.POLICY_LOCOMOTION): obs_spec.POLICY_MAP_LOCOMOTION,
            obs_spec.layout_dim(obs_spec.POLICY_JUMP): obs_spec.POLICY_MAP_JUMP,
        },
        critic_sources={
            obs_spec.layout_dim(obs_spec.CRITIC_LOCOMOTION): obs_spec.CRITIC_MAP_LOCOMOTION,
            obs_spec.layout_dim(obs_spec.CRITIC_JUMP): obs_spec.CRITIC_MAP_JUMP,
        },
    )
    for key, value in report.items():
        print(f"[info] {key}: {value}")

    # Build the optimizer through MoEPPO so the saved parameter-group structure is exactly the one
    # the runner will construct and check against. The state itself is left empty: Adam
    # re-initialises its moments lazily, and there are no meaningful moments to inherit for a
    # network that has just been assembled from two different runs.
    algorithm = MoEPPO(policy, device="cpu")
    optimizer_state = algorithm.optimizer.state_dict()
    optimizer_state["state"] = {}

    output = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "iter": 0,
        "infos": None,
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = os.path.join(log_root_from_task(args.task), f"{stamp}_moe_init")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "model_0.pt")
    torch.save(output, target_path)

    print(f"[info] wrote {target_path}")
    print(f"\nNext: python scripts/rsl_rl/train_and_aggregate.py --task {args.task} --resume")


if __name__ == "__main__":
    main()
