# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble the walk / fall / get-up mixture's starting checkpoint from two single-task runs.

Expert 0 of the actor <- the blind locomotion policy (117-column observation, loaded unchanged).
Expert 1 of the actor <- the get-up policy (42 columns), placed onto the newest frame of each
history block, with its action-scale difference folded into the weights:

    standalone:  target = default + 0.5  * a_g(x_g)        a_g reads its own last action a_prev
    here:        target = default + 0.25 * a(x)            the observation carries 2 * a_prev

so ``a = 2 * a_g`` -- the output layer is doubled -- and the last-action input columns are halved
to read ``a_prev`` back. The result commands bit-for-bit the joint targets the standalone get-up
policy did; ``--verify`` checks exactly that on random observations.

The critic and both gates stay randomly initialised (the gates' last layers are zero, so the
attitude prior alone routes at first). The shared action noise is the elementwise minimum of the
two policies' noise, the get-up one doubled into the new action units.

    python scripts/rsl_rl/build_recovery_moe_checkpoint.py \\
        --locomotion-checkpoint <run>/model_6497.pt --getup-checkpoint <run>/model_2999.pt
    python scripts/rsl_rl/train.py --task Unitree-Go2-Recovery-MoE-v0 --headless --resume --load_run '.*_moe_init'

With ``--getup-only`` it writes the get-up policy alone instead, as a plain actor-critic on the same
interface (``Unitree-Go2-GetUp-Unified-v0``), so get-up alone and the mixture run on one deploy
state::

    python scripts/rsl_rl/build_recovery_moe_checkpoint.py --getup-only --getup-checkpoint <run>/model_2999.pt

Runs on any interpreter with PyTorch and rsl_rl; it does not import Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import torch
from datetime import datetime

_PACKAGE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab")
)
sys.path.insert(0, os.path.join(_PACKAGE, "assets"))

from models import RecoveryMoEActorCritic, RecoveryMoEPPO  # noqa: E402
from models.modules.weight_surgery import expand_state_dict, scatter_observation  # noqa: E402
from models.recovery_moe import EXPERT_GETUP, EXPERT_WALK  # noqa: E402
from rsl_rl.algorithms import PPO  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from rsl_rl.networks import MLP  # noqa: E402

# Loaded by path: importing the task package would register gym environments and pull in Isaac Lab.
_spec = importlib.util.spec_from_file_location(
    "go2_recovery_obs_spec", os.path.join(_PACKAGE, "tasks", "locomotion", "robots", "go2_recovery", "obs_spec.py")
)
obs_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs_spec)


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def actor_params(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[len("actor.") :]: value.clone() for key, value in state_dict.items() if key.startswith("actor.")}


def getup_params(state_dict: dict[str, torch.Tensor], ratio: float) -> dict[str, torch.Tensor]:
    params = expand_state_dict(state_dict, obs_spec.POLICY_DIM, obs_spec.GETUP_POLICY_MAP, "actor")
    start, end = obs_spec.GETUP_LAST_ACTION_COLUMNS
    params["0.weight"][:, start:end] /= ratio
    last = max(int(key.split(".")[0]) for key in params)
    params[f"{last}.weight"] *= ratio
    params[f"{last}.bias"] *= ratio
    return params


@torch.no_grad()
def verify(policy: RecoveryMoEActorCritic, locomotion: dict, getup: dict, ratio: float) -> None:
    """Each expert must reproduce its source policy, in joint-target units, on random inputs."""
    hidden = [512, 256, 128]
    getup_src = MLP(obs_spec.GETUP_POLICY_DIM, 12, hidden, "elu")
    getup_src.load_state_dict(actor_params(getup))

    walk_err = 0.0
    if locomotion is not None:
        walk_src = MLP(obs_spec.POLICY_DIM, 12, hidden, "elu")
        walk_src.load_state_dict(actor_params(locomotion))
        x = torch.randn(256, obs_spec.POLICY_DIM)
        walk_err = (policy.actor.experts[EXPERT_WALK](x) - walk_src(x)).abs().max().item()

    # Standalone get-up observation, with its last action in its own (scale 0.5) units.
    x_g = torch.randn(256, obs_spec.GETUP_POLICY_DIM)
    placed = scatter_observation(torch.ones_like(x_g), obs_spec.POLICY_DIM, obs_spec.GETUP_POLICY_MAP) > 0
    # Columns the get-up expert does not read (command, older history) get noise: they must not matter.
    x_u = torch.where(placed, scatter_observation(x_g, obs_spec.POLICY_DIM, obs_spec.GETUP_POLICY_MAP), torch.randn(256, obs_spec.POLICY_DIM))
    start, end = obs_spec.GETUP_LAST_ACTION_COLUMNS
    x_u[:, start:end] *= ratio  # the same previous joint target, in this task's action units
    target_src = obs_spec.GETUP_ACTION_SCALE * getup_src(x_g)
    target_new = obs_spec.ACTION_SCALE * policy.actor.experts[EXPERT_GETUP](x_u)
    getup_err = (target_new - target_src).abs().max().item()

    if locomotion is not None:
        print(f"[verify] locomotion expert max |diff|: {walk_err:.2e}")
    print(f"[verify] get-up expert max |joint target diff| (rad): {getup_err:.2e}")
    if walk_err > 1e-5 or getup_err > 1e-4:
        raise SystemExit("[error] an expert does not reproduce its source policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="Log directory under logs/rsl_rl.")
    parser.add_argument("--locomotion-checkpoint", help="Unitree-Go2-Velocity-v1 model_*.pt (117 obs).")
    parser.add_argument("--getup-checkpoint", required=True, help="Unitree-Go2-GetUp-v0 model_*.pt (42 obs).")
    parser.add_argument("--gating-hidden-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--getup-only", action="store_true", help="Write the get-up policy alone, for Unitree-Go2-GetUp-Unified-v0."
    )
    args = parser.parse_args()
    if not args.getup_only and args.locomotion_checkpoint is None:
        parser.error("--locomotion-checkpoint is required unless --getup-only")
    experiment = args.experiment or ("go2_getup_unified" if args.getup_only else "go2_recovery_moe")

    ratio = obs_spec.GETUP_ACTION_SCALE / obs_spec.ACTION_SCALE
    locomotion = None if args.getup_only else load_state_dict(args.locomotion_checkpoint)
    getup = load_state_dict(args.getup_checkpoint)
    for name, sd, width in (
        ("locomotion", locomotion, obs_spec.POLICY_DIM),
        ("get-up", getup, obs_spec.GETUP_POLICY_DIM),
    ):
        if sd is None:
            continue
        if sd["actor.0.weight"].shape[1] != width:
            raise SystemExit(f"[error] {name} actor takes {sd['actor.0.weight'].shape[1]} inputs, expected {width}")

    from tensordict import TensorDict

    obs = TensorDict(
        {"policy": torch.zeros(2, obs_spec.POLICY_DIM), "critic": torch.zeros(2, obs_spec.CRITIC_DIM)}, batch_size=[2]
    )
    policy = RecoveryMoEActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=12,
        num_experts=2,
        gating_hidden_dims=tuple(args.gating_hidden_dims),
        actor_gravity_z_index=obs_spec.POLICY_GRAVITY_Z,
        critic_gravity_z_index=obs_spec.CRITIC_GRAVITY_Z,
    )
    if locomotion is not None:
        policy.actor.experts[EXPERT_WALK].load_state_dict(actor_params(locomotion))
    policy.actor.experts[EXPERT_GETUP].load_state_dict(getup_params(getup, ratio))

    getup_std = getup["std"] * ratio
    noise = getup_std if locomotion is None else torch.minimum(locomotion["std"], getup_std)
    with torch.no_grad():
        policy.std.copy_(noise)
    print(f"[info] action noise: get-up {getup_std.mean():.3f} (x{ratio:g}) -> {noise.mean():.3f}")

    if not args.no_verify:
        verify(policy, locomotion, getup, ratio)

    if args.getup_only:
        # A plain actor-critic whose actor is the converted get-up expert; the critic stays random.
        single = ActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        single.actor.load_state_dict(policy.actor.experts[EXPERT_GETUP].state_dict())
        with torch.no_grad():
            single.std.copy_(noise)
        policy, algorithm = single, PPO(single, device="cpu")
        suffix = "converted"
    else:
        # Optimizer built through the algorithm class so its parameter groups match what the runner
        # constructs; state left empty for Adam to initialise lazily.
        algorithm = RecoveryMoEPPO(policy, device="cpu")
        suffix = "moe_init"
    optimizer_state = algorithm.optimizer.state_dict()
    optimizer_state["state"] = {}

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = os.path.abspath(os.path.join("logs", "rsl_rl", experiment, f"{stamp}_{suffix}"))
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "model_0.pt")
    torch.save(
        {"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer_state, "iter": 0, "infos": None},
        target_path,
    )
    print(f"[info] wrote {target_path}")
    if args.getup_only:
        print(f"\nNext: python scripts/rsl_rl/export_policy.py --task Unitree-Go2-GetUp-Unified-v0 --checkpoint {target_path}")
    else:
        print("\nNext: python scripts/rsl_rl/train.py --task Unitree-Go2-Recovery-MoE-v0 --headless --resume --load_run '.*_moe_init'")


if __name__ == "__main__":
    main()
