"""Widen a trained rsl_rl checkpoint for observation terms appended after the existing ones.

The new input columns get zero weights, so the widened network computes exactly what the
original did until training gives the new columns a use. Adam's moments are widened the same way,
and the iteration counter is reset.

Used to start Go2w-Biped (two stances, + handstand_command) from Go2w-Biped-Hind:

    python scripts/rsl_rl/widen_policy_inputs.py \\
        logs/rsl_rl/go2w_biped_hind/<run>/model_999.pt \\
        logs/rsl_rl/go2w_biped/0000_init_from_hind/model_0.pt --extra 6
    python scripts/rsl_rl/train.py --task Go2w-Biped --headless \\
        --resume --load_run 0000_init_from_hind --checkpoint model_0.pt

``--extra`` counts flattened columns: handstand_command is 2 wide with history 3. The ``0000_``
prefix keeps the init folder sorted before the timestamped runs, which play.py and the deploy
controller pick from by name.
"""

import argparse
import os

import torch

FIRST_LAYERS = ("actor.0.weight", "critic.0.weight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--extra", type=int, required=True, help="Input columns appended to actor and critic.")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    weights = ckpt["model_state_dict"]
    widths = {weights[k].shape[1] for k in FIRST_LAYERS}

    def widen(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.zeros(t.shape[0], args.extra, dtype=t.dtype)], dim=1)

    for k in FIRST_LAYERS:
        weights[k] = widen(weights[k])
    for state in ckpt.get("optimizer_state_dict", {}).get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            t = state.get(key)
            if t is not None and t.dim() == 2 and t.shape[1] in widths:
                state[key] = widen(t)
    ckpt["iter"] = 0
    ckpt["infos"] = {"widened_from": os.path.abspath(args.src), "extra_columns": args.extra}

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(ckpt, args.dst)
    print({k: tuple(weights[k].shape) for k in FIRST_LAYERS}, "->", args.dst)


if __name__ == "__main__":
    main()
