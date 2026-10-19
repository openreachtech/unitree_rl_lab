#!/usr/bin/env python3
"""rsl_rl の checkpoint (.pt) から actor MLP の重みだけを .npz に書き出す。

MuJoCo 評価用 venv (/home/tanaka/isaacsim/anaguma/venv) には torch が無いので、torch のある
env_isaaclab の python でこれを呼んで numpy に落とす。wallcross_mujoco.py が必要なときに自動で呼ぶ。

    /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/mujoco/export_actor_npz.py <ckpt.pt> <out.npz>

書き出すもの: actor.{0,2,4,6}.{weight,bias}（ELU を挟む 4 層 MLP、rsl_rl ActorCritic の既定）。
観測正規化 (actor_obs_normalization) を使っている checkpoint は想定していない -- 見つけたら止める。
"""
import sys

import numpy as np
import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("actor_obs_normalizer") and "mean" in k for k in sd):
        raise SystemExit(f"{src}: actor_obs_normalizer がある checkpoint は未対応")
    out = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items() if k.startswith("actor.")}
    layers = sorted({int(k.split(".")[1]) for k in out})
    if layers != [0, 2, 4, 6]:
        raise SystemExit(f"{src}: 想定外の actor 層構成 {layers}")
    out["iter"] = np.array(ckpt.get("iter", -1))
    np.savez(dst, **out)
    print(f"wrote {dst}: input {out['actor.0.weight'].shape[1]}, output {out['actor.6.weight'].shape[0]}")


if __name__ == "__main__":
    main()
