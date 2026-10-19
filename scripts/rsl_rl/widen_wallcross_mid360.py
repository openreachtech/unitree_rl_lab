"""Carry a blind wall-crossing checkpoint over to the MID-360 task.

Local helper, same idea as ``widen_wallcross_heightmap.py`` with separate widths: the MID-360 task
appends ``mid360_map`` (609) to the actor and the analytic ``wall_heightmap`` (155) to the critic.
Zero columns at the end of each first layer, Adam's moments padded the same way, so the widened
network computes exactly the blind one until training moves the new weights.

    python scripts/rsl_rl/widen_wallcross_mid360.py --src <run13 model_17995.pt> --out_dir <log dir>
"""

import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--actor_extra", type=int, default=609)
ap.add_argument("--critic_extra", type=int, default=155)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
extra = {"actor.0.weight": a.actor_extra, "critic.0.weight": a.critic_extra}
by_shape = {}
for key, n in extra.items():
    w = sd[key]
    by_shape[tuple(w.shape)] = n
    sd[key] = torch.cat((w, torch.zeros(w.shape[0], n)), dim=1)
    print(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
if len(by_shape) != 2:
    raise SystemExit("actor and critic first layers have the same shape; cannot tell their Adam moments apart")
for st in ck["optimizer_state_dict"]["state"].values():
    for name, t in list(st.items()):
        if torch.is_tensor(t) and t.dim() == 2 and tuple(t.shape) in by_shape:
            st[name] = torch.cat((t, torch.zeros(t.shape[0], by_shape[tuple(t.shape)])), dim=1)
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, os.path.basename(a.src))
torch.save(ck, out)
print("wrote", out)
