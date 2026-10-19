"""Carry a blind wall-crossing checkpoint over to the height-map task.

Local helper. The height-map task appends ``wall_heightmap`` (155 columns) after the unified layout
on the actor and the critic, so the blind network is widened by zero columns at the end of each
first layer: the widened network computes exactly the blind one until training moves the new
weights. Adam's moments for the two widened matrices are padded the same way.

    python scripts/rsl_rl/widen_wallcross_heightmap.py --src <run13 model_17995.pt> --out_dir <log dir>
"""

import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--extra", type=int, default=155)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
old = {}
for key in ("actor.0.weight", "critic.0.weight"):
    w = sd[key]
    old[key] = tuple(w.shape)
    sd[key] = torch.cat((w, torch.zeros(w.shape[0], a.extra)), dim=1)
    print(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
for st in ck["optimizer_state_dict"]["state"].values():
    for name, t in list(st.items()):
        if torch.is_tensor(t) and t.dim() == 2 and tuple(t.shape) in old.values():
            st[name] = torch.cat((t, torch.zeros(t.shape[0], a.extra)), dim=1)
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, os.path.basename(a.src))
torch.save(ck, out)
print("wrote", out)
