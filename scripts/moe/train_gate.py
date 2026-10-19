"""Train the gate of a two-expert MoE (walking expert + wall-crossing expert) to reproduce the deploy
schedule from the observation alone (2026-10-13).

The experts stay frozen (they are the two trained policies); only the gate is learned, by supervised
classification of "which policy was running" on MuJoCo rollouts of the full walk -> rise -> cross -> land
-> walk sequence (``wallcross_mujoco.py --record_gate``) plus walking with random commands. The gate has the
form of Tamura-san's ``Gating`` (``assets/models/moe_actor.py``): softmax(MLP(obs) + prior_scale * prior), the
prior keyed off the command flags (jump enabled col 9, hind-stance on col 14) -> wall expert. The MLP has to
learn the one case the flags miss: the landing, where both flags are already off but the wall expert still runs.

    /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/moe/train_gate.py --rec reproduce_logs/moe/rec \\
        --walker <walker.pt> --wall <wall.pt> --out reproduce_logs/moe/moe_blind.npz
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn

JUMP_COL, HS_COL = 9, 14


def load_actor(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    return {f"{i}": (sd[f"actor.{j}.weight"].numpy(), sd[f"actor.{j}.bias"].numpy()) for i, j in enumerate((0, 2, 4, 6))}


# v2: the gate does not read the previous-action history (cols 88-123). With it the gate learned "whoever acted
# last keeps acting" (val. accuracy 99.8%), and in closed loop the wall expert never handed back: after the
# hand-over its weight stayed 0.2-0.5 and walking on fell from 94/70/67% to 75/37/35% at 0.60/0.70/0.80 m.
GATE_MASK = np.ones(124, dtype=np.float32)
GATE_MASK[88:124] = 0.0


class Gate(nn.Module):
    def __init__(self, hidden=128, prior_scale=5.0):
        super().__init__()
        self.register_buffer("mask", torch.tensor(GATE_MASK))
        self.mlp = nn.Sequential(nn.Linear(124, hidden), nn.ELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.prior_scale = prior_scale

    def forward(self, x):
        flag = torch.maximum(x[:, JUMP_COL], x[:, HS_COL]).clamp(0, 1)
        prior = torch.stack([1 - flag, flag], dim=1)
        return self.mlp(x * self.mask) + self.prior_scale * prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--walker", required=True)
    ap.add_argument("--wall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    X, Y, F, Tr, Tt = [], [], [], [], []
    for k, f in enumerate(sorted(glob.glob(os.path.join(a.rec, "*.npz")))):
        z = np.load(f); X.append(z["obs"]); Y.append(z["label"]); F.append(np.full(len(z["label"]), "walk" in os.path.basename(f)))
        Tr.append(z["trial"].astype(np.int64) + 1_000_000 * k); Tt.append(z["t"])
    X = np.concatenate(X).astype(np.float32); Y = np.concatenate(Y).astype(np.int64); W = np.concatenate(F)
    flag = np.maximum(X[:, JUMP_COL], X[:, HS_COL]) > 0.5
    # v2 labels: hand back to the walker as soon as the deploy FSM would declare "LANDED" (0.8 s after the cue,
    # tilt < 15 deg held 0.3 s), not 1.0 s later as the FSM does. That moment is visible in the state; the extra
    # 1.0 s is a timer the gate cannot see.
    T = np.concatenate(Tr); Tm = np.concatenate(Tt)
    tilt = np.arccos(np.clip(-X[:, 5], -1, 1))
    relabeled = 0
    for tr in np.unique(T):
        ii = np.where(T == tr)[0]
        ii = ii[np.argsort(Tm[ii])]
        fl = flag[ii]
        if not fl.any():
            continue
        cue = ii[np.where(X[ii, JUMP_COL] > 0.5)[0][0]] if (X[ii, JUMP_COL] > 0.5).any() else None
        if cue is None:
            continue
        t_cue = Tm[cue]; hold = 0
        for j in ii[Tm[ii] > t_cue]:
            hold = hold + 1 if (tilt[j] < 0.26 and Tm[j] - t_cue >= 0.8) else 0
            if hold >= 15 and Y[j] == 1 and not flag[j]:
                Y[j] = 0; relabeled += 1
    print("relabeled to walker after LANDED:", relabeled)
    cat = np.where(flag, 0, np.where(Y == 1, 1, 2))   # 0: flag on, 1: flags off but wall (landing), 2: flags off, walker
    print("steps", len(Y), "| flag on", (cat == 0).sum(), "| landing (flags off, wall)", (cat == 1).sum(), "| walker", (cat == 2).sum())
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Y)); nv = len(Y) // 10
    va, tr = idx[:nv], idx[nv:]
    # balance: the landing class is small and is what the MLP has to learn
    w = np.ones(len(Y), dtype=np.float32)
    w[cat == 1] = (cat == 2).sum() / max((cat == 1).sum(), 1) * 0.5
    Xt, Yt, Wt = map(torch.tensor, (X, Y, w))
    g = Gate()
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    for ep in range(a.epochs):
        perm = torch.tensor(rng.permutation(tr))
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            loss = (nn.functional.cross_entropy(g(Xt[b]), Yt[b], reduction="none") * Wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = torch.softmax(g(Xt[va]), 1)[:, 1].numpy() > 0.5
        acc = {k: float((p[cat[va] == c] == (Y[va][cat[va] == c] == 1)).mean()) for k, c in (("flag", 0), ("landing", 1), ("walker", 2))}
        print(f"epoch {ep:2d} loss {loss.item():.4f} val acc " + " ".join(f"{k} {v:.4f}" for k, v in acc.items()), flush=True)
    out = {"gate_mask": GATE_MASK, "gate_layers": np.array(2), "prior_scale": np.array(g.prior_scale), "jump_col": np.array(JUMP_COL), "hs_col": np.array(HS_COL)}
    out["gate.0.weight"] = g.mlp[0].weight.detach().numpy(); out["gate.0.bias"] = g.mlp[0].bias.detach().numpy()
    out["gate.1.weight"] = g.mlp[2].weight.detach().numpy(); out["gate.1.bias"] = g.mlp[2].bias.detach().numpy()
    for e, path in enumerate((a.walker, a.wall)):
        for i, (Wm, bm) in load_actor(path).items():
            out[f"expert{e}.{i}.weight"] = Wm; out[f"expert{e}.{i}.bias"] = bm
    np.savez(a.out, **out)
    torch.save(g.state_dict(), a.out[:-4] + "_gate.pt")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
