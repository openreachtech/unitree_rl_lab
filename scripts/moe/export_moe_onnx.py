"""Export the two-expert MoE (npz from train_gate.py) as one ONNX policy and check it against the numpy
reference used in MuJoCo (2026-10-13). Input [1, D] (D = 124 blind, 733 with the MID-360 map), output [1, 12].
The walking expert and the gate read the first 124 columns; the wall expert reads all D."""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn as nn


def mlp(W, b):
    layers = []
    for i, (w, bb) in enumerate(zip(W, b)):
        lin = nn.Linear(w.shape[1], w.shape[0]); lin.weight.data = torch.tensor(w); lin.bias.data = torch.tensor(bb)
        layers.append(lin)
        if i < len(W) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class MoEPolicy(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.walk = mlp([z[f"expert0.{i}.weight"] for i in range(4)], [z[f"expert0.{i}.bias"] for i in range(4)])
        self.wall = mlp([z[f"expert1.{i}.weight"] for i in range(4)], [z[f"expert1.{i}.bias"] for i in range(4)])
        n = int(z["gate_layers"])
        self.gate = mlp([z[f"gate.{i}.weight"] for i in range(n)], [z[f"gate.{i}.bias"] for i in range(n)])
        self.register_buffer("mask", torch.tensor(z["gate_mask"], dtype=torch.float32))
        self.prior_scale = float(z["prior_scale"]); self.jc = int(z["jump_col"]); self.hc = int(z["hs_col"])

    def forward(self, obs):
        base = obs[:, :124]
        flag = torch.maximum(base[:, self.jc], base[:, self.hc]).clamp(0.0, 1.0).unsqueeze(1)
        logits = self.gate(base * self.mask) + self.prior_scale * torch.cat([1.0 - flag, flag], dim=1)
        w = torch.softmax(logits, dim=1)
        return w[:, 0:1] * self.walk(base) + w[:, 1:2] * self.wall(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out"); ap.add_argument("--rec", default="reproduce_logs/moe/rec")
    a = ap.parse_args()
    z = np.load(a.npz)
    m = MoEPolicy(z).eval()
    D = z["expert1.0.weight"].shape[1]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.onnx.export(m, torch.zeros(1, D), a.out, input_names=["obs"], output_names=["actions"], opset_version=13)
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out)
    X = np.concatenate([np.load(f)["obs"] for f in sorted(glob.glob(os.path.join(a.rec, "*.npz")))])[:: 20].astype(np.float32)
    if D > 124:
        X = np.concatenate([X, np.random.default_rng(0).uniform(-0.3, 0.6, (len(X), D - 124)).astype(np.float32)], 1)
    with torch.no_grad():
        ref = m(torch.tensor(X)).numpy()
    out = np.concatenate([sess.run(None, {"obs": X[i:i + 1]})[0] for i in range(len(X))])
    rep = {"onnx": a.out, "input": [1, D], "output": list(out.shape[1:]), "n_checked": len(X), "max_abs_diff_torch_vs_onnx": float(np.abs(out - ref).max())}
    print(json.dumps(rep)); json.dump(rep, open(a.out[:-5] + "_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
