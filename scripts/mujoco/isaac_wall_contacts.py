#!/usr/bin/env python3
"""isaac_dump_sequence.py の記録から、合図後〜成功までに壁の近くで力を受けた部位を列挙する (numpy のみ)。

Isaac の接触センサは「何と触れたか」を持たないので、部位の位置が壁の厚み ±margin に入っていて
床から min_z 以上にあるときの力を「壁との接触」とみなす (足は別 body なので calf の力は脛の接触)。

    python scripts/mujoco/isaac_wall_contacts.py reproduce_logs/mujoco_wallcross/isaac_ref_hm6_nodr_delay0.npz
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--margin", type=float, default=0.06)
ap.add_argument("--min_z", type=float, default=0.06)
ap.add_argument("--n_env", type=int, default=8)
a = ap.parse_args()
z = dict(np.load(a.npz))
sn, bn = list(z["sensor_body_names"]), list(z["body_names"])
BP, BF = z["body_pos"], z["body_f"]
envs = [e for e in range(BF.shape[1]) if 0.6 <= z["height"][e] <= 0.72 and z["thick"][e] <= 0.1 and z["half"][e] >= 1.45]
for e in envs[: a.n_env]:
    ti = int(z["triggered"][:, e].argmax())
    su = z["success"][:, e]
    si = int(su.argmax()) if su.any() else ti + 100
    half, t = z["half"][e], z["thick"][e]
    hits = {}
    for k in range(ti, min(si, BF.shape[0])):
        for b, nm in enumerate(bn):
            p = BP[k, e, b]
            f = BF[k, e, sn.index(nm)]
            if f > 1 and half - a.margin < p[0] < half + t + a.margin and p[2] > a.min_z:
                h = hits.setdefault(nm, [round((k - ti) * float(z["dt"]), 2), 0.0, round(float(p[2]), 2)])
                h[1] = max(h[1], round(float(f)))
    print(f"env {e:2d} h {z['height'][e]:.2f} t {t:.3f}: " + str({k: tuple(v) for k, v in sorted(hits.items(), key=lambda kv: kv[1][0])}))
