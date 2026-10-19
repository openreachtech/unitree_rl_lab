#!/usr/bin/env python3
"""isaac_dump_sequence.py の npz を wallcross_mujoco.py と同じ指標で集計する (numpy だけで動く)。

    python scripts/mujoco/analyze_isaac_ref.py reproduce_logs/mujoco_wallcross/isaac_ref_run13_nodr.npz \
        [--min_h 0.6 --max_h 0.72 --max_t 0.1 --min_half 1.45]

壁の内面までの初期距離を 2.4 m に揃えて置いているので、内側半径 (half) が小さいタイルでは
反対側の壁に重なって始まる。--min_half でそれを除く。"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--min_h", type=float, default=0.6)
ap.add_argument("--max_h", type=float, default=0.72)
ap.add_argument("--max_t", type=float, default=0.1)
ap.add_argument("--min_half", type=float, default=1.45)
ap.add_argument("--handover_s", type=float, default=1.0)
ap.add_argument("--walk_after_s", type=float, default=3.0)
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()

z = np.load(a.npz)
dt = float(z["dt"])
T, N = z["done"].shape
rp, q, ff = z["root_pos"], z["root_quat"], z["feet_f"]
w, x, y, qz = np.moveaxis(q, -1, 0)
gx = -2 * (x * qz - w * y)
gz = -(1 - 2 * (x * x + y * y))
first_done = np.where(z["done"].any(0), z["done"].argmax(0), T)
sel = [e for e in range(N) if a.min_h <= z["height"][e] <= a.max_h and z["thick"][e] <= a.max_t and z["half"][e] >= a.min_half]
rows = []
for e in sel:
    fd = first_done[e]
    trig = z["triggered"][:fd, e]
    ti = int(trig.argmax()) if trig.any() else None
    su = z["success"][:fd, e]
    si = int(su.argmax()) if su.any() else None
    st = slice(int(3.3 / dt), int(4.2 / dt))
    rose = bool(((-gx[st, e] > 0.93) & (ff[st, e, 0] <= 1) & (ff[st, e, 1] <= 1)).any()) if fd > st.stop else False
    jumped = bool(ti is not None and (ff[ti:(si or fd), e] <= 1).all(-1).any())
    zt = rp[ti, e, 2] if ti is not None else np.nan
    zmax = rp[ti:(si or fd), e, 2].max() if ti is not None and ti < fd else np.nan
    fell = fd < T
    stood = si is not None and (not fell or fd >= si + a.handover_s / dt)
    h_i = int(si + a.handover_s / dt) if si is not None else None
    walked = None
    if h_i is not None and h_i < T and not fell:
        end = min(T - 1, int(h_i + a.walk_after_s / dt))
        walked = float(np.linalg.norm(rp[end, e, :2] - rp[h_i, e, :2]))
    rows.append(dict(e=e, rose=rose, cue=ti is not None, jumped=jumped, crossed=si is not None, fell=fell,
                     fell_t=fd * dt if fell else None, stood=stood, walked=walked, zt=zt, rise=zmax - zt,
                     t_succ=(si - ti) * dt if si is not None else None))
    if a.verbose:
        print(f"env {e:2d} h {z['height'][e]:.2f} t {z['thick'][e]:.3f} half {z['half'][e]:.2f}: {rows[-1]}")
n = len(rows)
c = lambda k: sum(bool(r[k]) for r in rows)  # noqa: E731
print(f"{a.npz}: {n} envs with wall h {a.min_h}-{a.max_h}, t <= {a.max_t}, half >= {a.min_half}")
print(f"rose {c('rose')}/{n} | cue {c('cue')}/{n} | jumped {c('jumped')}/{n} | crossed {c('crossed')}/{n} | "
      f"stood {c('stood')}/{n} | walked>0.9m(disp) {sum(r['walked'] is not None and r['walked'] > 0.9 for r in rows)}/{n} | fell {c('fell')}/{n}")
rise = [r["rise"] for r in rows if np.isfinite(r["rise"])]
print(f"z at cue mean {np.nanmean([r['zt'] for r in rows]):.3f}; rise p10/50/90 {np.percentile(rise, [10, 50, 90]).round(3).tolist()}; "
      f"time cue->success p50 {np.median([r['t_succ'] for r in rows if r['t_succ'] is not None]):.2f}s")
w_ = [r["walked"] for r in rows if r["walked"] is not None]
if w_:
    print(f"walked after handover (3 s) p10/50/90 {np.percentile(w_, [10, 50, 90]).round(2).tolist()} m")
