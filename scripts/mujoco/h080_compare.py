#!/usr/bin/env python3
"""合図の後の Isaac と MuJoCo の比較 (0.80 m 前後の薄い壁、2026-10-11 診断)。

Isaac: isaac_dump_sequence.py の npz (壁は env ごと: half / height / thick、ロボットは +x を向く)
MuJoCo: wallcross_mujoco.py --dump_npz の npz (複数可、壁は wall_x / wall_h / wall_t)

    python scripts/mujoco/h080_compare.py --isaac reproduce_logs/h080_diag/isaac_B.npz \
        --mujoco reproduce_logs/h080_diag/mj/B_s10*.npz

壁との接触: MuJoCo は壁 geom との接触力そのもの。Isaac は接触センサの合力しかないので、
「力 > 1 N で、その部位が床に届かない高さ (部位の原点と、すねなら足先、腿ならひざが z > 0.08 m) にあり、
壁から x 方向 0.35 m 以内」を壁との接触とみなす。
"""
import argparse
import glob

import numpy as np

GROUPS = {"base": ("base", "base_link", "Head_upper", "Head_lower"), "hip": ("_hip",), "thigh": ("_thigh",),
          "calf": ("_calf",), "foot": ("_foot",)}


def group_of(name):
    for g, keys in GROUPS.items():
        if any(name == k or (k.startswith("_") and name.endswith(k)) for k in keys):
            return g
    return None


def pitch_deg(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    bxz = 2 * (x * z - w * y)                 # z component of the body x axis
    return np.degrees(np.arcsin(np.clip(bxz, -1, 1)))


def rotate(q, v):
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    u = np.concatenate([x, y, z], -1)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def load_isaac(path):
    z = np.load(path)
    names = list(z["body_names"])
    snames = list(z["sensor_body_names"])
    sidx = [snames.index(n) for n in names]
    trials = []
    T, N = z["done"].shape
    for e in range(N):
        tr = np.where(z["triggered"][:, e])[0]
        if len(tr) == 0:
            continue
        ti = int(tr[0])
        dn = np.where(z["done"][:, e])[0]
        end = int(dn[0]) if len(dn) else T - 1
        half, h, th = float(z["half"][e]), float(z["height"][e]), float(z["thick"][e])
        bp = z["body_pos"][:, e]                                    # (T, B, 3) robot order
        bf = z["body_f"][:, e][:, sidx]                             # (T, B)
        low = bp[..., 2].copy()
        for i, n in enumerate(names):
            if n.endswith("_calf"):
                low[:, i] = np.minimum(low[:, i], bp[:, names.index(n[:2] + "_foot"), 2])
            if n.endswith("_thigh"):
                low[:, i] = np.minimum(low[:, i], bp[:, names.index(n[:2] + "_calf"), 2])
        near = np.abs(bp[..., 0] - (half + th / 2)) < 0.35
        wall_f = np.where((bf > 1.0) & (low > 0.08) & near, bf, 0.0)
        su = np.where(z["success"][:, e])[0]
        trials.append(dict(
            ti=ti, end=end, wall_x=half, h=h, t=th, names=names,
            pos=z["root_pos"][:, e].astype(np.float64), quat=z["root_quat"][:, e].astype(np.float64),
            vel_w=rotate(z["root_quat"][:, e].astype(np.float64), z["linvel_b"][:, e].astype(np.float64)),
            wy=z["angvel_b"][:, e, 1].astype(np.float64), body_pos=bp.astype(np.float64), wall_f=wall_f,
            success=int(su[0]) if len(su) else None, fell=end if len(dn) and (not len(su) or dn[0] < su[0]) else None,
        ))
    return trials


def load_mujoco(paths):
    trials = []
    for p in paths:
        z = np.load(p)
        names = list(z["body_names"])
        dt = float(z["dt"])
        for i in range(len(z["seed"])):
            if not z["triggered"][i]:
                continue
            L = int(z["length"][i])
            ti = int(round(z["t_trigger"][i] / dt)) - 1
            ts = z["t_success"][i]
            tf = z["fell_t"][i]
            trials.append(dict(
                ti=ti, end=L - 1, wall_x=float(z["wall_x"]), h=float(z["wall_h"]), t=float(z["wall_t"]), names=names,
                pos=z["pos"][i, :L], quat=z["quat"][i, :L], vel_w=z["linvel_w"][i, :L], wy=z["angvel_b"][i, :L, 1],
                body_pos=z["body_pos"][i, :L], wall_f=z["wall_f"][i, :L],
                success=None if np.isnan(ts) else int(round(ts / dt)) - 1,
                fell=None if np.isnan(tf) else int(round(tf / dt)) - 1, seed=int(z["seed"][i]),
            ))
    return trials


def per_trial(tr, horizon=150):
    ti, end = tr["ti"], tr["end"]
    k1 = min(end, ti + horizon)
    sl = slice(ti, k1 + 1)
    pos = tr["pos"][sl]
    wx, h, th = tr["wall_x"], tr["h"], tr["t"]
    pit = pitch_deg(tr["quat"][sl])
    kz = int(np.argmax(pos[:, 2]))
    names = tr["names"]
    wf = tr["wall_f"][sl]
    groups = {}
    for g in GROUPS:
        cols = [i for i, n in enumerate(names) if group_of(n) == g]
        f = wf[:, cols].max(axis=1)
        on = np.where(f > 1.0)[0]
        groups[g] = dict(first=(on[0] if len(on) else None), steps=len(on), peak=float(f.max()) if len(f) else 0.0)
    first_body = None
    on_any = np.where(wf.max(axis=1) > 1.0)[0]
    if len(on_any):
        k = on_any[0]
        first_body = names[int(np.argmax(wf[k]))]
    # where are the feet / knees when the base is highest
    bp = tr["body_pos"][sl]
    feet = [i for i, n in enumerate(names) if n.endswith("_foot")]
    fx_at_apex = bp[kz, feet, 0] - (wx + th)
    fz_at_apex = bp[kz, feet, 2]
    outer = wx + th
    past = np.where(pos[:, 0] > outer)[0]
    succ = tr["success"] is not None and tr["success"] - ti <= horizon
    fell = tr["fell"] is not None and (not succ) and tr["fell"] - ti <= horizon
    hang = (not succ) and (not fell)
    return dict(
        gap=wx - pos[0, 0], z_cue=pos[0, 2], vx_cue=tr["vel_w"][ti, 0], vz_cue=tr["vel_w"][ti, 2], pitch_cue=pit[0],
        wy_cue=tr["wy"][ti], zmax=pos[kz, 2], apex_over=pos[kz, 2] - h, t_apex=kz, x_apex=pos[kz, 0] - outer,
        pitch_apex=pit[kz], vzmax=tr["vel_w"][ti:k1 + 1, 2].max(), vx_max=tr["vel_w"][ti:k1 + 1, 0].max(),
        base_past=(past[0] if len(past) else None), groups=groups, first_body=first_body,
        fx_apex=fx_at_apex, fz_apex=fz_at_apex, succ=succ, fell=fell, hang=hang,
        x_end=pos[-1, 0] - outer, z_end=pos[-1, 2], pitch_end=pit[-1],
        prof=dict(x=pos[:, 0] - wx, z=pos[:, 2], pitch=pit), wall_top=bp[:, :, :],
    )


def summarize(label, trials, horizon):
    R = [per_trial(t, horizon) for t in trials]
    n = len(R)
    c = lambda k: sum(bool(r[k]) for r in R)  # noqa: E731
    print(f"\n### {label}: n={n}  success {c('succ')}  fell {c('fell')}  hang(no success, no fall) {c('hang')}"
          f"   walls h {np.min([t['h'] for t in trials]):.3f}-{np.max([t['h'] for t in trials]):.3f}")
    def q(key, sub=None):
        v = np.array([r[key] for r in R if (sub is None or sub(r)) and r[key] is not None], dtype=float)
        return "-" if len(v) == 0 else f"{np.median(v):+.3f} [{np.percentile(v, 10):+.3f},{np.percentile(v, 90):+.3f}]"
    for key in ("gap", "z_cue", "vx_cue", "vz_cue", "pitch_cue", "wy_cue", "zmax", "apex_over", "t_apex", "x_apex",
                "pitch_apex", "vzmax", "vx_max", "base_past", "x_end", "z_end", "pitch_end"):
        print(f"  {key:11s} all {q(key)}   succ {q(key, lambda r: r['succ'])}   hang {q(key, lambda r: r['hang'])}")
    print("  wall contact by part (share of trials touching / median first step after cue / median steps / median peak N):")
    for g in GROUPS:
        on = [r["groups"][g] for r in R if r["groups"][g]["first"] is not None]
        if on:
            print(f"    {g:6s} {len(on):3d}/{n}  first {np.median([o['first'] for o in on]):5.1f}  "
                  f"steps {np.median([o['steps'] for o in on]):5.1f}  peak {np.median([o['peak'] for o in on]):7.1f}")
        else:
            print(f"    {g:6s}   0/{n}")
    fb = {}
    for r in R:
        fb[r["first_body"]] = fb.get(r["first_body"], 0) + 1
    print(f"  first body to touch the wall: {fb}")
    fx = np.array([r["fx_apex"] for r in R])
    fz = np.array([r["fz_apex"] for r in R])
    print(f"  feet at base apex (x past outer face, FL FR RL RR order of the dump): median x {np.median(fx, 0).round(2).tolist()} "
          f"z {np.median(fz, 0).round(2).tolist()}")
    # mean profile
    ks = list(range(0, 61, 5))
    for key in ("x", "z", "pitch"):
        vals = []
        for k in ks:
            v = [r["prof"][key][k] for r in R if k < len(r["prof"][key])]
            vals.append(np.median(v) if v else np.nan)
        print(f"  profile {key:5s} (cue+{ks[0]}..{ks[-1]} every 5): " + " ".join(f"{v:+.2f}" for v in vals))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", nargs="*", default=[])
    ap.add_argument("--mujoco", nargs="*", default=[])
    ap.add_argument("--horizon", type=int, default=150, help="合図からこのステップ数以内の成功を数える (150 = 3 s)")
    a = ap.parse_args()
    for p in a.isaac:
        summarize(f"Isaac {p}", load_isaac(p), a.horizon)
    if a.mujoco:
        paths = sorted(sum([glob.glob(p) for p in a.mujoco], []))
        summarize(f"MuJoCo {paths[0]}..({len(paths)} files)", load_mujoco(paths), a.horizon)


if __name__ == "__main__":
    main()
