#!/usr/bin/env python3
"""Self-test of the MuJoCo MID-360 map (mid360_map.py).

1. ray casting: mj_multiRay (floor + wall only) against an analytic ray/plane/box intersection, every one of the
   200 scan windows, two static poses (four legs, hind stance leaning back 100 deg nose-up).
2. map: the same two poses held still in front of a wall for 1 s; per forward row of the central 5 columns,
   the height above the floor the map implies, next to the true geometry.
3. (--isaac npz from scripts/rsl_rl/dump_wallcross_mid360_rays.py) Isaac cross-check:
   a) ray origins / directions rebuilt from Isaac's base pose and window index vs Isaac's own,
   b) window index advances by one per control step,
   c) Isaac's hits and poses replayed through Mid360Map.update vs the map the Isaac policy saw.

    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/mid360_selftest.py [--isaac reproduce_logs/mid360/isaac_rays.npz]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mid360_map as M  # noqa: E402
from wallcross_mujoco import GO2_SCENE, build_model  # noqa: E402,F401


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def analytic_hits(origin, dirs, wall_x, wall_t, wall_h, wall_w):
    """First hit of each ray with the plane z=0 and the box [wall_x, wall_x+t] x [-w/2, w/2] x [0, h]."""
    n = len(dirs)
    t_best = np.full(n, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = -origin[2] / dirs[:, 2]
        t_best = np.where((dirs[:, 2] < 0) & (tp > 0), tp, t_best)
        lo = np.array([wall_x, -wall_w / 2, 0.0])
        hi = np.array([wall_x + wall_t, wall_w / 2, wall_h])
        t1 = (lo - origin) / dirs
        t2 = (hi - origin) / dirs
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        tb = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
    t_best = np.minimum(t_best, tb)
    t_best = np.where(t_best <= M.MAX_DISTANCE, t_best, np.inf)
    hits = origin + dirs * np.where(np.isfinite(t_best), t_best, 0.0)[:, None]
    hits[~np.isfinite(t_best)] = np.inf
    return hits


def set_pose(m, d, base_xyz, pitch_deg, joint_q=None):
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = base_xyz
    p = math.radians(pitch_deg)       # rotation about +y; negative = nose up
    d.qpos[3:7] = [math.cos(p / 2), 0, math.sin(p / 2), 0]
    if joint_q is not None:
        for name, v in joint_q.items():
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
    mujoco.mj_forward(m, d)


def static_test(a):
    wall_x, wall_t, wall_h, wall_w = 1.0, a.wall_t, a.wall_h, 4.0
    m = build_model(wall_x, wall_h, wall_t, wall_w, 0.0, 1.0)
    d = mujoco.MjData(m)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    lid = M.Mid360Map(m, base)
    stand = {f"{l}_{j}_joint": v for l in ("FR", "FL", "RR", "RL") for j, v in (("hip", 0.0), ("thigh", 0.8), ("calf", -1.5))}
    # hind stance: body leaning back 100 deg (nose up), hind legs under the hips, front legs folded
    hind = dict(stand)
    hind.update({"RR_thigh_joint": 2.4, "RL_thigh_joint": 2.4, "RR_calf_joint": -1.2, "RL_calf_joint": -1.2,
                 "FR_thigh_joint": 2.0, "FL_thigh_joint": 2.0, "FR_calf_joint": -2.6, "FL_calf_joint": -2.6})
    poses = [("four legs", (wall_x - a.gap4, 0.0, 0.32), 0.0, stand),
             ("hind stance 100deg nose-up", (wall_x - a.gaph, 0.0, 0.50), -100.0, hind)]
    ok = True
    for label, xyz, pitch, jq in poses:
        set_pose(m, d, xyz, pitch, jq)
        # 1. rays: every window vs analytic
        lid.reset(0, np.random.default_rng(0))
        maxerr, nhit, nwall, nmiss_mismatch = 0.0, 0, 0, 0
        for w in range(lid.nwin):
            origin, hits = lid.cast(d)
            ref = analytic_hits(origin, lid.windows[lid.win] @ d.xmat[base].reshape(3, 3).T, wall_x, wall_t, wall_h, wall_w)
            fin, fin_ref = np.isfinite(hits).all(1), np.isfinite(ref).all(1)
            nmiss_mismatch += int((fin != fin_ref).sum())
            both = fin & fin_ref
            if both.any():
                maxerr = max(maxerr, float(np.abs(hits[both] - ref[both]).max()))
            nhit += int(fin.sum())
            nwall += int((fin & (hits[:, 0] > wall_x - 1e-6)).sum())
        R = d.xmat[base].reshape(3, 3)
        origin = d.xpos[base] + R @ M.MOUNT_POS
        print(f"\n[{label}] base {np.round(d.xpos[base], 3).tolist()} pitch {pitch:+.0f} deg | sensor origin "
              f"{np.round(origin, 3).tolist()} | heading {math.degrees(M.yaw_from_quat(d.xquat[base])):+.1f} deg")
        print(f"  rays (200 windows x 2000): hit {nhit / (200 * 2000):.1%}, on the wall {nwall / max(nhit, 1):.1%}; "
              f"vs analytic: hit/miss mismatches {nmiss_mismatch}, max |hit diff| {maxerr:.2e} m")
        ok &= nmiss_mismatch == 0 and maxerr < 1e-6
        # 2. map held still for 1 s from a fresh reset (random window phase like the evaluator)
        lid.reset(17, np.random.default_rng(0))
        for _ in range(50):
            mp = lid.step(d)
        bz = d.xpos[base][2]
        above = (bz - mp - M.OFFSET).reshape(M.NUM_X, M.NUM_Y)
        meas = lid.measured.reshape(M.NUM_X, M.NUM_Y)
        gap = wall_x - d.xpos[base][0]
        print(f"  map after 1 s: measured {lid.measured.mean():.1%} of cells; wall inner face at base x {gap:+.3f}, "
              f"outer {gap + wall_t:+.3f}, height {wall_h}")
        print("   row x   | true terrain (cell bin ±0.025) | map: height above floor, central 5 cols (|y|<=0.1) | measured")
        floor_err = 0.0
        for ix in range(M.NUM_X):
            cx = M.X0 + ix * M.RESOLUTION
            in_wall = (cx + 0.025 > gap) and (cx - 0.025 < gap + wall_t)
            row = above[ix, 8:13]
            mrow = meas[ix, 8:13]
            truth = f"wall {wall_h:.2f}" if in_wall else "floor 0.00"
            if not in_wall:
                # a measured floor cell must read the floor exactly (plane z=0). Cells beyond the wall can only
                # see the floor if the beam passed over the top; otherwise they are never measured (flat fill).
                if mrow.any():
                    floor_err = max(floor_err, float(np.abs(above[ix][meas[ix]]).max()))
            print(f"   {cx:+.2f}   | {truth:10s}                   | " + " ".join(f"{v:5.2f}" for v in row)
                  + f" | {''.join('m' if v else '.' for v in mrow)}")
        wall_rows = [ix for ix in range(M.NUM_X)
                     if (M.X0 + ix * M.RESOLUTION + 0.025 > gap) and (M.X0 + ix * M.RESOLUTION - 0.025 < gap + wall_t)]
        wall_top = max(float(above[ix, 8:13][meas[ix, 8:13]].max()) if meas[ix, 8:13].any() else -9 for ix in wall_rows)
        print(f"  check: max |floor-cell height| = {floor_err:.2e} m (expect 0), highest wall-row reading "
              f"{wall_top:.3f} m (expect <= {wall_h}, ~{wall_h} if the top edge is in view)")
        ok &= floor_err < 1e-6 and wall_top <= wall_h + 1e-6
    return ok


def isaac_test(path):
    z = np.load(path)
    win, rp, rq = z["window"], z["root_pos"].astype(np.float64), z["root_quat"].astype(np.float64)
    starts, dirs, hits, mp, done = z["starts"], z["dirs"], z["hits"].astype(np.float64), z["map"], z["done"]
    T, N = rp.shape[:2]
    print(f"\n[isaac] {path}: {T} steps x {N} envs, windows {win[:6].tolist()} ... {win[-3:].tolist()}")
    adv = np.diff(win) % 200
    print(f"  window advance per control step: {sorted(set(adv.tolist()))} (expect [1])")
    W = M.load_windows()
    e_o = e_d = 0.0
    for t in range(T):
        for n in range(N):
            R = M.quat_to_mat(rq[t, n])
            o = rp[t, n] + R @ M.MOUNT_POS
            dd = W[win[t]] @ R.T
            e_o = max(e_o, float(np.abs(starts[t, n] - o).max()))
            e_d = max(e_d, float(np.abs(dirs[t, n] - dd).max()))
    print(f"  ray origins rebuilt from base pose: max |diff| {e_o:.2e} m; directions (window {win[0]}..): max |diff| {e_d:.2e}")
    # replay through the numpy map
    m = mujoco.MjModel.from_xml_path(GO2_SCENE)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    worst, frac, nstep = 0.0, [], 0
    for n in range(N):
        lid = M.Mid360Map(m, base)
        for t in range(T):
            if done[t, n]:
                lid.reset(0, np.random.default_rng(0))
            R = M.quat_to_mat(rq[t, n])
            out = lid.update(rp[t, n], rq[t, n], hits[t, n], rp[t, n] + R @ M.MOUNT_POS)
            diff = np.abs(out - mp[t, n])
            worst = max(worst, float(diff.max()))
            frac.append(float((diff > 1e-3).mean()))
            nstep += 1
    fin = np.isfinite(hits).all(-1)
    print(f"  map replay: {nstep} env-steps, max |numpy - isaac| {worst:.2e}, cells off by > 1 mm: mean {np.mean(frac):.3%}, "
          f"worst step {np.max(frac):.2%} | Isaac hit rate {fin.mean():.1%}")
    return e_o < 1e-4 and e_d < 1e-4 and list(set(adv.tolist())) == [1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isaac", default=None)
    ap.add_argument("--wall_h", type=float, default=0.65)
    ap.add_argument("--wall_t", type=float, default=0.05)
    ap.add_argument("--gap4", type=float, default=0.45, help="four legs: base origin to the wall's inner face")
    ap.add_argument("--gaph", type=float, default=0.40, help="hind stance: base origin to the wall's inner face")
    a = ap.parse_args()
    ok = static_test(a)
    if a.isaac:
        ok &= isaac_test(a.isaac)
    print("\nSELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
