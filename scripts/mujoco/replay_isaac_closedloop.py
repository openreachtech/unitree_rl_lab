#!/usr/bin/env python3
"""Isaac の状態から MuJoCo で閉ループを始め、どの区間で越えられなくなるかを測る (2026-10-11、0.80 m 診断)。

isaac_dump_sequence.py の npz (MID-360 版も可) を読み、各 env のジャンプ合図の ``pre`` ステップ前
(負なら合図の後) の Isaac の状態 (胴体の位置・姿勢・速度、関節角・角速度) を MuJoCo に写して、
壁越えポリシーを MuJoCo の状態で回す。指令の列 (速度指令・ジャンプ合図・jump_time・倒立指令 =
obs の 6:16 列) は Isaac の記録をそのまま使う (合図の時刻は Isaac と同じ)。

MID-360 版 (入力 733) の地図は MuJoCo 内のレイで作る (mid360_map.py)。地図は前の値を保持するので、
開始前の ``--warmup`` ステップぶん Isaac の姿勢を運動学的に置いて走査し、Isaac と同じ履歴を作ってから始める
(開始時の地図と Isaac の obs の差を表示する)。

成功は wallcross_mujoco.py と同じ判定 (胴体が外面 +0.2 m 先・4 足とも外面の先・3 足以上接地・高さ > 0.24 m・
傾き < 15° を 0.3 s 保持)、転倒は胴体/hip の壁以外への接触か高さ < 0.15 m。

    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/replay_isaac_closedloop.py \
        reproduce_logs/h080_diag/isaac_B.npz --ckpt logs/.../model_23993.pt --pre 40 20 10 0 -10 -20
"""
import argparse
import collections
import os
import sys

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wallcross_mujoco as W  # noqa: E402
from mid360_map import Mid360Map  # noqa: E402


VMAX = np.array([30.1] * 8 + [15.7] * 4)   # Isaac 順: hip x4, thigh x4, calf x4


def run_env(z, e, pre, pol, params, a, rng):
    ti = int(np.where(z["triggered"][:, e])[0][0])
    k0 = ti - pre
    half, h, th = float(z["half"][e]), float(z["height"][e]), float(z["thick"][e])
    if a.wall_height_override:
        h = a.wall_height_override
    m = W.build_model(half, h, th if not a.wall_thickness else a.wall_thickness, 6.0, a.timestep, a.wall_friction,
                      a.foot_friction, wall_priority=2 if a.isaac_wall else 0, wall_solref=a.wall_solref,
                      wall_solimp=a.wall_solimp, impratio=a.impratio,
                      spec_hook=lambda sp: [hk(sp) for hk, on in ((W.isaac_calf_geom_hook, a.isaac_calf_geom),
                                                                   (W.calf_capsule_hook, a.calf_capsule)) if on])
    if a.foot_condim:
        for f in W.FEET:
            m.geom_condim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f)] = a.foot_condim
    if a.margin is not None:
        m.geom_margin[:] = a.margin
    d = mujoco.MjData(m)
    rb = W.Robot(m, params)
    th_eff = th if not a.wall_thickness else a.wall_thickness

    def put(k, vel=True):
        d.qpos[0:3] = z["root_pos"][k, e]
        d.qpos[3:7] = z["root_quat"][k, e]
        d.qpos[rb.qadr] = z["jpos"][k, e]
        if vel:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            d.qvel[0:3] = R.reshape(3, 3) @ z["linvel_b"][k, e]
            d.qvel[3:6] = z["angvel_b"][k, e]
            d.qvel[rb.vadr] = z["jvel"][k, e]
        else:
            d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    mujoco.mj_resetData(m, d)
    lidar = None
    map_err = None
    if pol.in_dim == 733:
        lidar = Mid360Map(m, rb.base, noise=False)
        lidar.reset(int(rng.integers(0, 200)), rng)
        for k in range(max(0, k0 - a.warmup), k0):
            put(k, vel=False)
            lidar.step(d)
    put(k0)

    dsub = int(round(a.delay * W.ISAAC_PHYSICS_DT / m.opt.timestep))
    hold_n = max(1, int(round(a.pd_hold / m.opt.timestep))) if a.pd_hold > 0 else 1
    pending = collections.deque()
    sub = 0
    target = rb.default_q + z["act"][k0, e] * rb.action_scale
    o0 = z["obs"][k0 + 1, e].astype(np.float64)
    hist = [collections.deque(list(o0[c:c + 36].reshape(3, 12)), 3) for c in (16, 52, 88)]
    last = o0[112:124].copy()
    hold = 0
    succ_k = fell_k = None
    zmax = -1.0
    qd_peak = np.zeros(12)
    wall_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    pen_min = 0.0
    deep_steps = set()
    vmax_mj = np.full(3, -9.0)
    probe = None
    T = z["obs"].shape[0]
    k_end = min(T - 1, ti + a.horizon)
    for k in range(k0 + 1, k_end + 1):
        if k > k0 + 1:
            hist[0].append(d.qpos[rb.qadr] - rb.default_q)
            hist[1].append(d.qvel[rb.vadr] * 0.05)
            hist[2].append(last)
        Rm = d.xmat[rb.base].reshape(3, 3)
        o = np.concatenate([d.qvel[3:6] * 0.2, Rm.T @ np.array([0, 0, -1.0]), z["obs"][k, e, 6:16],
                            np.concatenate(hist[0]), np.concatenate(hist[1]), np.concatenate(hist[2])])
        if lidar is not None:
            lm = lidar.step(d)
            if k == k0 + 1:
                ref = z["obs"][k, e, 124:733].astype(np.float64)
                map_err = (float(np.abs(lm - ref).max()), float(np.abs(lm - ref).mean()))
            o = np.concatenate([o, lm])
        elif pol.in_dim == 279:
            o = np.concatenate([o, W.wall_heightmap(d.qpos[:2], d.qpos[3:7], half, th_eff, h, 100.0)])
        act = pol(o)
        last = act
        pending.append((sub + dsub, rb.default_q + act * rb.action_scale))
        illegal = []
        for _ in range(rb.decimation):
            while pending and pending[0][0] <= sub:
                target = pending.popleft()[1]
            if sub % hold_n == 0:
                # Isaac は PD トルクを物理刻み (5 ms) ごとに計算して保持する (--pd_hold 0.005 で真似る)
                tau = rb.kp * (target - d.qpos[rb.qadr]) - rb.kd * d.qvel[rb.vadr]
                if a.isaac_match:
                    d.ctrl[rb.act] = W.isaac_match_clip(tau, d.qvel[rb.vadr], a)
                else:
                    d.ctrl[rb.act] = np.clip(tau, -rb.tau_lim, rb.tau_lim)
            if a.vcap_damper > 0:
                # PhysX の関節最大速度 (USD: hip/thigh 30.1, calf 15.7 rad/s) を「超えた分にだけ効く強いダンパ」で近似
                qd = d.qvel[rb.vadr]
                d.qfrc_applied[rb.vadr] = -a.vcap_damper * (qd - np.clip(qd, -VMAX, VMAX))
            mujoco.mj_step(m, d)
            sub += 1
            feet, il = W.contact_state(m, d, rb, half, th_eff)
            illegal += il
            for ci in range(d.ncon):
                c = d.contact[ci]
                if wall_id in (c.geom1, c.geom2):
                    pen_min = min(pen_min, c.dist)
                    if c.dist < -0.010:
                        deep_steps.add(k)
        zb = d.qpos[2]
        if k >= ti:
            zmax = max(zmax, zb)
        if ti <= k <= ti + 12:
            qd_peak = np.maximum(qd_peak, np.abs(d.qvel[rb.vadr]))
            vmax_mj = np.maximum(vmax_mj, d.qvel[0:3])
        if k == ti + a.probe:
            probe = dict(z=d.qpos[2], x=d.qpos[0] - half, vx=d.qvel[0], vz=d.qvel[2], wy=d.qvel[4],
                         pitch=float(np.degrees(np.arcsin(np.clip(d.xmat[rb.base][6], -1, 1)))))
        grav = d.xmat[rb.base].reshape(3, 3).T @ np.array([0, 0, -1.0])
        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        outer = half + th_eff
        feet_x = np.array([d.xpos[b][0] for b in rb.foot_body])
        landed = (k >= ti and (d.qpos[0] - outer) > W.BASE_PAST_MARGIN and (feet_x > outer).all()
                  and feet.sum() >= W.MIN_FEET_IN_CONTACT and zb > W.MIN_SUCCESS_HEIGHT and tilt < W.MAX_TILT_RAD)
        hold = hold + 1 if landed else 0
        if a.trace_env == e:
            snames = list(z["sensor_body_names"])
            def iz_f(nm):
                return float(z["body_f"][k, e, snames.index(nm)])
            bn = list(z["body_names"])
            wl = []
            W.contact_state(m, d, rb, half, th_eff, wl)
            mjf = collections.defaultdict(float)
            for bname, fn, cz in wl:
                mjf[bname] = max(mjf[bname], fn)
            Ri = np.zeros(9)
            mujoco.mju_quat2Mat(Ri, z["root_quat"][k, e].astype(np.float64))
            vi = Ri.reshape(3, 3) @ z["linvel_b"][k, e]
            fi = z["body_pos"][k, e, bn.index("FL_foot")]
            fm = d.xpos[rb.foot_body[1]]
            print(f"   k{k - ti:+3d} base x {d.qpos[0] - half:+.2f}/{z['root_pos'][k, e, 0] - half:+.2f} z {zb:.2f}/{z['root_pos'][k, e, 2]:.2f} "
                  f"vx {d.qvel[0]:+.2f}/{vi[0]:+.2f} vz {d.qvel[2]:+.2f}/{vi[2]:+.2f} pitch "
                  f"{np.degrees(np.arcsin(np.clip(d.xmat[rb.base][6], -1, 1))):5.1f}/{np.degrees(np.arcsin(np.clip(Ri[6], -1, 1))):5.1f} | "
                  f"FL foot x-in {fm[0] - half:+.3f}/{fi[0] - half:+.3f} z-top {fm[2] - h:+.3f}/{fi[2] - h:+.3f} | "
                  f"F MJ(wall) FLcalf {mjf['FL_calf']:5.0f} FLfoot {mjf['FL_foot']:5.0f} FRcalf {mjf['FR_calf']:5.0f} FRfoot {mjf['FR_foot']:5.0f} "
                  f"base {mjf['base_link']:5.0f} | F Isaac(all) FLcalf {iz_f('FL_calf'):5.0f} FLfoot {iz_f('FL_foot'):5.0f} "
                  f"FRcalf {iz_f('FR_calf'):5.0f} FRfoot {iz_f('FR_foot'):5.0f} base {iz_f('base'):5.0f}")
            wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
            f6 = np.zeros(6)
            for ci in range(d.ncon):
                c = d.contact[ci]
                if wid not in (c.geom1, c.geom2):
                    continue
                g = c.geom2 if c.geom1 == wid else c.geom1
                mujoco.mj_contactForce(m, d, ci, f6)
                nrm = c.frame[:3] * (1 if c.geom1 == wid else -1)     # from the wall towards the robot
                print(f"        wall contact {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])}/geom{g} "
                      f"pos x-in {c.pos[0] - half:+.3f} z-top {c.pos[2] - h:+.3f} normal {np.round(nrm, 2).tolist()} "
                      f"dist {c.dist * 1000:+.1f} mm  F {np.linalg.norm(f6[:3]):.0f} N (normal {f6[0]:.0f})")
        if succ_k is None and hold >= W.SUCCESS_HOLD_STEPS:
            succ_k = k
        if fell_k is None and (illegal or zb < W.COLLAPSE_HEIGHT):
            fell_k = k
            break
        if succ_k is not None and k >= succ_k + 50:
            break
    su = np.where(z["success"][:, e])[0]
    return dict(e=e, h=h, succ=succ_k is not None and (fell_k is None or fell_k > succ_k + 50 or k_end < succ_k + 50),
                crossed=succ_k is not None, fell=fell_k is not None and succ_k is None, zmax=zmax,
                isaac_succ=len(su) > 0, map_err=map_err, x_end=d.qpos[0] - (half + th_eff), z_end=d.qpos[2],
                qd_peak=qd_peak, vmax=vmax_mj, probe=probe, ti=ti, pen=-pen_min, deep=len(deep_steps),
                fell_past=fell_k is not None and succ_k is None and d.qpos[0] > half + th_eff)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pre", type=int, nargs="+", default=[40, 20, 10, 0])
    ap.add_argument("--n_env", type=int, default=40)
    ap.add_argument("--hmin", type=float, default=0.0)
    ap.add_argument("--hmax", type=float, default=2.0)
    ap.add_argument("--isaac_success_only", action="store_true", help="Isaac で越えた env だけ使う")
    ap.add_argument("--horizon", type=int, default=150, help="合図からこのステップ数まで回す")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--delay", type=int, default=3, help="MuJoCo 側の遅延 (Isaac 物理ステップ数、dump の --fixed_delay と揃える)")
    ap.add_argument("--seed", type=int, default=0)
    # physics variants (wallcross_mujoco.py と同じ意味)
    ap.add_argument("--timestep", type=float, default=0.0)
    ap.add_argument("--wall_friction", type=float, default=1.0)
    ap.add_argument("--isaac_wall", action="store_true")
    ap.add_argument("--foot_friction", type=float, default=None)
    ap.add_argument("--foot_condim", type=int, default=None, help="足の球の condim (go2.xml は 6 = 転がり摩擦込み)")
    ap.add_argument("--wall_solref", type=float, nargs=2, default=None)
    ap.add_argument("--wall_solimp", type=float, nargs=3, default=None)
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--wall_thickness", type=float, default=0.0, help=">0: Isaac の壁の厚さの代わりにこれを使う")
    ap.add_argument("--wall_height_override", type=float, default=0.0)
    ap.add_argument("--isaac_match", action="store_true")
    ap.add_argument("--isaac_calf_geom", action="store_true", help="Isaac にだけあるすね下端の円柱を足す")
    ap.add_argument("--calf_capsule", action="store_true", help="すねの衝突円柱をカプセルに替える")
    ap.add_argument("--pd_hold", type=float, default=0.0, help=">0: PD トルクをこの周期 [s] でゼロ次ホールド (Isaac は 0.005)")
    ap.add_argument("--trace_env", type=int, default=-1, help="この env の毎ステップを MuJoCo/Isaac で並べて表示")
    ap.add_argument("--vcap_damper", type=float, default=0.0, help="関節速度上限を超えた分に掛けるダンパ [N m s/rad]")
    ap.add_argument("--probe", type=int, default=10, help="合図からこのステップ後の胴体の状態を Isaac と比べる")
    ap.add_argument("--knee_x1", type=float, default=7.0)
    ap.add_argument("--knee_x2", type=float, default=15.7)
    a = ap.parse_args()

    z = dict(np.load(a.npz))   # NpzFile は添字のたびに展開し直すので先に全部読む
    params = yaml.safe_load(open(os.path.join(W.rel(W.DEF_PARAMS), "deploy.yaml")))
    pol = W.MLP(a.ckpt)
    T, N = z["done"].shape
    envs = []
    for e in range(N):
        tr = np.where(z["triggered"][:, e])[0]
        if not len(tr) or not (a.hmin <= z["height"][e] <= a.hmax) or z["thick"][e] >= 0.1:
            continue
        dn = np.where(z["done"][:, e])[0]
        if len(dn) and dn[0] < tr[0] + 5:
            continue
        if a.isaac_success_only and not z["success"][:, e].any():
            continue
        envs.append(e)
    envs = envs[: a.n_env]
    print(f"[replay] {a.npz} | {pol.path} (in {pol.in_dim}) | {len(envs)} envs, walls "
          f"{z['height'][envs].min():.3f}-{z['height'][envs].max():.3f} m | Isaac crossed "
          f"{int(sum(z['success'][:, e].any() for e in envs))}/{len(envs)}", flush=True)
    for pre in a.pre:
        rng = np.random.default_rng([a.seed, pre + 1000])
        rows = [run_env(z, e, pre, pol, params, a, rng) for e in envs]
        n = len(rows)
        c = lambda k: sum(bool(r[k]) for r in rows)  # noqa: E731
        me = [r["map_err"] for r in rows if r["map_err"] is not None]
        print(f"  start {pre:+4d} steps before cue: crossed+stood {c('succ')}/{n} crossed {c('crossed')}/{n} "
              f"fell {c('fell')}/{n} hang {n - c('crossed') - c('fell')}/{n} | zmax p50 "
              f"{np.median([r['zmax'] for r in rows]):.3f} | wall penetration max p50 {1000 * np.median([r['pen'] for r in rows]):.1f} mm, "
              f"runs with >10 mm {sum(r['deep'] > 0 for r in rows)}/{n} | falls with base past the wall {c('fell_past')}"
              + (f" | map vs Isaac at start: max {np.median([x[0] for x in me]):.3f} mean {np.median([x[1] for x in me]):.4f} (p50 over envs)"
                 if me else ""), flush=True)
        pr = [r for r in rows if r["probe"] is not None]
        if pr:
            def isaac_probe(r):
                e, k = r["e"], r["ti"] + a.probe
                q = z["root_quat"][k, e].astype(np.float64)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, q)
                v = R.reshape(3, 3) @ z["linvel_b"][k, e]
                return dict(z=z["root_pos"][k, e, 2], x=z["root_pos"][k, e, 0] - z["half"][e], vx=v[0], vz=v[2],
                            wy=z["angvel_b"][k, e, 1], pitch=float(np.degrees(np.arcsin(np.clip(R[6], -1, 1)))))
            ip = [isaac_probe(r) for r in pr]
            print(f"      cue+{a.probe} (MJ / Isaac): " + " | ".join(
                f"{key} {np.median([p['probe'][key] for p in pr]):+.3f} / {np.median([q[key] for q in ip]):+.3f}"
                for key in ("x", "z", "vx", "vz", "wy", "pitch")))
            qi = np.array([np.abs(z["jvel"][r["ti"]:r["ti"] + 13, r["e"]]).max(0) for r in pr])
            qm = np.array([r["qd_peak"] for r in pr])
            print("      peak |qd| cue..cue+12, hip/thigh/calf (MJ / Isaac): "
                  + " | ".join(f"{nm} {np.median(qm[:, sl]):.1f} / {np.median(qi[:, sl]):.1f}"
                               for nm, sl in (("hip", slice(0, 4)), ("thigh", slice(4, 8)), ("calf", slice(8, 12)))))


if __name__ == "__main__":
    main()
