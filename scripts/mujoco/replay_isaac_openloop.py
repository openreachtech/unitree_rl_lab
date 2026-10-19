#!/usr/bin/env python3
"""Isaac の記録 (isaac_dump_sequence.py) を MuJoCo で開ループ再生し、跳躍の高さ・前進量を比べる。

ジャンプ合図の 10 ステップ前の Isaac の状態 (胴体の姿勢・速度、関節角・角速度) を MuJoCo に写し、
Isaac が実際に出した行動をそのまま 45 ステップ流す。物理の違いだけが残るので、
1 つずつ設定を変えて (--solref 等) どれが差を作っているかを調べるための道具。

    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/replay_isaac_openloop.py \
        reproduce_logs/mujoco_wallcross/isaac_ref_run13_nodr_delay0.npz [--solref 0.005 1] ...
"""
import argparse
import collections
import os
import sys

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallcross_mujoco as W  # noqa: E402

ISAAC_MASS_NPZ = os.path.join(W.REPO, "reproduce_logs/mujoco_wallcross/isaac_body_props.npz")
MJ_OF_ISAAC = {"base": "base_link"}   # Isaac body name -> MJCF body name (others identical)


def _q2m(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, q)
    return M


def _feet_force(m, d, rb):
    f = np.zeros(4)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            if g in rb.foot_geom:
                mujoco.mj_contactForce(m, d, i, f6)
                f[rb.foot_geom.index(g)] += np.linalg.norm(f6[:3])
    return f


def isaac_inertia_hook(a):
    """MjSpec の段階で Isaac の質量・重心・慣性テンソル (body 座標系, PhysX から読んだ値) に置き換える。
    コンパイル後の MjModel の body_inertia/iquat を書き換えると MuJoCo の派生量と食い違うので、こちらでやる。"""
    import re
    z = np.load(ISAAC_MASS_NPZ)
    names = list(z["bodies"])

    def hook(spec):
        for nm in names:
            if not re.fullmatch(a.isaac_inertia_bodies, nm):
                continue
            try:
                body = spec.body(MJ_OF_ISAAC.get(nm, nm))
            except Exception:
                body = None
            if body is None:
                continue
            i = names.index(nm)
            I = z["inertia"][i].reshape(3, 3).astype(np.float64)
            parts = a.isaac_inertia_parts
            if "m" in parts:
                body.mass = float(z["mass"][i])
            if "c" in parts:
                body.ipos = z["com"][i][:3].astype(np.float64)
            if "i" in parts:
                if nm.endswith("_foot"):
                    I = np.eye(3) * 1e-6
                I = 0.5 * (I + I.T)
                w_, v_ = np.linalg.eigh(I)
                if np.linalg.det(v_) < 0:
                    v_[:, 0] *= -1
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, v_.flatten())
                body.inertia = np.maximum(w_, 1e-7)
                body.iquat = q
            if nm.endswith("_foot") and "m" in parts:
                body.ipos = [0, 0, 0]
                if "i" not in parts:
                    body.inertia = [1e-6, 1e-6, 1e-6]
    return hook


def apply_variant(m, a):
    """MJCF をメモリ上で書き換える (元ファイルは触らない)。"""
    robot = set(b for b in range(1, m.nbody) if m.body_rootid[b] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in W.FEET]
    if a.solref is not None:
        m.geom_solref[:] = a.solref
    if a.solimp is not None:
        m.geom_solimp[:, :3] = a.solimp
    if a.margin is not None:
        m.geom_margin[:] = a.margin
    if a.foot_friction is not None:
        m.geom_friction[feet, 0] = a.foot_friction
    if a.foot_radius is not None:
        m.geom_size[feet, 0] = a.foot_radius
    if a.foot_z is not None:
        m.geom_pos[feet, 2] = a.foot_z
    if a.condim is not None:
        m.geom_condim[feet] = a.condim
    jnt = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    dof = [m.jnt_dofadr[j] for j in jnt]
    if a.joint_damping is not None:
        m.dof_damping[dof] = a.joint_damping
    if a.frictionloss is not None:
        m.dof_frictionloss[dof] = a.frictionloss
    if a.armature is not None:
        m.dof_armature[dof] = a.armature
    if a.limit_solref is not None:
        m.jnt_solref[jnt] = a.limit_solref
    if a.limit_solimp is not None:
        m.jnt_solimp[jnt, :3] = a.limit_solimp
    if a.impratio is not None:
        m.opt.impratio = a.impratio
    if a.cone is not None:
        m.opt.cone = {"pyramidal": 0, "elliptic": 1}[a.cone]
    if a.no_selfcollision is False:
        pass
    if a.calf_inertia_scale is not None:
        for f in W.FEET:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_calf")
            m.body_inertia[b] *= a.calf_inertia_scale
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--n_env", type=int, default=21)
    ap.add_argument("--closed_loop", default=None, help="ckpt: open-loop ではなく、このポリシーを MuJoCo の状態で回す")
    ap.add_argument("--pre", type=int, default=10, help="合図の何ステップ前から再生するか")
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--delay", type=int, default=0, help="MuJoCo 側の遅延 (Isaac 物理ステップ数)")
    ap.add_argument("--timestep", type=float, default=0.0)
    ap.add_argument("--pd_hold", type=float, default=0.0,
                    help=">0: PD トルクをこの周期 [s] で計算してゼロ次ホールド (Isaac は 0.005)")
    ap.add_argument("--isaac_clip", action="store_true", help="Isaac の Y1/Y2/X1/X2 トルク制限 (速度依存)")
    ap.add_argument("--solref", type=float, nargs=2, default=None)
    ap.add_argument("--solimp", type=float, nargs=3, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--foot_friction", type=float, default=None)
    ap.add_argument("--foot_radius", type=float, default=None)
    ap.add_argument("--foot_z", type=float, default=None)
    ap.add_argument("--condim", type=int, default=None)
    ap.add_argument("--joint_damping", type=float, default=None)
    ap.add_argument("--frictionloss", type=float, default=None)
    ap.add_argument("--armature", type=float, default=None)
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--knee_x1", type=float, default=13.5, help="--isaac_clip の膝の T-N 折れ点 [rad/s]")
    ap.add_argument("--knee_x2", type=float, default=30.0, help="--isaac_clip の膝の無負荷速度 [rad/s]")
    ap.add_argument("--vmax_calf", type=float, default=15.7)
    ap.add_argument("--vmax_hipthigh", type=float, default=30.1)
    ap.add_argument("--vcap_damper", type=float, default=0.0, help="関節速度上限を超えた分に掛けるダンパ係数 [N m s/rad]")
    ap.add_argument("--vclamp", action="store_true", help="関節速度を PhysX の上限 (30.1/15.7 rad/s) でクリップ")
    ap.add_argument("--limit_solref", type=float, nargs=2, default=None, help="関節リミットの solref (既定 0.02 1)")
    ap.add_argument("--limit_solimp", type=float, nargs=3, default=None)
    ap.add_argument("--cone", choices=["pyramidal", "elliptic"], default=None)
    ap.add_argument("--calf_inertia_scale", type=float, default=None)
    ap.add_argument("--isaac_inertia_spec", action="store_true",
                    help="Isaac の質量/重心/慣性を MjSpec で入れる (--isaac_inertia_parts/--isaac_inertia_bodies で絞る)")
    ap.add_argument("--isaac_inertia_bodies", default=".*", help="Isaac の body 名の正規表現 (対象を絞る)")
    ap.add_argument("--isaac_inertia_parts", default="mic", help="m=質量, i=慣性テンソル, c=重心位置 の組み合わせ")
    ap.add_argument("--no_wall", action="store_true")
    ap.add_argument("--no_selfcollision", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dump_env", type=int, default=-1)
    a = ap.parse_args()

    z = np.load(a.npz)
    params = yaml.safe_load(open(os.path.join(W.rel(W.DEF_PARAMS), "deploy.yaml")))
    envs = [e for e in range(z["done"].shape[1]) if 0.6 <= z["height"][e] <= 0.72 and z["thick"][e] <= 0.1
            and z["half"][e] >= 1.45][: a.n_env]
    VMAX = np.array([a.vmax_hipthigh] * 8 + [a.vmax_calf] * 4)
    Y1 = np.array([20.2] * 8 + [39.21735] * 4)
    Y2 = np.array([23.4] * 8 + [45.43] * 4)
    pol = W.MLP(a.closed_loop) if a.closed_loop else None
    rows = []
    for e in envs:
        ti = int(z["triggered"][:, e].argmax())
        k0 = ti - a.pre
        if k0 + a.steps + 2 >= z["obs"].shape[0]:
            continue
        m = W.build_model(z["half"][e], 0.0 if a.no_wall else z["height"][e], z["thick"][e], 6.0, a.timestep, 1.0,
                          spec_hook=isaac_inertia_hook(a) if a.isaac_inertia_spec else None)
        m = apply_variant(m, a)
        d = mujoco.MjData(m)
        rb = W.Robot(m, params)
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = z["root_pos"][k0, e]
        d.qpos[3:7] = z["root_quat"][k0, e]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        d.qvel[0:3] = R.reshape(3, 3) @ z["linvel_b"][k0, e]
        d.qvel[3:6] = z["angvel_b"][k0, e]
        d.qpos[rb.qadr] = z["jpos"][k0, e]
        d.qvel[rb.vadr] = z["jvel"][k0, e]
        mujoco.mj_forward(m, d)
        dsub = int(round(a.delay * W.ISAAC_PHYSICS_DT / m.opt.timestep))
        hold_n = max(1, int(round(a.pd_hold / m.opt.timestep))) if a.pd_hold > 0 else 1
        pending = collections.deque()
        sub = 0
        target = rb.default_q + z["act"][k0, e] * 0.25
        traj = []
        mj_contacts = collections.Counter()
        tau = np.zeros(12)
        if pol is not None:
            o0 = z["obs"][k0 + 1, e].astype(np.float64)
            hist = [collections.deque(list(o0[c:c + 36].reshape(3, 12)), 3) for c in (16, 52, 88)]
            last = o0[112:124].copy()
        for k in range(k0 + 1, k0 + 1 + a.steps):
            if pol is None:
                act = z["act"][k, e]
            else:
                # closed loop: Isaac の指令列 (cmd/jump/jump_time/handstand = obs の 6:16 列) を使い、状態は MuJoCo から作る
                if k > k0 + 1:
                    hist[0].append(d.qpos[rb.qadr] - rb.default_q)
                    hist[1].append(d.qvel[rb.vadr] * 0.05)
                    hist[2].append(last)
                Rm = d.xmat[rb.base].reshape(3, 3)
                o = np.concatenate([d.qvel[3:6] * 0.2, Rm.T @ np.array([0, 0, -1.0]), z["obs"][k, e, 6:16],
                                    np.concatenate(hist[0]), np.concatenate(hist[1]), np.concatenate(hist[2])])
                if pol.in_dim == 279:
                    o = np.concatenate([o, W.wall_heightmap(d.qpos[:2], d.qpos[3:7], z["half"][e], z["thick"][e],
                                                            0.0 if a.no_wall else z["height"][e], 100.0)])
                act = pol(o)
                last = act
            pending.append((sub + dsub, rb.default_q + act * 0.25))
            for _ in range(rb.decimation):
                while pending and pending[0][0] <= sub:
                    target = pending.popleft()[1]
                if sub % hold_n == 0:
                    tau = rb.kp * (target - d.qpos[rb.qadr]) - rb.kd * d.qvel[rb.vadr]
                    if a.isaac_clip:
                        qd = d.qvel[rb.vadr]
                        mx = np.where(qd * tau > 0, Y1, Y2)
                        X1 = np.array([13.5] * 8 + [a.knee_x1] * 4)
                        X2 = np.array([30.0] * 8 + [a.knee_x2] * 4)
                        mx = np.where(np.abs(qd) < X1, mx, np.clip(mx - mx / (X2 - X1) * (np.abs(qd) - X1), 0, None))
                        tau = np.clip(tau, -mx, mx)
                    else:
                        tau = np.clip(tau, -rb.tau_lim, rb.tau_lim)
                d.ctrl[rb.act] = tau
                if a.vcap_damper > 0:
                    # PhysX の関節最大速度を「超えた分にだけ効く強いダンパ」で近似する。
                    # qvel を後から書き換える (--vclamp) と運動量が消えるだけで他の関節・接地に反作用が出ない。
                    qd = d.qvel[rb.vadr]
                    d.qfrc_applied[rb.vadr] = -a.vcap_damper * (qd - np.clip(qd, -VMAX, VMAX))
                mujoco.mj_step(m, d)
                sub += 1
                if a.vclamp:
                    # PhysX の関節最大速度 (USD 由来: hip/thigh 30.1, calf 15.7 rad/s) を真似る
                    d.qvel[rb.vadr] = np.clip(d.qvel[rb.vadr], -VMAX, VMAX)
            for i in range(d.ncon):
                c = d.contact[i]
                for g in (c.geom1, c.geom2):
                    b = m.geom_bodyid[g]
                    if b in rb.robot_bodies:
                        mj_contacts[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] += 1
            traj.append((d.qpos[0], d.qpos[2], d.qvel[2]))
            if k == ti:
                g_mj = d.xmat[rb.base].reshape(3, 3).T @ np.array([0, 0, -1.0])
                at_cue = dict(gap_mj=z["half"][e] - d.qpos[0], gap_is=z["half"][e] - z["root_pos"][k, e, 0],
                              zc_mj=d.qpos[2], zc_is=z["root_pos"][k, e, 2],
                              gx_mj=g_mj[0], gx_is=z["obs"][k + 1, e, 3],
                              vx_mj=d.qvel[0], vx_is=(_q2m(z["root_quat"][k, e]).reshape(3, 3) @ z["linvel_b"][k, e])[0],
                              wy_mj=d.qvel[4], wy_is=z["angvel_b"][k, e, 1])
            if a.dump_env == e:
                kk = k
                feet_c = [any(m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ft}_calf")
                              for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2)) for ft in W.FEET]
                print(f"  k{kk - k0:3d} z is {z['root_pos'][kk, e, 2]:.3f} mj {d.qpos[2]:.3f} | "
                      f"RL thigh q is {z['jpos'][kk, e, 6]:+.2f} mj {d.qpos[rb.qadr[6]]:+.2f} qd is {z['jvel'][kk, e, 6]:+6.1f} mj {d.qvel[rb.vadr[6]]:+6.1f} | "
                      f"RL calf q is {z['jpos'][kk, e, 10]:+.2f} mj {d.qpos[rb.qadr[10]]:+.2f} qd is {z['jvel'][kk, e, 10]:+6.1f} mj {d.qvel[rb.vadr[10]]:+6.1f} | "
                      f"tau mj thigh {tau[6]:+5.1f} calf {tau[10]:+5.1f} | target-q calf {target[10] - d.qpos[rb.qadr[10]]:+.2f} | "
                      f"feetF is {np.round(z['feet_f'][kk, e]).astype(int)} mj {np.round(_feet_force(m, d, rb)).astype(int)}")
        traj = np.array(traj)
        iz = z["root_pos"][k0 + 1: k0 + 1 + a.steps, e]
        lv = z["linvel_b"][k0 + 1: k0 + 1 + a.steps, e]
        qq = z["root_quat"][k0 + 1: k0 + 1 + a.steps, e]
        vz_is = np.array([(lambda M: (M.reshape(3, 3) @ lv[i])[2])(_q2m(qq[i])) for i in range(len(lv))])
        kc = min(a.pre + 10, a.steps)      # 10 control steps after the cue: takeoff done, before any wall contact (Isaac: >= 10)
        su = z["success"][:, e]
        outer = z["half"][e] + z["thick"][e]
        cross_mj = (d.qpos[0] - outer > 0.2) and d.qpos[2] > 0.2
        kend = k0 + a.steps
        cross_is = (z["root_pos"][kend, e, 0] - outer > 0.2) and z["root_pos"][kend, e, 2] > 0.2
        perch_mj = abs(d.qpos[0] - z["half"][e]) < 0.15 and d.qpos[2] > 0.5
        rows.append(dict(**(at_cue if ti >= k0 + 1 else {}), cross_mj=cross_mj, cross_is=cross_is, perch_mj=perch_mj, x_end_mj=d.qpos[0] - outer,
                         x_end_is=z["root_pos"][kend, e, 0] - outer,
            e=e, zmax_mj=traj[:, 1].max(), zmax_is=iz[:, 2].max(),
            dx_mj=traj[-1, 0] - traj[0, 0], dx_is=iz[-1, 0] - iz[0, 0],
            vz_mj=traj[:, 2].max(),
            err20=np.abs(traj[:20, 1] - iz[:20, 2]).max(),
            z_c_mj=traj[kc - 1, 1], z_c_is=iz[kc - 1, 2], vz_c_mj=traj[kc - 1, 2], vz_c_is=vz_is[kc - 1],
            vzmax_mj=traj[:kc, 2].max(), vzmax_is=vz_is[:kc].max(),
        ))
        if a.verbose:
            print(f"env {e}: zmax mj {rows[-1]['zmax_mj']:.3f} is {rows[-1]['zmax_is']:.3f} | contacts {dict(mj_contacts)}")
    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    dz = r["zmax_mj"] - r["zmax_is"]
    ddx = r["dx_mj"] - r["dx_is"]
    print(f"n={len(rows)}  zmax(MJ)-zmax(Isaac): mean {dz.mean():+.3f} median {np.median(dz):+.3f} sd {dz.std():.3f} | "
          f"apex Isaac {r['zmax_is'].mean():.3f} MJ {r['zmax_mj'].mean():.3f} | forward travel (0.9 s) Isaac {r['dx_is'].mean():.3f} "
          f"MJ {r['dx_mj'].mean():.3f} (diff {ddx.mean():+.3f}) | max |z err| first 0.4 s {r['err20'].mean():.3f}")
    print(f"   end of window ({a.steps} steps): base past wall+0.2 & up: MJ {int(r['cross_mj'].sum())}/{len(rows)} "
          f"Isaac {int(r['cross_is'].sum())}/{len(rows)} | MJ perched on wall {int(r['perch_mj'].sum())} | "
          f"x past outer face MJ {r['x_end_mj'].mean():+.2f} Isaac {r['x_end_is'].mean():+.2f}")
    if "gap_mj" in r:
        print("   state at the cue (MJ / Isaac): " + " | ".join(
            f"{n} {r[n + '_mj'].mean():+.3f} / {r[n + '_is'].mean():+.3f}" for n in ("gap", "zc", "gx", "vx", "wy")))
    print(f"   at cue+10 steps (pre-wall): z Isaac {r['z_c_is'].mean():.3f} MJ {r['z_c_mj'].mean():.3f} (diff {np.mean(r['z_c_mj'] - r['z_c_is']):+.3f}) | "
          f"vz Isaac {r['vz_c_is'].mean():.2f} MJ {r['vz_c_mj'].mean():.2f} | peak vz Isaac {r['vzmax_is'].mean():.2f} MJ {r['vzmax_mj'].mean():.2f} "
          f"(ratio {np.mean(r['vzmax_mj']) / np.mean(r['vzmax_is']):.2f}) | ballistic apex from cue+10 Isaac {np.mean(r['z_c_is'] + np.maximum(r['vz_c_is'], 0)**2 / 19.62):.3f} "
          f"MJ {np.mean(r['z_c_mj'] + np.maximum(r['vz_c_mj'], 0)**2 / 19.62):.3f}")


if __name__ == "__main__":
    main()
