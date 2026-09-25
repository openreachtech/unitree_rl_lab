#!/usr/bin/env python3
"""踏切の仕事のうち「実機に無いトルク」で稼いでいる割合を測る（2026-09-16）。

mujoco の go2.xml は ctrlrange が速度非依存なので、関節が何 rad/s で回っていても満トルクが出る。
IsaacLab の UnitreeActuator は T-N 曲線を持ち、X1=13.5 rad/s から低下し X2=30 rad/s でゼロになる。
実モーター(GO-M8010-6)の無負荷回転数も 30 rad/s なので、**Isaac 側が実機に近い**。

したがって mujoco で測った飛距離は、実機には出せないトルクに乗っている可能性がある。
このスクリプトは mujoco を走らせながら、各物理ステップで
  1. mujoco が実際に加えたトルク tau_mj
  2. 同じ瞬間に Isaac の T-N 曲線が許すトルク tau_isaac
を両方求め、踏切区間（トリガ〜離陸）の**正の機械仕事** ∫max(tau*omega,0)dt を
関節速度の帯域別に積算する。tau_isaac/tau_mj の仕事比が「実機で残る割合」の目安になる。

注意: これは反事実の見積もりであって再シミュレーションではない。トルクを制限すれば
軌道そのものが変わるので、実機の飛距離を直接予測する数字ではない。
「踏切のエネルギーがどの速度帯で作られているか」を見るためのもの。
"""
import argparse
import os

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import onnxruntime as ort

REPO = "/home/tanaka/isaacsim/unitree_rl_lab"
SDK_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ACTION_SCALE = 0.25

# GO2_CORRECTED_ACTUATOR_CFG (longjump_base_env_cfg.py) の値
X1, X2 = 13.5, 30.0
Y1_HIP, Y2_HIP = 20.2, 23.4
Y1_CALF, Y2_CALF = 39.21735042735043, 45.43


def isaac_clip(tau, omega, y1, y2, derate=True):
    """UnitreeActuator._clip_effort と同じ計算。

    derate=False にすると T-N 曲線の速度依存だけを外し、Y1/Y2 の平坦な上限は残す。
    mujoco の ctrlrange は押す側も止める側も同じ値(膝 45.43)なので、Isaac の
    Y1(押す側 39.22) より **速度ゼロでも 16% 高い**。26% の内訳をこの2つに割るために使う。
    """
    same_dir = (omega * tau) > 0
    max_eff = np.where(same_dir, y1, y2)
    if derate:
        k = -max_eff / (X2 - X1)
        derated = np.clip(k * (np.abs(omega) - X1) + max_eff, 0.0, None)
        max_eff = np.where(np.abs(omega) < X1, max_eff, derated)
    return np.clip(tau, -max_eff, max_eff)


def named(m, objtype, *cands):
    for c in cands:
        i = mujoco.mj_name2id(m, objtype, c)
        if i >= 0:
            return i
    raise SystemExit(f"none of {cands} found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--scene", default="/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--vx", type=float, default=2.8)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--window", type=float, default=3.0)
    args = ap.parse_args()

    run = args.run if os.path.isabs(args.run) else os.path.join(REPO, args.run)
    cfg = yaml.safe_load(open(os.path.join(run, "params", "deploy.yaml")))
    sess = ort.InferenceSession(args.policy or os.path.join(run, "exported", "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    jmap = cfg["joint_ids_map"]
    names = [SDK_JOINTS[i] for i in jmap]
    default_q = np.array(cfg["default_joint_pos"])
    kp = np.array(cfg["stiffness"])[jmap]
    kd = np.array(cfg["damping"])[jmap]
    step_dt = float(cfg["step_dt"])
    hold = float(cfg["commands"]["jump_command"]["jump_hold_time_s"])
    horizon = float(cfg["commands"]["jump_command"]["max_jump_duration_s"])

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    decimation = max(1, int(round(step_dt / dt)))
    qadr = np.array([m.jnt_qposadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    vadr = np.array([m.jnt_dofadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    act_id = np.array([named(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in names])
    tau_lim = np.abs(m.actuator_ctrlrange[act_id]).max(axis=1)
    base_bid = named(m, mujoco.mjtObj.mjOBJ_BODY, "base", "base_link")
    feet = [named(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot", l) for l in ("FR", "FL", "RR", "RL")]
    floor_gid = named(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cmd = np.array([args.vx, 0.0, 0.0])

    is_calf = np.array([("calf" in n) for n in names])
    y1 = np.where(is_calf, Y1_CALF, Y1_HIP)
    y2 = np.where(is_calf, Y2_CALF, Y2_HIP)

    REAL_JUMP_AIRBORNE_S = 0.20
    REAL_JUMP_RISE_M = 0.08
    TAKEOFF_MAX_S = 0.40      # 踏切として遡る上限

    rows = []
    for trial in range(args.trials):
        rs = np.random.default_rng(trial)
        warmup = args.warmup + rs.uniform(-0.35, 0.35)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[qadr] = default_q + rs.normal(0, 0.01, 12)
        mujoco.mj_kinematics(m, d)
        d.qpos[2] = m.geom_size[feet[0]][0] - min(d.geom_xpos[f][2] for f in feet) + 0.002
        mujoco.mj_forward(m, d)

        last_action = np.zeros(12)
        jump_cmd, t_since, t = 0.0, 0.0, 0.0
        triggered = False
        # 物理ステップ単位の履歴（トリガ後のみ）
        H_t, H_w, H_mj, H_is, H_down, H_z = [], [], [], [], [], []
        H_fl = []

        for _ in range(int((warmup + args.window) / step_dt)):
            if not triggered and t >= warmup:
                triggered, jump_cmd, t_since = True, 1.0, 0.0
            q, qd = d.qpos[qadr], d.qvel[vadr]
            R = d.xmat[base_bid].reshape(3, 3)
            obs = np.concatenate([
                d.qvel[3:6] * 0.2, R.T @ np.array([0.0, 0.0, -1.0]), cmd,
                q - default_q, qd * 0.05, last_action,
                [jump_cmd], [t_since / horizon if jump_cmd > 0.0 else 0.0],
            ]).astype(np.float32)[None, :]
            action = sess.run(None, {in_name: obs})[0][0].astype(np.float64)
            last_action = action
            target = default_q + action * ACTION_SCALE

            for _ in range(decimation):
                omega = d.qvel[vadr].copy()
                tau = kp * (target - d.qpos[qadr]) - kd * d.qvel[vadr]
                tau_mj = np.clip(tau, -tau_lim, tau_lim)
                d.ctrl[act_id] = tau_mj
                mujoco.mj_step(m, d)
                if triggered:
                    touching = set()
                    for c in range(d.ncon):
                        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                        if floor_gid in (g1, g2):
                            touching.add(int(g2 if g1 == floor_gid else g1))
                    H_t.append(d.time)
                    H_w.append(omega)
                    H_mj.append(tau_mj)
                    H_is.append(isaac_clip(tau_mj, omega, y1, y2))
                    H_fl.append(isaac_clip(tau_mj, omega, y1, y2, derate=False))
                    H_down.append(any(f in touching for f in feet))
                    H_z.append(d.qpos[2])

            t += step_dt
            if jump_cmd > 0.0:
                t_since += step_dt
                if t_since > hold:
                    jump_cmd = 0.0

        if not H_t:
            continue
        W = np.array(H_w); TM = np.array(H_mj); TI = np.array(H_is); TF = np.array(H_fl)
        down = np.array(H_down); Z = np.array(H_z); T = np.array(H_t)

        # 滞空区間を列挙し、real_jump を満たす最初のものを跳躍とする
        jump_i0 = None
        i = 0
        while i < len(down):
            if not down[i]:
                j = i
                while j < len(down) and not down[j]:
                    j += 1
                air_s = T[j - 1] - T[i]
                rise = Z[i:j].max() - Z[i]
                if air_s >= REAL_JUMP_AIRBORNE_S and rise >= REAL_JUMP_RISE_M:
                    jump_i0 = i
                    break
                i = j
            else:
                i += 1
        if jump_i0 is None:
            print(f"  試行{trial:2d}: real_jump なし、skip")
            continue

        # 踏切 = 離陸直前の連続接地区間（最大 TAKEOFF_MAX_S まで遡る）
        k0 = jump_i0 - 1
        while k0 > 0 and down[k0 - 1] and (T[jump_i0] - T[k0 - 1]) <= TAKEOFF_MAX_S:
            k0 -= 1
        sl = slice(k0, jump_i0)
        if sl.stop - sl.start < 2:
            continue
        w = W[sl]; tm = TM[sl]; ti = TI[sl]; tf = TF[sl]
        a = np.abs(w)
        p_mj = np.maximum(tm * w, 0.0) * dt
        p_is = np.maximum(ti * w, 0.0) * dt
        p_fl = np.maximum(tf * w, 0.0) * dt
        w_mj = p_mj.sum(); w_is = p_is.sum(); w_fl = p_fl.sum()
        if w_mj <= 0:
            continue
        rows.append(dict(
            t=(T[jump_i0] - T[k0]), w_mj=w_mj, w_is=w_is, w_fl=w_fl,
            lo=p_mj[a < X1].sum(), mid=p_mj[(a >= X1) & (a < X2)].sum(), hi=p_mj[a >= X2].sum(),
            mj_calf=p_mj[:, is_calf].sum(), is_calf=p_is[:, is_calf].sum(),
            peak=a.max(), peak_calf=a[:, is_calf].max(),
            # Isaac 側は制御レート(step_dt)でしか観測できないので、比較用に
            # 同じ間引きで取ったピークも出す（物理サブステップのピークより低く出る）。
            peak_ctrl=a[::decimation].max() if len(a) >= decimation else a.max(),
            peak_calf_ctrl=(a[::decimation][:, is_calf].max() if len(a) >= decimation else a[:, is_calf].max()),
            t_hi=(a >= X2).any(axis=1).sum() * dt, t_mid=((a >= X1) & (a < X2)).any(axis=1).sum() * dt,
            sat=np.mean(np.any(np.abs(tm) > np.abs(ti) + 1e-9, axis=1)),
        ))
        r = rows[-1]
        print(f"  試行{trial:2d}: 踏切 {r['t']*1000:5.0f}ms  仕事 mj {r['w_mj']:6.1f}J "
              f"isaac {r['w_is']:6.1f}J ({100*r['w_is']/r['w_mj']:4.0f}%)  "
              f"帯域 <13.5:{100*r['lo']/r['w_mj']:3.0f}% 13.5-30:{100*r['mid']/r['w_mj']:3.0f}% "
              f">30:{100*r['hi']/r['w_mj']:3.0f}%  ピーク {r['peak']:5.1f} rad/s(膝 {r['peak_calf']:5.1f})")

    if not rows:
        print("踏切が取れなかった")
        return
    n = len(rows)
    S = lambda k: sum(r[k] for r in rows)
    W_ = S("w_mj")
    print(f"\n=== {n} 試行の合計 ===")
    print(f"踏切区間の長さ      平均 {1000*S('t')/n:.0f} ms")
    print(f"踏切の正の機械仕事  mujoco {W_/n:.1f} J/試行")
    print(f"  同じ瞬間に Isaac の T-N 曲線が許す分  {S('w_is')/n:.1f} J/試行 "
          f"= **{100*S('w_is')/W_:.0f}%**（残りの {100-100*S('w_is')/W_:.0f}% は実機に無いトルク）")
    print(f"  膝だけで見ると                        {100*S('is_calf')/S('mj_calf'):.0f}% が残る")
    print(f"  内訳: 速度デレートを外し Y1/Y2 の平坦上限だけ掛けると {100*S('w_fl')/W_:.0f}% が残る")
    print(f"        → 失われる {100-100*S('w_is')/W_:.0f}% のうち "
          f"{100-100*S('w_fl')/W_:.0f}pt は平坦な上限差(Y1/Y2 vs ctrlrange)、"
          f"{100*S('w_fl')/W_-100*S('w_is')/W_:.0f}pt が速度デレート由来")
    print(f"仕事の作られた速度帯  <13.5 rad/s {100*S('lo')/W_:.0f}%  "
          f"13.5-30 {100*S('mid')/W_:.0f}%  >30 {100*S('hi')/W_:.0f}%")
    print(f"関節速度ピーク        全体 {np.mean([r['peak'] for r in rows]):.1f} rad/s "
          f"(最大 {max(r['peak'] for r in rows):.1f})  膝 {np.mean([r['peak_calf'] for r in rows]):.1f} "
          f"(最大 {max(r['peak_calf'] for r in rows):.1f})")
    print(f"（制御レート間引きでのピーク  全体 {np.mean([r['peak_ctrl'] for r in rows]):.1f} rad/s  "
          f"膝 {np.mean([r['peak_calf_ctrl'] for r in rows]):.1f} rad/s）")
    print(f"30 rad/s 超の関節がいた時間  {100*S('t_hi')/S('t'):.0f}% / "
          f"13.5-30 にいた時間 {100*S('t_mid')/S('t'):.0f}%")
    print(f"T-N 曲線に当たっていた物理ステップの割合  {100*np.mean([r['sat'] for r in rows]):.0f}%")


if __name__ == "__main__":
    main()
