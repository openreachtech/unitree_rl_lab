#!/usr/bin/env python3
"""走り幅跳びポリシーを MuJoCo で N 回跳ばせ、着地率と飛距離を測る。

これまで mujoco の判定は unitree_mujoco(C++/DDS) + go2_ctrl をNoMachineで2枚の端末に
立ち上げ、キーボードで跳ばせて目で数えていた。「Isaac の指標は実機性能を予測しない」以上
候補は mujoco で選ぶしかないのに、1個体あたり十数回の手作業が要るのがループの律速だった。
これはその判定を自動化する。C++ 経路を置き換えるものではない -- あちらは実機と同じコードで
あることに意味がある -- が、候補を絞るところまでは機械にやらせられる。

観測は deploy.yaml の順そのまま (47次元):
    base_ang_vel*0.2 | projected_gravity | velocity_commands | joint_pos_rel
    | joint_vel_rel*0.05 | last_action | jump_command | jump_time
jump_command / jump_time の扱いは State_RLBase.cpp に合わせる -- コマンドは
jump_hold_time_s (0.85 s) の固定タイマで落ち、jump_time は立っている間だけ
time_since_trigger / max_jump_duration_s (2.2 s) を返す。

実行には mujoco の入った venv が要る: /home/tanaka/isaacsim/anaguma/venv/bin/python
"""
import argparse
import collections
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
REAL_JUMP_AIRBORNE_S = 0.20   # mdp の real_jump 判定と同じ
REAL_JUMP_RISE_M = 0.08


def named(m, objtype, *cands):
    for c in cands:
        i = mujoco.mj_name2id(m, objtype, c)
        if i >= 0:
            return i
    raise SystemExit(f"none of {cands} found")



_FONT = None


def _font(size=30):
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(path):
                _FONT = ImageFont.truetype(path, size)
                break
        else:
            _FONT = ImageFont.load_default()
    return _FONT


def _marker(scene, x, y, rgba):
    """地面に立てる小さな目印（踏切位置・着地位置）。ロボットと同じ y に置く
    ――y=0 固定にするとロボットが横にドリフトしたとき奥行きがずれて巨大に映る。"""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_BOX,
        np.array([0.02, 0.02, 0.12]), np.array([x, y, 0.12]),
        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def _draw_frame(renderer, frames, d, cam, seg, takeoff, land, t, vx_body, triggered, label=""):
    """1フレーム描いて frames に積む。踏切/着地の目印と、飛距離のテキストを載せる。"""
    from PIL import Image, ImageDraw
    renderer.update_scene(d, cam)
    tx = takeoff[1] if takeoff is not None else (seg["x0"] if seg is not None else None)
    if tx is not None:
        _marker(renderer.scene, tx, d.qpos[1], [1.0, 0.25, 0.15, 0.8])     # 踏切: 赤
    if land is not None:
        _marker(renderer.scene, land[1], d.qpos[1], [0.2, 0.9, 0.35, 0.8])  # 着地: 緑
    img = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(img)
    lines = ([label] if label else []) + [f"助走 {vx_body:4.2f} m/s" if not triggered else "踏切トリガ済み"]
    if takeoff is not None:
        lines.append(f"離陸 v=({takeoff[2]:.2f}, {takeoff[3]:.2f}) m/s")
    if takeoff is not None and land is not None:
        lines.append(f"飛距離 {land[1] - takeoff[1]:.3f} m")
    for i, line in enumerate(lines):
        dr.text((24, 20 + 38 * i), line, fill=(255, 255, 255), font=_font(30),
                stroke_width=3, stroke_fill=(0, 0, 0))
    frames.append(np.asarray(img))


def run_trial(m, d, sess, in_name, qadr, vadr, act_id, default_q, kp, kd, tau_lim,
              step_dt, decimation, cmd, cfg, base_bid, feet, floor_gid, seed,
              warmup_s, window_s, frames=None, cam=None, renderer=None, fps_every=1,
              action_delay=0, video_label=""):
    rs = np.random.default_rng(seed)
    # 踏切のタイミングを試行ごとに散らす。C++ 経路では人がキーボードで叩くので一定でなく、
    # 助走のどの位相で跳ぶかは着地の成否に効く。固定すると1つの位相しか測れない。
    warmup_s = warmup_s + rs.uniform(-0.35, 0.35)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    d.qpos[qadr] = default_q + rs.normal(0, 0.01, 12)
    mujoco.mj_kinematics(m, d)
    d.qpos[2] = m.geom_size[feet[0]][0] - min(d.geom_xpos[f][2] for f in feet) + 0.002
    mujoco.mj_forward(m, d)

    hold = float(cfg["commands"]["jump_command"]["jump_hold_time_s"])
    horizon = float(cfg["commands"]["jump_command"]["max_jump_duration_s"])

    last_action = np.zeros(12)
    # 制御レートの遅延。C++/DDS 経路や実機は「観測してから指令が効くまで」1〜2 ステップ遅れる。
    # ここを 0 のままにすると、python 経路だけが遅延ゼロの理想条件で測ることになる。
    delayed = collections.deque([np.zeros(12)] * action_delay, maxlen=max(1, action_delay))
    jump_cmd, t_since = 0.0, 0.0
    triggered = False
    t = 0.0
    takeoff = None        # (t, x, vx, vz)
    land = None           # (t, x)
    apex = 0.0
    flight_s = 0.0
    seg = None
    airborne_since = None
    base_hit = False
    approach = []
    n_ctrl = int((warmup_s + window_s) / step_dt)

    for k in range(n_ctrl):
        if not triggered and t >= warmup_s:
            triggered, jump_cmd, t_since = True, 1.0, 0.0
            trigger_x, trigger_z = d.qpos[0], d.qpos[2]
            if frames is not None:
                # ここでカメラを止める。以降は背景が動かないので、
                # 踏切から着地までの移動量が画面上でそのまま見える。
                cam.lookat[:] = [d.qpos[0] + 1.6, d.qpos[1], 0.35]

        q, qd = d.qpos[qadr], d.qvel[vadr]
        R = d.xmat[base_bid].reshape(3, 3)
        obs = np.concatenate([
            # 2026-09-15 修正: 以前は (R.T @ d.qvel[3:6]) としていたが、MuJoCo の
            # free joint の qvel[3:6] は **すでに胴体ローカル系** の角速度なので
            # R.T を掛けると二重回転になっていた。実測で確認:
            #   胴体を y 軸 90 度に傾けてワールド z 軸まわりに 1 rad/s 回すと
            #   qvel[3:6] = [0,0,1] / gyro センサ = [0,0,1] / R.T@qvel = [-1,0,0]
            # IsaacLab の base_ang_vel も deploy C++ が読む IMU gyro もローカル系なので
            # 掛けないのが正しい。姿勢が立っているときは R ≈ I で差が出ないが、
            # 空中で傾いている間 -- まさに跳躍を決める区間 -- でずれていた。
            d.qvel[3:6] * 0.2,
            R.T @ np.array([0.0, 0.0, -1.0]),
            cmd,
            q - default_q,
            qd * 0.05,
            last_action,
            [jump_cmd],
            [t_since / horizon if jump_cmd > 0.0 else 0.0],
        ]).astype(np.float32)[None, :]
        action = sess.run(None, {in_name: obs})[0][0].astype(np.float64)
        last_action = action
        if action_delay:
            applied = delayed[0]
            delayed.append(action)
        else:
            applied = action
        target = default_q + applied * ACTION_SCALE

        for _ in range(decimation):
            tau = kp * (target - d.qpos[qadr]) - kd * d.qvel[vadr]
            d.ctrl[act_id] = np.clip(tau, -tau_lim, tau_lim)
            mujoco.mj_step(m, d)
        t += step_dt
        if jump_cmd > 0.0:
            t_since += step_dt
            if t_since > hold:
                jump_cmd = 0.0

        # 接地の状態
        touching = set()
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if floor_gid in (g1, g2):
                other = g2 if g1 == floor_gid else g1
                touching.add(int(other))
                if m.geom_bodyid[other] == base_bid:
                    base_hit = True
        feet_down = any(f in touching for f in feet)

        if frames is not None and k % fps_every == 0:
            # 2026-09-16: 助走から撮る。以前はここが `if not triggered: continue` の
            # 後ろにあったため、**踏切の瞬間から始まる動画**しか撮れていなかった。
            # カメラはトリガまで追従し、トリガで固定する（追従したままだと
            # どれだけ前に進んだのかが画面から分からないため）。
            if not triggered:
                cam.lookat[:] = [d.qpos[0] + 1.0, d.qpos[1], 0.35]
            # 着地から 0.8 秒で撮り終える（その後は走り去るだけで情報が無い）
            if land is None or t <= land[0] + 0.8:
                _draw_frame(renderer, frames, d, cam, seg, takeoff, land, t,
                            float((R.T @ d.qvel[:3])[0]), triggered, video_label)

        if not triggered:
            approach.append(float((R.T @ d.qvel[:3])[0]))
            continue

        # 滞空は「区間」として扱う。区間ごとに離陸時の高さと区間内の最高点を持ち、
        # real_jump 条件を満たした最初の区間だけを踏切として採用する。
        # 通しの最大高さを見ると、着地後のバウンドや転倒して脚で突っ張った姿勢まで
        # 拾ってしまう (実測: 弾道頂点 0.44m のはずが 0.64m と出た)。
        if not feet_down:
            if airborne_since is None:
                airborne_since = t
                seg = {"t0": t, "x0": d.qpos[0], "z0": d.qpos[2],
                       "vx": d.qvel[0], "vz": d.qvel[2], "zmax": d.qpos[2]}
            seg["zmax"] = max(seg["zmax"], d.qpos[2])
        else:
            if airborne_since is not None:
                flight = t - seg["t0"]
                rise = seg["zmax"] - seg["z0"]
                if takeoff is None and flight >= REAL_JUMP_AIRBORNE_S and rise >= REAL_JUMP_RISE_M:
                    takeoff = (seg["t0"], seg["x0"], seg["vx"], seg["vz"])
                    land = (t, d.qpos[0])
                    apex = rise
                    flight_s = flight
                airborne_since = None

    tilt = np.degrees(np.arccos(np.clip(d.xmat[base_bid].reshape(3, 3)[2, 2], -1, 1)))
    ok = (takeoff is not None) and (not base_hit) and tilt < 45.0
    return {
        "jumped": takeoff is not None,
        "success": bool(ok),
        "distance": (land[1] - takeoff[1]) if takeoff else 0.0,
        "apex_rise": apex if takeoff else 0.0,
        "flight_s": flight_s if takeoff else 0.0,
        "v_x": takeoff[2] if takeoff else 0.0,
        "v_z": takeoff[3] if takeoff else 0.0,
        "approach": float(np.mean(approach[-25:])) if approach else 0.0,
        "final_tilt": float(tilt),
        "base_hit": base_hit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--scene", default="/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--vx", type=float, default=2.8)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--action-delay", dest="action_delay", type=int, default=0,
                    help="指令を N 制御ステップ遅らせる (1 = 20 ms)")
    ap.add_argument("--video", default=None, help="mp4 の出力先")
    ap.add_argument("--video-trials", dest="video_trials", default="0",
                    help="動画に入れる試行番号（カンマ区切り、複数指定で連結）")
    ap.add_argument("--video-label", dest="video_label", default="",
                    help="動画の左上に出すポリシー名。遅延条件は自動で併記する")
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

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    decimation = max(1, int(round(step_dt / m.opt.timestep)))
    qadr = np.array([m.jnt_qposadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    vadr = np.array([m.jnt_dofadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    act_id = np.array([named(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in names])
    tau_lim = np.abs(m.actuator_ctrlrange[act_id]).max(axis=1)
    base_bid = named(m, mujoco.mjtObj.mjOBJ_BODY, "base", "base_link")
    feet = [named(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot", l) for l in ("FR", "FL", "RR", "RL")]
    floor_gid = named(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cmd = np.array([args.vx, 0.0, 0.0])

    frames, cam, renderer, fps_every = None, None, None, 1
    if args.video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
        m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
        renderer = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        # 横視点。**azimuth 270 で world +x が画面の左向き** ＝ ロボットは画面の右から左へ走る
        # （ユーザー指定、2026-09-16。90 で試したら左→右だったので実測で決めた）。
        cam.distance, cam.azimuth, cam.elevation = 4.5, 270, -6
        frames, fps_every = [], max(1, int(round((1 / 30) / step_dt)))

    vset = {int(x) for x in args.video_trials.split(",") if x.strip() != ""}
    rows = []
    for i in range(args.trials):
        r = run_trial(m, d, sess, in_name, qadr, vadr, act_id, default_q, kp, kd, tau_lim,
                      step_dt, decimation, cmd, cfg, base_bid, feet, floor_gid, seed=i,
                      warmup_s=args.warmup, window_s=args.window, action_delay=args.action_delay,
                      video_label=(f"{args.video_label}  /  遅延 {args.action_delay} step"
                                   if args.video_label else ""),
                      frames=frames if (args.video and i in vset) else None,
                      cam=cam, renderer=renderer, fps_every=fps_every)
        rows.append(r)
        print(f"  試行{i:2d}: 跳躍={'o' if r['jumped'] else 'x'} 着地={'o' if r['success'] else 'x'} "
              f"飛距離={r['distance']:.3f}m 頂点={r['apex_rise']:.3f}m "
              f"v=({r['v_x']:.2f},{r['v_z']:.2f}) 滞空={r['flight_s']:.2f}s 助走={r['approach']:.2f} 傾き={r['final_tilt']:.0f}deg")

    jumped = [r for r in rows if r["jumped"]]
    good = [r for r in rows if r["success"]]
    print(f"\n=== {os.path.basename(run)} / {args.policy or 'exported/policy.onnx'} ===")
    print(f"試行 {len(rows)}  跳躍成立 {len(jumped)} ({100*len(jumped)/len(rows):.0f}%)  "
          f"着地成功 {len(good)} ({100*len(good)/len(rows):.0f}%)")
    if jumped:
        print(f"飛距離 平均 {np.mean([r['distance'] for r in jumped]):.3f} m / 最大 {max(r['distance'] for r in jumped):.3f} m")
        print(f"頂点   平均 {np.mean([r['apex_rise'] for r in jumped]):.3f} m  "
              f"(弾道の予想 {np.mean([r['v_z'] for r in jumped])**2/(2*9.81):.3f} m)")
        print(f"滞空   平均 {np.mean([r['flight_s'] for r in jumped]):.3f} s  "
              f"(弾道の予想 {2*np.mean([r['v_z'] for r in jumped])/9.81:.3f} s)")
        print(f"離陸   v_x {np.mean([r['v_x'] for r in jumped]):.2f}  v_z {np.mean([r['v_z'] for r in jumped]):.2f}")
        print(f"助走   {np.mean([r['approach'] for r in jumped]):.2f} m/s")
    if args.video and frames:
        import imageio.v3 as iio
        iio.imwrite(args.video, np.stack(frames), fps=30)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
