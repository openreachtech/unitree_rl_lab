#!/usr/bin/env python3
"""Go2 「四足歩行 → 後足で立つ → 薄い壁を跳び越える → 着地 → 歩く」 を MuJoCo で通す sim2sim 評価。

Isaac 側の同じ手順 ``scripts/rsl_rl/film_wallcross_sequence.py`` (``--walk_until 3.2``) を MuJoCo で
再現する。2 つのポリシーを時刻表で切り替える:

* 歩行ポリシー (Gallop Phase2, 統一観測 124 列)     ... 0 〜 rise_at、および成功の handover_s 後から
* 壁越えポリシー (run13 等, 124 列 / HM 版は 279 列) ... rise_at 〜 成功 + handover_s

時刻表 (既定値。Isaac スクリプトの ``--walk_until 3.2 --handover_s 1.0`` に対応):

    0.0 - 0.5 s  歩行ポリシー, 速度指令 0
    0.5 - 3.2 s  歩行ポリシー, 速度指令 0.6 m/s (止まらずに次へ)
    3.2 s -      壁越えポリシー, handstand_command = [1, -1] (後足), 速度指令 0.6 m/s
    t >= 4.2 s かつ 胴体と壁の内面の距離 <= 0.5 m
                 → ジャンプ合図: handstand = [0, 0], jump_command = [1, clip(h-0.25, 0.05, 1), 0, 0]
                   を 26 制御ステップ (0.52 s) 出す。jump_time は Isaac の JumpCommand と同じく
                   合図の 2 ステップ目から t^3/(1+t^3) (t = 0, 0.02, ..., 0.48)
    成功 (胴体が壁の外面 +0.2 m より先・4 足とも外面より先・3 足以上接地・胴体高さ > 0.24 m・
          傾き < 15 deg を 0.3 s 保持) の 1.0 s 後に歩行ポリシーへ戻し、0.6 m/s で 3 s → 0 で 1 s

観測は Isaac Lab の ``UnifiedObservationsCfg.PolicyCfg`` の順・スケールそのまま (ノイズ無し = play 設定):

    base_ang_vel*0.2 (3) | projected_gravity (3) | velocity_commands (3) | jump_command (4)
    | jump_time (1) | handstand_command (2) | joint_pos_rel x3 (36) | joint_vel_rel*0.05 x3 (36)
    | last_action x3 (36)                                     = 124
    [--heightmap] + wall_heightmap (31 x 5, x-major)          = 279
    [--mid360]    + mid360_map (29 x 21, x-major, MID-360 の点群から作る高さマップ) = 733
                    (scripts/mujoco/mid360_map.py: Isaac の RollingLivoxSensor + LidarElevationMap の写し。
                     床と壁だけにレイを当てる。マップは試行の最初から毎ステップ更新し続ける)

履歴は古い順・最新が最後。リセット直後は Isaac の CircularBuffer と同じく最初のフレームで全履歴を埋める。
関節の並び (Isaac 順) と既定姿勢・PD ゲイン・行動スケールは run の ``params/deploy.yaml`` を正とする。

MuJoCo は CPU で回す (GPU は学習に使っているので触らない)。描画は EGL。

    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/wallcross_mujoco.py --trials 20
    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/wallcross_mujoco.py --trials 3 \
        --video reproduce_logs/mujoco_wallcross/run13.mp4 --video_trials 0
    # 歩行ポリシーだけの確認 (10 s, 0.6 m/s)
    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/mujoco/wallcross_mujoco.py --walker_only 10
    # 高さマップ版
    ... --heightmap --wall_policy logs/rsl_rl/go2_multitask_wallcross_thinfirst_fromwalklandall_hm/<run>/model_X.pt
    # MID-360 版 (ジャンプ合図は --heightmap と同じ「壁までの真の距離」で出す。実機で合図を
    # 地図から出す問題は別。--mid360_noise で Isaac の LidarNoiseCfg を掛ける)
    ... --mid360 --wall_policy logs/rsl_rl/go2_multitask_wallcross_thinfirst_fromwalklandall_mid360/<run>/model_X.pt

.pt は torch の無いこの venv では読めないので、env_isaaclab の python で export_actor_npz.py を呼んで
~/.cache/wallcross_mujoco/ に .npz を作ってから使う (自動)。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import os
import subprocess
import sys

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

REPO = "/home/tanaka/isaacsim/unitree_rl_lab_multitask"
HERE = os.path.dirname(os.path.abspath(__file__))
TORCH_PY = "/home/tanaka/isaacsim/env_isaaclab/bin/python"
CACHE = os.path.expanduser("~/.cache/wallcross_mujoco")
GO2_SCENE = "/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml"

DEF_WALKER = "logs/rsl_rl/go2_multitask_gallop_phase2/2026-09-26_02-43-47/model_2500.pt"
DEF_WALL = "logs/rsl_rl/go2_multitask_wallcross_thinfirst_fromwalklandall/2026-10-01_00-15-06/model_17995.pt"
DEF_PARAMS = "logs/rsl_rl/go2_multitask_wallcross_thinfirst_fromwalklandall/2026-10-01_00-15-06/params"

SDK_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
FEET = ("FR", "FL", "RR", "RL")          # Isaac の foot_body_names の順
TRUNK_BODIES = ("base_link", "FR_hip", "FL_hip", "RR_hip", "RL_hip")   # Isaac: base, .*_hip

# -- Isaac 側の定数 (env.yaml / commands.py から。値の出所はコメント) ---------------------------
STANCE_HIND = -1.0
JUMP_COMMAND_DURATION_S = 0.5     # JumpCommandCfg.command_duration_s (env.yaml)
TARGET_HEIGHT_OFFSET = -0.25      # WallCrossCommandCfg.target_height_offset
TARGET_HEIGHT_CLIP = (0.05, 1.0)
BASE_PAST_MARGIN = 0.2
MIN_FEET_IN_CONTACT = 3
MIN_SUCCESS_HEIGHT = 0.24
MAX_TILT_RAD = 0.26
SUCCESS_HOLD_STEPS = 15           # success_hold_s 0.3 / step_dt 0.02
CONTACT_THRESHOLD = 1.0           # N
ILLEGAL_MARGIN = 0.15             # illegal_contact_off_wall の margin / min_height
ILLEGAL_MIN_HEIGHT = 0.15
COLLAPSE_HEIGHT = 0.15            # terminations.collapsed
WALL_TIMEOUT_S = 8.0
DEPLOY_MIN_CUE_TO_HANDOVER_S = 0.8   # deploy_patch: min_cue_to_handover_s
HANDSTAND_SUCCESS_ALIGN = 0.93    # handstand success_alignment
ISAAC_PHYSICS_DT = 0.005          # actuator delay は物理ステップ単位 (min_delay 0, max_delay 6)
ISAAC_MAX_DELAY_STEPS = 6

# -- wall_heightmap (tasks/wallcross/mdp/observations.py) --------------------------------------
HM_X = np.arange(-0.20, 1.30 + 1e-6, 0.05)    # 31
HM_Y = np.arange(-0.20, 0.20 + 1e-6, 0.10)    # 5
HM_CELL_HALF = 0.025
HM_GX, HM_GY = [a.reshape(-1) for a in np.meshgrid(HM_X, HM_Y, indexing="ij")]   # x-major

STAGES = {
    "stand": "停止 (歩行ポリシー)",
    "walk": "四足歩行 0.6 m/s (歩行ポリシー)",
    "rise": "後足で立ち上がり・接近 (壁越えポリシー)",
    "jump": "ジャンプ合図 → 跳躍 (壁越えポリシー)",
    "land": "着地・立位 (壁越えポリシー)",
    "walk2": "歩行に戻る 0.6 m/s (歩行ポリシー)",
    "stop": "停止 (歩行ポリシー)",
}


def rel(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


# =============================================================================================
# policy
# =============================================================================================


class MLP:
    """rsl_rl ActorCritic の actor (Linear-ELU x3 - Linear) の決定論的推論 (mean)。"""

    def __init__(self, path):
        npz = to_npz(path)
        z = np.load(npz)
        self.W = [z[f"actor.{i}.weight"] for i in (0, 2, 4, 6)]
        self.b = [z[f"actor.{i}.bias"] for i in (0, 2, 4, 6)]
        self.in_dim = self.W[0].shape[1]
        self.path = path
        self.iter = int(z["iter"]) if "iter" in z else -1

    def __call__(self, x):
        h = x
        for i in range(3):
            h = self.W[i] @ h + self.b[i]
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
        return self.W[3] @ h + self.b[3]


class MoE:
    """Mixture of experts in the form of Tamura-san's MoEActorCritic (``assets/models/moe_actor.py``): every expert
    runs, a softmax gate weights them, the action is the weighted sum. Expert 0 = walking, expert 1 = wall crossing.
    Gate logits = MLP(obs) + prior_scale * prior, prior from the command flags (jump or hind-stance flag -> 1,
    otherwise -> 0), as the multitask gate does from its flags. The npz holds the gate MLP and the two experts."""

    def __init__(self, path):
        z = np.load(path)
        self.GW = [z[f"gate.{i}.weight"] for i in range(int(z["gate_layers"]))]
        self.Gb = [z[f"gate.{i}.bias"] for i in range(int(z["gate_layers"]))]
        self.experts = []
        for e in (0, 1):
            m = MLP.__new__(MLP)
            m.W = [z[f"expert{e}.{i}.weight"] for i in range(4)]
            m.b = [z[f"expert{e}.{i}.bias"] for i in range(4)]
            m.in_dim = m.W[0].shape[1]
            self.experts.append(m)
        self.prior_scale = float(z["prior_scale"])
        self.jump_col, self.hs_col = int(z["jump_col"]), int(z["hs_col"])
        self.mask = z["gate_mask"] if "gate_mask" in z.files else np.ones(124)

    def weights(self, x):
        h = x * self.mask
        for i in range(len(self.GW) - 1):
            h = self.GW[i] @ h + self.Gb[i]
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
        logit = self.GW[-1] @ h + self.Gb[-1]
        flag = max(x[self.jump_col], x[self.hs_col])
        logit = logit + self.prior_scale * np.array([1.0 - flag, flag])
        e = np.exp(logit - logit.max())
        return e / e.sum()

    def __call__(self, x, extra=None):
        """``x`` = the 124 unified columns; ``extra`` = the wall expert's extra input (the MID-360 map), if any."""
        w = self.weights(x)
        xw = x if self.experts[1].in_dim == x.shape[0] else np.concatenate([x, extra])
        return w[0] * self.experts[0](x) + w[1] * self.experts[1](xw), w


def to_npz(path):
    path = rel(path)
    if path.endswith(".npz"):
        return path
    os.makedirs(CACHE, exist_ok=True)
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.realpath(path)}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:16]
    out = os.path.join(CACHE, f"{os.path.basename(path)[:-3]}_{key}.npz")
    if not os.path.exists(out):
        # GPU を使わせない (学習中)。torch.load は map_location=cpu。
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        subprocess.run([TORCH_PY, os.path.join(HERE, "export_actor_npz.py"), path, out], check=True, env=env)
    return out


# =============================================================================================
# scene
# =============================================================================================


def build_model(wall_x, wall_h, wall_t, wall_w, timestep, wall_friction, foot_friction=None, spec_hook=None,
                wall_priority=0, wall_solref=None, wall_solimp=None, impratio=None):
    """scene_flat.xml (go2.xml を include) を読んで、壁の箱を足す。元の XML には触らない。"""
    spec = mujoco.MjSpec.from_file(GO2_SCENE)
    if spec_hook is not None:
        spec_hook(spec)
    if timestep:
        spec.option.timestep = timestep
    if wall_h > 0:
        g = spec.worldbody.add_geom()
        g.name = "wall"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [wall_t / 2, wall_w / 2, wall_h / 2]
        g.pos = [wall_x + wall_t / 2, 0.0, wall_h / 2]
        g.rgba = [0.85, 0.45, 0.15, 0.6]   # 半透明: 横から撮ると手前に伸びる壁がロボットを隠すため
        g.friction = [wall_friction, 0.005, 0.0001]
        g.condim = 3
        # 既定 (0) では足の球 (priority=1, friction 0.4) との接触は足側の 0.4 になる。Isaac は壁も足も 1.0
        # (multiply)。2 にすると壁側の摩擦 (wall_friction) が全部の接触に使われる (--isaac_wall)。
        g.priority = wall_priority
        if wall_solref is not None:
            g.solref = wall_solref
        if wall_solimp is not None:
            g.solimp = list(wall_solimp) + [0.5, 2.0]
    if impratio is not None:
        spec.option.impratio = impratio
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    m = spec.compile()
    if foot_friction is not None:
        # 足 (priority=1 の球) の滑り摩擦だけ差し替える。MJCF 既定は 0.4、Isaac は 1.0 (学習中は 0.3-1.2 をランダム化)
        for f in FEET:
            m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f), 0] = foot_friction
    return m


def isaac_calf_geom_hook(spec):
    """Isaac の go2.usd にだけある「すねの下端の太い円柱」(FL_calf/collisions/mesh_2: 半径 0.0155, 長さ 0.03,
    すね原点から (0.008, 0, -0.1875)、y 軸まわり 30.4 度) を go2.xml の 4 本のすねに足す (2026-10-11 診断)。"""
    for f in FEET:
        g = spec.body(f"{f}_calf").add_geom()
        g.name = f"{f}_calflower1"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [0.0155, 0.015, 0.0]
        g.pos = [0.008013, 0.0, -0.18745]
        g.quat = [0.96509, 0.0, 0.26191, 0.0]
        g.group = 3


def calf_capsule_hook(spec):
    """go2.xml のすねの衝突円柱 (各 2 本) をカプセル (同じ半径・同じ半長) に替える (2026-10-11 診断)。
    円柱–箱は汎用凸ソルバで 1 点しか接触を作らず、薄い壁の上端をまたぐとすねが壁の中に入り込んで挟まる。"""
    for f in FEET:
        for g in spec.body(f"{f}_calf").geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                g.type = mujoco.mjtGeom.mjGEOM_CAPSULE


def named(m, objtype, name):
    i = mujoco.mj_name2id(m, objtype, name)
    if i < 0:
        raise SystemExit(f"{name} not found in model")
    return i


# =============================================================================================
# observation helpers
# =============================================================================================


def heading_xy(q):
    """observations.heading_xy の写し: body x - body z を水平面に射影 (後足立ちでも定義される)。"""
    w, x, y, z = q
    bx = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x)])
    f = bx - bz
    return f / max(np.linalg.norm(f), 1e-6)


def wall_heightmap(base_xy, quat, wall_x, wall_t, wall_h, wall_w):
    """wall_heightmap の意味を直線の壁で再現。

    Isaac はリング状の壁に対して Chebyshev 距離で「セル (半幅 0.025) が壁の内面〜外面と重なるか」を
    見る。直線の壁 (x = wall_x 〜 wall_x+wall_t, |y| <= wall_w/2) では、セル中心の x に同じ ±0.025 を
    付けて重なりを見るのと同じ。値は重なれば壁の高さ、そうでなければ 0。"""
    f = heading_xy(quat)
    left = np.array([-f[1], f[0]])
    px = base_xy[0] + HM_GX * f[0] + HM_GY * left[0]
    py = base_xy[1] + HM_GX * f[1] + HM_GY * left[1]
    hit = (px + HM_CELL_HALF >= wall_x) & (px - HM_CELL_HALF <= wall_x + wall_t) & (np.abs(py) <= wall_w / 2)
    return hit.astype(np.float64) * wall_h


# --isaac_match: Isaac の UnitreeActuator (UNITREE_GO2_CORRECTED_CFG) の速度依存トルク制限 + 膝の速度上限。
# 既定の評価では使わない (go2.xml の ctrlrange だけ)。2026-10-04 の開ループ再生で、MuJoCo の踏切が Isaac より
# 強い (鉛直速度ピーク 1.5 倍) 主因が Isaac 側の PhysX 関節最大速度 (URDF の calf velocity=15.7 rad/s) だと
# 分かった。PhysX のその上限は MuJoCo では再現できない (qvel を後から切っても効かない) ので、膝のトルク–速度
# 曲線を「15.7 rad/s で 0」になるように折って近似する。開ループ再生で鉛直速度ピークの比 1.51 → 1.15。
ISAAC_Y1 = np.array([20.2] * 8 + [45.43 * 20.2 / 23.4] * 4)   # Isaac 順: hip x4, thigh x4, calf x4
ISAAC_Y2 = np.array([23.4] * 8 + [45.43] * 4)


ISAAC_VMAX = np.array([30.1] * 8 + [15.7] * 4)   # Isaac 順: hip x4, thigh x4, calf x4 (PhysX の関節最大速度)


def isaac_match_clip(tau, qd, a):
    x1 = np.array([13.5] * 8 + [a.knee_x1] * 4)
    x2 = np.array([30.0] * 8 + [a.knee_x2] * 4)
    mx = np.where(qd * tau > 0, ISAAC_Y1, ISAAC_Y2)
    mx = np.where(np.abs(qd) < x1, mx, np.clip(mx - mx / (x2 - x1) * (np.abs(qd) - x1), 0.0, None))
    return np.clip(tau, -mx, mx)


def jump_time_encoding(t):
    return t ** 3 / (1.0 + t ** 3)


# =============================================================================================
# simulation of one trial
# =============================================================================================


class Robot:
    def __init__(self, m, params):
        self.m = m
        dep = params
        jmap = dep["joint_ids_map"]
        self.names = [SDK_JOINTS[i] for i in jmap]                    # Isaac の関節順
        self.default_q = np.array(dep["default_joint_pos"], dtype=np.float64)
        self.kp = np.array(dep["stiffness"], dtype=np.float64)       # Isaac 順で書かれている
        self.kd = np.array(dep["damping"], dtype=np.float64)
        self.action_scale = np.array(dep["actions"]["JointPositionAction"]["scale"], dtype=np.float64)
        self.step_dt = float(dep["step_dt"])
        self.qadr = np.array([m.jnt_qposadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names])
        self.vadr = np.array([m.jnt_dofadr[named(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names])
        self.act = np.array([named(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in self.names])
        self.tau_lim = np.abs(m.actuator_ctrlrange[self.act]).max(axis=1)
        self.decimation = int(round(self.step_dt / m.opt.timestep))
        assert abs(self.decimation * m.opt.timestep - self.step_dt) < 1e-9, "step_dt must be a multiple of timestep"
        self.base = named(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.foot_geom = [named(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FEET]
        self.foot_body = [named(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_foot") for f in FEET]
        self.trunk_body = [named(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRUNK_BODIES]
        self.robot_bodies = set()
        for b in range(m.nbody):
            if b != 0 and m.body_rootid[b] == self.base:
                self.robot_bodies.add(b)


def contact_state(m, d, rb, wall_x, wall_t, wall_log=None):
    """Isaac の接触センサ相当 (環境 = 床・壁 との接触力 > 1 N)。

    戻り値: (各足が接地しているか[4], 胴体/hip の illegal contact があったか, 触れた部位名)"""
    feet = np.zeros(4, dtype=bool)
    illegal = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        r1, r2 = b1 in rb.robot_bodies, b2 in rb.robot_bodies
        if r1 == r2:            # self contact or world-world
            continue
        geom = c.geom1 if r1 else c.geom2
        body = b1 if r1 else b2
        mujoco.mj_contactForce(m, d, i, f6)
        if np.linalg.norm(f6[:3]) <= CONTACT_THRESHOLD:
            continue
        if wall_log is not None:
            other = c.geom2 if r1 else c.geom1
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) == "wall":
                # the foot sphere is a geom of the calf body in go2.xml: report it as the (empty) foot body,
                # as Isaac's contact sensor does
                bname = (f"{FEET[rb.foot_geom.index(geom)]}_foot" if geom in rb.foot_geom
                         else mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body))
                wall_log.append((bname, float(np.linalg.norm(f6[:3])), float(d.contact[i].pos[2])))
        if geom in rb.foot_geom:
            feet[rb.foot_geom.index(geom)] = True
        if body in rb.trunk_body:
            p = d.xpos[body]
            near_wall = (p[0] > wall_x - ILLEGAL_MARGIN) and (p[0] < wall_x + wall_t + ILLEGAL_MARGIN)
            raised = p[2] >= ILLEGAL_MIN_HEIGHT
            if not (near_wall and raised):
                illegal.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body))
    return feet, illegal


_FONT = {}


def font(size):
    if size not in _FONT:
        from PIL import ImageFont
        for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            if os.path.exists(path):
                _FONT[size] = ImageFont.truetype(path, size)
                break
        else:
            from PIL import ImageFont
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]


class WallDetector:
    """Automatic rise and jump cue from the MID-360 points (2026-10-14, the hardware wants the robot to go over a
    wall it walks up to without the j key). The policy's map only reaches 0.7 m ahead, too short to rise in time
    (rises happen 1.1-1.2 m before the wall), so this reads the raw returns: points higher than ``min_z`` above the
    floor, ahead along the body heading (``heading_xy``, valid in the hind stance) within ``max_x`` and ``half_w``
    of the centre line, over the last ``keep`` casts. The wall distance is the 5th percentile of their forward
    distance; the wall top is the running maximum of the points near that face (seen once the stance lifts the
    sensor)."""

    def __init__(self, min_z=0.15, max_x=3.0, half_w=0.3, keep=10, min_pts=8):
        self.min_z, self.max_x, self.half_w, self.min_pts = min_z, max_x, half_w, min_pts
        self.buf = collections.deque(maxlen=keep)
        self.top = 0.0

    def update(self, hits, base, quat):
        h = hits[np.isfinite(hits).all(axis=1)]
        self.buf.append(h[h[:, 2] > self.min_z])
        p = np.concatenate(self.buf) if self.buf else np.zeros((0, 3))
        f = heading_xy(quat)
        rel = p[:, :2] - base[:2]
        xf = rel @ f
        yl = rel @ np.array([-f[1], f[0]])
        m = (xf > 0.15) & (xf < self.max_x) & (np.abs(yl) < self.half_w)
        if m.sum() < self.min_pts:
            return float("inf")
        gap = float(np.percentile(xf[m], 5))
        near = m & (xf < gap + 0.2)
        self.top = max(self.top, float(p[near, 2].max()))
        return gap


def run_trial(m, d, rb, walker, wallp, a, seed, wall_x, renderer=None, frames=None, hm_ref=None, lidar=None):
    moe, gate_rec, moe_stats = a._moe, a._gate_rec, a._moe_stats
    rand_cmd = None
    cmd_rng = np.random.default_rng(seed + 777)
    rise_at, walk_until, approach_at = a.rise_at, a.walk_until, a.approach_at
    auto = WallDetector() if a.auto_trigger else None
    if auto is not None:
        # --auto_trigger: walk on until the MID-360 points show a wall ahead, then rise and cue by itself
        rise_at = walk_until = approach_at = float("inf")
    auto_gap = float("inf")
    rs = np.random.default_rng(seed)
    dt = rb.step_dt
    hm = a.heightmap
    if hm and wallp.in_dim != 279:
        raise SystemExit(f"--heightmap needs a 279-input wall policy, got {wallp.in_dim}")
    if lidar is not None and wallp.in_dim != 733:
        raise SystemExit(f"--mid360 needs a 733-input wall policy, got {wallp.in_dim}")
    if not hm and lidar is None and wallp.in_dim != 124:
        raise SystemExit(f"wall policy has {wallp.in_dim} inputs; use --heightmap for 279, --mid360 for 733")

    # ---- reset (Isaac: default root state z=0.4, default joints, zero velocity) ----
    mujoco.mj_resetData(m, d)
    yaw = np.radians(rs.uniform(-a.yaw_jitter_deg, a.yaw_jitter_deg))
    d.qpos[0] = rs.uniform(-a.pos_jitter, a.pos_jitter)
    d.qpos[1] = rs.uniform(-a.pos_jitter, a.pos_jitter)
    d.qpos[2] = 0.4
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[rb.qadr] = rb.default_q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    if lidar is not None:
        # 走査列の位相 (Isaac では全 env 共通の通し番号で、エピソード開始時の位相は任意) と雑音は
        # 別の乱数列から引く: 姿勢の揺らぎ・遅延は --heightmap と同じ seed で同じになる。
        lrs = np.random.default_rng([seed, 360])
        w0 = int(lrs.integers(0, 200)) if a.mid360_window0 < 0 else a.mid360_window0
        lidar.reset(w0, lrs)

    # actuator delay: Isaac の DelayedPDActuator は環境ごとに 0..6 物理ステップ (5 ms) を引く
    if a.delay_steps < 0:
        k = int(rs.integers(0, ISAAC_MAX_DELAY_STEPS + 1))
    else:
        k = a.delay_steps
    delay_sub = int(round(k * ISAAC_PHYSICS_DT / m.opt.timestep))
    target_q = rb.default_q.copy()
    pending = collections.deque()          # (apply_at_substep, target)
    substep = 0

    hist = {"q": None, "qd": None, "a": None}
    last_action = np.zeros(12)
    jump_on, jump_obs_i, jump_target = False, 0, 0.0
    hs_on = False
    triggered = False
    t_trigger = z_trigger = z_max = None
    success_hold, t_success = 0, None
    fell_t = fell_stage = fell_why = None
    rose = False
    max_align = -1.0
    airborne_steps = max_airborne = 0
    x_at_handover = None
    yaw_at_handover = None
    wall_hits = {}
    gap_at_rise = None
    hm_maxdiff = 0.0
    lmap = None
    lmap_hist = collections.deque(maxlen=51)     # (base z, map) of the last 51 steps, for the cue log
    trace_bids = sorted(rb.robot_bodies)
    trace_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in trace_bids]
    trace = {k: [] for k in ("pos", "quat", "linvel_w", "angvel_b", "jpos", "jvel", "act", "wall_f", "body_pos", "feet",
                             "tau_abs", "foot_f", "floor_f")}
    log = []
    end_t = None
    T = a.max_s
    n_steps = int(round(T / dt))
    stage = "stand"

    for k_step in range(n_steps):
        t = k_step * dt
        # ---- schedule ----
        after = (t - t_success) if t_success is not None else None
        if a.walker_only:
            use_wall = False
            cmd_x = a.walk_speed if t >= 0.5 else 0.0
            stage = "walk" if t >= 0.5 else "stand"
            if a.walk_random_cmd and t >= 0.5:
                # --walk_random_cmd: a new (vx, vy, wz) every 2 s, as the walking expert is commanded on the robot
                if k_step % int(round(2.0 / dt)) == 0 or rand_cmd is None:
                    rand_cmd = [cmd_rng.uniform(-0.5, 1.0), cmd_rng.uniform(-0.5, 0.5), cmd_rng.uniform(-1.0, 1.0)]
                cmd_x = rand_cmd[0]
        else:
            use_wall = t >= rise_at - 1e-9 and not (after is not None and after >= a.handover_s - 1e-9)
            if t < 0.5:
                cmd_x, stage = 0.0, "stand"
            elif t < walk_until - 1e-9:
                cmd_x, stage = a.walk_speed, "walk"
            elif t >= approach_at - 1e-9:
                cmd_x, stage = a.approach_speed, "rise"
            else:
                cmd_x, stage = 0.0, "stand"
            if not hs_on and not triggered and t >= rise_at - 1e-9:
                hs_on = True
                gap_at_rise = wall_x - d.qpos[0]
            if triggered:
                stage = "jump" if t_success is None and (t - t_trigger) < 1.0 else ("land" if t_success is None else "land")
            if after is not None and after >= a.handover_s - 1e-9:
                if after < a.handover_s + a.walk_after_s - 1e-9:
                    cmd_x, stage = a.walk_speed, "walk2"
                else:
                    cmd_x, stage = 0.0, "stop"
                if x_at_handover is None:
                    x_at_handover = d.qpos[:2].copy()
                    yaw_at_handover = float(np.degrees(np.arctan2(*heading_xy(d.qpos[3:7])[::-1])))
        if fell_t is not None:
            stage = "fell"

        # ---- observation ----
        q = d.qpos[rb.qadr] - rb.default_q
        qd = d.qvel[rb.vadr].copy()
        R = d.xmat[rb.base].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        angvel = d.qvel[3:6].copy()          # free joint の角速度は body 系 (= Isaac root_ang_vel_b)
        if jump_on:
            # 合図のステップと次のステップは 0 (trigger_step 未設定 → 立ち上がりエッジで 0 から)
            jt = jump_time_encoding(max(jump_obs_i - 1, 0) * dt)
            jcmd = [1.0, jump_target, 0.0, 0.0]
        else:
            jt, jcmd = 0.0, [0.0, 0.0, 0.0, 0.0]
        hcmd = [1.0, STANCE_HIND] if hs_on else [0.0, 0.0]
        for key, val in (("q", q), ("qd", qd * 0.05), ("a", last_action)):
            if hist[key] is None:
                hist[key] = collections.deque([val.copy()] * 3, maxlen=3)
            else:
                hist[key].append(val.copy())
        cmd_y, cmd_wz = (rand_cmd[1], rand_cmd[2]) if (a.walker_only and a.walk_random_cmd and rand_cmd is not None) else (0.0, 0.0)
        obs = np.concatenate([
            angvel * 0.2, grav, [cmd_x, cmd_y, cmd_wz], jcmd, [jt], hcmd,
            np.concatenate(hist["q"]), np.concatenate(hist["qd"]), np.concatenate(hist["a"]),
        ])
        assert obs.shape[0] == 124
        if lidar is not None:
            # 地図は歩行ポリシーの間も毎ステップ更新する (実機の地図生成と同じく常時動いている)
            lmap = lidar.step(d)
            lmap_hist.append((float(d.qpos[2]), lmap))
            if auto is not None:
                auto_gap = auto.update(lidar.last_hits, d.qpos[:3], d.qpos[3:7])
                if os.environ.get("AUTO_DEBUG") and k_step % 5 == 0 and hs_on:
                    print(f"        dbg t {t:5.2f} auto_gap {auto_gap:.2f} true {wall_x - d.qpos[0]:.2f} z {d.qpos[2]:.2f} hs {hs_on}")
                if not hs_on and not triggered and t >= 0.5 and auto_gap <= a.auto_rise_gap and rise_at == float("inf"):
                    # stop first, rise a.auto_stop_s later -- the schedule the policies were checked with
                    # (walk -> stop at ~1.2 m -> rise 0.7 s later); --auto_stop_s 0 rises straight from the walk
                    walk_until = t
                    rise_at = approach_at = t + a.auto_stop_s
                    log.append((t, f"auto rise (wall {auto_gap:.2f} m ahead by MID-360, true gap {wall_x - d.qpos[0]:.2f})"))
        if gate_rec is not None:
            gate_rec["obs"].append(obs.copy()); gate_rec["label"].append(1 if use_wall else 0)
            gate_rec["trial"].append(seed); gate_rec["t"].append(t)
        if moe is not None:
            # --moe: one policy for the whole sequence (walking expert + wall expert mixed by the learned gate);
            # the schedule still sets the commands, but no longer picks the policy.
            if lidar is not None and lmap is None:
                lmap = lidar.step(d)
            action, wgt = moe(obs, lmap)
            moe_stats["n"] += 1
            moe_stats["wrong"] += int((wgt[1] > 0.5) != use_wall)
            moe_stats["by_stage"].setdefault(stage, [0, 0])
            moe_stats["by_stage"][stage][0] += 1
            moe_stats["by_stage"][stage][1] += float(wgt[1])
        elif use_wall:
            if lidar is not None:
                # --mid360_blank: Isaac の Mid360Blank と同じ診断 (全セル flat_fill = 平らな床)
                action = wallp(np.concatenate([obs, np.full(609, 0.32 - 0.273175) if a.mid360_blank else lmap]))
            elif hm:
                hmap = wall_heightmap(d.qpos[:2], d.qpos[3:7], wall_x, a.wall_thickness, a.wall_height, a.wall_width)
                obs_w = np.concatenate([obs, hmap])
                action = wallp(obs_w)
                if hm_ref is not None:
                    hm_maxdiff = max(hm_maxdiff, float(np.abs(action - hm_ref(obs)).max()))
            else:
                action = wallp(obs)
        else:
            action = walker(obs)
        last_action = np.asarray(action)
        new_target = rb.default_q + action * rb.action_scale

        # ---- physics (decimation substeps, explicit PD as in Isaac's actuator) ----
        pending.append((substep + delay_sub, new_target))
        feet_any = np.zeros(4, dtype=bool)
        illegal = []
        if a.dump_npz:
            trace_wf = np.zeros(len(trace_names))
            trace_tau = np.zeros(len(rb.act))          # 制御 1 ステップ内の |トルク| の最大 (Isaac 関節順)
            trace_ff = np.zeros(4)                     # 足の球の床・壁からの力の最大 [N]
            trace_fl = np.zeros(len(trace_names))      # 足以外の部位の床 (壁以外) からの力の最大 [N]
        for _ in range(rb.decimation):
            while pending and pending[0][0] <= substep:
                target_q = pending.popleft()[1]
            tau = rb.kp * (target_q - d.qpos[rb.qadr]) - rb.kd * d.qvel[rb.vadr]
            if a.isaac_match:
                d.ctrl[rb.act] = isaac_match_clip(tau, d.qvel[rb.vadr], a)
            else:
                d.ctrl[rb.act] = np.clip(tau, -rb.tau_lim, rb.tau_lim)
            if a.vcap_damper > 0:
                # PhysX の関節最大速度 (go2.usd: hip/thigh 30.1, calf 15.7 rad/s) を「超えた分にだけ効く強いダンパ」で近似
                qd = d.qvel[rb.vadr]
                d.qfrc_applied[rb.vadr] = -a.vcap_damper * (qd - np.clip(qd, -ISAAC_VMAX, ISAAC_VMAX))
            mujoco.mj_step(m, d)
            substep += 1
            if a.dump_npz:
                trace_tau = np.maximum(trace_tau, np.abs(d.actuator_force[rb.act]))
                f6 = np.zeros(6)
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    if (b1 in rb.robot_bodies) == (b2 in rb.robot_bodies):
                        continue
                    geom, body, other = (c.geom1, b1, c.geom2) if b1 in rb.robot_bodies else (c.geom2, b2, c.geom1)
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fn = abs(f6[0])
                    if geom in rb.foot_geom:
                        k_ = rb.foot_geom.index(geom); trace_ff[k_] = max(trace_ff[k_], fn)
                    elif mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) != "wall":
                        bi = trace_bids.index(body); trace_fl[bi] = max(trace_fl[bi], fn)
            wl = [] if (a.wall_contacts or a.dump_npz) else None
            f, il = contact_state(m, d, rb, wall_x, a.wall_thickness, wl)
            if a.dump_npz and wl:
                for bname, fn, cz in wl:
                    bi = trace_names.index(bname)
                    trace_wf[bi] = max(trace_wf[bi], fn)
            if wl and triggered and t_success is None:
                for bname, fn, cz in wl:
                    st = wall_hits.setdefault(bname, [round(t + dt - t_trigger, 2), 0.0, cz])
                    st[1] = max(st[1], fn)
            feet_any |= f       # Isaac の success は最終状態の接触力だが、ここは最後の substep を使う
            illegal += il
        feet_now = f
        t_next = t + dt
        if a.dump_npz:
            trace["pos"].append(d.qpos[:3].copy())
            trace["quat"].append(d.qpos[3:7].copy())
            trace["linvel_w"].append(d.qvel[:3].copy())
            trace["angvel_b"].append(d.qvel[3:6].copy())
            trace["jpos"].append(d.qpos[rb.qadr].copy())
            trace["jvel"].append(d.qvel[rb.vadr].copy())
            trace["act"].append(action.copy())
            trace["wall_f"].append(trace_wf.copy())
            trace["body_pos"].append(np.array([d.xpos[b] for b in trace_bids]))
            trace["feet"].append(feet_now.copy())
            trace["tau_abs"].append(trace_tau.copy())
            trace["foot_f"].append(trace_ff.copy())
            trace["floor_f"].append(trace_fl.copy())

        # ---- bookkeeping after the step (= Isaac command/termination update) ----
        z = d.qpos[2]
        R = d.xmat[rb.base].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        gap = wall_x - d.qpos[0]
        if jump_on:
            jump_obs_i += 1
            # 合図の obs + 25 (= 0.5 s / 0.02) で切れる → 26 制御ステップ
            if (jump_obs_i - 1) * dt >= JUMP_COMMAND_DURATION_S - 1e-9:
                jump_on = False
        if hs_on and not triggered and use_wall:
            align = STANCE_HIND * grav[0]
            max_align = max(max_align, align)
            if align > HANDSTAND_SUCCESS_ALIGN and not feet_now[0] and not feet_now[1]:
                rose = True
        cue_gap = auto_gap if auto is not None else gap
        if (not a.walker_only and not triggered and hs_on and t_next >= rise_at + a.trigger_delay - 1e-9
                and cue_gap <= a.trigger_gap):
            triggered, t_trigger = True, t_next
            hs_on = False
            jump_on, jump_obs_i = True, 0
            wall_h_est = auto.top if auto is not None else a.wall_height
            jump_target = float(np.clip(wall_h_est + TARGET_HEIGHT_OFFSET, *TARGET_HEIGHT_CLIP))
            if auto is not None:
                log.append((t_next, f"auto cue (wall {auto_gap:.2f} m by MID-360, true {gap:.2f}; top {auto.top:.2f} m, true {a.wall_height:.2f})"))
            z_trigger = z_max = z
            log.append((t_next, f"jump cue (gap {gap:.2f} m, z {z:.2f} m, target {jump_target:.2f})"))
            if hm:
                hmap = wall_heightmap(d.qpos[:2], d.qpos[3:7], wall_x, a.wall_thickness, a.wall_height, a.wall_width)
                rows_hit = [f"{HM_X[i]:+.2f}" for i in range(len(HM_X)) if hmap.reshape(31, 5)[i].any()]
                log.append((t_next, f"heightmap at cue: wall in rows x={rows_hit} (cells hit {int((hmap > 0).sum())}/155)"))
            if lidar is not None and lmap_hist:
                # 合図の直前と 1 s 前に方策が見た地図: 中央 5 列 (|y| <= 0.1) の各前方行の「床からの高さ」最大値
                # (probe_wallcross_mid360.py と同じ集計)
                for lab, (bz, mp) in (("at cue", lmap_hist[-1]), ("1 s before", lmap_hist[0])):
                    above = (bz - mp - 0.273175).reshape(29, 21)[:, 8:13].max(axis=1)
                    log.append((t_next, f"mid360 map {lab:10s} height above floor x=0.00..0.70: "
                                + " ".join(f"{v:.2f}" for v in above[14:])))
        if triggered and t_success is None:
            z_max = max(z_max, z)
            if not feet_now.any():
                airborne_steps += 1
                max_airborne = max(max_airborne, airborne_steps)
            else:
                airborne_steps = 0
            outer = wall_x + a.wall_thickness
            feet_x = np.array([d.xpos[b][0] for b in rb.foot_body])
            if a.handover_rule == "deploy":
                # deploy_patch の WallCross FSM と同じ: 合図から 0.8 s 以降、傾き 15° 未満を 0.3 s 保てば「着地」。
                # 実機は壁の位置も足の接地も見ない。越えたかどうかは最後の位置で判定する (下の crossed)
                landed = t_next - t_trigger >= DEPLOY_MIN_CUE_TO_HANDOVER_S and tilt < MAX_TILT_RAD
            else:
                landed = ((d.qpos[0] - outer) > BASE_PAST_MARGIN and (feet_x > outer).all()
                          and feet_now.sum() >= MIN_FEET_IN_CONTACT and z > MIN_SUCCESS_HEIGHT and tilt < MAX_TILT_RAD)
            success_hold = success_hold + 1 if landed else 0
            if success_hold >= SUCCESS_HOLD_STEPS:
                t_success = t_next
                log.append((t_next, "success"))
        if fell_t is None and (illegal or z < COLLAPSE_HEIGHT):
            fell_t = t_next
            fell_stage = stage
            fell_why = ("+".join(sorted(set(illegal))) if illegal else "") + (" low" if z < COLLAPSE_HEIGHT else "")
            log.append((t_next, f"fell [{stage}] {fell_why.strip()}"))
        if frames is not None and (frames["k"] % frames["stride"] == 0):
            draw(renderer, frames, d, rb, a, seed, t_next, stage, wall_x, cmd_x, use_wall, jump_on, gap, triggered,
                 t_success)
        if a.trace and k_step % a.trace == 0:
            print(f"    t {t_next:5.2f} {stage:5s} x {d.qpos[0]:5.2f} z {z:.3f} gx {grav[0]:+.2f} tilt {np.degrees(tilt):5.1f} "
                  f"feet {''.join('1' if f else '0' for f in feet_now)} vx {(R.T @ d.qvel[:3])[0]:+.2f} "
                  f"yaw {np.degrees(np.arctan2(*heading_xy(d.qpos[3:7])[::-1])):+5.0f} |a| {np.abs(action).max():.2f} jt {jt:.3f} jc {jcmd[0]:.0f} hs {hcmd[0]:.0f}")
        if frames is not None:
            frames["k"] += 1
        # ---- end of trial ----
        if end_t is None:
            if fell_t is not None:
                end_t = fell_t + (1.5 if frames is not None else 0.0)
            elif a.walker_only and t_next >= a.walker_only:
                end_t = t_next
            elif t_success is not None and t_next - t_success >= a.handover_s + a.walk_after_s + a.stop_after_s:
                end_t = t_next
            elif triggered and t_success is None and t_next - t_trigger >= WALL_TIMEOUT_S:
                log.append((t_next, "wall timeout"))
                end_t = t_next
        if end_t is not None and t_next >= end_t - 1e-9:
            break

    walked = None
    if x_at_handover is not None and fell_t is None:
        walked = float(np.linalg.norm(d.qpos[:2] - x_at_handover))   # 向きは着地で変わるので水平変位の大きさ
    return {
        "seed": seed, "delay_ms": k * ISAAC_PHYSICS_DT * 1000, "gap_at_rise": gap_at_rise,
        "rose": rose, "max_align": max_align, "triggered": triggered, "t_trigger": t_trigger,
        "rise": (z_max - z_trigger) if triggered else None, "z_trigger": z_trigger, "z_max": z_max,
        "airborne_s": max_airborne * dt, "jumped": triggered and max_airborne > 0,
        "crossed": t_success is not None and (a.handover_rule != "deploy"
                                              or d.qpos[0] - (wall_x + a.wall_thickness) > BASE_PAST_MARGIN),
        "t_success": t_success,
        "fell_t": fell_t, "fell_stage": fell_stage, "fell_why": fell_why,
        "stood": t_success is not None and (fell_t is None or fell_t >= t_success + a.handover_s)
                 and (a.handover_rule != "deploy" or d.qpos[0] - (wall_x + a.wall_thickness) > BASE_PAST_MARGIN),
        "walked": walked, "walked_ok": walked is not None and walked > 0.5 * a.walk_speed * a.walk_after_s
                                       and (a.handover_rule != "deploy" or d.qpos[0] - (wall_x + a.wall_thickness) > BASE_PAST_MARGIN),
        "final_x": float(d.qpos[0]), "final_y": float(d.qpos[1]), "final_z": float(d.qpos[2]), "t_end": t_next,
        "hm_maxdiff": hm_maxdiff, "log": log, "wall_hits": wall_hits,
        "yaw_at_handover": yaw_at_handover, "final_yaw": float(np.degrees(np.arctan2(*heading_xy(d.qpos[3:7])[::-1]))),
        "trace": {k: np.array(v) for k, v in trace.items()} if a.dump_npz else None, "trace_names": trace_names,
    }


def draw(renderer, frames, d, rb, a, seed, t, stage, wall_x, cmd_x, use_wall, jump_on, gap, triggered, t_success):
    from PIL import Image, ImageDraw
    cam = renderer._wc_cam
    # 横から追従 (x だけ滑らかに追う)。azimuth 90: +x が画面の右 (要確認)。
    cam.lookat[0] += 0.3 * (d.qpos[0] + 0.3 - cam.lookat[0])
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.38
    renderer.update_scene(d, cam)
    img = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(img)
    label = "転倒" if stage == "fell" else STAGES.get(stage, stage)
    if a.moe:
        label = label.split(" (")[0]          # one policy: the stage, not which policy runs
    lines = [
        (f"{label}", 40),
        (f"t = {t:5.2f} s   速度指令 {cmd_x:.1f} m/s   壁まで {gap:+.2f} m" + ("   ジャンプ合図 ON" if jump_on else ""), 26),
        (f"MuJoCo sim2sim  試行 {seed}   壁 高さ {a.wall_height:.2f} m × 厚さ {a.wall_thickness * 100:.0f} cm"
         + ("   1本のポリシー（MoE：歩行＋壁越え）" if a.moe else f"   {'壁越え' if use_wall else '歩行'}ポリシー動作中") + ("   (HM)" if a.heightmap else "")
         + ("   (MID-360)" if a.mid360 else ""), 22),
    ]
    y = 18
    for text, size in lines:
        dr.text((24, y), text, fill=(255, 255, 255), font=font(size), stroke_width=3, stroke_fill=(0, 0, 0))
        y += int(size * 1.35)
    frames["writer"].append_data(np.asarray(img))
    frames["n"] += 1


# =============================================================================================
# main
# =============================================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walker", default=DEF_WALKER)
    ap.add_argument("--wall_policy", default=DEF_WALL)
    ap.add_argument("--params", default=DEF_PARAMS, help="run の params/ (deploy.yaml を読む)")
    ap.add_argument("--heightmap", action="store_true", help="壁越えポリシーに wall_heightmap 155 列を足す (279 入力)")
    ap.add_argument("--mid360", action="store_true",
                    help="壁越えポリシーに mid360_map 609 列を足す (733 入力)。地図は mid360_map.py で MuJoCo 内のレイから作る")
    ap.add_argument("--mid360_noise", action="store_true", help="--mid360: Isaac の MID360_NOISE_CFG (弱/中/強) を掛ける")
    ap.add_argument("--mid360_blank", action="store_true",
                    help="--mid360 の診断: 地図を全セル flat_fill (平らな床) にして方策に渡す (Isaac の *-Mid360-Blank と同じ)")
    ap.add_argument("--mid360_window0", type=int, default=-1,
                    help="--mid360: 試行開始時の走査窓 (0..199)。-1 = 試行ごとに乱数")
    ap.add_argument("--hm_selftest", default=None,
                    help="--heightmap 時、同じ観測の先頭 124 列をこの blind ckpt にも通し、行動の最大差を報告")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--wall_height", type=float, default=0.65)
    ap.add_argument("--wall_thickness", type=float, default=0.05)
    ap.add_argument("--wall_width", type=float, default=4.0, help="壁の横幅 (Isaac のリングの一辺は 2-4 m)")
    ap.add_argument("--wall_distance", type=float, default=2.4,
                    help="初期位置 (胴体原点) から壁の内面 (手前の面) までの距離 [m]")
    ap.add_argument("--wall_friction", type=float, default=1.0)
    ap.add_argument("--isaac_wall", action="store_true",
                    help="壁との接触の摩擦を Isaac と同じ 1.0 にする (既定では足–壁が go2.xml の足の 0.4 になる)")
    ap.add_argument("--wall_solref", type=float, nargs=2, default=None, help="壁 geom の solref (既定 0.02 1)")
    ap.add_argument("--wall_solimp", type=float, nargs=3, default=None, help="壁 geom の solimp (既定 0.9 0.95 0.001)")
    ap.add_argument("--impratio", type=float, default=None, help="摩擦の impratio (go2.xml は 100、全接触に効く)")
    ap.add_argument("--foot_friction", type=float, default=None,
                    help="足の摩擦係数を上書き (既定: go2.xml の 0.4 のまま。Isaac は 1.0、学習時 0.3-1.2)")
    ap.add_argument("--yaw_jitter_deg", type=float, default=5.0)
    ap.add_argument("--pos_jitter", type=float, default=0.05)
    ap.add_argument("--delay_steps", type=int, default=-1,
                    help="アクチュエータ遅延 (Isaac の 5 ms 物理ステップ単位)。-1 = 試行ごとに 0..6 を一様に引く (学習と同じ)")
    ap.add_argument("--isaac_match", action="store_true",
                    help="Isaac のトルク–速度制限 + 膝の速度上限 (下の knee_x1/x2) を足す。既定は off (go2.xml のまま)")
    ap.add_argument("--vcap_damper", type=float, default=0.0,
                    help=">0: 関節速度が PhysX の上限 (hip/thigh 30.1, calf 15.7 rad/s) を超えた分にこの係数 [N m s/rad] の"
                         "ダンパを掛ける (Isaac の関節速度上限の近似、2026-10-11 診断で 20 を使用)")
    ap.add_argument("--knee_x1", type=float, default=7.0, help="--isaac_match: 膝トルクが落ち始める速度 [rad/s]")
    ap.add_argument("--knee_x2", type=float, default=15.7, help="--isaac_match: 膝トルクが 0 になる速度 [rad/s] (URDF の velocity)")
    ap.add_argument("--timestep", type=float, default=0.0, help="MuJoCo の物理刻み (0 = モデル既定 0.002)")
    # schedule (film_wallcross_sequence.py --walk_until 3.2 に合わせる)
    ap.add_argument("--walk_speed", type=float, default=0.6)
    ap.add_argument("--approach_speed", type=float, default=0.6)
    ap.add_argument("--walk_until", type=float, default=3.2)
    ap.add_argument("--rise_at", type=float, default=3.2)
    ap.add_argument("--approach_at", type=float, default=3.2)
    ap.add_argument("--trigger_delay", type=float, default=1.0, help="立ち上がりからこの時間はジャンプ合図を出さない")
    ap.add_argument("--trigger_gap", type=float, default=0.5, help="胴体〜壁内面の距離がこれ以下で合図")
    ap.add_argument("--handover_s", type=float, default=1.0)
    ap.add_argument("--walk_after_s", type=float, default=3.0)
    ap.add_argument("--stop_after_s", type=float, default=1.0)
    ap.add_argument("--max_s", type=float, default=25.0)
    ap.add_argument("--walker_only", type=float, default=0.0, help="歩行ポリシーだけで N 秒歩かせる (壁なし)")
    ap.add_argument("--walk_random_cmd", action="store_true", help="--walker_only: 2 s ごとに (vx, vy, wz) をランダムに変える")
    ap.add_argument("--record_gate", default=None, help="毎制御ステップの観測 124 列と、そのとき動いていたポリシー (1=壁越え) を npz に保存")
    ap.add_argument("--auto_trigger", action="store_true",
                    help="--mid360: 歩き続けて、MID-360 の点で壁を見つけたら自動で立ち上がり (auto_rise_gap)、合図も自動 (trigger_gap、壁の高さも点から)")
    ap.add_argument("--auto_rise_gap", type=float, default=1.15, help="--auto_trigger: 壁までこの距離で止まる (立ち上がるのは auto_stop_s 後)")
    ap.add_argument("--auto_stop_s", type=float, default=0.0, help="--auto_trigger: 止まってから立ち上がるまで [s]")
    ap.add_argument("--moe", default=None, help="MoE (歩行＋壁越えの専門家＋ゲート) の npz。時刻で切り替える代わりに 1 本で通す")
    ap.add_argument("--video", default=None)
    ap.add_argument("--video_trials", default="0", help="動画に入れる試行 (カンマ区切り, 'success' = 最初の成功試行)")
    ap.add_argument("--video_fps", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--wall_contacts", action="store_true", help="跳躍後〜成功までに壁に触れた部位・時刻・最大力を表示")
    ap.add_argument("--trace", type=int, default=0, help="N 制御ステップごとに状態を表示")
    ap.add_argument("--handover_rule", choices=["isaac", "deploy"], default="isaac",
                    help="歩行へ切り替える条件。isaac = 学習の成功判定 (壁の先・3 足接地・高さ > 0.24 m・傾き < 15° を 0.3 s)、"
                         "deploy = 実機の WallCross FSM (合図から 0.8 s 以降、傾き < 15° を 0.3 s。越えたかは最後の位置で判定)")
    ap.add_argument("--dump_npz", default=None,
                    help="全試行の毎ステップの状態 (胴体・関節・行動・各部位の壁との接触力の最大値) を npz に保存")
    a = ap.parse_args()
    if a.mid360 and a.heightmap:
        raise SystemExit("--mid360 and --heightmap are exclusive")

    params = yaml.safe_load(open(os.path.join(rel(a.params), "deploy.yaml")))
    wdir = os.path.join(os.path.dirname(rel(a.walker)), "params", "deploy.yaml")
    if os.path.exists(wdir):
        wp = yaml.safe_load(open(wdir))
        for key in ("joint_ids_map", "step_dt", "stiffness", "damping", "default_joint_pos", "actions", "observations"):
            if wp[key] != params[key]:
                print(f"[warn] walker deploy.yaml differs from wall params in {key!r}")

    walker = MLP(a.walker)
    wallp = MLP(a.wall_policy)
    a._moe = MoE(a.moe) if a.moe else None
    a._gate_rec = {"obs": [], "label": [], "trial": [], "t": []} if a.record_gate else None
    a._moe_stats = {"n": 0, "wrong": 0, "by_stage": {}}
    hm_ref = MLP(a.hm_selftest) if a.hm_selftest else None
    print(f"[policy] walker {a.walker} (iter {walker.iter}, in {walker.in_dim})")
    print(f"[policy] wall   {a.wall_policy} (iter {wallp.iter}, in {wallp.in_dim})" + ("  +heightmap" if a.heightmap else "")
          + (f"  +mid360{' (noise)' if a.mid360_noise else ''}" if a.mid360 else ""))
    assert walker.in_dim == 124

    wall_x = a.wall_distance          # robot starts near x=0 facing +x
    m = model = build_model(wall_x, 0.0 if a.walker_only else a.wall_height, a.wall_thickness, a.wall_width,
                            a.timestep, a.wall_friction, a.foot_friction,
                            wall_priority=2 if a.isaac_wall else 0, wall_solref=a.wall_solref,
                            wall_solimp=a.wall_solimp, impratio=a.impratio)
    d = mujoco.MjData(m)
    rb = Robot(m, params)
    print(f"[mujoco] timestep {m.opt.timestep} decimation {rb.decimation} | joints (Isaac order) {rb.names}")
    print(f"[mujoco] kp {rb.kp[0]} kd {rb.kd[0]} action_scale {rb.action_scale[0]} tau_lim {rb.tau_lim.tolist()}")
    lidar = None
    if a.mid360:
        sys.path.insert(0, HERE)
        from mid360_map import Mid360Map
        lidar = Mid360Map(m, rb.base, noise=a.mid360_noise)

    vids = set()
    want_success_video = False
    if a.video:
        for tok in a.video_trials.split(","):
            tok = tok.strip()
            if tok == "success":
                want_success_video = True
            elif tok:
                vids.add(int(tok))

    rows = []
    for i in range(a.trials):
        seed = a.seed0 + i
        r = run_trial(m, d, rb, walker, wallp, a, seed, wall_x, hm_ref=hm_ref, lidar=lidar)
        rows.append(r)
        if a.walker_only:
            print(f"  trial {seed:2d}: x {r['final_x']:.2f} y {r['final_y']:+.2f} z {r['final_z']:.3f} "
                  f"fell {r['fell_t']} {r['fell_why'] or ''}  delay {r['delay_ms']:.0f}ms", flush=True)
            continue
        fell = f"fell {r['fell_t']:.2f}s [{r['fell_stage']}] {r['fell_why']}" if r["fell_t"] is not None else "no fall"
        print(f"  trial {seed:2d}: delay {r['delay_ms']:2.0f}ms gap@rise {r['gap_at_rise'] if r['gap_at_rise'] is None else round(r['gap_at_rise'], 2)} "
              f"rose={'o' if r['rose'] else 'x'}(align {r['max_align']:.2f}) "
              f"cue={'o' if r['triggered'] else 'x'} jumped={'o' if r['jumped'] else 'x'}"
              f"(air {r['airborne_s']:.2f}s, rise {r['rise'] if r['rise'] is None else round(r['rise'], 2)}, "
              f"zmax {r['z_max'] if r['z_max'] is None else round(r['z_max'], 2)}) "
              f"crossed={'o' if r['crossed'] else 'x'} stood={'o' if r['stood'] else 'x'} "
              f"walked={'o' if r['walked_ok'] else 'x'}({r['walked'] if r['walked'] is None else round(r['walked'], 2)} m) | {fell}"
              + (f" | hm_maxdiff {r['hm_maxdiff']:.2e}" if hm_ref is not None else ""), flush=True)
        if a.auto_trigger:
            for tt, msg in r["log"]:
                if msg.startswith("auto"):
                    print(f"      {tt:5.2f} {msg}")
        if a.wall_contacts:
            print("      wall contacts (body: t since cue, peak N, contact z):",
                  {k: (v[0], round(v[1]), round(v[2], 2)) for k, v in sorted(r["wall_hits"].items(), key=lambda kv: kv[1][0])})
        if a.verbose:
            for t, e in r["log"]:
                print(f"      {t:6.2f}s {e}")

    if a.dump_npz:
        L = max(len(r["trace"]["pos"]) for r in rows)
        out = {"seed": np.array([r["seed"] for r in rows]), "body_names": np.array(rows[0]["trace_names"]),
               "dt": np.array(rb.step_dt), "wall_x": np.array(wall_x), "wall_h": np.array(a.wall_height),
               "wall_t": np.array(a.wall_thickness), "joint_names": np.array(rb.names),
               "length": np.array([len(r["trace"]["pos"]) for r in rows])}
        for key in ("triggered", "crossed", "stood", "jumped"):
            out[key] = np.array([bool(r[key]) for r in rows])
        for key in ("t_trigger", "t_success", "fell_t", "delay_ms"):
            out[key] = np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=np.float64)
        for key in rows[0]["trace"]:
            arrs = [r["trace"][key] for r in rows]
            pad = np.full((len(rows), L) + arrs[0].shape[1:], np.nan)
            for i, x in enumerate(arrs):
                pad[i, :len(x)] = x
            out[key] = pad
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_npz)), exist_ok=True)
        np.savez_compressed(a.dump_npz, **out)
        print(f"wrote {a.dump_npz}")

    if a.video:
        # 決定的 (同じ seed で同じ軌道) なので、撮る試行だけ描画しながらもう一度回す。
        if want_success_video:
            ok = [r["seed"] for r in rows if r["crossed"] and r["stood"]] or [r["seed"] for r in rows if r["crossed"]]
            if ok:
                vids.add(ok[0])
            else:
                print("[video] no successful trial; 'success' ignored")
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.video)), exist_ok=True)
        stride = max(1, int(round(1.0 / (a.video_fps * rb.step_dt))))
        writer = imageio.get_writer(a.video, fps=1.0 / (rb.step_dt * stride), codec="libx264", quality=8,
                                    macro_block_size=8)
        renderer = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.azimuth, cam.elevation = 4.2, 90.0, -10.0
        renderer._wc_cam = cam
        rec = {"writer": writer, "stride": stride, "n": 0, "k": 0}
        for seed in sorted(vids):
            cam.lookat[:] = [0.3, 0.0, 0.38]
            r2 = run_trial(m, d, rb, walker, wallp, a, seed, wall_x, renderer, rec, hm_ref, lidar)
            ref = [r for r in rows if r["seed"] == seed]
            if ref and (ref[0]["crossed"], ref[0]["fell_t"]) != (r2["crossed"], r2["fell_t"]):
                print(f"[video] warning: re-run of trial {seed} differs from the first run")
        writer.close()
        print(f"wrote {a.video} (trials {sorted(vids)}, {rec['n']} frames)")

    n = len(rows)
    if a.walker_only:
        fell = sum(r["fell_t"] is not None for r in rows)
        dist = np.array([r["final_x"] for r in rows])
        T = a.walker_only - 0.5
        print(f"\n=== walker only: {n} trials, fell {fell}, mean speed {dist.mean() / T:.3f} m/s "
              f"(cmd {a.walk_speed}), lateral drift |y| mean {np.mean([abs(r['final_y']) for r in rows]):.2f} m ===")
    else:
        c = lambda k: sum(bool(r[k]) for r in rows)  # noqa: E731
        print(f"\n=== {n} trials{' | MID-360' + (' noise' if a.mid360_noise else '') if a.mid360 else ''} | wall h {a.wall_height} m, t {a.wall_thickness} m, distance {a.wall_distance} m | "
              f"delay {'0-30ms random' if a.delay_steps < 0 else f'{a.delay_steps * 5}ms'} ===")
        print(f"rose onto hind legs {c('rose')}/{n} | jump cue {c('triggered')}/{n} | jumped (4 feet off) {c('jumped')}/{n} | "
              f"crossed (Isaac success) {c('crossed')}/{n} | stood until handover {c('stood')}/{n} | "
              f"walked after handover {c('walked_ok')}/{n}")
        stages = collections.Counter(r["fell_stage"] for r in rows if r["fell_t"] is not None)
        print(f"falls by stage: {dict(stages)}")
        rises = [r["rise"] for r in rows if r["rise"] is not None]
        if rises:
            print(f"base rise after cue: p10/50/90 {np.percentile(rises, [10, 50, 90]).round(3).tolist()} "
                  f"(needed for apex 0.15 above wall: {a.wall_height + 0.15 - np.mean([r['z_trigger'] for r in rows if r['triggered']]):.3f})")
        if hm_ref is not None:
            print(f"hm selftest: max |action(widened, HM obs) - action(blind, 124 obs)| = {max(r['hm_maxdiff'] for r in rows):.3e}")
    if a._gate_rec is not None:
        g = a._gate_rec
        os.makedirs(os.path.dirname(os.path.abspath(a.record_gate)), exist_ok=True)
        np.savez_compressed(a.record_gate, obs=np.array(g["obs"], dtype=np.float32), label=np.array(g["label"], dtype=np.int8),
                            trial=np.array(g["trial"]), t=np.array(g["t"], dtype=np.float32))
        print(f"wrote {a.record_gate} ({len(g['label'])} steps, wall share {np.mean(g['label']):.3f})")
    if a._moe is not None:
        st = a._moe_stats
        print(f"moe: steps {st['n']}, gate disagrees with the schedule on {st['wrong'] / max(st['n'], 1):.4f} of them | "
              "mean wall weight by stage: " + ", ".join(f"{k} {v[1] / v[0]:.3f}" for k, v in sorted(st["by_stage"].items())))


if __name__ == "__main__":
    main()
