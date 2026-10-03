"""モデル由来の粗い基準軌道（reference trajectory）生成器。

実測モーションを使わず、身体パラメータ・指令・歩容から
「重心と足をどう運べば物理的に成立しやすいか」だけを計算する。
関節軌道 q_ref は生成しない（そこは RL に任せる）。

このモジュールは純粋関数だけを置く。位相などの状態は
mdp/commands/gait_reference.py の GaitReferenceCommand が持つ。
将来ここを重心動力学の軌道最適化に差し替える場合も、
関数の入出力を保てば報酬側は無変更で済む。

用語（デューティ比は「1周期 = 2歩」を基準にした標準的な定義）:
    delta > 0.5 : 両脚支持あり = 歩行
    delta = 0.5 : 境界
    delta < 0.5 : 滞空あり   = 走行
"""

from __future__ import annotations

import math

import torch

GRAVITY = 9.81

# --- ステップ長モデル -------------------------------------------------
# 無次元ステップ長  L / l = A + B * sqrt(Fr),   Fr = v^2 / (g l)
# 人間の歩行(1.3 m/s)・走行(2.35 m/s)の実測から当てた1次近似。
STEP_LEN_RATIO_A = 0.55
STEP_LEN_RATIO_B = 0.57
STEP_LEN_RATIO_MIN = 0.50
STEP_LEN_RATIO_MAX = 1.30

STEP_PERIOD_MIN = 0.25  # s  1歩あたりの下限
STEP_PERIOD_MAX = 0.60  # s  1歩あたりの上限
MIN_SPEED = 0.15  # m/s  これ以下は周期を上限に張り付かせる

# --- デューティ比（1周期に対する片足の接地割合）-----------------------
# GaitID: 0 STAND / 1 WALK / 2 WALK_TO_STAND / 3 RUN / 4 RUN_TO_WALK
DUTY_BY_GAIT = (1.00, 0.60, 0.70, 0.35, 0.50)
RAIBERT_GAIN = 0.10
STANCE_WIDTH = 0.20  # m  左右足の基準間隔（骨盤幅相当）

# --- 地形問い合わせのパラメータ ---
# 足裏の寸法（G1 の足は概ね 0.20 x 0.10 m）。平坦判定はこの矩形で行う。
# 「足裏全体が同じ面に乗るか」が段鼻を踏むかどうかの判定そのものなので、
# 円盤で半径を決めるより物理的に意味がはっきりする。
# 実寸 (0.10, 0.05) に、スキャン格子の半セル 0.025 m を足して少し保守側に取る。
# レイは点サンプルなので、段鼻の位置は格子解像度の半分までしか分からない。
# この余裕を入れても、着地点は最大 0.025 m 程度は段鼻側へずれうる。
FOOT_HALF_LENGTH = 0.125      # m  (実寸 0.10 + 格子半セル 0.025)
FOOT_HALF_WIDTH = 0.075       # m  (実寸 0.05 + 格子半セル 0.025)

# 遊脚のクリアランス判定に使う前方の見込み範囲
CLEARANCE_BACK = 0.05         # m  足より後ろ
CLEARANCE_FRONT = 0.12        # m  足より前（これから当たりうる蹴上）
CLEARANCE_HALF_WIDTH = 0.08   # m

FOOTHOLD_SEARCH_SPAN = 0.12   # m  踏面スナップの前後探索幅
FOOTHOLD_NUM_CAND = 5         # 候補点数（前後方向）
FOOTHOLD_FORWARD_BIAS = 0.15  # 同点のとき前方の踏面を選ぶための弱いバイアス
FOOTHOLD_SIGMA_NEAR = 0.10    # m  名目位置への近さの許容
FOOTHOLD_SIGMA_FLAT = 0.03    # m  平坦とみなす局所高さばらつき


def step_period(speed: torch.Tensor, leg_length: float) -> torch.Tensor:
    """指令速度 [m/s] -> 1歩あたりのステップ周期 T_step [s]。

    leg_length=0.6 のとき、おおよそ
        v=1.0 -> 0.47 s / 0.47 m,  v=2.0 -> 0.31 s / 0.61 m,
        v=2.5 -> 0.27 s / 0.68 m
    となる。固定 period=0.8 s（=ステップ 0.4 s）より明確に速い。
    """
    fr_sqrt = speed / math.sqrt(GRAVITY * leg_length)
    ratio = torch.clamp(
        STEP_LEN_RATIO_A + STEP_LEN_RATIO_B * fr_sqrt,
        STEP_LEN_RATIO_MIN,
        STEP_LEN_RATIO_MAX,
    )
    period = ratio * leg_length / torch.clamp(speed, min=MIN_SPEED)
    return torch.clamp(period, STEP_PERIOD_MIN, STEP_PERIOD_MAX)


def duty_factor(gait_id: torch.Tensor) -> torch.Tensor:
    """歩容 ID -> デューティ比（周期基準）。"""
    table = torch.tensor(DUTY_BY_GAIT, device=gait_id.device, dtype=torch.float)
    return table[gait_id.long().clamp(0, len(DUTY_BY_GAIT) - 1)]


def contact_schedule(phase: torch.Tensor, duty: torch.Tensor) -> torch.Tensor:
    """期待される接地状態 (N, 2) = [左, 右]。

    phase は [0, 1) で 1周期 = 2歩。左足は [0, delta)、右足は 0.5 ずらし。
    delta < 0.5 なら「両足とも 0」の区間（滞空）が自動的に現れ、
    delta > 0.5 なら両脚支持が現れる。feet_gait の threshold 固定と違い、
    歩容ごとに接地パターンそのものが変わる。
    """
    ph = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=-1)
    return (ph < duty.unsqueeze(-1)).float()


def flight_fraction(duty: torch.Tensor) -> torch.Tensor:
    """1回の滞空区間の長さ（位相単位）。delta >= 0.5 なら 0。"""
    return torch.clamp(0.5 - duty, min=0.0)


def flight_time(duty: torch.Tensor, cycle_period: torch.Tensor) -> torch.Tensor:
    """1回あたりの滞空時間 [s]。"""
    return flight_fraction(duty) * cycle_period


def takeoff_vz(duty: torch.Tensor, cycle_period: torch.Tensor) -> torch.Tensor:
    """滞空が成立するために離地時点で必要な重心鉛直速度 v_z = g t_f / 2 [m/s]。"""
    return 0.5 * GRAVITY * flight_time(duty, cycle_period)


def flight_progress(phase: torch.Tensor, duty: torch.Tensor) -> torch.Tensor:
    """現在の滞空区間内での進捗 [0, 1]。滞空していないときの値は使わない。"""
    seg = torch.where(phase < 0.5, phase - duty, phase - 0.5 - duty)
    length = torch.clamp(flight_fraction(duty), min=1e-6)
    return torch.clamp(seg / length, 0.0, 1.0)


def com_vz_reference(
    phase: torch.Tensor, duty: torch.Tensor, cycle_period: torch.Tensor
) -> torch.Tensor:
    """滞空中の重心鉛直速度の基準 [m/s]。

    接触力ゼロなら r_ddot = g なので、滞空中の重心鉛直速度は
        vz(s) = vz0 * (1 - 2s),  s = 滞空区間内の進捗
    と閉形式で決まる（近似ではなく厳密）。立脚中は拘束しない。
    """
    return takeoff_vz(duty, cycle_period) * (1.0 - 2.0 * flight_progress(phase, duty))


def foot_placement(
    base_lin_vel_b: torch.Tensor,
    cmd_vel_b: torch.Tensor,
    duty: torch.Tensor,
    cycle_period: torch.Tensor,
    lateral_sign: torch.Tensor,   
    stance_width: float = STANCE_WIDTH,
    gain: float = RAIBERT_GAIN,
) -> torch.Tensor:
    """左右それぞれの着地点 (N, 2, 2) を body frame（root 原点）で返す。Raibert 則。

    第1項 = 支持期に進む距離の半分、第2項 = 速度誤差の補正。
    左右には stance_width/2 の横オフセットを乗せる（[左 +y, 右 -y]）。
    """
    stance_time = (duty * cycle_period).unsqueeze(-1)
    xy = 0.5 * stance_time * cmd_vel_b + gain * (base_lin_vel_b - cmd_vel_b)  # (N, 2)
    out = xy.unsqueeze(1).repeat(1, 2, 1)  # (N, 2feet, 2)
    lateral = torch.tensor([0.5 * stance_width, -0.5 * stance_width], device=xy.device)
    out[..., 1] = out[..., 1] + lateral.unsqueeze(0)
    return out


def phase_mask(contact_ref: torch.Tensor, gait_id: torch.Tensor) -> torch.Tensor:
    """重心・角運動量の基準を適用する区間 (N,)。

    走行 (RUN / R2W) は滞空期、歩行・階段は単脚支持期。停止中は 0。
    階段には飛翔期がないので、in_flight だけで絞ると基準が常にゼロになる。
    """
    n = contact_ref.sum(dim=-1)
    flight = (n < 0.5).float()
    single = ((n > 0.5) & (n < 1.5)).float()
    is_run = ((gait_id == 3) | (gait_id == 4)).float()
    moving = (gait_id != 0).float()
    return moving * (is_run * flight + (1.0 - is_run) * single)


# ----------------------------------------------------------------------
# 地形問い合わせ（height_scanner の ray_hits_w はワールド座標なので、
# グリッドの並び順に依存せず xy 距離で引ける）
# ----------------------------------------------------------------------


def scan_box(
    hits_w: torch.Tensor,
    query_xy: torch.Tensor,
    forward_xy: torch.Tensor,
    x_back: float,
    x_front: float,
    half_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """進行方向に沿った矩形領域での地形高さ。

    hits_w (N, R, 3) / query_xy (N, Q, 2) / forward_xy (N, 2) ->
        z_near  最近傍レイの高さ          (N, Q)
        z_max   矩形内の最大高さ          (N, Q)
        z_range 矩形内の高低差            (N, Q)

    ray_hits_w はワールド座標なので、グリッドの並び順に依存しない。
    矩形は進行方向に沿って [-x_back, +x_front] x [-half_width, +half_width]。
    """
    hx = hits_w[:, None, :, 0]
    hy = hits_w[:, None, :, 1]
    hz = hits_w[:, None, :, 2]
    qx = query_xy[:, :, None, 0]
    qy = query_xy[:, :, None, 1]
    fx = forward_xy[:, None, None, 0]
    fy = forward_xy[:, None, None, 1]

    ex = hx - qx
    ey = hy - qy
    dx = ex * fx + ey * fy
    dy = -ex * fy + ey * fx
    del ex, ey

    d2 = dx * dx + dy * dy
    z = hz.expand_as(d2)
    valid = torch.isfinite(z) & torch.isfinite(d2)

    far = torch.full_like(d2, torch.finfo(d2.dtype).max)
    d2m = torch.where(valid, d2, far)
    idx = d2m.argmin(dim=-1, keepdim=True)
    z_near = torch.gather(z, -1, idx).squeeze(-1)
    z_near = torch.where(torch.isfinite(z_near), z_near, torch.zeros_like(z_near))

    inside = valid & (dx >= -x_back) & (dx <= x_front) & (dy.abs() <= half_width)
    lo = torch.full_like(z, torch.finfo(z.dtype).min)
    hi = torch.full_like(z, torch.finfo(z.dtype).max)
    z_max = torch.where(inside, z, lo).amax(dim=-1)
    z_min = torch.where(inside, z, hi).amin(dim=-1)

    empty = ~inside.any(dim=-1)
    z_max = torch.where(empty, z_near, z_max)
    z_min = torch.where(empty, z_near, z_min)
    return z_near, z_max, z_max - z_min


def foot_footprint_query(
    hits_w: torch.Tensor, query_xy: torch.Tensor, forward_xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """足裏の矩形での地形問い合わせ。z_range が大きい = 段鼻をまたいでいる。"""
    return scan_box(
        hits_w, query_xy, forward_xy, FOOT_HALF_LENGTH, FOOT_HALF_LENGTH, FOOT_HALF_WIDTH
    )


def clearance_query(
    hits_w: torch.Tensor, query_xy: torch.Tensor, forward_xy: torch.Tensor
) -> torch.Tensor:
    """遊脚が越えるべき地形高さ (N, Q)。足のやや前方まで含めた最大。"""
    _, z_max, _ = scan_box(
        hits_w, query_xy, forward_xy, CLEARANCE_BACK, CLEARANCE_FRONT, CLEARANCE_HALF_WIDTH
    )
    return z_max


def snap_foothold(
    hits_w: torch.Tensor,
    nominal_xy: torch.Tensor,
    forward_xy: torch.Tensor,
    span: float = FOOTHOLD_SEARCH_SPAN,
    num_cand: int = FOOTHOLD_NUM_CAND,
    sigma_near: float = FOOTHOLD_SIGMA_NEAR,
    sigma_flat: float = FOOTHOLD_SIGMA_FLAT,
    forward_bias: float = FOOTHOLD_FORWARD_BIAS,
) -> torch.Tensor:
    """Raibert の名目着地点を、足裏が乗る平坦面（踏面）へ寄せた (N, F, 3) を返す。

    名目位置の前後 ±span に候補を並べ、

        score = exp(-offset^2 / sigma_near^2) * exp(-z_range^2 / sigma_flat^2)
                * (1 + forward_bias * offset / span)

    が最大の候補を選ぶ。z_range は「その位置に足裏を置いたときに足裏矩形が
    またぐ高低差」なので、段鼻に掛かる候補は score が落ちる。
    Dai らの「接触を点ではなくパッチで扱う」定式化の簡易版にあたる。

    重み付き平均ではなく argmax にしてあるのは、段差をまたぐと候補の高さが
    二峰性になり、平均すると蹴上の中腹という実在しない高さを指すため。
    forward_bias は手前と奥の踏面が同点のとき奥（進行方向）を選ばせる。
    上りでは上段、下りでは下段になり、どちらも「次に乗る段」になる。

    候補はループで回す。まとめて (N, F, M, R) にすると一時テンソルが
    数百 MB になり、メッシュ地形で VRAM を圧迫するため。
    """
    offsets = torch.linspace(-span, span, num_cand, device=nominal_xy.device)

    best_score = None
    best_xy = nominal_xy
    best_z = torch.zeros_like(nominal_xy[..., 0])

    for m in range(num_cand):
        off = offsets[m]
        cand = nominal_xy + off * forward_xy[:, None, :]
        z_near, _, z_range = foot_footprint_query(hits_w, cand, forward_xy)
        score = (
            math.exp(-float(off) ** 2 / sigma_near**2)
            * torch.exp(-(z_range**2) / sigma_flat**2)
            * (1.0 + forward_bias * float(off) / span)
        )
        if best_score is None:
            best_score, best_xy, best_z = score, cand, z_near
        else:
            take = score > best_score
            best_score = torch.where(take, score, best_score)
            best_xy = torch.where(take.unsqueeze(-1), cand, best_xy)
            best_z = torch.where(take, z_near, best_z)

    return torch.cat([best_xy, best_z.unsqueeze(-1)], dim=-1)


def terrain_slope(
    hits_w: torch.Tensor,
    root_xy: torch.Tensor,
    forward_xy: torch.Tensor,
    span: float = 0.60,
    num_probe: int = 7,
) -> torch.Tensor:
    """進行方向の局所勾配 dz/dx (N,)。階段では 蹴上/踏面 に相当する。

    前方 span まで num_probe 点を置き、最小二乗で直線を当てる。
    2 点差分だと踏面 0.30 m に対してプローブ距離が半端なときに
    「何段ぶん乗ったか」で量子化され、0.45 m 前方の 1 点では
    0.15/0.45 = 0.33 となって真の 0.50 から外れる。
    span を踏面の整数倍に近い 0.60 m 取って回帰すると量子化が均される。
    """
    d = torch.linspace(0.0, span, num_probe, device=root_xy.device)
    q = root_xy[:, None, :] + d.view(1, -1, 1) * forward_xy[:, None, :]
    z, _, _ = scan_box(hits_w, q, forward_xy, 0.06, 0.06, 0.06)

    dm = d - d.mean()
    zm = z - z.mean(dim=-1, keepdim=True)
    return (
        (dm.view(1, -1) * zm).sum(dim=-1) / (dm * dm).sum().clamp(min=1e-6)
    ).clamp(-1.0, 1.0)


# ----------------------------------------------------------------------
# 重心・角運動量のユーティリティ（報酬側とコマンド側で共用）
# ----------------------------------------------------------------------


def cached_body_masses(env, asset) -> torch.Tensor:
    """リンク質量 (N, B, 1) をキャッシュ付きで返す。

    既存の arm_leg_momentum_penalty と同じ env._body_masses を共有する。
    """
    body_pos = asset.data.body_pos_w
    expected = (body_pos.shape[0], body_pos.shape[1], 1)
    masses = getattr(env, "_body_masses", None)
    if masses is None or tuple(masses.shape) != expected:
        raw = asset.root_physx_view.get_masses()
        if raw.ndim == 1:
            raw = raw.unsqueeze(0).expand(body_pos.shape[0], -1)
        masses = raw.unsqueeze(-1).to(body_pos.device)
        env._body_masses = masses
    return masses


def com_position_velocity(
    body_pos_w: torch.Tensor, body_lin_vel_w: torch.Tensor, masses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """全身重心の位置と速度 (N, 3) を返す。"""
    total = masses.sum(dim=1)
    return (body_pos_w * masses).sum(dim=1) / total, (body_lin_vel_w * masses).sum(dim=1) / total


def angular_momentum_com(
    body_pos_w: torch.Tensor, body_lin_vel_w: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    """重心まわりの角運動量 (N, 3) [kg m^2/s]。

    移送項 sum m_i (c_i - r) x (v_i - r_dot) のみ。各リンクの自転項
    I_i omega_i は四肢では移送項より小さいので初期実装では省く。
    既存の arm_leg_momentum_penalty と違い、重心速度を引いているので
    前進運動由来のバイアスが乗らない。
    """
    com, com_vel = com_position_velocity(body_pos_w, body_lin_vel_w, masses)
    r = body_pos_w - com.unsqueeze(1)
    v = body_lin_vel_w - com_vel.unsqueeze(1)
    return torch.cross(r, masses * v, dim=-1).sum(dim=1)