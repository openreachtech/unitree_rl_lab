"""モデル生成の粗い基準軌道を保持する CommandTerm。

役割は「位相を持つこと」と「毎ステップ reference.py を呼ぶこと」だけ。
基準軌道の中身は mdp/reference.py 側にあるので、将来そこを
重心動力学の軌道最適化に差し替えても、このクラスと報酬側は変わらない。

位相は phi += dt / T(v) と積分する。T が速度依存なので
phi = t / T では計算できない（feet_gait はこの形だったため周期固定だった）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from .. import reference as ref

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitReferenceCommand(CommandTerm):
    """歩容と指令速度から、粗い基準軌道を生成して保持する。

    公開する量（報酬側はこれを読む）:
        phase          (N,)    1周期 = 2歩 の位相 [0, 1)
        duty           (N,)    デューティ比（周期基準）
        cycle_period   (N,)    1周期の時間 [s]
        contact_ref    (N, 2)  期待接地 [左, 右] の 0/1
        in_flight      (N,)    基準上、両足が浮いているべきなら 1
        phase_mask     (N,)    重心・角運動量の基準を課す区間
                               走行 = 滞空期 / 歩行・階段 = 単脚支持期
        com_vz_ref     (N,)    重心鉛直速度の基準 [m/s]
                               滞空中 = 弾道、立脚中 = 地形勾配 x 指令速度
        h_ref          (N, 3)  角運動量の基準（滞空中 = 離地時の値 / 立脚中 = 0）
        h_tol          (N,)    角運動量の許容幅（立脚中は腕振りを潰さないための不感帯）
        foot_ref_w     (N, 2, 3) 地形へスナップした着地点（ワールド座標）
        forward_xy     (N, 2)  進行方向の単位ベクトル（ヨーのみ、ワールド）
        terrain_slope  (N,)    進行方向の局所勾配 dz/dx
    """

    cfg: GaitReferenceCommandCfg

    def __init__(self, cfg: GaitReferenceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        num_envs, device = self.num_envs, self.device

        # 接触センサ内の [左足, 右足] インデックスをここで解決しておく
        self.foot_sensor_cfg = cfg.foot_sensor_cfg.copy()
        self.foot_sensor_cfg.resolve(env.scene)
        self._foot_ids = self.foot_sensor_cfg.body_ids

        self.foot_asset_cfg = cfg.foot_asset_cfg.copy()
        self.foot_asset_cfg.resolve(env.scene)
        self._foot_body_ids = self.foot_asset_cfg.body_ids
        asset_init: Articulation = env.scene[cfg.asset_name]
        foot_names = [asset_init.body_names[i] for i in self._foot_body_ids]
        self._lateral_sign = torch.tensor(
            [1.0 if "left" in n.lower() else -1.0 for n in foot_names], device=device
        )
        print(f"[GaitReferenceCommand] foot order = {foot_names} -> "
              f"lateral sign = {self._lateral_sign.tolist()}")

        self.phase = torch.zeros(num_envs, device=device)
        self.duty = torch.full((num_envs,), ref.DUTY_BY_GAIT[1], device=device)
        self.cycle_period = torch.full((num_envs,), 2.0 * ref.STEP_PERIOD_MAX, device=device)
        self.contact_ref = torch.ones(num_envs, 2, device=device)
        self.in_flight = torch.zeros(num_envs, device=device)
        self.phase_mask = torch.zeros(num_envs, device=device)
        self.com_vz_ref = torch.zeros(num_envs, device=device)
        self.h_ref = torch.zeros(num_envs, 3, device=device)
        self._h_takeoff = torch.zeros(num_envs, 3, device=device)
        self.h_tol = torch.zeros(num_envs, device=device)
        self.foot_ref_w = torch.zeros(num_envs, 2, 3, device=device)
        self.terrain_slope = torch.zeros(num_envs, device=device)
        self.forward_xy = torch.zeros(num_envs, 2, device=device)
        self.forward_xy[:, 0] = 1.0
        self.com_vz_mask = torch.zeros(num_envs, device=device)   # ← 追加


        # height_scanner が無い構成でも落ちないようにしておく（平地なら z=0 扱い）
        self._scanner = env.scene.sensors.get(cfg.terrain_sensor_name, None) \
            if cfg.terrain_sensor_name else None

        self.metrics["cycle_period"] = torch.zeros(num_envs, device=device)
        self.metrics["flight_fraction"] = torch.zeros(num_envs, device=device)
        self.metrics["terrain_slope"] = torch.zeros(num_envs, device=device)
        self.metrics["contact_match"] = torch.zeros(num_envs, device=device)
        self.metrics["angmom_norm"] = torch.zeros(num_envs, device=device)   # ← 追加
        self.metrics["angmom_yaw"] = torch.zeros(num_envs, device=device)
        self.metrics["angmom_pitch"] = torch.zeros(num_envs, device=device)


    def __str__(self) -> str:
        return f"GaitReferenceCommand: leg_length={self.cfg.leg_length}, duty={ref.DUTY_BY_GAIT}"

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """(N, 4) = [cos(2 pi phi), sin(2 pi phi), duty, in_flight]。

        観測に入れる場合に備えて露出しているだけで、既定では使わない
        （ブラインド条件を保つため）。
        """
        angle = 2.0 * torch.pi * self.phase
        return torch.stack([torch.cos(angle), torch.sin(angle), self.duty, self.in_flight], dim=-1)

    """
    Implementation
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # エピソード開始時の位相をばらけさせる（全 env が同位相で始まらないように）
        self.phase[env_ids] = torch.rand(self.phase[env_ids].shape, device=self.device)

    def _update_command(self):
        env = self._env
        asset: Articulation = env.scene[self.cfg.asset_name]

        vel_cmd = env.command_manager.get_command(self.cfg.velocity_command_name)
        speed = torch.norm(vel_cmd[:, :2], dim=-1)

        # gait_env.py が保持する現在の歩容 ID。存在しない場合は WALK 扱い。
        gait_id = getattr(env, "current_gait_id", None)
        if gait_id is None:
            gait_id = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        self.duty = ref.duty_factor(gait_id)
        self.cycle_period = 2.0 * ref.step_period(speed, self.cfg.leg_length)

        # --- 位相を積分（STAND では進めない）---
        moving = (gait_id != 0).float()
        self.phase = torch.remainder(
            self.phase + moving * env.step_dt / self.cycle_period, 1.0
        )

        prev_in_flight = self.in_flight
        self.contact_ref = ref.contact_schedule(self.phase, self.duty)
        self.in_flight = (self.contact_ref.sum(dim=-1) == 0).float()
        self.phase_mask = ref.phase_mask(self.contact_ref, gait_id)

        # --- 進行方向（ヨーのみ）---
        q = asset.data.root_quat_w  # (w, x, y, z)
        siny = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
        cosy = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        yaw = torch.atan2(siny, cosy)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        forward_xy = torch.stack([cy, sy], dim=-1)
        self.forward_xy = forward_xy

        # --- 着地点: Raibert 名目 (body) -> ワールド -> 地形の踏面へスナップ ---
        foot_ref_b = ref.foot_placement(
            asset.data.root_lin_vel_b[:, :2],
            vel_cmd[:, :2],
            self.duty,
            self.cycle_period,
            self._lateral_sign,
            self.cfg.stance_width,
        )  # (N, 2, 2)
        nom_x = cy[:, None] * foot_ref_b[..., 0] - sy[:, None] * foot_ref_b[..., 1]
        nom_y = sy[:, None] * foot_ref_b[..., 0] + cy[:, None] * foot_ref_b[..., 1]
        nominal_w = torch.stack([nom_x, nom_y], dim=-1) + asset.data.root_pos_w[:, None, :2]

        hits = self._terrain_hits()
        if hits is None:
            # 平地フォールバック: 地形 z = 0、勾配 0
            self.foot_ref_w = torch.cat(
                [nominal_w, torch.zeros_like(nominal_w[..., :1])], dim=-1
            )
            self.terrain_slope = torch.zeros_like(speed)
        else:
            self.foot_ref_w = ref.snap_foothold(hits, nominal_w, forward_xy)
            self.terrain_slope = ref.terrain_slope(
                hits, asset.data.root_pos_w[:, :2], forward_xy
            )

        # --- 重心鉛直速度の基準 ---
        # 滞空中: 接触力ゼロなので弾道軌道が厳密に決まる
        # 立脚中: 指令水平速度 x 地形勾配（階段では 蹴上/踏面 に相当）
        vz_flight = ref.com_vz_reference(self.phase, self.duty, self.cycle_period)
        vz_stance = speed * self.terrain_slope
        self.com_vz_ref = torch.where(self.in_flight > 0.5, vz_flight, vz_stance)
         # 立脚中の基準は勾配が立っているときだけ課す。平地では基準が 0 になるが、
        # 単脚支持中の重心上下動は倒立振子として自然に起きるものなので、
        # 0 を要求すると膝を曲げたまま歩く不自然な歩容へ誘導してしまう。
        has_slope = (self.terrain_slope.abs() > 0.05).float()
        self.com_vz_mask = torch.where(
            self.in_flight > 0.5, self.phase_mask, self.phase_mask * has_slope
        )

        # --- 角運動量の基準 ---
        # 滞空中は外力がないので h は変えられない（離地時の値を保存すべき）。
        # 立脚中は接触力で h を変えられるので、値を追わせず「許容幅」で縛る。
        # 0 に張り付かせると腕振りまで潰すため、不感帯を置く。
        took_off = (prev_in_flight == 0) & (self.in_flight == 1)
        if bool(took_off.any()):
            masses = ref.cached_body_masses(env, asset)
            h_now = ref.angular_momentum_com(
                asset.data.body_pos_w, asset.data.body_lin_vel_w, masses
            )
            self._h_takeoff = torch.where(took_off.unsqueeze(-1), h_now, self._h_takeoff)

        flight = self.in_flight.unsqueeze(-1)
        self.h_ref = flight * self._h_takeoff
        self.h_tol = (1.0 - self.in_flight) * self.cfg.angmom_tolerance

    def _terrain_hits(self) -> torch.Tensor | None:
        """地形スキャンのレイ命中点 (N, R, 3) ワールド座標。無ければ None。"""
        if self._scanner is None:
            return None
        return self._scanner.data.ray_hits_w

    def _update_metrics(self):
        max_steps = self._env.max_episode_length
        self.metrics["cycle_period"] += self.cycle_period / max_steps
        self.metrics["flight_fraction"] += self.in_flight / max_steps
        self.metrics["terrain_slope"] += self.terrain_slope.abs() / max_steps

        asset: Articulation = self._env.scene[self.cfg.asset_name]
        masses = ref.cached_body_masses(self._env, asset)
        h = ref.angular_momentum_com(
            asset.data.body_pos_w, asset.data.body_lin_vel_w, masses
        )
        self.metrics["angmom_norm"] += torch.norm(h, dim=-1) / max_steps
        self.metrics["angmom_yaw"] += h[:, 2].abs() / max_steps
        self.metrics["angmom_pitch"] += h[:, 1].abs() / max_steps

        sensor: ContactSensor = self._env.scene.sensors[self.foot_sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, self._foot_ids, :]
        actual = (torch.norm(forces, dim=-1).amax(dim=1) > 1.0).float()
        match = (actual == self.contact_ref).float().mean(dim=-1)
        self.metrics["contact_match"] += match / max_steps

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_foot_marker"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/gait_reference",
                    markers={
                        "stance": sim_utils.SphereCfg(
                            radius=0.045,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.85, 0.2)
                            ),
                        ),
                        "flight": sim_utils.SphereCfg(
                            radius=0.045,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.9, 0.2, 0.15)
                            ),
                        ),
                        "target": sim_utils.SphereCfg(
                            radius=0.035,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.15, 0.4, 0.95)
                            ),
                        ),
                    },
                )
                self._foot_marker = VisualizationMarkers(marker_cfg)
            self._foot_marker.set_visibility(True)
        elif hasattr(self, "_foot_marker"):
            self._foot_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """基準を可視化する。

        足元の球   緑 = 基準上いま接地しているべき / 赤 = 浮いているべき
        青い球     地形へスナップした基準着地点 foot_ref_w

        走行で両足が赤になる区間が出れば滞空が基準に入っている。
        階段では、青い球が段鼻ではなく踏面の上に乗っているかを見る。
        """
        if not hasattr(self, "_foot_marker"):
            return
        asset: Articulation = self._env.scene[self.cfg.asset_name]
        pos = asset.data.body_pos_w[:, self._foot_body_ids, :].clone()
         # 左右の横オフセットの符号を、解決後のリンク名から決める。
        # SceneEntityCfg は名前リストの順ではなく USD のボディ順で解決するので、
        # 「0 番が左足」を仮定すると左右が入れ替わり、基準が脚を交差させる。
      
        idx = (self.contact_ref < 0.5).long()  # 0 = stance(緑), 1 = flight(赤)

        target = self.foot_ref_w.clone()
        target[..., 2] += 0.02
        target_idx = torch.full(target.shape[:2], 2, dtype=torch.long, device=self.device)

        self._foot_marker.visualize(
            translations=torch.cat([pos, target], dim=1).reshape(-1, 3),
            marker_indices=torch.cat([idx, target_idx], dim=1).reshape(-1),
        )


@configclass
class GaitReferenceCommandCfg(CommandTermCfg):
    """GaitReferenceCommand の設定。"""

    class_type: type = GaitReferenceCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """位相はエピソード内で再サンプルしない（リセット時のみ）。"""

    asset_name: str = "robot"
    velocity_command_name: str = "base_velocity"
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=[".*_ankle_roll_link"]
    )
    """[左足, 右足] の順に解決される接触センサ設定。順序は feet_gait の
    offset=[0.0, 0.5] と同じく、名前のソート順（left, right）に一致する。"""

    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=[".*_ankle_roll_link"]
    )
    """debug_vis で球を置くリンク。"""

    leg_length: float = 0.6
    """股関節から足裏までの長さ [m]。ステップ長・周期の無次元化に使う。"""

    stance_width: float = 0.20
    """左右足の基準間隔 [m]。Raibert の着地点に横オフセットとして乗せる。"""

    terrain_sensor_name: str | None = "foothold_scanner"
    """地形高さを引く RayCaster センサー名。

    シーンに無ければ平地（z=0, 勾配0）として動く。観測には入れないので、
    ポリシーはブラインドのまま、報酬だけが地形を知る構成になる。
    """

    angmom_tolerance: float = 3.0
    """立脚中の角運動量の許容幅 [kg m^2/s]。

    立脚中は接触力で h を変えられるので値を追わせない。0 に張り付かせると
    腕振りまで潰すため、この幅を超えた分だけを罰する不感帯にする。
    """