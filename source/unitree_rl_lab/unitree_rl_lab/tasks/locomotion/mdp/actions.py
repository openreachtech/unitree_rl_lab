"""Action terms for this project.

2026-09-16: 行動遅延つきの関節位置アクション。

なぜ要るか。このリポジトリの学習環境には**行動遅延・観測遅延のモデルが1つも無く**、
トルク・質量・摩擦だけを DR していた。python の mujoco 評価も同期ループなので、
学習も評価も「観測した同じ制御周期に指令が効く」という同じ理想化を共有していた。

実機と、このリポジトリの mujoco C++/DDS 経路は、構造的に最低1ステップ（20 ms）遅れる。
遅延を1ステップ入れて測ると、走り幅跳びポリシーは着地 48% -> 5%、飛距離 1.338 -> 0.359 m、
飛距離最良個体に至っては跳躍成立 90% -> 0% になった（2026-09-16、docs/go2_overnight_20260916.md）。
実際に C++ 経路で跳ばせると5回とも転倒する。**遅延に対する耐性がゼロだった。**
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """関節位置アクションを env ステップ単位で遅らせる（環境ごとにランダムな遅延）。

    遅延はエピソードごとに `delay_steps_range` から引き直す。1 ステップ = env の
    step_dt（このタスクでは 0.02 s）で、sim.dt を変えても decimation 側で保たれる。

    policy が観測する `last_action` は**遅らせない**。実機でも policy は自分が最後に
    出力した指令を知っているので、遅れるのは「指令が関節に届くまで」だけ。
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        lo, hi = cfg.delay_steps_range
        if not (0 <= lo <= hi):
            raise ValueError(f"delay_steps_range が不正: {cfg.delay_steps_range}")
        self._delay_lo, self._delay_hi = int(lo), int(hi)
        # hist[k] = k ステップ前に計算された指令
        self._hist = torch.zeros(self._delay_hi + 1, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._applied = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.reset()
        print(
            f"[longjump] action delay = U[{self._delay_lo}, {self._delay_hi}] step "
            f"({self._delay_lo * 20}-{self._delay_hi * 20} ms 相当)",
            flush=True,
        )

    def _default_target(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        """遅延バッファの初期値。既定関節姿勢（= offset）を入れておく。"""
        if isinstance(self._offset, torch.Tensor):
            return self._offset[env_ids]
        return torch.full((self.num_envs, self.action_dim), float(self._offset), device=self.device)[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._hist[:, ids, :] = self._default_target(ids)
        self._applied[ids] = self._default_target(ids)
        n = self.num_envs if env_ids is None else len(env_ids)
        self._delay[ids] = torch.randint(
            self._delay_lo, self._delay_hi + 1, (n,), device=self.device, dtype=torch.long
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._hist = torch.roll(self._hist, shifts=1, dims=0)
        self._hist[0] = self._processed_actions
        idx = self._delay.view(1, -1, 1).expand(1, -1, self.action_dim)
        self._applied = torch.gather(self._hist, 0, idx).squeeze(0)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._applied, joint_ids=self._joint_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """`DelayedJointPositionAction` の設定。

    `delay_steps_range = (0, 0)` なら遅延なし＝上流の `JointPositionAction` と同じ挙動。
    """

    class_type: type[ActionTerm] = DelayedJointPositionAction

    delay_steps_range: tuple[int, int] = (0, 2)
    """指令が関節に届くまでの遅延 [env ステップ]。エピソードごとに一様分布から引く。"""
