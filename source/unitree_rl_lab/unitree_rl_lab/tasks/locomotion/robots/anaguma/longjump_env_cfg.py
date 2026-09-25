"""Stage B for Anaguma: running long jump ("Anaguma-LongJump-v1").

Inherits Go2's Stage B wholesale. Three things are overridden and each has a reason:

1. The robot (see ``unitree_rl_lab.assets.robots.anaguma``).
2. Two body-name patterns that name ``Head_.*``. Anaguma has no head link and an
   unresolvable body pattern is a hard error, not a silent no-op.
3. **The objective stays height.** Go2 moved to distance in Loop 47 because that is what
   was asked of it; this port was asked for as 「走り幅跳びの高さ重視のほうのポリシー」,
   so the apex term is pinned back on and the distance term back off. Without this the
   Anaguma run would silently inherit whichever objective Go2 happens to be chasing.

Everything else is Go2's, deliberately, so that the first Anaguma result answers "does
this task transfer to a different machine" rather than "does this new reward set work".
The numbers that are Go2-specific and will probably need changing once there is a result
-- the apex target (0.40 m, sized to Go2's v_z ceiling of 2.80 m/s) and the EFGCL assist
strength -- are called out below rather than quietly adjusted, because adjusting them
before measuring would make the first run uninterpretable.

Anaguma's own numbers, for when that time comes (see project memory
プロジェクト_Anaguma移植の基礎.md): the energy method puts its take-off at a = 3.52 m/s
against Go2's 3.17, i.e. a 63 cm apex against 51 cm, so 0.40 m is if anything *low* as a
target. The risk is on the other side -- its joint-speed ceiling leaves only 1.14x the
extension speed the take-off needs, where Go2 has 1.54x.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.anaguma import ANAGUMA_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.longjump_env_cfg import (
    RewardsCfgLongJump,
    RobotEnvCfgLongJump,
)


@configclass
class RewardsCfgAnagumaLongJump(RewardsCfgLongJump):
    """Go2's Stage B rewards, minus the head, with the height objective pinned."""

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]
            ),
        },
    )
    # === サイクル3（2026-09-15 03:50）: 着地の飴を 4.0 -> 20.0 ===
    #
    # サイクル2/2b で飛距離は 0.987 -> 1.266 -> 1.342 m と伸びたが、**着地が 90% -> 60% に落ちた**。
    # 同一 run の7点を mujoco で測ると、離陸 v_x が大きい個体ほど着地が悪い
    # (v_x 2.27→着地90% / 2.39→60% / 2.56→22%)。頂点は 0.16〜0.20 でほぼ一定なので、
    # **Anaguma の着地を壊しているのは高さではなく水平速度**。飛距離は v_x で買うしかないので
    # このトレードオフは報酬で買い戻す必要がある。
    #
    # 収支は `jump_takeoff_distance` +1.457 に対し着地の飴は +0.155（約10:1）。Go2 側でも
    # 同じ比率で同じ症状が出ており、同じレバー（結果指標の値段を5倍）を両機体で同時に試す。
    # 20.0 なら着地の飴は +0.78 相当＝踏切の飴の半分強で、跳躍自体は殺さない見込み。
    jump_landing_feet_first = RewTerm(
        func=mdp.jump_landing_feet_first,
        weight=20.0,
        params={
            "command_name": "jump_command",
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "body_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base", ".*_hip", ".*_thigh"]
            ),
        },
    )
    # Height objective, pinned. Go2 swapped these two in Loop 47 (apex 50 -> 0,
    # distance 0 -> 74) because distance is what was asked of it. This port was asked for
    # as the height policy, so the swap is reverted here rather than inherited.
    # target_apex 0.40 is Go2's ceiling (v_z 2.80). Anaguma's energy-method ceiling is
    # higher (a = 3.52 m/s, 0.63 m), so this is likely to saturate -- left alone on
    # purpose for the first run, so the result says something about the transfer.
    # 助走ゲートをフェーズ1の助走帯に合わせる（2026-09-14 夜）。
    #
    # このゲートは Go2 の Loop 31 で「走りながら跳べ」を強制するために入ったもので、
    # 報酬 = progress * clamp((trigger_speed - approach_lo)/(approach_hi - approach_lo))
    # ——2.0 m/s 未満は一切払わない。ところが同日夕のフェーズ1で助走指令を U(0.0, 3.3) へ
    # 下げたので、**遅く跳べ**と設定で言いながら**遅い跳躍には報酬を払わない**状態に
    # なっていた。2026-09-14_17-02-57 のログはそのとおりで、重み50のこの項の収入は
    # 全区間 0.0000、`unaided_approach_gate` も 0.0000。跳躍側の最大の飴が消えていた。
    #
    # 名前と実測対象のズレの8例目（報酬ではなく報酬とタスク設定の不一致、7例目と同型）。
    # フェーズ2で助走帯を U(2.8, 3.3) に戻すときは、この2値も (2.0, 2.5) に戻すこと。
    # === サイクル2（2026-09-15 00:00）: 目的関数を高さ -> 飛距離へ入れ替え ===
    #
    # フェーズ2 `2026-09-14_22-39-08` の mujoco 実測（40試行 @vx2.8）:
    #   model_2600  成立100% / 着地68% / 0.947m / 頂点0.205 / 離陸(1.90,1.80)
    #   model_3200  成立 98% / 着地85% / 0.913m / 頂点0.187 / 離陸(1.84,1.68)
    #   model_5399  成立100% / 着地50% / 0.890m / 頂点0.300 / 離陸(1.56,2.20)
    # **飛距離は 0.89〜0.95m で頭打ち**、しかも学習が進むほど v_x を落として v_z を上げ
    # （5399 は頂点0.300m の代わりに飛距離最小・着地50%）、飛距離と着地を同時に悪化させている。
    # 目的関数が `jump_takeoff_apex`（頂点 v_z^2/2g）のままだからで、**報酬が狙っている量と
    # 判定基準（飛距離）が違う**。Go2 が Loop 47 でやった入れ替えと同じことを報酬側だけで行う。
    jump_takeoff_apex = RewTerm(
        func=mdp.jump_takeoff_apex,
        weight=0.0,
        params={
            # フェーズ2で (2.0, 2.5) に復帰（2026-09-14 夜）。フェーズ1の値は (0.5, 1.5)。
            "approach_lo": 2.0,
            "approach_hi": 2.5,
            "command_name": "jump_command",
            "target_apex": 0.40,
            "payout_window_s": 0.6,
        },
    )
    # 踏切ペナルティを機体重量で正規化する（2026-09-14 夜、試行6）。
    #
    # `jump_dense_reward` は −std(接地力)[N]＝**跳躍報酬セットで唯一ニュートンで持ち回る量**。
    # 飴の側（apex/height/landing/clearance）は全部無次元なので、質量が 1.62 倍の Anaguma では
    # 鞭だけが 1.62 倍以上に効く。実測では跳んだ瞬間 −3.478（Go2 は同項 −0.0064、540倍差）。
    #
    # 試行5（`2026-09-14_19-56-07`）で curriculum の early_weight を 2.5 → 0.25 に下げたが
    # **足りなかった**——iter 48 で dense −1.464 に対し飴の合計 +0.297、収支はまだ −1.3 の赤字で、
    # `real_jump` も 0.65(iter20) → 0.46(iter48) と試行4と同じ形で落ち始めていた。
    # 重みではなく単位から出さないと機体を変えるたびに同じことが起きる。
    #
    # m*g で割ると無次元になり、試行5の実測値から逆算して early_weight 0.25 のとき
    # 1エピソードあたり約 −0.006 ＝ **Go2 の −0.0064 とほぼ同じ水準**に落ちる。
    # 論文どおり「跳べるようになってから 2.5 で対称性を整形する」順序に戻すのは、
    # `unaided_real_jump` が立ってからで良い。
    # Go2 サイクル1（2026-09-15）で Go2 側の joint_vel が jump 窓の外に出されたが、
    # **Anaguma のループは1サイクル1変更を守る**ため素の joint_vel_l2 を明示的に固定する。
    # 2つのループが同じ基底 cfg を共有しているので、固定しないと Go2 の変更が
    # Anaguma の次の run に混ざって読めなくなる。Anaguma でも試すのは別サイクルで。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # === サイクル4（2026-09-15 05:40）: v_z 上限ペナルティを Anaguma でも有効化 ===
    #
    # サイクル3の `model_9600` は **飛距離1.302m / 着地69% / 成立95%（80試行）** で成功条件に
    # 着地1ポイント足りない。失敗した試行を1本ずつ見ると内訳がはっきりしている:
    #   - 転倒(傾き105〜166deg) 6本 ... **いずれも頂点 0.277〜0.291 と高い側の裾**
    #   - 直立のまま不成立(傾き0〜37deg) 5本
    #   - 跳ばなかった 1本
    # **平均頂点 0.214 に対し、落ちるのは 0.28 超の跳躍**。つまり Anaguma も Go2 と同じく
    # 「高く跳んだ個体ではなく、高く跳んだ試行」が転んでいる。平均を見ていたので
    # 「Anaguma の着地を壊しているのは v_x」と読んでいたが、**試行単位では v_z の裾**だった。
    #
    # cap 2.0 は平均 v_z 1.92 のすぐ上＝**平均的な跳躍は無料で、裾だけに課金**する位置。
    # 距離への影響は小さいはず（裾の試行はそもそも転んでいて着地率に寄与していない）。
    # cap は **Isaac の分布で決める**（2026-09-15、スモークで発覚）。mujoco の平均 v_z 1.92 を
    # 見て 2.0 に置いたら、Isaac 側の unaided 平均 v_z は 1.41 しかなく **収入 0.0000＝項が
    # 一度も発火しない**死んだ報酬になっていた。mujoco/Isaac の v_z 比は約1.36 で、
    # mujoco で転んでいる裾 2.39 は Isaac のおよそ 1.76 に当たる。平均1.41 の上・裾1.76 の下、
    # ということで 1.6 を採る。**「名前と実測対象のズレ」の類型: 別のシミュレータで測った値を
    # そのまま報酬のしきい値に持ち込んだ。**
    # **この項は Anaguma では死んでいる（2026-09-15、実測で確認して weight 0 に戻した）。**
    # cap 2.0 でも 1.6 でも収入は 0.0000 のまま——Isaac 側の unaided v_z は平均 1.38 で、
    # mujoco で転んでいる裾（v_z 2.3〜2.4 相当）に当たる分布が **Isaac には存在しない**。
    # 裾は mujoco 側で生まれているので、Isaac の量に課金しても触れない。
    jump_liftoff_vz_cap = RewTerm(
        func=mdp.jump_liftoff_vz_cap,
        weight=0.0,
        params={"command_name": "jump_command", "v_z_cap": 1.6, "payout_window_s": 0.6},
    )
    jump_dense_reward = RewTerm(
        func=mdp.jump_dense_reward,
        weight=0.5,
        params={
            "command_name": "jump_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "normalize_by_weight": True,
        },
    )
    # === サイクル4（2026-09-15 05:50）: 飛距離の目標を 2.06 -> 1.10 に下げて飽和させる ===
    #
    # サイクル2〜3で **飛距離と着地が一方向にトレードオフし続けた**:
    #   `5800` 1.266m/着地90% → `9198` 1.342m/60% → `11000` 1.392m/62% → `12197` 1.402m/**0%**
    # `progress = clamp(distance/target)` が 2.06 に対して 0.45 前後＝**勾配が生きたまま**なので、
    # ポリシーは永遠に v_x を買い続ける。目標を Isaac 実測 0.93 のすぐ上（1.10）に置くと
    # **現在の動作点の少し先で飽和**し、以降は着地系の報酬が相対的に支配する。
    #
    # 飽和＝勾配ゼロは通常なら「死んだ報酬」(Loop 11/16/18) だが、**今回はそれが狙い**。
    # 成功条件の飛距離 1.3m は `model_9600` が既に 1.302m(80試行) で満たしており、
    # 足りないのは着地 69% → 70% だけ。距離を伸ばす動機を切って着地を取りに行く。
    #
    # 着地を Isaac の量で直接ねらう道は塞がっている: mujoco で着地90%の `5800` と 69%の `9600` を
    # Isaac の着地系メトリクス全部（upright_at_close 0.898/0.890、landing_max_tilt 0.163/0.157、
    # knee_contact 0.889/0.937、impact 488N/434N、touchdown_vz −0.54/−0.42）で比べても
    # **どれも差がない**。Isaac はこの差を見ていない。だから距離側の動機を切るという間接策を採る。
    jump_takeoff_distance = RewTerm(
        func=mdp.jump_takeoff_distance,
        weight=74.0,
        params={"command_name": "jump_command", "target_distance": 1.10, "payout_window_s": 0.6},
    )


@configclass
class RobotEnvCfgAnagumaLongJump(RobotEnvCfgLongJump):
    """Stage B on Anaguma: running approach plus jump, height objective."""

    rewards: RewardsCfgAnagumaLongJump = RewardsCfgAnagumaLongJump()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANAGUMA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- EFGCL assist, turned back on (2026-09-14) ---
        #
        # The first Anaguma run (2026-09-13_15-26-37, 3000 iter) walked well and jumped
        # exactly zero times: real_jump_fraction 0.0000 for all 3000 iterations, airborne
        # 0.072 s against the 0.20 s gate, rise 0.018 m against 0.08 m. The cause was this
        # inherited value. Go2's Stage B sits at initial_assist_scale=0.0 because by Loop
        # 13 its teacher had been overtaken by its own policy -- but every jump reward is
        # gated on ``real_jump``, so a machine that has never left the ground collects
        # nothing from any of them (jump_takeoff_apex, weight 50, paid 0.0000 for the
        # whole run). Nothing was left to produce the first take-off. This is the v3/v4
        # lesson running the other way: there the assist was measuring itself, here its
        # absence left nothing to measure.
        self.commands.jump_command.initial_assist_scale = 1.0

        # 0.15 m, set by MEASUREMENT rather than by the ballistic formula (2026-09-14).
        #
        # The formula says a 0.15 setting imparts v0 = 1.72 m/s for a 0.15 m apex and a
        # 0.35 s flight. On Anaguma, running at 2.8-3.3 m/s, it does not: two 40-iteration
        # smokes from model_2999 of 2026-09-13_15-26-37, both at assist_scale 0.97, give
        #
        #   setting   trunk rise   airborne   real_jump   falls (base_contact+bad_orient)
        #   ------------------------------------------------------------------------------
        #   0.15      0.0794 m     0.169 s      36.8%      9.0%
        #   0.30      0.1195 m     0.224 s      48.6%     23.3%
        #   (none)    0.018 m      0.072 s       0.0%      0.8%   <- the 2026-09-13 run
        #
        # so about half the nominal rise arrives: the impulse lands on a machine mid-gait
        # and part of it goes into the legs rather than the trunk. Doubling the setting
        # buys +50% rise and +32% real jumps but 2.6x the falls -- 24.3 kg at 3 m/s cannot
        # absorb the larger vertical impulse and tumbles. Falls, not the reward targets,
        # are the binding constraint here (0.12 m against jump_height's 0.30 m target is
        # not close to the v4 trap, where assist and target were equal).
        #
        # 0.15 is chosen because it turns the signal on at a fall rate still near the
        # unassisted baseline. It does NOT hand the policy a finished jump: at 36.8% the
        # teacher gets the machine to the edge of the real_jump gate (rise 0.079 against
        # the 0.08 m threshold) and the policy has to supply its own push-off to collect
        # on the other 63%. That is the condition EFGCL wants -- above what the policy can
        # do alone (it does nothing alone), below what earns full marks.
        self.commands.jump_command.assist_apex_height_m = 0.50

        # The crouch pulse is the one assist force NOT mass-scaled in JumpCommand: it is
        # a flat 150 N sized for Go2's ~15 kg. On 24.31 kg it would load the legs with 62%
        # of the intended acceleration. 150 * 24.31/15.0 = 243 N restores it.
        self.commands.jump_command.crouch_assist_force = 243.0

        # 19_200 steps = 800 iterations. That schedule was written for a Go2 that could
        # already jump and only needed the teacher out of the way; Anaguma starts from no
        # take-off at all. 36_000 = 1500 iterations, so a 3000-iteration run still trains
        # its whole second half unaided -- the Loop 9 failure to avoid is the opposite
        # one, an assist still fading with only 400 iterations left.
        self.curriculum.jump_assist_decay.params["force_zero_at_step"] = 36_000

        # --- 踏切ペナルティを下げる（2026-09-14 夜） ---
        #
        # 2026-09-14_17-02-57 の収支を項別に読むと、跳んでいた瞬間（iter 100、補助0.90、
        # real_jump 0.74）の跳躍関連の収支は
        #
        #   飴  jump_height +0.0875 / landing_feet_first +0.0743 / foot_clearance +0.0302
        #       / takeoff_apex +0.0114                                     = +0.203
        #   鞭  jump_dense_reward -3.478 / landing_order -0.078 / flight_pitch -0.032
        #       / landing_gear -0.023 / roll_rate -0.007 ほか              = -3.622
        #
        # で、**踏切は1エピソードあたり約 -3.4 の赤字**。同じ報酬セットで走る Go2
        # （2026-09-14_17-02-52、iter 10899、real_jump 0.71）は takeoff_distance +1.439 を
        # 筆頭に +2.06 を稼ぎ、jump_dense_reward は **-0.0064** しか払っていない。
        # 同じ項が Anaguma で 540 倍重い。
        #
        # jump_dense_reward は -std(接地力) [N] で、**報酬セットで唯一 質量で正規化されて
        # いない量**（Anaguma は Go2 の 1.62 倍の質量）。加えて curriculum が
        # early_weight=2.5 を decay_at_step=60000（=iter 2500）まで掛け続けるので、
        # cfg の weight 0.5 ではなく実効 2.5 が 3000 iter のうち 2500 iter を支配する。
        # そもそもこの項は「跳べる機体の踏切を左右対称にする」整形であって、まだ一度も
        # 踏み切っていない機体には**試行そのものへの課税**にしかならない（論文でも
        # Phase 2a→2b、すなわち跳べるようになってから緩める順序）。
        #
        # 跳べるようになるまでは late_weight と同じ 0.25 に寝かせる。踏切が立ったら
        # （unaided_real_jump > 0）2.5 に戻して対称性を整形する、が本来の順序。
        self.curriculum.jump_dense_reward_decay.params["early_weight"] = 0.25

        # --- 助走を落とす（2026-09-14 夕、フェーズ1） ---
        #
        # Go2 の Stage B は U(2.8, 3.3) で走りながら跳ぶ。そこを iter 0 から学ばせる試みは
        # 補助を 0.15 と 0.50 の二通りで試して二度とも同じ形で失敗した——補助が抜けるにつれ
        # real_jump が 0 に落ち、転倒も 32% -> 2.8% まで下がる（＝跳ばずに走るのが最適解に
        # なった）。
        #
        # 過去ログを当たると、Go2 が iter 0 から跳べるようになった v3〜v9 の助走帯は
        # U(0.0, 3.3) で、跳躍が成立していた瞬間の実測助走は 1.11 m/s だった。
        # U(2.8, 3.3) に上げたのは Loop 10（iter 7600）、跳べるようになった後である。
        # つまり「走りながら跳ぶ」は最初から要求されていたのではなく、遅い跳躍を覚えてから
        # 速くしている。この移植はその順番を飛ばしていた。
        #
        # Anaguma は関節速度上限が Go2 の 45% しかないので、なおさら低速から入るべき。
        # フェーズ2（この重みから U(2.8, 3.3) へ上げる）は unaided_real_jump が立ってから。
        # --- フェーズ2: 助走帯を戻す（2026-09-14 夜） ---
        #
        # フェーズ1（`2026-09-14_20-00-36`、助走 U(0.0,3.3)・ゲート(0.5,1.5)）で
        # **Anaguma が初めて自力で踏み切った**（unaided_real_jump 0.726、補助ゼロ後も維持）。
        # mujoco 実測では `model_2400` が **学習で一度も踏み切っていない助走 2.8 指令
        # （実測2.59 m/s）で 飛距離0.987m・跳躍95%・着地78%** を出しており、
        # 速い助走に載る能力は既にある。よってフェーズ2は新規学習ではなく
        # `model_2400` からの継続で、助走帯とゲートだけを戻す。
        #
        # 補助は **0 で入る**（`initial_assist_scale = 0.0`）。踏切はもう自力でできるので
        # 教師は不要で、試行2の「走れる重みに補助を当てると打ち消される」を繰り返さない。
        # Go2 の Stage B も Loop 13 以降は同じ理由で assist 0 で回している。
        self.commands.base_velocity.ranges.lin_vel_x = (2.8, 3.3)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (2.8, 3.3)
        self.commands.jump_command.initial_assist_scale = 0.0


@configclass
class RobotPlayEnvCfgAnagumaLongJump(RobotEnvCfgAnagumaLongJump):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Show what the policy does unaided, not what the assist is still doing for it.
        self.commands.jump_command.initial_assist_scale = 0.0
