# Go2 走り幅跳び(Running Long Jump)ポリシー開発 — 進捗まとめ

**発表予定: 2026-09-03(木)**
**位置づけ**: ORT「Go2限界性能タスク」①番(垂直ジャンプ=田村さんチームの水平版)。詳細背景は永続メモリ `プロジェクト_Go2限界性能タスク.md` 参照。

このファイルは作業を進めるたびに追記していく進捗ログ。発表資料のベースとして使う想定。

---

## 1. 方針決定の経緯

### 1.1 参考論文の選定
- 当初候補 **Curriculum-Based Reference-free Jumping**(Atanassov et al., arXiv:2401.16337, Go1で前方90cm)を検討したが、**YouTube上の実機映像を確認したところ「その場跳躍」起点の立ち幅跳び寄りの動作**と判明(2026-08-27)。
- 「助走→踏切」を伴う**本当の走り幅跳び**を実現するため再調査し、**Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot**(Guan et al., arXiv:2404.15096, MIT Mini Cheetah Vision)を採用。助走2m/sで重心移動96cm(隙間55cm)を参照軌道なしで達成。
- 両論文とも**コードは非公開**と確認(著者・研究室のGitHubを直接調査)。Impedance Matching論文がベースにしたKAIST先行研究(Ji et al. 2022, arXiv:2202.05481「Explicit Estimator」)のサードパーティ実装が`genesis_lr`にGo2向けで存在することを発見、実装の参考書として活用。
- 追加候補として **EFGCL**(Yoneda et al., arXiv:2605.10063、外部力ガイド付きカリキュラム学習)も調査済み。学習が局所解に陥った場合の代替/補完手段として保留。

### 1.2 Go2の理論限界飛距離の試算
公式URDF(`unitreerobotics/unitree_ros`)から確認した実機スペック(大腿0.213m・下腿0.213m、Calf関節トルク45.43N·m等)を用い、エネルギー法で踏切速度を推定:

| 助走速度 | 理論飛距離 |
|---|---|
| 2 m/s(論文と同条件) | 約2.02 m |
| 5.3 m/s(Go2実測最大速度) | 約3.89 m |

論文の実達成値(96cm)との2〜4倍のギャップは、田村さんチームの垂直ジャンプ(理論値51cm相当 vs 実測40cm安全/70cm一発)と同じ「物理限界と実制御性能の差」の構図。詳細は `プロジェクト_走り幅跳び理論値計算.md`。

### 1.3 タスク設計方針
- **blind policy**(height-scan/視覚なし、固有受容ベース)で進める。理由: 本タスクは「平地・既知条件での理論限界性能測定」が目的で、地形認識が必要な場面が想定しにくいため。
- **Stage A/B の2段階構成**(Net2Net、Impedance Matching論文の設計を踏襲):
  - **Stage A**(`Unitree-Go2-LongJump-Base-v1`): 平地走行のみをゼロから学習。既存の混合地形チェックポイントは地形履歴(平地/荒地/階段)が不明瞭なため不採用と決定。
  - **Stage B**(`Unitree-Go2-LongJump-v1`): Stage Aの重みに、ジャンプコマンド入力ニューロンをNet2Net(ゼロ初期化)で追加し、歩行/跳躍を統合学習。
- **Explicit Estimator**(KAIST Ji et al. 2022ベース): 基部速度・全17リンク接触状態・4脚の足高さを固有受容の観測履歴から推定するネットワークを、Actorの入力に連結。PPO本体とは完全に独立したoptimizerで学習。
- **jump_command**: 0(通常走行)/1(跳躍モード)の二値コマンド。ランダムトリガー(学習時)またはキーボード`j`キー(実機)で0→1、着地検知で自動的に1→0(1エピソード内で連続跳躍可能)。
- **報酬**: Dense Jump Reward(-std(4脚接地力)、対称的な踏切を促進)、Sparse Jump Reward(離陸速度と目標値の一致度、目標値はカリキュラムで段階的に引き上げ——論文の固定2.5m/sとは異なり、理論限界に迫ることを意図した意図的な変更)。

---

## 2. 実装内容(2026-08-29完了)

| コンポーネント | ファイル | 状態 |
|---|---|---|
| Stage A環境 | `source/.../robots/go2/longjump_base_env_cfg.py` | ✅ 実装・スモークテスト・本番学習実行中 |
| Stage B環境 | `source/.../robots/go2/longjump_env_cfg.py` | ✅ 実装・スモークテスト完了 |
| jump_commandコマンド | `mdp/commands/jump_command.py` | ✅ 実装・動作確認 |
| Estimator観測(link_contact_bool, foot_height) | `mdp/observations.py` | ✅ 実装 |
| Jump報酬(dense/sparse) | `mdp/rewards.py` | ✅ 実装・動作確認(離陸イベント検知を確認) |
| 跳躍目標速度カリキュラム | `mdp/curriculums.py` | ✅ 実装(閾値判定は疎な報酬に未最適化、要検証) |
| ジャンプ時のみ緩和する終了条件 | `mdp/terminations.py` | ✅ 実装 |
| ActorCriticEE / PPOEE(Estimator本体) | `tasks/locomotion/agents/rsl_rl_ee.py` | ✅ 実装・スモークテスト完了(estimator loss記録確認) |
| Net2Net重み移植スクリプト | `scripts/rsl_rl/longjump_net2net_transplant.py` | ✅ 実装・実`--resume`パイプラインで検証済み |
| キーボードjump trigger(C++) | `deploy/include/.../keyboard.h`, `deploy/robots/go2/src/State_RLBase.cpp` | ✅ 実装・ビルド確認済み(固定時間タイムアウト方式、接触センサ配線は将来課題) |

### 実装中に発見・解決した技術的な問題
1. **共有マシンの`/tmp/isaaclab/logs/`権限エラー** — 他ユーザー所有ディレクトリで書き込み不可。`TMPDIR`環境変数で回避。
2. **PPO損失とEstimator損失の計算グラフ競合** — `estimator_output`をActor入力側で`.detach()`していなかったため「2回目のbackward」エラー。detachすることで、論文通りの完全独立学習を実現しつつ解決。
3. **2つの環境を同一プロセスで構築すると無限再帰** — `configclass`の検証処理で発生。Net2Net移植スクリプトはStage A側の環境構築を回避し、checkpointのテンソル形状から直接次元を読み取る設計に変更して解決。

---

## 3. 現在の状況(2026-08-30更新)

- **2026-08-29 20:29開始のStage A本番学習が、iter 3400/10000で停止していたことを2026-08-30 10:13に発見**。プロセス自体が消えており(GPU上にも該当プロセスなし)、クラッシュログ/OOMの痕跡は見当たらず。原因は「バックグラウンドタスクがセッション境界を跨げず、プロセスが道連れで落ちた」と推測(明確な確定はできていない)。停止直前のtensorboard記録では`mean_reward`10.29・`track_lin_vel_xy`+0.80・entropy 4.89と健全な学習曲線だった。
- **10:15に`model_3400.pt`から`--resume`で再開**。今回は`setsid nohup ... & disown`で完全デタッチ起動(PPID=1に再親化、親シェル/セッションが死んでも道連れにならない構成)にし、ログはリポジトリ内`logs/rsl_rl/unitree_go2_longjump_base_v1/resume_stdout_*.log`に出力。`--max_iterations`はrsl_rlの仕様(`tot_iter = 再開iter + 指定値`)を踏まえ**6600**を指定し、合計10000で完走するよう設定。10:15台に`Learning iteration 3400/10000`のログで再開を確認済み。新しい学習ログは実行ディレクトリ`2026-08-30_10-15-30`配下に保存される(重み自体は`2026-08-29_20-29-43/model_3400.pt`から正しく引き継ぎ)。
- 残り6600iter、実測ペース(~3.9秒/iter)で**本日(8/30)18時頃に完走見込み**。
- **注意(発見): `--resume`はActor/Critic重みのみ復元し、カリキュラム進捗は復元されない**。`Curriculum/lin_vel_cmd_levels`が停止前(iter3450時点で0.1→3.2まで到達済み)から再開直後に**0.1へリセット**されているのを確認(2026-08-30)。再開直後にmean_reward(10.9→35.7)・entropy(+4.6→-6.0)が急変しているのはこのカリキュラムリセットにより一時的にタスクが易化しているためで、**真の収束ではない**。学習を打ち切る判断は`Curriculum/lin_vel_cmd_levels`が以前の到達水準を超えて安定してから行うこと。同じ仕組み(`jump_vel_target_levels`)を使うStage Bでも、中断・再開時に同じ現象が起きる見込み。
- **2026-08-31追記: Stage B本番学習が完走**(2026-08-30 15:26開始 → 2026-08-31 04:23終了、iter9999)。最終チェックポイント`logs/rsl_rl/unitree_go2_longjump_v1/2026-08-30_15-26-16/model_9999.pt`。最終値: `lin_vel_cmd_levels`3.8、`bad_orientation`58.4%、`jump_vel_target_levels`1.5(想定通り全期間不動)。iter4142頃から速度・転倒率とも実質横ばいで、残り約6000iterの伸びは小さかった(tanaka氏の判断でこのまま完走まで見送り、次はmujoco確認へ)。

## 4. 次の一手

- [x] Stage A本番学習 — **2026-08-30 15:xx、iter 7100(全体の71%)で打ち切り**。`Curriculum/lin_vel_cmd_levels`が3.2で長く横ばいになったことを確認し、tanaka氏判断で終了。最終チェックポイント`logs/rsl_rl/unitree_go2_longjump_base_v1/2026-08-30_10-15-30/model_7100.pt`。
  - **原因の考察(tanaka氏)**: `lin_vel_cmd_levels`のレベルアップ条件が`track_lin_vel_xy`(前後+左右速度追従の合成報酬)の80%ラインで、前後・左右・回転を**同時に**満たさないと上がらない設計になっていたため、前進最大速度が(理論値やAB論文の設定ほど)伸びなかったと推測。次回Stage Aをやり直す際は、可動方向・範囲を絞って少ないiterationで学習させる方針が良さそう(要メモ化・次回反映)。
- [x] Net2Net移植スクリプトでStage Bへ重み移植 — 完了(2026-08-30 15:20)
- [x] **Stage B本番学習を実行 — 完走**(2026-08-30 15:26開始 → 2026-08-31 04:23終了、iter9999)。最終チェックポイント`logs/rsl_rl/unitree_go2_longjump_v1/2026-08-30_15-26-16/model_9999.pt`。
- [x] **学習途中のIsaac Sim動作確認を実施(2026-08-30)**: `--livestream 2`(WebRTC)でtanaka氏が学習中のポリシーを直接観察。「ほとんど動かない/転ぶ」「助走してから跳ぶ個体がいない」という報告——前者はPlay configが速度上限を即座に5.5m/sまで開放してしまい未習熟の速度域でテストしていたのが原因(暫定的に-1.0〜4.0m/sへキャップして解消確認)、後者は既知の`jump_vel_target_levels`固定バグの症状と判明。**完走後、暫定キャップは削除済み**(`longjump_env_cfg.py`は`limit_ranges`(5.5m/s)そのままの正規状態に復帰済み)。
- [x] **Stage B完走後のmujoco sim2sim確認 — 実施**(詳細はセクション5)。「後退はうまく歩くが、前進コマンドですぐ転倒する」という重大な問題を発見。
- [x] **forward-only修正を実装**(セクション5)。Stage A/B双方のconfigで`lin_vel_x`の後退方向を完全に排除。
- [x] **Stage A v2(forward-only)を再学習 — 完走**(2026-09-02、詳細はセクション5)。
- [x] **Net2Net移植 → Stage B v2(forward-only)を起動**(2026-09-02、詳細はセクション5)。**進行中**。
- [x] **旧Stage B(forward-only修正前, model_9999.pt)をmujoco実機で再検証、動画収録**(2026-09-02、失敗ケース=前進での転倒も含めて収録。詳細はセクション5)。
- [ ] **Stage B v2(forward-only)完走を待つ**(9/2 16:13時点でETA目安 約4時間44分、GPU共有状況により変動)。
- [ ] Stage B v2完走後、mujoco実機で再検証(forward-onlyで前進が安定するか)。
- [ ] 実際の跳躍距離を計測し、理論値(2.0〜3.9m、助走速度依存)と比較
- [ ] (必要なら)`jump_vel_target_levels`カリキュラムの閾値判定を疎な報酬向けに見直し

---

## 5. mujoco実機検証で判明した前進転倒問題とforward-only修正(2026-09-02追記)

### 5.1 Stage B(v1)のmujoco実機検証で判明した問題

Stage B完走(2026-08-31 04:23)後のmujoco sim2sim確認で、**「後退はうまく歩くが、前進コマンドを与えるとすぐ転倒する」**ことが判明した。

**原因**: `mdp.curriculums.lin_vel_cmd_levels`のレベルアップ条件`track_lin_vel_xy`(前後+左右速度追従の合成報酬)が、Go2のこの体格だと**後退の方が楽な歩様らしく、後退だけでも満たせてしまう**ため、カリキュラムレベル(=速度コマンド範囲)は`lin_vel_cmd_levels`3.2〜3.8まで順調に伸びていたにもかかわらず、**前進方向は実質学習不足のまま見過ごされていた**。セクション4で触れた「前後・左右・回転を同時にこなせないと速度が伸びない」という当初の考察に加え、**「同時にこなす」以前に後退という抜け道自体があった**、という追加の知見。

### 5.2 forward-only修正

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/longjump_base_env_cfg.py`(Stage A)・`longjump_env_cfg.py`(Stage B)の速度コマンド設定で、**`lin_vel_x`のみ**`ranges`/`limit_ranges`の下限を`0.0`に固定し、後退コマンドをサンプリング対象から完全に排除した(`lin_vel_y`・`ang_vel_z`は変更なし)。

### 5.3 Stage A/B v2(forward-only)の学習経緯

1. **初回起動(2026-08-31 17:40〜18:30)**: `--max_iterations 6000`で起動するも、iteration 750/6000まで進んだところで(原因不明、セッション境界の可能性)プロセスが強制終了させられ、シャットダウン処理中にIsaac Sim内部バグでクラッシュ。**その後41時間、誰にも気づかれず放置**されていた。
2. **2026-09-02、tanaka氏指示で仕切り直し**: 「iterationは3000でいい、途中(model_700.pt)からではなく最初から。3000終わったら自動でStage Bへ、Stage Bも3000でいい」との指示を受け、Stage A(ゼロから、3000iter)→完走後自動でNet2Net移植→Stage B(3000iter)まで一気通貫の自動化スクリプト(`scripts/run_longjump_pipeline_v2_scratch.sh`)を作成し、セッション境界を跨いでも死なないよう完全デタッチで起動。
3. **Stage A v2 完走**(2026-09-02 14:56、iteration 2999/3000)。
4. **Net2Net移植でハング発生**: 移植プロセスが47分経っても完了せず(GPU競合と見られたが、後の検証で実際には**移植処理自体は数分で正常完了しており、その後の`simulation_app.close()`だけがハングしていた**と判明)。強制終了→出力済み`model_0.pt`を検証(重み・NaN無し・Net2Netのゼロ初期化列を確認)→**正常な成果物と確認できたためやり直し不要**、そのままStage Bを手動起動する形に切り替えた。
5. **Stage B v2(forward-only)、現在学習中**(2026-09-02 15:47開始、`--max_iterations 3000`)。uchida氏の別ジョブとGPU共有中のためやや低速。

### 5.4 旧Stage B(forward-only修正前)のmujoco実機再検証と動画収録(2026-09-02)

NoMachineリモートデスクトップ経由で、旧Stage B(`model_9999.pt`, forward-only修正**前**のモデル)をmujoco + `go2_ctrl`で再検証。**前進コマンドでの転倒が実機シミュレーション上でも再現することを確認**し、5.1の問題を目視で裏付けた。**失敗ケース(転倒の様子)も含めて動画収録済み**——発表資料への添付を想定。

---

## 6. 用語メモ(発表時の混同注意)

- 本ドキュメントの「Stage A/B」は**歩行→跳躍のNet2Net統合段階**を指す。既存`scripts/run_all_phases.sh`の「Phase1/2/3」は**地形カリキュラムレベル**であり別概念(混同注意、2026-08-27に判明)。
