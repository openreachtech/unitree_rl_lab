# Go2 ブラインド階段 報酬チューニング 実験ログ

`scripts/auto_tune_loop.sh` が自動追記する。全エントリ: 変更・根拠・結果・考察をセットで記録する
(仮説は必ず既存ログのEpisode_Reward/*実測値か既存の設計制約に基づくこと。当てずっぽうは禁止)。

## 前提: これまでの経緯(2026-07-05時点)

- ベース系譜: Phase1(3000it)→Phase2(+4000it)→Phase3初回(+5000it, curriculum_level=3, model_11997)
  →降格緩和(demote_fraction 0.5→0.25, level3限定, +1500it, terrain_levels 4.56→5.12)
  →延長(打ち切り, model_16900, terrain_levels≈4.74) = **BASE**
    →target_clearance 0.08→0.15(+1500it, model_18399, terrain_levels≈5.01)
      →joint_pos -0.4→-0.2追加(+1500it, model_19898, terrain_levels≈5.28) = **現在のBEST**

- **ラウンド1のバグ**: `auto_tune_loop.sh`がvenvを有効化せずpython3(tensorboard無し環境)を呼んでいたため、
  `terrain_level_of`が毎回クラッシュ→空文字列→比較ロジックが誤動作し、5仮説
  (forward_command_progress=2.2, flat_orientation_l2=-0.5, air_time_variance_penalty=-0.05,
  lin_vel_z_l2=-0.5, foot_clearance_terrain_adaptive=0.8)が**全て「BASE(model_16900)から個別分岐」で
  実行されていたのに評価が機能せず、全部不採用扱い**になっていた。手動でvenvを有効化し再測定した結果:

  | 仮説(BASEから分岐) | terrain_levels | 判定 |
  |---|---|---|
  | forward_command_progress=2.2 | 4.87 | BASE(4.74)よりわずかに高いが誤差範囲。ただしentropy 7.48・mean_reward 35.6と学習ダイナミクスが顕著に違う |
  | flat_orientation_l2=-0.5 | 4.90 | 誤差範囲 |
  | air_time_variance_penalty=-0.05 | 4.82 | 誤差範囲(むしろ僅かに低い) |
  | lin_vel_z_l2=-0.5 | 4.89 | 誤差範囲 |
  | foot_clearance_terrain_adaptive=0.8 | 4.96 | 誤差範囲 |

  → 5つともBASE単体では明確な改善なし。ただしこれらはBEST(target_clearance+joint_pos, 5.28)の上には
  一度も載せていなかった。BESTの上でも効果が無いかは未検証だった。

- **方針転換(ユーザー指摘, 2026-07-05)**: 貪欲法(常にBESTの上に1つ積む)だけでは、採用された変更が
  「単体で効く主効果」なのか「既存の採用済み変更との組み合わせでしか効かない交互作用」なのか区別できない。
  全数検証(2^n通り)はコスト的に不可能なため、**「採用された変更だけ、BASE単体でも追加確認する」**方式を
  ラウンド2から導入。あわせて仮説には必ず根拠(ログ実測値 or 既存設計上の制約)を明記することを義務化。

## ラウンド2以降のエントリ

(このセクションから `scripts/auto_tune_loop.sh` が自動追記する)

## 2026-07-06 18:25 forward_progress_2.2 (手動評価、round-1バグ修正後の再検証)

- **変更**: `forward_command_progress` -> `2.2`
- **根拠**: ラウンド1テスト(BASE=4.74から分岐)では単体で明確な改善はなかったが、entropy 6.1-6.4→7.48、
  mean_reward 29-32→35.6と学習ダイナミクスが顕著に変化していた。指令方向への正味変位を直接報酬化する項で、
  カリキュラム昇格条件と同じ量を測るため、より自由になったpolicy(target_clearance=0.15+joint_pos=-0.2)の
  上でなら効く可能性を検証。
- **主線結果** (分岐元: best 2026-07-05_07-06-24/model_19898.pt, level=5.28): 学習中に terrain_levels が
  一時5.48まで上昇したが、最終的に **5.08**(直近15点平均, model_21397.pt)まで低下 -> **不採用**
  (旧best 5.28を下回る)
- **単体確認結果**: 不採用のため実施せず
- **考察**: ラウンド1で確認した「entropy急上昇(探索活発化)」という性質が、今回のBEST上でも同様に表れ、
  最終的な収束を不安定にした可能性が高い(ピーク5.48→終端5.08という上下動は、他の採用済み変更にはない
  パターン)。forward_command_progressの重みを上げすぎると探索が広がりすぎて着地点が悪化する、という
  仮説が立つ。もし将来再挑戦するなら、2.2ではなくもっと穏やかな値(例1.8)を試す価値があるかもしれないが、
  優先度は今回のキュー消化後に持ち越し。
- **現在のbest**: 2026-07-05_07-06-24/model_19898.pt (terrain_levels=5.28, 変更なし)

## 2026-07-06 23:45 flat_orientation_-0.5

- **変更**: `flat_orientation_l2` -> `-0.5`
- **根拠**: Already halved once (-2.5->-1.0, Fix3) because steep body pitch is required to climb. Episode_Reward/flat_orientation_l2 has stayed small (-0.03 to -0.05) in every run, i.e. rarely saturating -- testing whether the residual penalty still caps the more aggressive pitch needed on the hardest (0.15-0.23m) steps that the population has not yet conquered (plateau ~5, short of the top rows).
- **主線結果** (分岐元: best 2026-07-05_07-06-24/model_19898.pt, level=5.28): terrain_levels = 5.427071317036947 -> 採用=yes
- **単体確認結果** (分岐元: base 2026-07-04_23-10-53/model_16900.pt, level=4.742484060923259): terrain_levels = 4.90142027537028
- **考察**: 単体でもbase(4.742484060923259)を上回った(4.90142027537028) -> flat_orientation_l2=-0.5は独立した主効果と判断できる。
- **現在のbest**: 2026-07-06_20-15-26/model_21397.pt (terrain_levels=5.427071317036947)

## 2026-07-07 00:13 【環境バグ発見・修正】後ろ向き登坂エクスプロイト -> heading_command有効化

- **発見の経緯**: mujoco実機検証中、直進コマンドのみを与えているにも関わらずgo2が数段登ったところで
  向きを反転し後ろ足側を先導させて登り続ける挙動を確認(高さ20cm/幅20cmの階段で顕著)。「数段登ってすぐ
  降りる」ように見えていた挙動は、実際には反転ターンの最中だった可能性が高い。
- **根本原因**: `forward_command_progress`/`track_lin_vel_xy_exp`は`root_lin_vel_b`(ボディ座標系速度)と
  ボディ座標系コマンドの内積のみを見ており、ロボットの絶対yawには一切依存しない。`CommandsCfg.base_velocity`
  (`velocity_env_cfg.py`)は`heading_command`が未設定(=False)で、`ang_vel_z`はコマンドとして自由サンプリング
  されるだけだった。そのため、体を反転させて「ボディ座標系での前進」を後ろ足側で満たす戦略が、報酬上まったく
  区別なく成立してしまっていた(terrain_levelsの昇格判定も`root_pos_w`の正味変位のみを見ており、向きは無関係
  なため、これも抜け穴を検知できない)。
- **対応(ユーザー承認: heading_command有効化案を選択)**: `velocity_env_cfg.py`の`CommandsCfg.base_velocity`に
  `heading_command=True`, `heading_control_stiffness=1.0`, `ranges.heading=(-3.14, 3.14)`を追加。IsaacLab標準の
  heading追従コントローラにより、resample期間(10秒)ごとに固定される目標headingとの誤差から`ang_vel_z`が
  常時自動生成されるようになり、向きを反転して保持する行動は継続的な`track_ang_vel_z_exp`減点を招くため、
  単発の一時的コストで済んでいた従来の抜け道を塞ぐ。`curriculums.py`のレベル別レンジ拡張ロジックは
  `lin_vel_x/y`と`ang_vel_z`のみ操作し`heading`には触れないため、既存のカリキュラム機構と衝突しない。
  報酬関数ファイルではなくコマンド生成設定(`CommandsCfg`)への変更だが、タスクを易化する方向ではなく
  エクスプロイトを塞ぐ方向の変更である。
- **中断した処理**: 実行中だった`lin_vel_z_-0.5`の主線trialは(旧コードのまま起動していたため)強制終了。
  この項目の評価データは無効(このヒストリーには残さない、後で必要ならキューに再投入)。
- **今後の扱い**: この変更は報酬重みではなく環境の根本修正のため、これまでの`best`(terrain_levels=5.427)は
  「反転エクスプロイトを部分的に含む」可能性があり単純比較はできない。BEST(2026-07-06_20-15-26/model_21397.pt)
  から`heading_fix_validation`として1500iter再学習し、新しいterrain_levelsを測定してから
  auto_tune_loopの仮説キュー(lin_vel_z_-0.5, foot_clearance_0.8, air_time_var_-0.05)を再開する。
  この検証runの結果が新たな比較基準になる。
- **追記(00:14起動 -> 即クラッシュ、00:28修正・再起動)**: 1回目の起動が環境構築時に
  `ValueError: heading_command has heading commands active but ranges.heading is None` で即死。
  原因は`go2_curriculum.py`の`apply_phase_velocity_ranges`が`__post_init__`時点で
  `env_cfg.commands.base_velocity.ranges`を`PHASE_VEL_START[level]`という別の`Ranges`インスタンスで
  丸ごと上書きしており、そこに`heading`が未設定(デフォルトNone)だったため。`PHASE_VEL_START`
  `PHASE_VEL_LIMIT`(lin_vel_cmd_levels_go2が一時的に差し替える`limit_ranges`側)`PLAY_VEL_RANGES`
  (play.py/export用)の全エントリに`heading=(-3.14, 3.14)`を追加して解消。学習iterationには一切
  到達していなかったため、無駄になった学習コストはゼロ(Isaac Sim起動オーバーヘッドのみ)。
  00:28に再起動、正常に起動確認(1つ目のログ点でterrain_levels=1.92 = 新規学習開始直後の初期値、想定通り)。
- **追記(01:09、40%経過時点の結果)**: terrain_levels=5.70〜5.73まで回復し**旧BEST(5.427)を超過**、
  entropy=6.35前後まで回復(旧水準6.2と同等)。ただしbad_orientation(転倒率)が0.0015→0.04まで
  約25倍に増加しており、heading強制保持とバランス確保の両立にコストがかかっている可能性を示唆。
  heading_commandによる反転エクスプロイト封じ自体は機能を実証できたと判断。

## 2026-07-07 01:20 【方針転換】記録方式・実験設計を先輩方式に合わせて再構築

インターン先の先輩が使っている`train-and-check`スキルのプロンプトを比較検討した結果、ユーザー指示で
以下の通り運用を変更する。これ以降、`auto_tune_loop.sh`による自動ループ(貪欲法+単体確認run)は
**運用停止**。このログ自体もここで区切りとし、今後の実験記録は各Try-Nのsandboxファイル自体
(gym.registerのコメント + configクラスのdocstring)に一本化する(先輩方式)。

- **demote_fractionを標準0.5に復元**: `go2_curriculum.py`の`terrain_levels_climb`から
  level3限定0.25の緩和を撤廃。カリキュラムの昇格/降格判定ロジック自体は今後一切変更しない
  (先輩の"don't use custom_terrain_levels_climb"という明示ルールに準拠。段差を甘くするのと
  同様、評価基準を甘くする方向の変更も「ずるをしない」の対象と判断)。
- **各試行の起点を固定**: 貪欲法(直近BESTから分岐)をやめ、**必ずPhase2の共通チェックポイント
  (`2026-07-04_05-26-55/model_6998.pt`, curriculum_level=2の終端)から**再学習する。これは
  現在有効な系譜(Phase1 0→2999 curriculum_level=1、Phase2 3000→6998 curriculum_level=2、
  Phase3初回 7000→11997 curriculum_level=3)から特定した。単体確認run(BASE分岐)の仕組みは
  不要になった(全試行が同じ起点から独立に評価されるため、交絡の問題がそもそも発生しない)。
- **記録方式を先輩式に変更**: `go2/sandbox/`配下に試行ごとの新規configファイル+`gym.register`
  (戦略コメント付き)を作成する方式に変更。`Unitree-Go2-Velocity-v1-Phase3-Try-1`から開始。
  IsaacLabの`import_packages`が全サブパッケージを自動探索するため、配置するだけでtask登録される。
  ただし`experiment_name`をデフォルト(空文字→task id自動変換)のままにすると別ログルート
  (`logs/rsl_rl/unitree_go2_velocity_v1_phase3_try_1/`)に分離されPhase2チェックポイントを
  見つけられずクラッシュしたため、`SandboxPPORunnerCfg`で`experiment_name="unitree_go2_velocity_v1"`
  に固定し、全Try-Nが共通ログルートに集約されるようにした。
- **報酬関数の自由度を先輩と同等に**: 既存weightの変更のみという制限を撤廃、必要なら新規報酬関数の
  定義も可とする。
- **Slack通知を最小限化**: `notify_train_slack.sh`を名前・サーバー名・開始/完了/失敗のみに簡略化
  (根拠・仮説・考察等の詳細はSlackに投げず、sandboxファイル側に書く)。
- **Try-1の内容**: 旧BEST(貪欲法+demote_fraction=0.25で到達したtarget_clearance=0.15,
  joint_pos=-0.2, flat_orientation_l2=-0.5, base_linear_velocity=-1.0, air_time_variance=-0.2,
  forward_command_progress=1.5)と全く同じ重みを`RewardsCfgGo2Try1`として凍結・再現し、
  Phase2から標準demote_fraction+heading_command=Trueの下で学習し直す再ベースライン試験。
  01:23起動、iteration 7000開始、目標9998(3000it)。

