# マルチターン化の変更点（2026-09-14/15）

単発の「指示 → 返事 + プログラム」を、ロボットが動いている間も続く会話にした。
目的は「もう一回」「あ、ストップ」「（走行中に）ハンドスプリングして！」に答えること。

## 1. 出力形式: `action` 行を足した

```
<返事 1 行>

action: none | cancel | replace | insert | append
program: [...]
```

| action | 意味 | program |
|---|---|---|
| none | 今の動きに触らない（雑談・質問・提案・返事待ち） | `[]` |
| cancel | 止めて、残りも捨てる | `[]` |
| replace | 今のを止めて差し替え。待機中の新しい指示もこれ | 非空 |
| insert | 今すぐやって、終わったら中断した動きに戻る（走行中に技や旋回だけ言われた） | 非空 |
| append | 今のが終わったら続けて | 非空 |

`program: []` が「何もしない」と「止める」の両方を意味してしまうのを避けるため。
none/cancel ⇔ `[]` の対応は `render_output` が拒み、GBNF では書けない。定義は `chat_format.py` の 1 か所。

## 2. 状態ブロック: 毎 user ターンの先頭 1 行

```
[状態] 実行中 6.1s/10.0s: 前へ2m(済) → 左側転×1(済) → 右斜め前へ3m(いま 残り2.9s) / 中断可 / 走りながらの技: 前方回転
[状態] 待機中 / 直前(完了): 前へ5m → バク転×1
[状態] 待機中 / 直前(中断): 前へ5m(済) → バク転×1(途中) → 左回り90度
```

- system prompt ではなく user ターンに置く → 履歴が append-only で KV キャッシュが壊れない。
- 「直前」は「もう一回」「続けて」の参照先。中断後は済/途中の印で残りが分かる。
- 「走りながらの技: 〜」は移動中だけ。方向→技の表引きを executor がやってモデルに渡す（v2→v3 で走行中の判断が 90%→99% になった要因）。
- 生成は `chat_format.render_state(RobotState)`、タイムラインからは `state_from_timeline(program, timeline, t)`。

**コントロール側の責務（決定）**: 状態ブロックは「今」ではなく **LLM の返答が届く時刻の予測状態**で作る
（`state_from_timeline(..., elapsed_s + 推論時間)`）。緊急停止は LLM を通さず制御側で処理する。
履歴は 5 ターンに絞る（伸びは無視）。

## 3. 履歴の持ち方: 生の書き足し

Qwen3 のテンプレートは過去 assistant ターンの `<think></think>` を消して描き直すので、そのまま使うと
毎ターン接頭辞が変わる。代わりに生成テキストをそのまま繋ぐ（`chat_format.conversation_text`）。
trainer / eval / 実機クライアントが同じ関数を使い、学習時の loss 境界 = 推論時の境界。

## 4. 走りながらの技（文法・コンパイラ）

- `flip` に `"running": true`。直前の `move` の終わりで止まらずに発火し、技で終わる（余計に走らない）。
- 種類は進行方向で決まる（ポリシーの `_select_motion_for_direction`）: 前系 → 前方回転、後系 → バク転、左 → 左側転、右 → 右側転。fast では不可（上限 1.0 m/s）。
- 合わない技・fast 後・move 以外の後 → コンパイラが置き換える（`normalise_running_flips`、adjustments に記録）。executor にエラーは届かない。
- 走りながらの前方回転は連続すると失敗（2回目 0.38）→ `count` を 1 に切る。返事で言う。
- `compile_program(..., context=今の move, resume=True)` で insert をコンパイル。
- calibrate: 走りながらの成功率 backflip 0.995 / sideflip 0.95–1.0 / frontflip 0.68。着地直後の停止指令で転倒は増えない（recover 0 = 1.5）。

## 5. データ

- 行の形を `turns`（user/assistant 交互）に統一。単発も 1 ターンの会話。user には `content`（状態ブロック込み）と `state`（チェック用データ）。
- `dialogues.py` の 15 シナリオで 2〜3 ターンを生成: repeat / interrupt / resume / insert（技・旋回・速すぎ）/ insert_propose / append / correction / 雑談中 / status / stopped / jump提案 / 走りながら提案 / running_fast。
- 判断はシナリオ・文法・能力表から、言い回しは表から。単発と同じ分離。
- 7,360 会話 / 10,140 ターン（単発 5,000 + 対話 2,500）。`check_dataset.py` 10 項目、往復検証、GBNF 全通過。
- 手書きテスト `handwritten_eval.py`（35 件、フレーズバンク外の言い回し）。関西弁は要追加。

## 6. 学習・評価・配布

- `train_sft.py`: 全 assistant ターンに loss、ターンごとに別トークン化。batch 4 × accum 4、max_len 2304。
- `eval_model.py`: ターン単位、前ターンは正解で埋める。action 正解率・「止めるか否か」を追加。
- `export_gguf.py`: マージ → f16 → Q4_K_M（1.1 GB）→ llama-completion + GBNF で確認。llama.cpp は `/home/tak/isaacsim/llama.cpp`。

## 7. 結果（v3、Qwen3-1.7B DoRA、102 分）

| | 生成 eval（928 ターン） | 手書き（55 ターン） |
|---|---|---|
| action 正解 | 98.4% | 94.5% |
| action + program 一致 | 98.1% | 89.1% |
| 止めるか否か | 91.9% | 83.3% |
| 形式 / 断る技の漏れ | 100% / 0 | 100% / 0 |

走行中の判断（insert・提案・速すぎ）は 98.5–100%。残るのは待機中の一文に方向と合わない
走りながらの技が混ざるケース（72.7%、コンパイラが置き換えるので実機では動く）と、語彙の汎化。
学習側の改善は後回し（決定）。

## 8. 残り

Jetson での計測（CUDA ビルド、GPU/CPU の取り合いは後で）、実機 executor（insert/append/cancel、
状態報告、技の発火は実測速度 < 0.3 m/s、cancel は次の安全点）、生成プログラムの sim end-to-end。
