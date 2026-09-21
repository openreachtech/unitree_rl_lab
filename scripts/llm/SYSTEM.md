# 全体設計: LLM を含めた実行系（2026-09-15）

実装状況: 1〜3 は実装済み（`scripts/llm/conductor.py`, `deploy/robots/go2/{include,src}/ProgramLink.*`、
偽コントローラ相手に UDP で検証済み）。4（MuJoCo 実演と settle の実測）と 5（Jetson）が残り。

前提（決定済み）: 既存の C++ コントローラ（`deploy/robots/go2`, `State_Multitask`）に統合する。
指令はコンソールにキーボードで文字入力。会話履歴はメモリ上、保存しない。MuJoCo（`unitree_mujoco`）でも動かす。

## 1. 構成

```
 ┌──────────────── conductor (Python, scripts/llm/conductor.py) ────────────────┐
 │  console thread   : 1 行入力 → 緊急語なら即 cancel（LLM を通さない）→ それ以外は LLM へ   │
 │  llm thread       : 予測状態で状態ブロック → llama-server (/completion, grammar, cache) │
 │                     → parse_output → action を queue に適用                          │
 │  executor loop 50Hz: queue（実行中 program + カーソル + insert スタック）を進め、        │
 │                     vx,vy,wz と flip/stance イベントを UDP で送る                       │
 │  receiver thread  : 制御側の状態パケット（FSM 状態、技/立ちの進行、手動停止、転倒）      │
 └───────────────▲───────────────────────────────┬──────────────────────────────────┘
                 │ UDP localhost  状態 50Hz        │ UDP  "VEL vx vy wz" 50Hz / "FLIP handspring" / "STANCE front|hind|off"
 ┌───────────────┴───────────────────────────────▼──────────────────────────────────┐
 │  go2_ctrl (C++, 既存)   State_Multitask + 新規 ProgramLink（受信スレッド ~150 行）      │
 │    velocity: ProgramLink が新鮮(<0.5s)なら network 目標、なければキーボード目標          │
 │    flip    : command_->request(...)（既存、ステップ境界で発火）  stance: handstand_->request/cancel │
 │    キーボード [space]=速度0 / [0]=Passive はそのまま生きる（手動の安全弁）                │
 └───────────────────────────────▲──────────────────────────────────────────────────┘
                                 │ DDS (unitree_sdk2)
                   実機 Go2  または  unitree_mujoco（interface lo）
 ┌────────────── llama-server (llama.cpp, 別プロセス) ──────────────┐
 │  model-Q4_K_M.gguf + output.gbnf、cache_prompt で KV 保持、temp 0   │
 └──────────────────────────────────────────────────────────────────┘
```

**なぜ Python を挟むか**: コンパイラ・文法・状態ブロック・会話の書き足しは `unitree_rl_lab.program` と
`chat_format.py` に既にあり、学習データと同じコードで動かすことに価値がある。C++ に移植すると実装が
2 つになり、ズレた瞬間に「学習時と違う状態ブロック」を渡すことになる。C++ 側に足すのは
「速度目標と技のトリガを外から受ける口」だけ。FSM・ポリシースレッド・転倒ガード・キーボードは無変更。

**MuJoCo と実機の差は conductor から見えない**: どちらも go2_ctrl の向こう側。`--network lo` で
unitree_mujoco、実機は eth の interface。

## 2. conductor の中身

| 部品 | 中身 | 既存コードの再利用 |
|---|---|---|
| Queue | 実行中の `Timeline` + 経過時刻 + カーソル、`insert` スタック、`append` 待ち行列、直前の program と結果 | `compile_program(context=, resume=)` |
| 状態ブロック | `state_from_timeline(program, timeline, t_now + 推論時間の見込み)` — **返答が届く時刻の予測状態**で作る（決定）。推論時間は直近の実測の EMA | `chat_format.render_user_turn` |
| 会話 | user/assistant のテキスト列。**5 ターンで打ち切り**（決定）。`conversation_text` で生プロンプトを組む | `chat_format.conversation_text` |
| LLM 呼び出し | llama-server `/completion`: prompt（生テキスト）, grammar（`output.gbnf`）, `cache_prompt: true`, `temperature: 0`, stop `<|im_end|>` | `gbnf_grammar()` |
| action の適用 | none: 何もしない / cancel: 即停止・キュー空 / replace: 停止して差し替え / insert: 今の step を止めて挿入、終わったら残り時間から再開 / append: 待ち行列へ | — |
| 緊急語ホットパス | 「ストップ／止まって／止まれ／待って／やめて」を含む行は **LLM を待たず** cancel を送る。LLM にも渡す（返事は「止まるで」になる。状態は 中断 と書いて渡す） | — |
| 実行 | 50 Hz でタイムラインの現在セグメントを読み、`VEL` を毎ステップ送信。flip セグメントの先頭で `FLIP <motion>`、stance の入/切で `STANCE` | `Timeline.segments` / `events()` |

### 技の発火タイミング

実機には基準速度の実測が無い（`lowstate` は IMU の角速度のみ、`base_lin_vel` は観測に無い）。
なので「実測速度 < 0.3 m/s で発火」はできない。代わりに **静止からの技の前の settle を実機用に長めに**
（`CompilerConfig.pre_flip_settle_s`、fast からは 1.0 s 程度）し、MuJoCo で 2.0 m/s → 停止の減速時間を
測って値を決める。走りながらの技は速度指令を保ったまま発火するのでこの問題は無い。

### cancel の安全点

`中断不可`（技の窓・立ちの保持）の間に cancel が来たら、conductor は「窓が閉じたら速度 0」に予約する。
技は 1.0 s で終わるので待てる。立ちは `STANCE off` で降ろしてから停止（降りは 4 割転ぶが、それでも
高い所から力を抜くより良い）。

### 手動との共存

キーボードの [space] は C++ 側で速度 0 にし、状態パケットに `manual_stop` を立てる。conductor は
それを見てキューを捨て、次の状態ブロックを「待機中 / 直前(中断)」にする。[0] Passive も同様。

## 3. C++ 側の追加（ProgramLink）

- `deploy/robots/go2/include/ProgramLink.h` / `src/ProgramLink.cpp`: UDP 受信スレッド、
  `std::atomic` の vx,vy,wz と最終受信時刻、`FLIP`/`STANCE` はキューに積んで `State_Multitask::run()` が
  既存の `command_->request` / `handstand_->request/cancel` に渡す（ステップ境界で発火する既存の流儀のまま）。
- `keyboard_velocity_commands`: ProgramLink が新鮮なら `target` を network 値に。それ以外は今のまま。
- 送信: 50 Hz で 1 行 `STATE fsm=Multitask flip=1,0.42 stance=0,-1 vel=0.98,0.00,0.00 manual_stop=0 fallen=0`。
- 文字列プロトコル（1 行 1 コマンド）。JSON 不要、`nlohmann/json` はあるが使わない。
- config.yaml: `program_link: {port: 7777, state_port: 7778, timeout_s: 0.5}`。

## 4. 起動手順（MuJoCo）

```bash
# 1. シム
cd ~/unitree/unitree_mujoco/simulate/build && ./unitree_mujoco          # interface: lo
# 2. LLM（すでにビルド済みの CPU 版。-c 4096 は 5 ターン分の余裕、--keep -1 でシステムプロンプトを固定）
~/isaacsim/llama.cpp/build/bin/llama-server -m logs/llm/gguf/v3/model-Q8_0.gguf \
    --port 8080 -t 16 -c 4096 --keep -1
# 3. コントローラ（FixStand → [6] Multitask まではキーボードで）
cd deploy/robots/go2/build && ./go2_ctrl --network lo
# 4. conductor
python scripts/llm/conductor.py --gguf-dir logs/llm/gguf/v3 --llm http://localhost:8080
> 前に3mくらい走ってから、そのままハンドスプリングして
```

LLM 抜きで実行系だけ確かめるときは `--llm` を外して `/fwd 3`、`/flip backflip running`、
`/stance front 5`、`/append <json>`、`/cancel`、`/state`。文法・コンパイラ・キューは同じものを通る。

### プロトコル（1 行 1 コマンド、UDP）

```
conductor -> 7777   VEL <vx> <vy> <wz>      50 Hz。これが鮮度のハートビートも兼ねる
                    FLIP <kind>             backflip|frontflip|sideflip_left|sideflip_right
                    STANCE front|hind|off
                    STOP                    即ゼロ＋キュー破棄
                    RESUME                  スペース停止ラッチの解除
7778 <- go2_ctrl    STATE fsm=Multitask link=1 flip=1,0.42,1.00 stance=0,-1.00,10.00
                          vel=0.98,0.00,0.00 manual_stop=0
```

技の名前は文法側（frontflip）と config 側（handspring）の両方を受ける。同じ 4 つの技で語彙が
違うだけなので、`State_Multitask::find_motion` が読み替える。

## 5. 順序

1. ProgramLink（C++）+ 速度だけ流す conductor の骨組み → MuJoCo で「前に3m」が動く
2. flip/stance イベント、insert/append/cancel のキュー、状態パケット
3. LLM 接続（llama-server）、予測状態、緊急語ホットパス、5 ターン履歴
4. MuJoCo で手書きテストの対話を実演 → settle の実機値決め
5. Jetson AGX Orin 64GB に全部載せる: llama-server（CUDA ビルド、全層 GPU）+ conductor + go2_ctrl。
   メモリは Q8_0 でも 2〜3 GB で問題なし（Q8_0 を既定、Q4_K_M は速度が要るときの予備）。
   ポリシーの ONNX は CPU（onnxruntime aarch64 CPU ビルド）で、LLM は GPU なので取り合いは主に
   メモリ帯域。50 Hz の制御ループのジッタを LLM 生成中に測ってから判断する。
