# VLFM Nav 概要設計

Go2 で VLFM (Vision-Language Frontier Maps, `doc/papers/VLFM_*.md`) 方式の
Zero-Shot 探索を行うシステムの概要設計。低レベル歩行は Gym タスク
`Go2-Blind-GRU-Phase4` のブラインドポリシー(段差・壁をある程度越えられる)を使い、
その上に ROS 2 のマッピング/ナビゲーション/VLFM 層を載せる。
将来 Anaguma (tsubame_robot) へ移植することを前提に、ロボット固有部を分離する。

## 1. 全体アーキテクチャ

```
┌────────────────────────────────────────────────────────┐
│ ロボット層 (robot-specific)                              │
│  Sim:  Isaac Lab + play_ros2.py                         │
│        - Phase4 ポリシー実行、/cmd_vel 購読               │
│        - LowState / PointCloud2 / RGB / clock publish    │
│  実機: unitree_ros2 + deploy C++ (cmd_vel DDS購読を追加)  │
└──────────────┬─────────────────────────────────────────┘
               │ ロボット抽象コントラクト (§3)
               ▼
┌────────────────────────────────────────────────────────┐
│ オドメトリ (robot-specific)                              │
│  Go2:     go2_odometry (脚運動学 + IMU InEKF)            │
│  Anaguma: rko_lio (LiDAR-inertial、既存)                 │
│        → odom→base_link TF, /odometry/filtered          │
└──────────────┬─────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────┐
│ マッピング/ナビ層 (robot-agnostic, パラメータのみ差替)     │
│  pointcloud_to_laserscan: 点群を base 高さの z バンドで   │
│    切り出して /scan へ                                   │
│  slam_toolbox: /scan + odom→base_link → /map, map→odom  │
│  Nav2: /map + /scan → 経路計画 → /cmd_vel               │
└──────────────┬─────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────┐
│ VLFM Core (robot-agnostic, 自作)                        │
│  Perception: /map, RGB, TF 購読                          │
│  Frontier:   探索済み/未踏の境界検出・クラスタリング        │
│  ValueMap:   VLM (SigLIP 2) スコアで意味マップ            │
│  Decision:   探索継続 / 接近 / 終了                       │
│  NavBridge:  nav2_simple_commander で goToPose()         │
└────────────────────────────────────────────────────────┘
```

## 2. パッケージ構成(3層)

VLFM/Nav 層はロボット固有型 (unitree_go 等) を一切 import しない。

```
unitree_rl_lab/
├── ros2/                       # ament パッケージ置き場 (colcon は外の ws で実行)
│   ├── patches/                # 依存リポジトリ用パッチ (適用手順は同 README)
│   ├── vlfm_nav/               # [共通] VLFM Core ノード (ament_python)
│   ├── vlfm_nav_bringup/       # [共通] slam_toolbox / Nav2 / p2l の launch + 共通 params
│   └── go2_nav_bringup/        # [Go2] config yaml + launch (go2_odometry 起動、remap)
├── scripts/ros2/play_ros2.py   # [Go2/sim] Isaac 側ブリッジ (env_isaaclab 内で実行)
└── deploy/robots/go2/          # [Go2/実機] cmd_vel の DDS 購読を追加予定
```

colcon ワークスペースは `~/isaacsim/go2_nav_ws`。`src/` には上記 `ros2/` の
パッケージと `go2_odometry`、依存リポジトリを symlink/clone して束ねる。

## 3. ロボット抽象コントラクト

ロボット層が提供する義務。VLFM/Nav 層はこれだけを前提にする。

| 提供物 | 型/形式 | Go2 | Anaguma |
|---|---|---|---|
| `odom→base_link` TF + オドメトリ | tf2 + `nav_msgs/Odometry` | **rko_lio**(`go2_sim.launch.py` デフォルト。実機も同構成予定)/ `--gt_odom`=sim直出し(デバッグ用)※ | rko_lio (既存) |
| LiDAR 点群 | `sensor_msgs/PointCloud2` | sim bridge / L1 実機 | livox driver (既存) |
| RGB カメラ | `sensor_msgs/Image` + `CameraInfo` | sim bridge / 実機 | 実機カメラ |
| 速度コマンド受理 | `geometry_msgs/Twist` on `/cmd_vel` | sim: play_ros2.py が購読 / 実機: deploy C++ に DDS 購読追加 | twist→joy 変換ノード (§6) |
| ロボット config | yaml 1 枚 | go2.yaml | anaguma.yaml |

ロボット config yaml に載るもの(= ロボット別に変わるものはここに集約):
フレーム名、footprint、速度レンジ(ポリシー学習レンジで clamp)、
点群→/scan の z バンド、センサトピック名、カメラ FoV。

Nav2/slam_toolbox のパラメータは「共通 yaml + ロボット別 yaml で上書き」方式。
ロボット別に出るのは footprint・速度リミット・z バンド程度の想定。

## 4. 依存ライブラリの整理

### ROS 側 (システム Python 3.10)

| 種別 | もの | 入手 |
|---|---|---|
| apt | navigation2, nav2-bringup, nav2-simple-commander, slam-toolbox, pointcloud-to-laserscan, rmw-cyclonedds-cpp(実機のみ) | `ros-humble-*` |
| clone + patch | unitree_ros2 (unitree_go/unitree_api msgのみ), rosidl_dds (humble) | `ros2/patches/README.md` 参照 |
| clone + patch | invariant-ekf (inria fork) | 同上 |
| clone | unitree_description (inria fork), go2_odometry | — |
| clone + build flag | rko_lio (PRBonn) — `colcon build --packages-select rko_lio --cmake-args -DRKO_LIO_FETCH_CONTENT_DEPS=ON -DCMAKE_BUILD_TYPE=Release` | github.com/PRBonn/rko_lio |
| pip (システム) | pin (pinocchio, 導入済み), torch + transformers 5.x + torchaudio cu128 (SigLIP 2。§7.1) | — |

パッチ(upstream 更新時は再適用):
- `unitree_ros2-msg-build-deps.patch`: unitree_go/unitree_api の package.xml に
  `rosidl_generator_dds_idl` の build_depend を追加(colcon のビルド順保証)
- `invariant-ekf-cmake-colcon.patch`: cmake_minimum_required 3.22 へ /
  build_type を `cmake` へ(local_setup.bash "not found" 警告の解消)

### Isaac 側 (env_isaaclab venv, Python 3.11)

追加依存なし。ROS 通信は Isaac Sim 5.1 同梱の `isaacsim.ros2.bridge`
(py3.11 ビルドの Humble ライブラリ) を使う。**システム ROS を source しない**。
同梱ライブラリは `LD_LIBRARY_PATH` 等をプロセス起動前に要求するため、
play_ros2.py は未設定なら環境を整えて自分自身を exec し直す(手動設定は不要)。

### 環境の分離

- ROS シェル: `go2ros` (~/.bashrc の関数。venv を抜けて Humble + go2_nav_ws を source)
- Isaac シェル: デフォルト (全シェルが env_isaaclab venv で起動する)
- 通信条件は同一 `ROS_DOMAIN_ID` (未設定=0) と同一 RMW (sim は FastDDS 同士)。
  tsubame の ROS ノードと同居させる場合はドメイン分離を検討。

### VLM の分離

VLFM Core と VLM 推論 (SigLIP 2) は**プロセス**分離。重い依存を隔離し、推論が
sim ループを止めないようにする。venv は分けない — Go2 の `~/isaacsim/env_isaaclab` と
Anaguma の `~/tsubame/env_isaaclab` が同一構成 (py3.11.15 / torch 2.7.0+cu128 /
isaacsim.ros2.bridge 同梱) なので、両方に transformers を入れるだけで環境差が出ない。
VLM ノードは `play_ros2.py` と同じ再 exec で Isaac 同梱 rclpy を掴む (この 10 行は
共有モジュールに切り出すこと)。実機オンボードへ載せる段になったら ONNX 書き出しを検討。

## 5. Sim / 実機の差分

| | Sim | 実機 |
|---|---|---|
| 関節/IMU | play_ros2.py が標準msg(/sim/joint_states, /sim/imu)を publish。robot_state_publisher は /sim/joint_states を直接購読 | 実機ブリングアップ時に /lowstate→/joint_states 変換を用意(unitree_go msg が py3.10 側で必要になるのはこのとき) |
| 点群 | RollingLivoxSensor (velocity_env_cfg_mid360.py の資産) から publish | L1/MID-360 ドライバ |
| /cmd_vel | play_ros2.py が購読しポリシーの command term へ | deploy C++ が DDS (`rt/cmd_vel`) で購読 — 既存 HeightScanUpdater と同じ dds_wrapper パターン |
| clock | `/clock` publish, ROS 側 `use_sim_time: true` | 実時間 |

sim/実機で rko_lio・slam_toolbox・Nav2・VLFM は無改造で共用する。

## 6. Anaguma 移植

やることは tsubame_robot 側に **アダプタ 1 パッケージを足すだけ** にする。

1. `anaguma_nav_bringup` を tsubame_robot に新設:
   - `config/anaguma.yaml`(footprint・速度レンジ・フレーム・z バンド)
   - `launch/anaguma_nav.launch.py`(rko_lio / livox driver / 共通 nav_stack.launch.py を include)
   - `twist_to_joy` ノード: `/cmd_vel` → `/joy`。anaguma_main は仮想 Joy
     (キーボード teleop も同じ経路) で動くため、この 1 ノードで
     コントラクトを満たせる。teleop と VLFM の調停 (mux) もここで行う
2. `vlfm_nav` / `vlfm_nav_bringup` を tsubame の ws (`~/tsubame/ros2_ws/src`) に symlink
3. オドメトリは rko_lio がそのまま `odom→base_link` を満たす(新規開発なし)
4. unitree 系依存 (unitree_go, go2_odometry, invariant-ekf...) は
   `go2_nav_bringup` 以下に閉じているため tsubame ws には不要

### オドメトリの実測メモ(2026-09-20)

※ 脚運動学 InEKF(go2_odometry)は短時間走行では機能したが、壁接触時の足滑りで
ヨーが飛び、数分を超える探索で slam_toolbox の探索窓を超えて地図がフォークした。
実機計画は Go2(height-mapスタック)/Anaguma とも RKO-LIO のため、2026-09-24 に
InEKF 経路と /lowstate 合成チェーンを削除し RKO-LIO に一本化(git 履歴に実装あり)。
2026-09-24: 本物の rko_lio を sim に統合済み(`odom:=lio`、点群は生/バンドの2系統に分離)。
実測誤差 1.9cm / 1.0°(2.87m歩行)。この構成でフロンティア探索 17/19 ゴール・
カバレッジ99.3%を確認。正解地図との比較は `gt_map_compare` ノード(/gt_map)。

2026-09-29 追記(rko_lio_go2.yaml に反映済み):
- `initialization_phase: false`(sim限定)— 初期化はスキャン2枚間の~3 IMU
  サンプル+静止仮定で行われ、launch起動のCPUスパイクと重なると壊滅的な
  バイアス推定(実測 z 7.88 m/s²)→地図崩壊。sim の IMU は真のバイアスが
  ゼロなので推定不要。実機(Anaguma含む)は true のままにする
- スループット: 10Hz 入力に対し処理 3.8Hz(2→6スレッドで不変=並列化が
  効かない)。処理落ち中の移動は変位を最大1/3取りこぼし(実測 GT比0.66)、
  ズレ登録が地図に幻の占有スペックルを撒く(足元に落ちると出発セルが
  inscribed になり全ゴール計画不能)。対策: `double_downsample: true`・
  `max_range: 15`・初回スピン減速(0.9→0.45 rad/s)。効果検証中

## 7. MLモデル構成(2026-09-20 確定 / 2026-09-30 改訂)

**SigLIP 2 (NaFlex) 1本**で価値マップと目標物検出を兼ねる。論文の Grounding DINO /
Mobile-SAM / ZoeDepth は載せない。深度カメラも使わない。

2026-09-30 の改訂は **モデル名・画像と地図の紐づけ方・スコアの尺度** の3点のみ。
「1モデルで両方兼ねる / 検出器なし / 深度カメラなし」という 09-20 の構成は維持する。

### 7.1 モデル

| | |
|---|---|
| チェックポイント | `google/siglip2-so400m-patch16-naflex` |
| 推論基盤 | PyTorch + transformers 5.x。既存の `env_isaaclab` に `pip install` するだけ |
| VRAM | 重み ~1.6GB + CUDA コンテキスト ~0.5GB = 約 2.2GB。Isaac(~6-8GB)と 16GB GPU に同居 |
| プロセス | `play_ros2.py` と同じ再 exec で Isaac 同梱 rclpy を掴む**別プロセス** |

BLIP-2 ITM から乗り換えた理由: VRAM が 4GB → 1.6GB(設計時の見積もりより GPU の
余裕が小さかった)、**sigmoid 出力がそのまま 0-1**(下記)、多言語。ただし BLIP-2 の
ITM はクロスアテンションのヘッドで dual-encoder のコサインより識別が鋭い
(VLFM が ITC でなく ITM を選んだ理由)。**価値スコアが方向で分離しない症状が出たら
ここを疑い、BLIP-2 ITM に戻す**。ITM も match/no-match の2クラスなので 0-1 が出る。
差し替えは `value_map.py` のスコアラー境界の内側で完結する。

**`torchaudio` は cu128 ビルドを入れること**(2026-10-02)。transformers 5.x は
`loss_utils.py` で `torchaudio` を import し、PyPI 既定の `torchaudio 2.7.0`(cu126
ビルド)は `torch 2.7.0+cu128` との不一致を検出して例外を投げる。その巻き添えで
`Siglip2Model` の import が落ちる。

```bash
pip install --force-reinstall --no-deps torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

一度 `transformers==4.57.6` へのダウングレードで回避しようとしたが、**それは
`scripts/download_paper.py` を壊す**(`marker-pdf` が `transformers>=5.12.1` を要求)。
不一致そのものを直すのが正しく、ダウングレードは不要。`--force-reinstall --no-deps`
なのは、バージョン番号が同じで pip が差し替えないのと、`torch` を巻き込まないため。

Isaac の venv に `pip install -U` を使わないこと。`networkx==3.3` / `sympy==1.13.3` は
`isaacsim-core` が固定しており、`-U` は巻き添えで上げてしまう。

### 7.2 スコアの尺度: sigmoid で 0-1

SigLIP は学習時からペア単位の二値判定(Sigmoid loss)なので、比較候補の集合なしに
確率が出る。

```
p = sigmoid(logit_scale * cos + logit_bias)   # scale/bias はチェックポイント同梱
```

CLIP 系のコサインは softmax で候補と競わせないと確率にならないが、SigLIP は単独で
0-1 を返す。**ただし絶対値は極端に低い**(2026-09-30 実測: 画面いっぱいのベッドで
帯 0.39、全体画像 0.048。設計当初に書いた「閾値 0.5〜0.7」は 1 桁違いだった)。

- **検出**は絶対閾値でよい。分離が 111 倍あるので **0.1 前後**に置ける
  (該当 0.390 / 非該当 0.0035)
- **価値マップ**はダイナミックレンジが狭すぎるので、**パーセンタイル正規化が要る**。
  生の値をセルに書き、`score()` で取り出すときに正規化する(走行中に正規化パラメータが
  変わっても過去に塗った値と整合が取れる)

### 7.3 プロンプト(起動引数で1回だけ)

```bash
ros2 launch vlfm_nav_bringup vlm.launch.py target:=toilet
```

テキスト側は起動時に1回エンコードしてキャッシュする。プロンプトは2本持つ —
画像側との内積は行列積1回なので**2本目のコストはゼロ**。

| 用途 | 文言 | 根拠 |
|---|---|---|
| 検出(帯) | `This is a photo of a {target}.` | 実測最良。下表 |
| 価値(帯) | 同上(暫定) | VLFM の文言は SigLIP では 40 分の 1 |

**2026-09-30 実測(kujiale, ベッドが画面いっぱいのフレーム / 写っていないフレーム)**

| テンプレート | 全体 | 帯の最大 | 非該当フレームの帯 |
|---|---|---|---|
| `Seems like there is a {} ahead.`(VLFM) | 0.0095 | 0.0639 | 0.0042 |
| **`This is a photo of a {}.`** | 0.0484 | **0.3898** | 0.0035 |
| `a photo of a {}` | 0.0009 | 0.0144 | 0.0001 |
| `{}` | 0.0180 | 0.0546 | 0.0002 |
| `a bedroom with a {}` | 0.0192 | 0.0116 | 0.0002 |
| `an interior rendering of a {}` | **0.5071** | 0.1065 | 0.0130 |

- **帯は全体画像の 8 倍**のスコアを出す(0.390 vs 0.048)。4 分割の設計はここで裏打ちされた
- **VLFM の `Seems like there is a X ahead.` は使わない**。BLIP-2 ITM 向けの文言で、SigLIP では最良比 40 分の 1
- `an interior rendering of a {}` が全体画像で突出するのは、Isaac のレンダが学習分布(ウェブ写真)から外れているのを文言で埋めているため。ただし帯では逆に弱く、対照(`elephant`)も浮く。**全体画像を使うなら**こちら
- 対照の `elephant` は良いテンプレートで全フレーム 0.0000。**モデルの識別は健全で、BLIP-2 への切り替えは不要**
**2026-10-01 全走行での検証**(kujiale 25 分 / 2,465 フレーム、`This is a photo of a {}.`、
帯 4 分割、閾値 0.1)

| 対象 | 帯 max | p99 | >0.1 枚数 | 区間数 | 事前の出現情報 |
|---|---|---|---|---|---|
| いす | 0.712 | 0.311 | 156 (6.3%) | 21 | 複数個・頻出 |
| 机 | 0.762 | 0.252 | 59 (2.4%) | 10 | 複数個・頻出 |
| ベッド | 0.461 | 0.297 | 277 (11.2%) | 12 | 複数個・頻出 |
| シャンデリア | 0.446 | 0.071 | **15 (0.6%)** | 4 | 1 個・頻出 ← **不一致** |
| ソファ | 0.613 | 0.314 | 94 (3.8%) | 7 | 1 個・1 期間 |
| 洗濯機 | **0.982** | 0.768 | 72 (2.9%) | 3 | 1 個・1 期間 |
| 冷蔵庫 | 0.682 | 0.320 | 41 (1.7%) | 5 | 1 個・1 期間 |
| キッチン | 0.220 | 0.074 | **10 (0.4%)** | 3 | 1 個・1 期間 ← **不一致** |
| **便器** | **0.892** | 0.395 | 94 (3.8%) | 7 | 1 個・1 期間 |
| **象(対照)** | **0.011** | 0.002 | **0** | — | 存在しない |

- **対照の象は 1 枚も閾値を超えない。** モデルの識別は健全で、BLIP-2 への切り替えは不要
- **閾値 0.1 は妥当。** 実在する物体は出現頻度に見合った枚数を拾う
- **便器 0.892**、上位フレームは目視で本物の便器(浴室、鏡とタイル床)。M5c の前提が立った
- 「複数個・頻出」(いす 21 区間 / 机 10 区間)と「1 個・1 期間」(洗濯機 3 / 冷蔵庫 5)が
  区間数にそのまま出る

**不一致 2 件、原因は別物**

- **キッチン(0.4%)は「物体」でなく「部屋の種類」。** 横 1/4 に切った帯は「カウンターの
  断片」で、`This is a photo of a kitchen.` が期待する広角の全景にならない。**帯による
  検出はオブジェクトには効くがシーン概念には効かない** → §7.4 の運用参照
- **シャンデリア(0.6%)は語彙の疑い。** `ceiling light` / `pendant lamp` の方が当たる
  可能性。bag があるので再生だけで検証できる(未実施)

`processor(text=..., padding="max_length")` は**必須**。SigLIP は固定長64トークンの
パディングで学習されており、既定のパディングではスコアが狂う。

### 7.4 1観測の処理

```
0.5m 移動 or 0.35rad 回転(上限 2Hz)
   ↓
768x384 を縦4分割 → 192x384 の帯4枚
   ↓
SigLIP 2 前向き 1回(バッチ4)
   ↓  i_emb @ t_emb.T → sigmoid → (4帯, 2プロンプト) すべて 0-1
   ├─ 価値: 各帯のスコアを、対応する 22.5 度の小扇形に塗る
   └─ 検出: 閾値を超えた帯の重み付き重心 → 方位 → 壁までレイキャスト → 目標点
```

**全体画像も 1 枚バッチに入れる(2026-10-01 改訂)。** 「帯だけで足りる」としていたが、
全走行の実測で用途が割れた。小部屋を占める物体は全体画像でも強く出る(便器 0.800、
冷蔵庫 0.597、洗濯機 0.416)が、広い部屋の物体は帯が圧倒的(いす 全体 0.081 / 帯 0.712、
ベッド 0.037 / 0.461)。そして**シーン概念(「キッチン」のような部屋の種類)は帯では
拾えない** — 価値マップが読みたいのはまさにこれ。テキスト側はキャッシュ済みで内積 1 回
なので、5 枚目のコストはほぼゼロ。

- **検出** = 帯 4 枚の max(オブジェクト)
- **価値** = 全体画像(部屋の種類・文脈)、4 つの小扇形には各帯のスコアを塗る

NaFlex はアスペクト比を保つ(`patch_size=16`, `max_num_patches=256` 既定)。
192x384 の帯は約 176x352(11x22=242パッチ)に整形され、**歪みが入らない**。
`get_image_features(pixel_values, pixel_attention_mask, spatial_shapes)` を取るので
`**inputs` で展開する。

### 7.5 画像と地図の紐づけ

**カメラの位置と向きは TF で分かるので、見えている範囲は地図上の扇形として描ける。**
VLFM は扇形全体に単一のスカラーを塗るが、ここでは **90 度を4分割し 22.5 度ずつ
別のスコアを塗る**(同じ計算量で方位分解能が4倍)。

| | 量 | 紐づけ | 深度の役割 | グリッド |
|---|---|---|---|---|
| 価値 | 帯ごとに1スカラー | 対応する小扇形のセル全部に塗る | 遮蔽でマスクを削る | 粗く 0.2m |
| 定位 | 帯の重み付き重心 | 方位1本 → 壁セル | 距離 | SLAM と同じ |

**可視レイキャストが両者の共通土台。** カメラ位置から HFOV 内に1度刻みで光線を飛ばし、
**占有または未知**に当たるまで進んで方位ごとの `r_max` を得る。深度画像ではなく
`/map` を舐める — 論文(VLFM/CoW)は Habitat のエージェントで LiDAR も信頼できる
大域地図も持っていないから深度に頼ったが、我々は両方持っている。`/map` は複数
スキャンの累積なので穴が少なく、最初から map 系にいる。目標点の距離も同じ
`r_max` を引くだけで出る(追加計算ゼロ)。

**この遮蔽カットが無いと壁の向こうまで価値を塗る。** 「隣の部屋が浴室っぽい」が
染み出して frontier 選択が壊れるため省略できない。

信頼度とブレンド(VLFM IV-B):

```
c_curr = cos(dbearing / (HFOV/2) * pi/2) ** 2      # 光軸上=1、視野端=0
v_new  = (c_curr*s + c_prev*v_prev) / (c_curr + c_prev)
c_new  = (c_curr**2 + c_prev**2)    / (c_curr + c_prev)   # 高い方に寄る
```

frontier のスコアは**そのクラスタのゴール点1つ**の周囲 0.5m を信頼度で重み付け平均
して読む(評価した場所と行く場所を一致させるため。クラスタ全体の平均や最大は取らない)。
レイキャストは未知セルで止まるので frontier セル自体が扇形の外縁として塗られるが、
LiDAR は 360 度でカメラは 90 度なので**一度も見ていない frontier** がある。信頼度が
閾値未満なら中立値を返し、距離だけで選ばれる(= 現状の nearest-first に degrade)。

### 7.6 定位と目標ナビゲーション

縦に割るのは**縦のピクセルを使わないから**。地面に投影するので高さは要らず、これが
深度カメラ不使用が成立する理由でもある(縦を使う瞬間に深度が要る)。

帯どうしの softmax は使わない。合計が1になるため**目標が写っていなくても必ず
どれかが最大になる**。sigmoid 出力に絶対閾値を当てる。

ゴールは**フロンティア探索と同一の形**にそろえる。両方が同じゴール配置コードと
同じ到達判定を通るので、片方を直せば両方直る。

| モード | ゴール座標 | 到達 |
|---|---|---|
| 探索 | フロンティアセル | 0.4m |
| 目標 | 壁セル(= 目標物) | 0.4m |

```
EXPLORE --(指示)--> SEARCH --(帯スコアが閾値超)--> GOAL_NAV --(0.4m)--> DONE
```

### 7.7 ループ閉じ込み対策

**理論上の懸念ではない。** グローバルコストマップで同じことが起きた(§ nav2_common.yaml
のコメント参照。`map->odom` の補正のたびに古いマークが誤った位置に残り、自由空間
52m^2 を lethal に塗り潰してロボットを幻の 1.5m^2 のポケットに封じ込めた)。価値マップと
目標点は構造的に同じもので、対策なしなら同じ事故を再現する。

| 対象 | 保持形式 | 補正が来たら | 理由 |
|---|---|---|---|
| 価値マップ | map 系のラスタ | **捨てる** | 順序付けにしか使わない。数観測(2-5秒の歩行)で再構築される |
| 目標点 | **odom 系の点**。使う瞬間に map へ変換 | 自動的に正しくなる | ゴールそのもの。補正はロボットの直近軌跡全体に効くので、同じ軌跡で観測した点にも同じ補正が効くのが正しい |

odom 自体のドリフトは検出から到達まで数十秒なので無視できる(RKO-LIO 実測
1.9cm / 2.87m)。

グリッドは `/map` に追随させない(`/map` は動的にリサイズされる)。起動時に原点固定の
広いグリッドを1枚確保する: `60m x 60m / 0.2m = 300x300 x 2ch x float32 = 720KB`。

画像は sim タイムスタンプ付きで publish し、TF は**その時刻で引く**(`"now"` では
引かない)。`/clock` があるので sim では厳密にでき、実機で必要になるコードを最初から
書くことになる。

### 7.8 カメラ

| 項目 | 値 | 根拠 |
|---|---|---|
| 位置 | `(0.32, 0, 0.03)` from base | L1 は `radar_joint` で `(0.28945, 0, -0.046825)`。その少し前・上に置きドームが画角下端を遮らないようにする |
| ピッチ | **0 度(水平)** | **実機に合わせる**(2026-10-02)。本物の Go2 は前面カメラを鼻先の垂直面にフラットに付けており、sim で勝手に傾けると bag・閾値・プロンプトの測定値が実機とズレる。一時 +10 度(上向き)にしていた — カメラ高が 0.35m しかなく水平だと床が画面下半分を占めるため — が、実機一致を優先。水平にすると近距離の床も戻る(見え始めが約 1.2m → 0.70m) |
| HFOV | 90 度 | 4分割して 22.5 度ずつ。魚眼は扇形投影が破綻するため矩形 |
| 解像度 | 768 x 384 | NaFlex により 2:1 の縛りは無くなったが、帯がパッチ予算にちょうど収まり帯域も軽い |
| 親リンク | base (fixed) | Go2 の頭部はトランクと剛結。ネックジョイントが無い |
| 深度 | なし | § 7.5 |

Isaac Lab 側。画角は `width`/`height` ではなく焦点距離とアパーチャの比で決まる。
`vertical_aperture=None` にするとアスペクト比から正方ピクセルになるよう自動計算される。

```python
TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.32, 0.0, 0.03),
        rot=(1.0, 0.0, 0.0, 0.0),             # 水平(実機と同じ)
        convention="world",                   # forward +X, up +Z (URDF の rpy と同じ規約)
    ),
    data_types=["rgb"],
    spawn=PinholeCameraCfg(focal_length=10.4775, horizontal_aperture=20.955),
    width=768, height=384,
)
# HFOV = 2*atan(20.955 / (2*10.4775)) = 90 度
# VFOV = 2*atan(0.5) = 53.1 度 (vertical_aperture は自動)
```

ヘッドレスでカメラを使うには `--enable_cameras` が要る。`TiledCamera` を使うこと
(素の `Camera` より軽い)。

`go2.urdf` には `radar_joint` と同じ形で1ブロック足す:

```xml
<joint name="camera_joint" type="fixed">
  <origin xyz="0.32 0 0.03" rpy="0 0 0"/>
  <parent link="base"/><child link="camera"/>
</joint>
```

### 7.9 配信トピック

| トピック | 型 | 確認できること |
|---|---|---|
| `/vlfm/camera/image_raw` | sensor_msgs/Image | 生の視界 |
| `/vlfm/vlm/debug_image` | sensor_msgs/Image | 4帯の境界線と各スコア、検出マークを焼き込んだ画像 |
| `/vlfm/value_map` | nav_msgs/OccupancyGrid | 目標の方向が高くなっているか(0-1 を x100) |
| `/vlfm/value_confidence` | nav_msgs/OccupancyGrid | **壁の向こうに染み出していないか**。遮蔽カットの検証はこれでしかできない |
| `/vlfm/detection` | geometry_msgs/PoseStamped | 閾値を超えた検知 1 件。**ゴールではない** |
| `/vlfm/target` | geometry_msgs/PoseStamped | 確定した目標物の座標 = 実際に向かうゴール |
| `/vlfm/status` | std_msgs/String | モード遷移 |

**デバッグ画像は `<名前空間>/image_raw` + `<名前空間>/camera_info` の対で出すこと**
(2026-10-02 に判明、10-05 に A/B で確定)。Foxglove の Image パネルはこの形のものしか
候補に出さない。`/vlfm/vlm/debug_image` という名前では、メッセージが生カメラと完全に
同一(768x384 rgb8, step 2304, 同じ frame_id)で、トピックが advertise されていて、
配信レートも出ていても、ドロップダウンに現れない。

一度「名前ではなくパネルの設定が原因」と結論して名前を戻したが、戻した途端にまた
消えた。改名すると出る / 戻すと消える、が両方向で再現している。**フラットな名前に
"整理" し直さないこと。**

値と信頼度を2本出すのが重要 — 値だけでは「見た結果として低い」のか「まだ誰も
見ていない」のかが区別できない。`cv_bridge` は使わず、既存の `PointCloud2` と同じく
numpy 配列を `.tobytes()` して手で詰める。

**画像だけは RELIABLE QoS**(点群の `qos_profile_sensor_data` ではない)。768x384 rgb8 は
884 kB で DDS が多数の UDP に分割するため、BEST_EFFORT では断片が 1 つ欠けるとサンプル
ごと落ちる。実測 2026-09-30 で **70% が欠落**(bag に届いたのは 143 枚中 42 枚。同じ関数で
同時に publish している CameraInfo は小さく分割されないので全数到着)。5 Hz なら再送は
安く、ここでの欠落は次のスキャンが埋める古い点群と違って VLM の証拠そのものの穴になる。
なお受信側で `ros2 topic hz` を使うときは `--qos-reliability` を合わせること。

### 7.10 計算コストと実装量

| 1観測あたり | 時間 |
|---|---|
| SigLIP 2 前向き(4帯バッチ) | 20-30 ms(支配的) |
| 可視レイキャスト 90本 | 0.1 ms |
| 扇形マスクと信頼度 | 0.05 ms |
| ブレンド | 0.05 ms |
| 方位 → 目標点 | ~0 ms(レイキャスト再利用) |
| **紐づけ処理 合計** | **< 0.3 ms**(VLM の 1/50。最適化する意味はない) |

`decision.py` と `vlfm_node.py` の探索ロジックは**無変更**。09-20 に
`score(points_xy) -> [0,1]` の契約を切ってあるため、価値マップの差し込みは
`UniformValueMap`(M4 のプレースホルダ)の置き換えだけで済んだ。実装後はそのクラスも
不要になり削除 — `GridValueMap` はグリッド未受信の間ずっと全点 0 を返すので、VLM を
起動しない構成でも切り替えなしに nearest-first へ落ちる。

**テスト**: `ros2/vlfm_nav/test/test_value_map.py`(ROS もモデルも不要、25 項目)。
守っているのは「黙って壊れる」種類の失敗だけ — 扇形の左右反転、壁抜けの塗り、
退化した分布の引き伸ばし、ループ閉じ込み補正、距離とのトレードオフ。どれも例外も
ログもレートの異常も出さないので、ここで止めるしかない。

### 7.11 未解決事項

| 項目 | 内容 |
|---|---|
| **プランナのゴール到達性** | 起動直後のログで `failed to create plan with tolerance 0.30` が 4 連続して出るのを確認(2026-10-02)。ゴールがフロンティアセル/壁セルの上にあり、`inflation_radius` 0.35 に対し許容 0.3 では経路を終える自由セルが無い。`tolerance` を 0.5 に上げる案は**却下**(経路計画の合格距離を緩めたくない)。残る案は **Nav2 に渡すときだけゴールを手前に落とす** — 探索ノードが追うゴールはフロンティア/壁セルのまま、`NavigateToPose` に載せる座標だけを数十 cm 手前にする。未実装 |
| **起動直後のゴール乗り換え**(保留) | **原因は特定済み、対処は保留**(2026-10-02)。初回スピン直後、LiDAR は既に半径 12m を見終わっているのに `/map` は `map_update_interval: 1.0` で遅れて追いつく。探索ノードはその遅れた地図を見て「まだ未知」の場所をゴールに選び、1〜4 秒後に地図が追いついて `frontier consumed` で取り消す — **選んだ時点で実体の無いフロンティアを掴んでいる**。実測で最初の 7 ゴール中 4 つがこれで消え、ロボットは 18 秒間ほぼ原点から動かないまま、フロンティア数だけが 29 → 44 と増え続けた。判定も取消も正しく動いており、誤っているのは選択側。<br>試した対処と却下理由: **時間ゲート**(ゴール送出から N 秒は consumed 判定をしない) — 終盤の「到達できないフロンティアを潰す」作業まで一律に遅くなる。**移動距離ゲート**(N メートル動くまで判定しない) — 同上の懸念が残る。<br>本筋は選択側で、「複数回の地図更新をまたいで存続したフロンティアだけをゴールにする」あたり。実体のあるフロンティアは何度更新しても残るので終盤に影響せず、追いつき中の幻だけが落ちる。未実装 |
| ~~Isaac のレンダ負荷~~ | **解決(2026-09-30 実測)**。カメラのコストは **0.64 ms/step(ベースライン比 2.7%)**。ただし `cfg.sim.render_interval` を `GO2_CAM_HZ` に揃えることが必須 — 既定の 1 は env step あたり `decimation` 回描画し、6.4 ms/step を食っていた(`update_period` は描画を止めない)。`--enable_cameras` 自体は無料(43.2 → 42.8 step/s) |
| **sim がリアルタイムに届いていない** | **ROS スタック同居で 10.9 step/s まで落ちる**(2026-09-30 実測: `/clock` 1437 件 / 131.8 秒。sim 時間は 28.7 秒しか進まず **0.22x**)。Isaac 単体なら 43.2 step/s / 要件 50(0.86x)。M5a とは無関係の既存の状態。心当たりは CPU ガバナが `powersave` であること(Isaac が起動時に警告を出す)と MID-360 レイキャスタのコスト。MPPI を 20Hz → 10Hz に落とした経緯とも符合する |
| bag を録る世界 | **Kujiale でなければ意味がない**。explore の地形はマテリアルが無く壁が真っ黒で、VLM に見せても評価にならない(そもそも便器もソファも無い)。Kujiale は explore を継承しているのでカメラは付いている |
| ~~検出閾値~~ | **解決(2026-10-01)**。2,465 フレームで **0.1** が妥当と確認(対照の象 0 枚、実在物体は出現頻度どおり)。§7.3 |
| シャンデリアの語彙 | 1 個・頻出のはずが 15 枚(0.6%)しか拾えない。`ceiling light` / `pendant lamp` を bag 再生で試す。未実施 |
| `max_num_patches` | 既定 256 は固定384版より実効解像度が低い。足りなければ処理時キーワードで 512 等に上げる |

### 7.12 マイルストーン

| | 内容 | 検証 | VLM |
|---|---|---|---|
| M5a | カメラ追加 → publish → Foxglove 表示 → **bag 記録** | **FPS が落ちないか**(最大のリスク)、TF と画像の整合、自機の写り込み | なし |
| M5b | SigLIP 2 ノード。価値マップ + 信頼度 + debug 画像。frontier 選択に value を流す | 目標の方向が高くなるか。壁の向こうに染み出していないか | あり |
| M5c | 指示(起動引数)と GOAL_NAV モード | `target:=toilet` で実際に到達するか | あり |

M5a を単独で先にやるのは、Isaac のレンダリング負荷が全計画を壊しうる唯一の未知数
だから。ここで FPS が持たないなら解像度・レート・構成を先に見直す必要があり、VLM を
書いてから発覚すると手戻りが大きい。

そして M5a で録った bag があれば、**M5b の価値マップと定位の検証は Isaac 抜きで
何度でも回せる**。プロンプト・閾値・`max_num_patches`・モデルの差し替えはいずれも
試行回数が勝負で、同じ走行に対して条件だけ変えないと比較にならない。

```bash
ros2 bag record -o run01 /vlfm/camera/image_raw /tf /tf_static /map /clock
```

## 8. 探索ロジック(2026-09-29 時点)

正は `vlfm_nav/vlfm_node.py` 冒頭 docstring。設計原則と要点のみ記す。

```
WAIT → SPIN → SELECT ⇄ NAVIGATE → DONE
 ↑    (初回のみ)  ↑          |
 └ goal REJECTED └─ ESCAPE ←─┘ (スタック検知が唯一の入口)
```

- **ゴールは口実**: フロンティアを見に行くための点。道中のスキャンで
  ゴール近傍(0.9m)からフロンティアが消えたら到達前でも完了
  (`frontier consumed en route`)。到達判定(0.15m)は消えない
  フロンティア(窓越し等)へのフォールバック
- **時計で諦めない**: ゴールタイムアウト・進捗監視なし。行けないゴールは
  物理証拠で終わる — スタック(5秒/0.15m)→ESCAPE→動けなかった方向に
  接触障害物(赤点、寿命10分)→経路が塞がる→Nav2 abort
- **記憶は2種類、どちらも恒久**(2026-09-29):
  **ブラックリスト**=失敗。Nav2 abort 時のみ登録(出発セルが inscribed/lethal
  ならゴールの罪ではないので登録しない — vlfm が global costmap を購読して判定)。
  半径0.15mの円をゴール選択で除外。全滅が45秒続いたら直近180秒分を1回だけ恩赦。
  **訪問済み**=成功。到達したゴールは別リストに記録し、その0.9m近傍を
  **フロンティア検出自体からマスク**(境界セルごと消えるので、再配置ジッタで
  隣に湧き直すことがない — 消化不能フロンティア2点の永久往復対策)。
  道中消化で完了した場合は立っていないので記録なし。恩赦対象外
- **終了は「消化可能なフロンティアの枯渇」1本**(検出ゼロ×3 or 恩赦後も
  全滅)+ ESCAPE 3周全敗(物理的詰み)の別枠
- **NavBridge は世代番号でコールバックを検証**: preempt された旧ゴールの
  死亡通知を新ゴールの abort と誤認する連鎖(1.2秒周期で未挑戦ゴールを
  焼き続ける)を 2026-09-29 に修正
- 可視化 `/vlfm/markers`: シアン=フロンティア / 青=候補 / 緑=現在ゴール /
  赤=接触障害物 / オレンジ円盤=ブラックリスト圏(実半径)

### ポリシー追従性の実測(2026-09-28, ステップ応答・整形なし)

| 指令 | Phase4 (model_7300) | Phase1 (model_999) |
|---|---|---|
| vx 0.2〜0.5 | 0.00(完全停止) | ほぼ 1:1 |
| vx 0.8 | ~85% | ~99% |
| vy(横歩き) | 0.00(不能) | ~80% |
| wz 0.5(その場) | 0.04 rad/s(不動) | 0.46 |
| 歩行中の旋回 | 良好 | 良好 |

Phase4 のデッドバンドは後半カリキュラムの獲得癖。ブリッジの持ち上げ
デフォルト(0.8/0.9)は Phase4 用で、**Phase1 で走らせる場合は
`--min_walk_speed 0 --min_walk_wz 0` を付ける**。Phase1 は平地専用で
障害物に弱く、接触転倒→リセット→play 即終了(仕様)に注意。

## 9. 未決事項


- 2D SLAM (slam_toolbox) は段差踏破と原理的に相性が悪い(乗り越え時の
  ピッチで /scan が汚れる)。まず z バンドフィルタで様子見、破綻したら
  elevation map / 3D 系 (rko_lio + 別マッピング) を検討
- rko_lio のスループット不足(§6実測メモ)— double_downsample の効果検証。
  不足なら MPPI vx_max を地図品質に合わせて下げる
- 窓越し等「到達できるが消化できない」フロンティアは abort しないため
  ブラックリストされず、再訪が続き探索が終了しない可能性(設計上の
  既知トレードオフ、証拠が出たら対策)
- 出発点封鎖(inscribed)からの自律離脱手段がない(ESCAPE 入口は
  スタックのみ)— 恩赦→誤 DONE で終わるケースの扱い
- 実機 Go2 の RGB カメラ選定とキャリブレーション(LiDAR との外部パラメータ)
