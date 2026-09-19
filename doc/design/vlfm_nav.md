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
│  ValueMap:   VLM (BLIP-2/CLIP) スコアで意味マップ         │
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
| `odom→base_link` TF + オドメトリ | tf2 + `nav_msgs/Odometry` | go2_odometry (InEKF) | rko_lio (既存) |
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
| pip (システム) | pin (pinocchio, 導入済み), torch + BLIP-2/CLIP 系 (VLM 実装時に確定) | — |

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

VLFM Core と VLM 推論 (BLIP-2/CLIP) はプロセス分離 (ROS service/action)。
重い依存を隔離し、将来推論だけ別マシン/別 env へ逃がせるようにする。

## 5. Sim / 実機の差分

| | Sim | 実機 |
|---|---|---|
| LowState | play_ros2.py が標準msg(/sim/joint_states, /sim/imu, /sim/foot_forces)を publish → go2_nav_bringup の `sim_lowstate_bridge`(py3.10側)が `/lowstate` に合成。unitree_go の Python バインディングは py3.10 ビルドで Isaac 同梱 rclpy (py3.11) から import できないため | unitree_ros2 |
| 点群 | RollingLivoxSensor (velocity_env_cfg_mid360.py の資産) から publish | L1/MID-360 ドライバ |
| /cmd_vel | play_ros2.py が購読しポリシーの command term へ | deploy C++ が DDS (`rt/cmd_vel`) で購読 — 既存 HeightScanUpdater と同じ dds_wrapper パターン |
| clock | `/clock` publish, ROS 側 `use_sim_time: true` | 実時間 |

sim/実機で go2_odometry・slam_toolbox・Nav2・VLFM は無改造で共用する
(そのために sim でも実機と同じ LowState 型で出す)。

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

## 7. 未決事項

- 2D SLAM (slam_toolbox) は段差踏破と原理的に相性が悪い(乗り越え時の
  ピッチで /scan が汚れる)。まず z バンドフィルタで様子見、破綻したら
  elevation map / 3D 系 (rko_lio + 別マッピング) を検討
- VLM の選定 (論文準拠なら BLIP-2 ITM)。GPU メモリと Isaac 併走の相性を実測して決める
- Nav2 controller: Go2 は全方向移動できるため holonomic 対応 (MPPI/DWB) を第一候補に
- VLFM の探索終了条件・対象物検出 (論文は Grounding DINO / Mobile-SAM) をどこまで再現するか
