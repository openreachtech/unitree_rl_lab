# Anaguma を unitree_mujoco でキーボード操作できるようにする手順

## 次回以降の使い方（構築はもう済んでいる。ここだけ読めば動く）

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab

# 1) 動かしたいポリシーを指定（go2 でも同じ。ONNX が無ければ自動で書き出す）
bash scripts/mujoco_load_policy.sh anaguma 2026-09-15_03-50-07 model_10100

# 2) 2窓で起動（古いプロセスは自動で落とす）
DISPLAY=:1 bash scripts/mujoco_launch.sh anaguma
```

操作は **`1`（起立）→ `Enter`（ポリシー）→ `b`（助走2.8m/s）→ 4〜5秒待つ → `j`（跳躍）**。
`Space` は停止。**転んだら mujoco のウィンドウで Backspace**（シムのリセット）→ `1` から。

`mujoco_load_policy.sh` は毎回次を自動で確認する。ここで警告が出たら動かさないこと。
- ONNX の入力次元 と deploy.yaml の観測合計 の一致（不一致は例外を出さずにヒープを読み越して壊れる）
- **ONNX の重みが指定した .pt と一致するか**（同名で中身の違う ONNX の事故を2回やっている）
- `keyboard_velocity_commands` があるか（無ければキーが policy に届かない）
- 助走コマンドの実際の値（`keyboard_vel_scale` 既定 0.5 のままだと学習帯の半分になる）

**別のポリシーを載せるときの条件**:
- 固有感覚のみ（47次元）であること。**height_scan を policy に入れて学習したものは載らない**
  （deploy C++ が height_scan を供給しない）
- 学習時に `--deploy-keyboard-commands` を付けてあること（deploy.yaml に
  `keyboard_velocity_commands` が出る）。忘れた場合は deploy.yaml の
  `velocity_commands:` ブロック名をリネームすれば救える
- **default 関節姿勢が変わるタスクを作った場合は `deploy/robots/<robot>/config/config.yaml` の
  FixStand の `qs` も直す**（起立姿勢はロボットではなくタスク設定に紐づく）

以下は構築時の記録。同じことを別の機体でやるとき、または上の確認で引っかかったときに読む。


2026-09-15 作成。Go2 でやっている「mujoco 本体 + C++ コントローラの2端末構成」を Anaguma でも
できるようにした作業の記録と、再現手順。**Anaguma 以外の機体を足すときもこの順で通るはず。**

ツバメ側（`tsubame_robot`、ROS2）にも Anaguma のキーボード操作はあるが、あれは別環境で
リポジトリの権限が要る。ここで作るのは **手元だけで完結する経路**。

## 何のために必要か

`scripts/mujoco_jump_eval.py`（40試行の自動評価）とは役割が違う。

| | 自動評価 (python) | この経路 (C++) |
|---|---|---|
| 経路 | mujoco を直接叩く | **実機と同じ C++ / DDS / Unitree SDK** |
| トリガ | プログラムが自動で打つ | **自分でキーボード** |
| 用途 | 候補の足切り（1本40秒） | 挙動を見る・実機前の確認 |

自動評価は速いが「go2_ctrl の C++ を通らない」ので、実機判断は従来どおりこちら。

## 全体像

```
端末1: unitree_mujoco     … 物理シミュ。DDS で LowState を配信し LowCmd を受ける
端末2: anaguma_ctrl       … ポリシー(ONNX)を回して関節指令を作る。キーボードを読む
```

2プロセスに分かれているのは実機の構成と同じだから。`anaguma_ctrl` は端末の stdin を
termios で直読みするので、**必ず端末ウィンドウの中で起動する**（デタッチするとキーが届かない）。

---

# A. 一度だけやる作業

## A-1. MJCF が Go2 と同じ並びかを確認する

ここが違うと以降すべて壊れるので最初に見る。**確認するのは2つの並び**。

```bash
cd /home/tanaka/isaacsim/unitree_mujoco/unitree_robots
grep -oE '<joint name="[^"]+"' anaguma/anaguma.xml | head -13   # 関節の定義順
grep -E "<motor" anaguma/anaguma.xml | head -14                 # アクチュエータ順
```

期待する結果（Anaguma は両方 Go2 と一致していた）:

- 関節の定義順 … `FL, FR, RL, RR` × (hip, thigh, calf)
- アクチュエータ順 … **`FR, FL, RR, RL`** × (hip, thigh, calf) ← SDK の並び

**この2つが食い違っているのは Go2 も同じ**なので、食い違い自体は正常。`go2.xml` と比べて
同じ構造ならコントローラの C++ はそのまま流用できる。違っていたら、そこを直すのが最初の仕事になる。

## A-2. 起動時の姿勢（spawn）を直す

**ここが今回いちばんハマった。** `unitree_mujoco` は `mj_resetData()` でリセットするので
**MJCF の `<keyframe>` を読まない**。つまり起動時は「全関節 0、胴体は `<body name="base">` の
`pos` の高さ」で置かれる。`go2.xml` には `home` キーフレームがあるが、あれも読まれていない。

Anaguma は `pos="0 0 0.40"` だったので、**全関節 0 のとき最も低い足が z = −0.044 m**、
つまり**床に 4.4 cm 埋まった状態で始まっていた**。起動直後に押し出されて跳ね、
`1` を押す前から崩れていた。これが「すっと立ち上がらずに転ぶ」の主因。

```bash
# 現状の確認（足の最下点を出す）
/home/tanaka/isaacsim/anaguma/venv/bin/python /home/tanaka/isaacsim/unitree_rl_lab/scripts/anaguma_fixstand_tune.py
```

`[spawn] base z=... 最下足 z=...` が負なら埋まっている。その分だけ base を上げる:

```bash
cd /home/tanaka/isaacsim/unitree_mujoco/unitree_robots/anaguma
cp -n anaguma.xml anaguma.xml.bak_before_spawn_height
sed -i 's|<body name="base" pos="0 0 0.40"|<body name="base" pos="0 0 0.45"|' anaguma.xml
```

`mujoco_jump_eval.py` は毎試行 `qpos` を自分で設定する（足が接地する高さを計算している）ので、
**この変更は自動評価の結果に影響しない。**

## A-3. コントローラを作る（go2 のコピー）

C++ は1行も書かない。`State_RLBase.cpp` は deploy.yaml を読んで動く汎用実装なので、
機体差は全部 config と deploy.yaml 側にある。

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab/deploy/robots
mkdir -p anaguma
cp -r go2/CMakeLists.txt go2/main.cpp go2/include go2/config anaguma/
mkdir -p anaguma/src && cp go2/src/State_RLBase.cpp anaguma/src/
sed -i 's/project(go2_controller)/project(anaguma_controller)/; s/add_executable(go2_ctrl main.cpp)/add_executable(anaguma_ctrl main.cpp)/' anaguma/CMakeLists.txt
sed -i 's/     Go2 Controller /     Anaguma Controller /' anaguma/main.cpp
```

`include/Types.h` は **go2 のまま使う**。中身は Unitree SDK の四足用 DDS 型
（`unitree::robot::go2::publisher::LowCmd` 等）の typedef で、12関節の四足なら共通。

## A-4. config.yaml を2箇所だけ直す

`deploy/robots/anaguma/config/config.yaml`。

**(1) ポリシーの置き場所**

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab/deploy/robots/anaguma
sed -i 's|policy_dir: ../../../logs/rsl_rl/unitree_go2_longjump_v1|policy_dir: ../../../logs/rsl_rl/anaguma_longjump_v1|' config/config.yaml
```

**(2) FixStand（`1` を押したときの起立動作）** ← Go2 の値では立てない

直す理由が3つある。

- **姿勢の符号が Go2 と逆。** Go2 は thigh 0.8 / calf −1.5 だが、Anaguma は
  thigh −0.25（前脚）/ +0.0517（後脚）、calf +0.44 / +0.446。学習時の
  `default_joint_pos` をそのまま使う。
- **ゲインが足りない。** 24.3 kg（Go2 の約2倍）。実測は下表。
- **中間の屈伸姿勢は入れない。** Go2 は「畳んでから伸ばす」2段構えだが、Anaguma の屈伸の
  符号を推測する根拠がないので、現在姿勢 → 立ち姿勢へ補間するだけにする。

`qs` に書く並びは **motor_cmd の順 = アクチュエータ順 = `FR, FL, RR, RL`**。
Go2 は4脚が同じ値だったのでこの差が出なかったが、Anaguma は前後で thigh/calf が違うので効く。

書き換え後（このまま貼れる）:

```yaml
  FixStand:
    kp: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    kd: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    ts: [0, 2.5]
    qs: [
      [],
      [0.0, -0.25, 0.44, 0.0, -0.25, 0.44, 0.0, 0.0517, 0.446, 0.0, 0.0517, 0.446],
    ]
```

ゲインは目で決めずに測った（`scripts/anaguma_fixstand_tune.py` が FixStand と同じ PD ランプを
python で再現する）。3秒後の胴体高と傾き:

| kp | kd | 胴体高 | 傾き |
|---|---|---|---|
| 80 | 3 | 0.349 m | **7.1 deg**（ぐらつく＝最初の失敗） |
| 150 | 8 | 0.361 m | 2.1 deg |
| **200** | **10** | **0.365 m** | **1.3 deg**（ramp 2.5 s なら 1.0 deg） |
| 250 | 12 | 0.368 m | 1.0 deg |
| 60 | 2 | 0.331 m | 5.7 deg（学習時のゲイン。保持用で起立用ではない） |

`qs[0]` は空 `[]` のままにする。`State_FixStand::enter()` が現在の指令値で埋めてくれる。

## A-5. ビルド

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab/deploy/robots/anaguma
mkdir -p build && cd build && cmake .. && make -j4
```

`anaguma_ctrl` ができる。

## A-6. ポリシーを置く

`anaguma_ctrl` は **`policy_dir` の直下の `exported/` しか見ない**（`param.h:95`、サブディレクトリは
探索しない）。なので直下に symlink を張る。

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab
RUN=2026-09-15_03-50-07                      # 使う run
CKPT=model_10100                             # 使うチェックポイント
ROOT=logs/rsl_rl/anaguma_longjump_v1

mkdir -p ${ROOT}/${RUN}/exported
cp -f eval_anaguma_cycle3/C3k_${RUN}_${CKPT}.onnx ${ROOT}/${RUN}/exported/policy.onnx
echo "${RUN}/${CKPT}  loaded $(date '+%F %T')" > ${ROOT}/${RUN}/exported/CURRENT_CANDIDATE.txt
ln -sfn ./${RUN}/exported ${ROOT}/exported
ln -sfn ./${RUN}/params   ${ROOT}/params
readlink ${ROOT}/exported ${ROOT}/params     # 指し先を必ず確認
```

ONNX が無ければ `play.py` で書き出す（終わったらループに入るので kill する）:

```bash
/home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Anaguma-LongJump-v1 --headless --num_envs 1 \
  --checkpoint $PWD/logs/rsl_rl/anaguma_longjump_v1/2026-09-15_03-50-07/model_10100.pt
```

**次元の検算を必ずやる。** 不一致でも例外が出ず、ヒープを読み越して即おかしな姿勢になる
（Go2 で実際に踏んだ）:

```bash
/home/tanaka/isaacsim/env_isaaclab/bin/python - <<'EOF'
import onnx, yaml
R="/home/tanaka/isaacsim/unitree_rl_lab/logs/rsl_rl/anaguma_longjump_v1"
g=onnx.load(f"{R}/exported/policy.onnx").graph
dim=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
obs=yaml.safe_load(open(f"{R}/params/deploy.yaml"))["observations"]
tot=sum(len(v["scale"])*(v.get("history_length",1) or 1) for v in obs.values() if isinstance(v,dict))
print("ONNX入力", dim, "/ deploy.yaml 観測合計", tot, "->", "OK" if dim[-1]==tot else "*** 不一致 ***")
print("keyboard_velocity_commands:", "あり" if "keyboard_velocity_commands" in obs else "*** 無し（キーが効かない）***")
EOF
```

## A-7. 助走コマンドのスケールを直す

`keyboard_velocity_commands` は **`keyboard_vel_scale`（既定 0.5）× `commands.base_velocity.ranges`**
で速度を作る（`State_RLBase.cpp:33`）。学習帯が U(2.8, 3.3) なら既定のままでは
`f` が 0.5 × 3.3 = **1.65 m/s** しか出ず、**学習帯の半分の動作点で見ることになる**。

```bash
cd /home/tanaka/isaacsim/unitree_rl_lab
DEP=logs/rsl_rl/anaguma_longjump_v1/params/deploy.yaml
cp -n $DEP ${DEP}.bak_before_vel_scale
sed -i '0,/^  base_velocity:$/{s/^  base_velocity:$/  base_velocity:\n    keyboard_vel_scale: 1.0/}' $DEP
grep -n keyboard_vel_scale $DEP
```

範囲が [2.8, 3.3] のように両方プラスだと、**`f` が上限 3.3、`b` が下限 2.8** になる
（`b` は本来後退キー）。自動評価は vx 2.8 なので、**比較するなら `b`**。

---

# B. 毎回やる作業

## B-1. 起動（NoMachine で画面に出す）

```bash
# 端末1: シミュレータ（更地シーン。既定の scene.xml は障害物あり）
export DISPLAY=:1
cd /home/tanaka/isaacsim/unitree_mujoco/simulate/build
setsid nohup ./unitree_mujoco -r anaguma -s scene_flat.xml > /tmp/mj_anaguma.log 2>&1 < /dev/null & disown

# 端末2: コントローラ（キー入力のため必ず端末ウィンドウの中で起動する）
export DISPLAY=:1
setsid nohup gnome-terminal --title="anaguma_ctrl" --geometry=110x30 -- \
  bash -c 'cd /home/tanaka/isaacsim/unitree_rl_lab/deploy/robots/anaguma/build && ./anaguma_ctrl --network lo; read' & disown
```

`DISPLAY` はデスクトップの番号。分からなければ `ps -ef | grep gnome-session` で確認（実績は `:1`）。

## B-2. 操作

**`anaguma_ctrl` のウィンドウにフォーカスして**打つ。

| キー | 動作 |
|---|---|
| `1` | 起立（FixStand）。2.5 秒かけて立ち姿勢へ |
| `Enter` | ポリシー起動（Velocity 状態へ） |
| `b` | 前進 2.8 m/s（レンジ下限。自動評価と同じ動作点） |
| `f` | 前進 3.3 m/s（レンジ上限） |
| **`j`** | **跳躍トリガ**（`keyboard.h:210`。速度キーは保持されたまま） |
| `Space` | **停止**（跳躍ではない。押した速度キーが全部クリアされる） |
| `0` | Passive（脱力） |

## B-3. シミュレータを再起動したらコントローラも入れ直す

FSM の状態がずれるため。停止は **PID 指定で**（理由は下の落とし穴を参照）:

```bash
ps -eo pid,args | grep -E "anaguma_ctrl|unitree_mujoco" | grep -v grep
kill -9 <pid> <pid>
```

## B-4. ポリシーの差し替え

A-6 の symlink と `policy.onnx` を張り替えるだけ。**最後に書き出した ONNX が
`exported/policy.onnx` に残る**ので、意図した候補で必ず上書きすること（同名で中身の違う
ファイルが残るのが事故のもと。`CURRENT_CANDIDATE.txt` に何を入れたか書いておく）。

---

# C. 踏んだ落とし穴（全部実際に踏んだ）

1. **spawn が床に埋まっていた**（A-2）。`unitree_mujoco` は keyframe を読まない。
   新しい機体を足したら最初にここを見る。
2. **FixStand のゲインが足りず、`1` で立たずに転ぶ**（A-4）。Go2 の値をそのまま使うと
   傾き 7 deg でぐらつく。**目で決めずに python で測る。**
3. **`joint_ids_map` の向きを間違えた。** `deploy.yaml` の `default_joint_pos` は
   **policy 順**で、`qs` に書くのは **SDK 順**。変換は
   `sdk[jmap[i]] = policy[i]`（逆向きにすると意味不明な姿勢になる）。
   検算方法: Go2 の deploy.yaml に同じ変換をかけて `[0, 0.8, -1.5] × 4` 相当になるか見る。
4. **跳躍キーは `j`。`Space` は停止。** 逆に覚えていて「跳ばない」と誤診した。
5. **`keyboard_vel_scale` の既定 0.5**（A-7）。半分の助走で測っていた。
6. **`pkill -f "anaguma_ctrl"` で自分のシェルを殺す。** 自分のコマンドラインに同じ文字列が
   入るため。`[a]naguma_ctrl` と書いても、同じコマンドの後半に素の文字列があると当たる。
   **プロセス停止は PID 指定で行う。**

---

# D. 実機挙動の調整（2026-09-15 夕、実際に動かして分かったこと）

## D-1. シミュレーションの刻みが Go2 の 2.5 倍粗かった

`anaguma.xml` が `<option timestep="0.005">`（200 Hz）を明示していた。**`go2.xml` は
`timestep` を書いていないので mujoco の既定 0.002（500 Hz）**で回っている。
接触が絡む剛い PD 制御を 5 ms 刻みで回すと震えるので、Go2 に合わせて 0.002 にした。

```bash
cd /home/tanaka/isaacsim/unitree_mujoco/unitree_robots/anaguma
cp -n anaguma.xml anaguma.xml.bak_before_timestep
sed -i 's/timestep="0.005"/timestep="0.002"/' anaguma.xml
```

**自動評価の結果が変わらないことを先に確認してから変えた**（刻みを変えると評価値が動く可能性があるので）:

| timestep | 飛距離 | 着地 | 成立 |
|---|---|---|---|
| 0.005（変更前） | 1.349 m | 92% | 95% |
| **0.002（変更後）** | **1.397 m** | 82% | 98% |

40試行のばらつきの範囲内で、ポリシーは刻みに対して頑健。**2 ms を採用。**

## D-2. FixStand のゲインは飽和していなかった

「プルプル震える」のを見てトルク飽和を疑ったが、`scripts/anaguma_fixstand_tune.py` で測ると
**kp 200 / kd 10 でもピークトルク 17 N·m（上限 70）で飽和率 0%**。震えの原因は
ゲインではなく D-1 の時間刻みだった。ゲイン表（3秒後、ramp 2.5 s）:

| kp | kd | 胴体高 | 傾き | ピークτ | 飽和率 |
|---|---|---|---|---|---|
| 60 | 2 | 0.330 | 5.7 deg | 21 N·m | 0% |
| 120 | 10 | 0.355 | 2.5 deg | 19 N·m | 0% |
| **200** | **10** | **0.365** | **1.0 deg** | **17 N·m** | **0%** |
| 250 | 10 | 0.368 | 0.8 deg | 17 N·m | 0% |

**例外**: kp 60 / kd 10 だけピーク 147 N·m・飽和率 28.5% になる（低 kp と高 kd の組み合わせで
共振する）。**kp と kd は片方だけ動かさないこと。**

## D-3. `j` で跳ばないときは助走が足りていない

自動評価は `--warmup 4.0`、つまり**コマンドを出してから4秒走らせてから**跳躍をトリガしている。
このポリシーは助走 2.6 m/s 前後で跳ぶよう学習しているので、`b` を押した直後に `j` を押しても
跳ばない。**`b` を押して 4〜5 秒、速度が乗ってから `j`。**

## D-4. 経路ごとの違いを潰した記録（どこが同じでどこが違うか）

C++ 経路と python 自動評価で挙動が違うときの切り分け。確認して**同じ**だったもの:

- MJCF の関節定義順（FL,FR,RL,RR）とアクチュエータ順（FR,FL,RR,RL）… go2.xml と同一構造
- `<sensor>` の並び … `FR_hip_pos, FR_thigh_pos, ...` の SDK 順で go2.xml と同一。
  `unitree_sdk2_bridge.h:183` の PD は `ctrl[i] = tau + kp*(q_des - sensordata[i]) + kd*(...)`
  と**アクチュエータ番号とセンサ番号を同一視している**ので、この2つの並びが崩れると壊れる
- ブリッジにトルク–速度曲線は入っていない（`unitree_sdk2_bridge.h.bak_before_tn_curve` との
  diff は空。以前の移植実験は revert 済み）
- FSM のループは 1 kHz（`CtrlFSM.h:80` の `dt = 0.001`）

違っていたのは D-1（時間刻み）と D-3（助走時間）だけだった。

## D-5. 社内ドキュメントに書かれている Anaguma と Go2 の違い

移植で効く数値はここにまとまっている（`reference_internal_docs_index` 参照）。

- **[AnagumaとGo2の違い](https://chiho-shared.openreach.tech/organization/v1/documents/sHpdOrph1jFssguNmic7YSLDY2rMXI)**
  - **base_link が前端にあり、CoM は約30cm後方（x = −0.295）。** Go2 は base_link 原点 ≈ CoM（x = +0.021）。
    → 報酬は `base_height_l2` ではなく `base_com_height_l2` を使う
  - 太もも 30.75cm（Go2 21.3）、ふくらはぎ 25.8cm（21.3）、Hip abd軸→HFE軸 6.25cm（9.55）
  - Isaac のソルバ反復: Anaguma 4/0、Go2 8/4
- **[Anagumaの性能限界：早く走らせる](https://chiho-shared.openreach.tech/organization/v1/documents/pRSMbzJC9SxjoEg2VyQpI3S4uj5rya)**
  - 5.66 m/s（Go2 5.31）。**関節速度上限 13.4 / 15.6 rad/s、平均 6.0、ピークで超過あり**
  - **トルク 70 N·m は大幅に余裕**（D-2 の実測とも一致）
  - 歩幅 1.0 m / 歩行周波数 5.8 Hz。律速は「脚を振り切る速度」
  - **注意**: mujoco には関節速度上限が無い（トルク–速度曲線も無い）。
    跳躍で mujoco 側が上限を超えて回している可能性があり、実機ではその分出ない
- **[Anagumaの性能限界：高くJumpさせる](https://chiho-shared.openreach.tech/organization/v1/documents/hm45EJMXZevtwIS0kQ0CeTWD9tG9wU)**
  - Try1 は「まったく Jump せず」→ **Phase3 から Resume して 60cm → 70cm → 1m**
- 他: Jump/Backflip/Sideflip の移植、倒立、側転2回転、ラフ地形、平地の綺麗な歩行

---

# E. 奮闘記（2026-09-15、半日でここまで来た経緯）

うまくいかなかった順に並べる。**どれも「原因はここだろう」という推測が外れて、実測で別の場所が出てきた。**

**1. 「Anagumaはmujocoで操作できない」と言い切った。** 嘘だった。社内にはツバメ環境
（`tsubame_robot` の `anaguma_main_entry.py` + `keyboard_teleop_entry.py`）に同じ機能が既にあり、
しかもそのドキュメントは自分の永続メモリに9日前から入っていた。答える前に自分のメモを引くべきだった。
（ただしツバメ側はリポジトリがprivateで権限が無く、自作の跳躍ポリシーを載せるにはアダプタも要るので、
手元で作る判断自体は間違っていなかった。）

**2. 作ってみたら、C++は1行も書かずに済んだ。** `anaguma.xml` の関節定義順・アクチュエータ順・
センサ順が3つとも `go2.xml` と同一構造だったため、`deploy/robots/go2` をコピーして
config を2箇所直すだけで動いた。Anaguma移植では「MJCFの順とSDKの順のズレ」を踏んでいたので
いちばん警戒していた場所が、実は無傷だった。

**3. `1` を押すと転ぶ → 原因は起動時の姿勢。** ゲインを疑ったが違った。`unitree_mujoco` は
`mj_resetData` でリセットするので **MJCFの keyframe を読まない**。Anagumaは全関節0の姿勢で
**足が床に4.4cm埋まった状態**で始まっていて、押し出されて跳ね、`1` を押す前から崩れていた。

**4. プルプル震える → 原因は時間刻み。** トルク飽和を疑って測ったら、kp200/kd10 でも
ピーク17 N·m（上限70）で**飽和率0%**。ゲインは無罪で、真因は `anaguma.xml` が
`timestep="0.005"` を明示していたこと（`go2.xml` は無指定＝既定0.002）。2.5倍粗い刻みで
接触つきの剛いPDを回していた。

**5. 「変な力で左回転しながら沈み込む」→ 2つの別の話だった。**
沈み込みは**コントローラの安全装置**で、傾きが1.0 rad（57度）を超えると自分でPassiveに落ちる仕様。
壊れていたのではなく、意図的に指令を切っていた。左回転のほうは python 経路でも
**+11度 / 10秒**出るので、C++のバグではなく**ポリシー本来の癖**だった。

**6. `j` で跳ばない → 信号は正常だった。** デバッグ出力を仕込んで確認したら
`[jump] trigger received` → `command raised`、bit clock はちょうど0.86秒（`jump_hold_time_s`）、
`phase` も `t/2.2` で正しい。足りなかったのは**助走時間**で、自動評価は `--warmup 4.0`、
つまり4秒走らせてから跳ばせていた。押すのが早すぎただけ。

**7. 「立てなくなった」→ 自分が転ばせたまま放置していた。** 疑似端末での自動テストで
キーが一度に流れ込み、助走なしで跳ばせて**傾き180度＝裏返り**のままシムに残っていた。
裏返った24kgは `1` では起き上がれない。**Backspace でリセットすれば直る。**

**8. 副産物として、評価スクリプトのバグが出た。** 経路差を潰す過程で
`scripts/mujoco_jump_eval.py` が観測の角速度を**二重回転**させていたのを見つけた
（MuJoCoのfree jointの`qvel[3:6]`は既に胴体ローカル系）。
**このプロジェクトの全mujoco数値がこのバグ込みで測られていた**ので、
Go2の据え置きベストは 1.929m → 1.674m、Loop10系起点は 1.354m → 1.601m に変わり、
両者の差は42%から5%になった。**Anagumaを動かそうとしただけで、Go2の評価の土台が1つ直った。**

## この日の教訓

- **推測した場所は4回連続で外れた**（ゲイン → 時間刻み、飽和 → 安全装置、C++のバグ → ポリシーの癖、
  信号 → 助走時間）。**毎回、実測が別の場所を指した。**
- **自分のメモリを引かずに「できない」と答えた**のがいちばんの無駄だった。
- 「動かない」の半分は**シムの状態**（埋まった足、裏返ったまま）で、コードではなかった。
