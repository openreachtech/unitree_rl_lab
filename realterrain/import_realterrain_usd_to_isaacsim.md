# 実地形USD/USDZ を Isaac Sim に取り込み、Go2 を立たせる手順

**このドキュメントについて**: 実際の現場をスキャン／CADから作った3D地形（USD/USDZ）を Isaac Sim に読み込み、衝突判定を付けて四足歩行ロボット Go2 を立たせ・歩かせるまでの手順をまとめる。同じことをやりたい人が再現できることを目的とする。

## 1. 目的
横浜国大の実測地形（衝突判定を焼き込んだ `Yokokoku_collider.usd`）または福岡造船ブロックのUSDZを Isaac Sim に配置し、Go2 が地形の上に正しく立って歩くところまでを確認する。結果を写真・動画で残す。

## 2. 前提・背景
CAD やスキャン由来の3Dモデルは、見た目のメッシュはあっても「衝突判定（collider）」が無いことが多く、そのままだとロボットの足が地面をすり抜ける。最短で“立っている画”を作るには、衝突判定を事前計算済みの `Yokokoku_collider.usd` を使うのがよい。

## 3. 参考ドキュメント（社内 chiho）
- USD(Z)環境の取り込みとGo2のポリシー走行手順: https://chiho-shared.openreach.tech/organization/v1/documents/0qbvOh2LOeYLIXjWcUsGPkZL8qiUlQ
- デジタルツイン × Go2 行動シミュレーション: https://chiho-shared.openreach.tech/organization/v1/documents/1AUh6ogPm0NW98eWjJzZDVY9sDmbgS
- USDZファイルを開く方法と起動手順: https://chiho-shared.openreach.tech/organization/v1/documents/NIDj5U0RKpRgOM1014UjRQADgdt65E
- Isaac Sim 5.1(pip版) ストリーミング起動マニュアル: https://chiho-shared.openreach.tech/organization/v1/documents/cUTWMwsDRVTgKOObmpMpx18rmYgBMv

## 4. 手順
今回の素材: `~/ダウンロード/fukuoka_test.usdz`（福岡造船の 2S1＋5S1 ブロック）。生の変換品なので**衝突判定が入っていない**。そこで「①衝突判定を焼き込んだ実地形USDを作る（ヘッドレススクリプト） → ②play.pyでGo2を載せる」の2段で進める。

### (A) 衝突判定を焼き込んだ実地形USDを作る（ヘッドレススクリプト・1回だけ）
> **注記**: 社内ドキュメントの正規手順は、Isaac Sim **デスクトップアプリのGUI**で「Import → Instanceable解除 → `Add→Physics→Colliders Preset`」を手作業で行うもの（社内chiho「福岡造船呉工場のブロックの3D DXFデータをIsaacSimに取り込む」: https://chiho-shared.openreach.tech/organization/v1/documents/O4N3h7WICjdheB08Q0H689lTMh3FBy ）。
> 本機は**デスクトップアプリが未インストール**（pip版のみ）で、共有サーバのため**勝手にインストールしてよいか判断がつかず**インストールは見送った。そのため、同じ結果（Instanceable解除＋衝突判定付与）を得られる**ヘッドレススクリプトによる代替手段**として `bake_collider.py` を作成した。デスクトップアプリが使える環境では、社内docどおりGUIで行うのが正規のやり方。

GUIで手作業せず、`realterrain/bake_collider.py` で焼き込む:
```bash
cd /home/tanaka/isaacsim/unitree_rl_lab
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
python realterrain/bake_collider.py
```
処理内容: `~/ダウンロード/fukuoka_test.usdz`（2S1＋5S1）を読み → **Instanceable を解除**（これをしないとメッシュに触れない）→ 全メッシュに**三角メッシュのCollider**付与 → **原点中心・床z=0** に平行移動 → `realterrain/assets/fukuoka_collider.usd` を出力。
- 確認値の例: `meshes with collider = 2`, `size ≈ 45.75 × 23.7 × 1.78m`（実ブロック相当）。
- 巨大に出たら mm由来 → `--scale 0.001`。中心寄せ不要なら `--no_center`。
- 出力の検証は bbox とCollider数がレポートに出る（横国の `Yokokoku_collider.usd` に相当する再利用可能な実地形）。

### (B) Go2 を実地形に載せて歩かせる（play.py）
play用cfg `RobotPlayEnvCfgGo2` は既に上記USDを読むよう変更済み（`terrain_type="usd"` / `usd_path=…fukuoka_collider.usd`）。学習済み blind ポリシーで走らせる:
```bash
cd /home/tanaka/isaacsim/unitree_rl_lab
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
# --checkpoint はフルパスで渡す（play.py は checkpoint をそのままファイルパスとして開く）
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity-v1 --num_envs 1 \
    --checkpoint /home/tanaka/isaacsim/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_v1/2026-07-09_05-46-45/model_9997.pt
# もしくは --checkpoint を外して最新を自動選択:
#   python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity-v1 --num_envs 1 --load_run 2026-07-09_05-46-45
```
Go2 がブロック上に現れ、姿勢を保って立つ／歩く。矢印キー等でコマンドを与えて歩かせ、スクショ・短動画を撮る。

**よくある詰まり**: ①足がすり抜ける→(A4)の衝突判定が未付与か近似が粗い。②Go2が落下/宙に浮く→(A5)の設置高さズレ。③極端な大小→(A3)のスケール。

## 作業ログ
- **2026-07-16**:
  - Isaac Sim デスクトップアプリが未インストール（pip版のみ）と判明 → 衝突判定付与を GUI ではなくヘッドレススクリプト `bake_collider.py` で実施する方針に。
  - 初回焼き込みで `collider = 0` → メッシュが **Instanceable** の中に隠れて触れていなかった。Instanceable 解除処理を追加して再焼き込み、`meshes with collider = 2` を確認。
  - 地形bbox = 45.75 × 23.7 × 1.78m（実ブロック相当、スケール正常）。原点中心・床 z=0 に整形して `fukuoka_collider.usd` 出力。
  - play cfg `RobotPlayEnvCfgGo2` を USD 地形（`terrain_type="usd"`）に差し替え。`--checkpoint` はフルパス指定が必要と判明。
  - play 初回、Go2 が中央(0,0)にスポーン → **2S1 と 5S1 は離れた2ブロックで、中央は空隙**のため落下。次に上面(z≈1.78)付近に置くと構造に埋まる。
  - **下向きレイキャストで面をマッピング**：5S1 の y<0 側に平らなデッキ(z≈1.59)、y=0 は溝(z≈0.2)、y>0 側は開口。スポーンを **5S1 デッキ上 (-17, -6, 2.2)** に修正。
  - 共有GPUが他ユーザのジョブで満杯（空き330MB）で一時OOM → GPU が空いてから再実行し、**Go2 が 5S1 デッキ上に着地・自立**するのを確認、スクリーンショット取得。

## 5. 結果・成果物（発表用）
福岡造船 **5S1 ブロックのデッキ上に Go2 が立った**（blind ポリシー）。実地形を Isaac Sim に忠実再現し、その上で四足ロボットを動かす第一歩を達成。

> 📷 **ここにスクリーンショットを貼る**：福岡造船 5S1 ブロックのデッキ上に立つ Go2
> （できれば動画も：デッキ上で立つ／歩く様子）

- **使った地形ファイル**: `realterrain/assets/fukuoka_collider.usd`（`fukuoka_test.usdz` に衝突判定を焼き込み・原点整形したもの）
- **タスク / ポリシー**: `Unitree-Go2-Velocity-v1` ＋ 学習済み blind ポリシー `model_9997.pt`
- **実行コマンド**: 上記 (B) の `play.py`
- **スポーン位置**: 5S1 デッキ（x≈-17, y≈-6, 上面 z≈1.59）
- **気づき**: 2S1 / 5S1 は離れた2ブロックで中央は空隙、中央に置くと落下した。どこなら立てるか探すために地面の高さを測ったところ、平らな面・一段低い溝・穴（測っても面が無い箇所）が混在しており、単純な平地ではないと分かった。データ上「二重底ブロック」と呼ばれているものだが、構造の詳細な対応づけ（デッキ/溝/開口の由来）はまだ未確認。スケール（45.75×23.7×1.78m）は実ブロック相当で単位ズレなし。

## 6. 用語集
- **Isaac Sim / Isaac Lab**: NVIDIA の物理シミュレータ（Sim）と、その上でロボットの強化学習を行うフレームワーク（Lab）。
- **USD / USDZ**: Isaac Sim が使う3Dシーンの記述形式。USDZ はシーン一式を zip で1ファイルにまとめたもの。
- **collider / 衝突判定(collision)**: 物体同士が物理的にぶつかる形状定義。付けないと足が地面をすり抜ける。
- **メッシュ(mesh)**: 3Dモデルの見た目を表す三角形の集まり。衝突判定とは別物。
- **胴体高さ(stand_h)**: Go2 が立ったときの胴体の高さ。実機で約 0.33m。
- **ストリーミング起動**: 画面のないサーバ上で Isaac Sim を動かし、映像を別PCへ配信する起動方法。
- **Instanceable（インスタンス）**: 同じ形状を軽量に複製するためのUSDの仕組み。中のメッシュは“共有の型”扱いで直接編集できないため、衝突判定を付ける前に解除が要る。
- **レイキャスト(raycast)**: ある点から光線を飛ばし、最初にぶつかる面の位置を求める処理。ここでは真下に飛ばして「地面の高さ」を測るのに使った。
- **スポーン(spawn)**: シミュレーション開始時にロボットを配置すること。配置座標が地形の穴や隙間だと落下する。
- **二重底ブロック(double bottom)**: 受領データ上そう呼ばれている、船底の二重構造の一区画。一般的には外板・内底板・その間の骨組みから成るとされるが、今回測った「平らな面・溝・開口」がその骨組みのどの部分に対応するかは未確認（今後の実測・図面確認で詰める）。

## 7. 詰まった点・次のステップ
**詰まった点（と対処）**:
- メッシュが Instanceable で衝突判定が付かない → **Instanceable を解除**してから付与（`bake_collider.py` に組み込み済み）。
- 中央スポーンが 2S1/5S1 の隙間で落下、上面付近だと構造に埋まる → **レイキャストで平らなデッキを特定**し、スポーンを 5S1 デッキ上に修正。
- `--checkpoint` がファイル名だと見つからない → **フルパス**で渡す。
- 共有GPUが他ユーザのジョブで満杯（空き330MB）で OOM → GPU が空いてから実行（コード側の不具合ではない）。

**次のステップ（W2）**:
- この 5S1 の溝・段差（敷居）を Go2 が**乗り越えられる**ように学習シーン化する（W2-1）。
- 新地形で **height_scan（187次元）を有効化**し、blind と比較する（W2-2）。
