# 実地形で height_scan（187次元ハイトマップ）を有効化し、blind と比較する

**このドキュメントについて**: 足元の地面高さを測る観測「height_scan（ハイトマップ）」を実地形の学習で有効化し、正しく地面を捉えているかを確認する方法と、外界センサ無し（blind）との比較の始め方をまとめる。

## 1. 目的
既存の height_scan 実装（0.1m間隔・17×11=187点）を新しい実地形で有効化し、可視化して地面を正しく捉えているか確認する。その上で blind（固有受容のみ）を基準に短時間の学習を1本回し、height-scan 導入の効果を測る比較の土台を作る。

## 2. 前提・背景
最終ゴールは blind と height-scan の比較。まずは「height_scan がこの地形で成立するか」を最小限のテスト（スモークテスト）で確認し、blind の学習曲線という“物差し”を用意する。ここでは完成ではなく「回り始める」ことを目標にする。

## 3. 参考ドキュメント（社内 chiho）
- Isaac labでのGo2 LIDAR再現とHeight map作成: https://chiho-shared.openreach.tech/organization/v1/documents/TBg77AQEg7Rx9QgHdXJi1IS8kBxgLm
- Go2 LiDAR 入出力インターフェース仕様: https://chiho-shared.openreach.tech/organization/v1/documents/NSkajOrNxVRJfbmori6jMHlj07FT4L
- Go2のLiDARからの情報をPolicyに渡すパイプライン: https://chiho-shared.openreach.tech/organization/v1/documents/dbXoniR5yTfDZ73wAGGeQN0Pd36v23

## 4. 手順・作業ログ
- 新地形で height_scan を有効化する
- ハイトマップを可視化し、地形の凹凸を正しく捉えているか確認する
- blind（固有受容のみ）で短時間の学習を1本回し、基準の数値を取る

## 5. 結果・成果物（発表用）
> 📷 ① height_scan の可視化（地形の上に格子状の高さ点が乗っている図）
> 📷 ② 学習の報酬／完走率カーブ（blind ベースライン）
> - 学習コマンド・iteration数・並列環境数:
> - height_scan が地面を正しく捉えているか（所見）:
> - blind ベースラインの初期数値:

## 6. 用語集
- **height_scan（187点）**: 足元の地面高さを 0.1m 間隔・17×11 の格子で測った観測。測れなかった点は −1.0 で埋める。地形を“見る”入力。
- **可視化**: 数値データを図にして目で確認できるようにすること。ここでは高さの格子点を地形の上に重ねて表示する。
- **スモークテスト**: 「とりあえず成立して動くか」を見る最小テスト。性能ではなく成立性の確認。
- **blind ベースライン**: 外界センサ無し（固有受容のみ）で学習した基準ポリシー。height-scan の効果を測る比較対象。
- **iteration（イテレーション）**: 強化学習の更新1回。数千回まわして徐々に上達する。
- **並列環境数(num_envs)**: 同時に走らせる仮想ロボットの数（例 4096）。多いほど学習が速く安定する。
- **報酬(reward) / 完走率**: 報酬は望ましい動きに与える点数。完走率は転ばず地形を渡り切れた割合で、地形歩行の主要な評価指標。

## 7. 詰まった点・次のステップ
- 詰まり:（記入）
- 次: 本格的なチューニングと、blind と height-scan の定量比較へ。
