# Go2 実地形歩行プロジェクト 技術サーベイ ― height_scan・地形取り込み・既存資産

**このドキュメントについて**: 四足歩行ロボット Go2 を「実際の現場の地形」で歩かせるために必要な技術要素と、社内に既にある再利用できる資産を1枚にまとめた入門・調査資料。これからこのテーマに関わる人が最初に読むと全体像がつかめることを目的とする。

## 1. 背景と目的
これまでは、外界センサを使わず自分の関節情報だけで歩く「blind（ブラインド）」なポリシーで階段歩行に取り組んできたが、平地歩行と階段の両立が頭打ちになっていた。そこで方針を「実環境（横浜国立大で使う敷居のある段差ブロック）を忠実に再現し、その上で歩けるポリシーを作る」ことに切り替えた。ロボットが地形を“見る”か“見ない”かで2アプローチを用意し、比較する。

## 2. 全体像：2つのアプローチ
- **blind**: 外界センサを使わず、関節角やIMUなど自分の体の情報（固有受容）だけで歩く。センサが汚れても頑健なのが利点。
- **height-scan**: 足元まわりの地面の高さを格子状に測った情報（ハイトマップ）を使って歩く。一般に難しい地形に強い。

両方を同じ実地形で学習し、性能・頑健性を比較するのがプロジェクトのゴール。

## 3. 分かったこと（社内に既にある資産）
ゼロから作る必要はなく、以下が既に社内に存在する:
- **height_scan（ハイトマップ）の観測仕様が確定済み**: 0.1m間隔・**17×11 = 187点**、範囲は前方 X −0.6〜1.0m / 左右 Y −0.5〜0.5m。
- **シミュレータ上のハイトマップ生成が実装済み**: 社内Gitブランチ `feat/lidar_input`。導入により胴体の揺れ・関節トルク・消費電力が下がり、完走率96%という結果が報告されている。
- **実地形データと取り込み手順**: 福岡造船のブロック（2S1／5S1）のCAD(DXF)データと取り込み報告書がある。横浜国大の環境は、衝突判定を事前計算した地形ファイル `Yokokoku_collider.usd` が既に用意されている。
- **知覚移動のアーキテクチャ**: 地形の正解情報を使う“先生モデル”と、それを実機で使えるモデルに写す belief-encoder（外界情報を内部状態に圧縮する仕組み）が文書化されている。

## 4. 参考ドキュメント（社内 chiho）
- Go2 LiDAR 入出力インターフェース仕様: https://chiho-shared.openreach.tech/organization/v1/documents/NSkajOrNxVRJfbmori6jMHlj07FT4L
- Isaac labでのGo2 LIDAR再現とHeight map作成: https://chiho-shared.openreach.tech/organization/v1/documents/TBg77AQEg7Rx9QgHdXJi1IS8kBxgLm
- Go2教師モデルのトレーニング: https://chiho-shared.openreach.tech/organization/v1/documents/4syFkIahViVpy4fQ20xY9ugGRixwM8
- belief-encoderの実装を理解する: https://chiho-shared.openreach.tech/organization/v1/documents/Yp48az6yRwIP9ljeNiWhGQPDSF3gCC
- 福岡造船ブロック3D DXFデータのIsaacSim取り込み報告書: https://chiho-shared.openreach.tech/organization/v1/documents/oQGexYJdKovmVAhhZkrSNoo18QJmvS
- デジタルツイン × Go2 行動シミュレーション: https://chiho-shared.openreach.tech/organization/v1/documents/1AUh6ogPm0NW98eWjJzZDVY9sDmbgS

## 5. 図（発表用）
> 📷 ①「既存資産マップ」: 何が揃っていて何を作るのかを1枚に。
> 📷 ② blind と height-scan の違いを示す模式図。

## 6. 用語集
- **固有受容(こゆうじゅよう / proprioception)**: 関節角・角速度・IMU など「自分の体」から得る内部感覚。
- **外受容(exteroception)**: LiDAR・カメラなど外界センサから得る情報。
- **blind ポリシー**: 外受容を使わず固有受容だけで歩くポリシー。
- **height_scan / ハイトマップ(height map)**: 足元の地面の高さを格子状に測った観測。ここでは 0.1m 間隔・17×11=187点。
- **LiDAR(ライダー)**: レーザーで周囲までの距離を測るセンサ。点群から地面の高さを求められる。
- **強化学習(RL)**: 試行錯誤で「報酬」が高くなる行動を学ぶ機械学習。歩行の学習に使う。
- **地形カリキュラム**: 学習中に地形の難易度を段階的に上げる仕組み。
- **teacher-student / belief-encoder**: 地形の正解など“特権情報”で賢い先生を作り、実機で使える生徒モデルに知識を写す手法。belief-encoder は外界情報を内部状態に圧縮するネットワーク。

## 7. 次のステップ
実地形を Isaac Sim に取り込み Go2 を立たせる → 敷居地形を学習シーン化 → height_scan を有効化して blind と比較、の順に進める。
