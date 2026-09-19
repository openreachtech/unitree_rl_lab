# vlfm_nav_bringup

VLFM ナビスタックのロボット非依存部分の launch / パラメータ。全体設計は
`doc/design/vlfm_nav.md`。

## ロボット抽象コントラクト

このスタック(`vlfm_nav` / `vlfm_nav_bringup`)はロボット固有の型・トピックを
一切知らない。ロボット層(`go2_nav_bringup`、将来 `anaguma_nav_bringup`)が
提供する義務は次の5つ:

| 提供物 | 形式 |
|---|---|
| `odom→base` TF + オドメトリ | tf2 + `nav_msgs/Odometry` |
| LiDAR 点群 | `sensor_msgs/PointCloud2`(frame は TF ツリーに接続していること) |
| RGB カメラ | `sensor_msgs/Image` + `CameraInfo` |
| 速度コマンド受理 | `geometry_msgs/Twist` on `/cmd_vel` |
| ロボット config | yaml(フレーム名・footprint・速度レンジ・スキャン z バンド) |

ロボット層はこれらを満たした上で、この パッケージの launch にパラメータを
渡して include する。逆方向の依存(このパッケージ → ロボットパッケージ)を
作らないこと。

## Launch

- `mapping.launch.py` (M2): 点群 → `/scan` (pointcloud_to_laserscan) →
  slam_toolbox → `/map`, `map→odom` TF。引数: `pointcloud_topic`,
  `base_frame`, `odom_frame`, `min_height`, `max_height`, `use_sim_time`
- `nav.launch.py` (M3, 予定): Nav2 (MPPI, holonomic)
- `vlfm.launch.py` (M4/M5, 予定): VLFM core + VLM サービス
