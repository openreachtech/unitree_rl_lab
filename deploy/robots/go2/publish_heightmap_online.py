# python publish_heightmap_online.py --interface eth0
#
# 有線 DDS から点群と機体姿勢を取得し、ハイトマップを計算して、
# C++ 側が期待する「rt/height_scan」トピック（DDS PointCloud2形式）に 10Hz で配信するスクリプト。

import sys
import os
import time
import argparse
import numpy as np
import torch
import yaml
import struct
import scipy.spatial.transform as transform
import threading

# パスの解決と共通モジュールの追加
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
HEIGHTMAP_SRC_DIR = os.path.join(DEPLOY_DIR, "unitree_go2_locomotion_heightmap")
sys.path.append(HEIGHTMAP_SRC_DIR)

from lidar_processor import HeightmapProcessor

# Unitree SDK2 imports
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except ImportError:
    try:
        from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointCloud2_ as PointCloud2_
        from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointField_ as PointField_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_
    except ImportError:
        print("エラー: IDLのインポートに失敗しました。unitree_sdk2pyのインストールとIDLパスを確認してください。")
        exit(1)

class RealtimeHeightmapPublisher:
    def __init__(self, interface: str, lidar_topic: str, state_topic: str, publish_topic: str):
        self.interface = interface
        self.lidar_topic = lidar_topic
        self.state_topic = state_topic
        self.publish_topic = publish_topic
        
        # 1. 仕様書のロード
        yaml_path = os.path.join(HEIGHTMAP_SRC_DIR, "heightmap_spec.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"heightmap_spec.yaml が見つかりません: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.spec_cfg = yaml.safe_load(f)

        # LiDAR取り付けオフセット
        extrinsics = self.spec_cfg['frame']['lidar_extrinsics']
        self.trans_offset = np.array(extrinsics['translation_m'])
        rpy_deg = extrinsics['rpy_deg']
        self.lidar_rot_rel = transform.Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()

        # 累積時間窓
        self.window_sec = float(self.spec_cfg['accumulation']['window_sec'])

        # 2. ハイトマッププロセッサの初期化
        self.processor = HeightmapProcessor(config_yaml_path=yaml_path, device="cpu")
        self.nx, self.ny = self.processor.nx, self.processor.ny
        self.num_cells = self.nx * self.ny

        # 3. 共有データ
        self.lock = threading.Lock()
        self.frame_buffer = []  # (timestamp, points_base)
        self.latest_accumulated_points = np.zeros((0, 3))
        self.latest_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # [w, x, y, z]
        self.latest_heightmap_1d = np.full((self.num_cells,), self.processor.unknown_fill, dtype=np.float32)

        self.new_data_available = False

        # 4. DDS初期化
        print(f"[INFO] Initializing DDS Factory with interface: {self.interface}")
        ChannelFactoryInitialize(0, self.interface)
        
        print(f"[INFO] Subscribing to LiDAR: {self.lidar_topic}")
        self.lidar_sub = ChannelSubscriber(self.lidar_topic, PointCloud2_)
        self.lidar_sub.Init(self.LidarMessageHandler, 10)
        
        print(f"[INFO] Subscribing to LowState: {self.state_topic}")
        self.state_sub = ChannelSubscriber(self.state_topic, LowState_)
        self.state_sub.Init(self.LowStateMessageHandler, 10)

        print(f"[INFO] Advertising Heightmap: {self.publish_topic}")
        self.hmap_pub = ChannelPublisher(self.publish_topic, PointCloud2_)
        self.hmap_pub.Init()

    def LidarMessageHandler(self, msg: PointCloud2_):
        """LiDAR点群受信コールバック"""
        raw_data = bytes(msg.data)
        point_step = msg.point_step
        num_points = len(raw_data) // point_step
        
        if num_points == 0:
            return
            
        current_time = time.time()
        
        # デコード
        points = []
        for i in range(num_points):
            offset = i * point_step
            x, y, z = np.frombuffer(raw_data[offset:offset+12], dtype=np.float32)
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
                
        if len(points) == 0:
            return
            
        points = np.array(points)
        points_base = (self.lidar_rot_rel @ points.T).T + self.trans_offset
        
        with self.lock:
            self.frame_buffer.append((current_time, points_base))
            # 累積窓のクリーンアップ
            self.frame_buffer = [f for f in self.frame_buffer if current_time - f[0] <= self.window_sec]
            
            if len(self.frame_buffer) == 0:
                return
                
            accumulated_pts = np.vstack([f[1] for f in self.frame_buffer])
            self.latest_accumulated_points = accumulated_pts
            
            # ハイトマップ生成
            points_base_t = torch.from_numpy(accumulated_pts).float().unsqueeze(0)
            root_pos_t = torch.zeros((1, 3), dtype=torch.float32)
            q = self.latest_quat
            root_quat_t = torch.tensor([[q[0], q[1], q[2], q[3]]], dtype=torch.float32) # [w, x, y, z]
            
            heightmap_t = self.processor.process(
                pos_w=points_base_t,
                root_pos_w=root_pos_t,
                root_quat_w=root_quat_t,
                randomize=False
            )
            self.latest_heightmap_1d = heightmap_t.squeeze(0).numpy() # [187]
            self.new_data_available = True

    def LowStateMessageHandler(self, msg: LowState_):
        """低レベル状態受信 (クォータニオンのみ抽出)"""
        with self.lock:
            self.latest_quat = np.array(msg.imu_state.quaternion, dtype=np.float32) # [w, x, y, z]

    def create_pointcloud2_msg(self, heightmap: np.ndarray) -> PointCloud2_:
        """187次元ハイトマップを C++ 側がパース可能な PointCloud2 メッセージにパックする"""
        msg = PointCloud2_()
        msg.height = self.nx # 17
        msg.width = self.ny  # 11
        
        # 各点の構造: x(float), y(float), z(float)
        # C++側はフィールド "z" のoffsetを読み出してmemcpyするため、位置は正確に指定する
        msg.point_step = 12
        msg.row_step = msg.height * msg.width * msg.point_step
        
        field_x = PointField_("x", 0, 7, 1) # name, offset, datatype(7=FLOAT32), count
        field_y = PointField_("y", 4, 7, 1)
        field_z = PointField_("z", 8, 7, 1)
        msg.fields = [field_x, field_y, field_z]
        
        msg.is_bigendian = False
        msg.is_dense = True
        
        # バッファデータをパック
        data_bytes = bytearray(msg.row_step)
        for i in range(self.num_cells):
            z_val = float(heightmap[i])
            # x, y はダミー値 (C++側で無視されるが、z値はオフセット8に詰める)
            struct.pack_into('<fff', data_bytes, i * 12, 0.0, 0.0, z_val)
            
        msg.data = list(data_bytes)
        return msg

    def run_publish_loop(self):
        """10Hz (0.1秒周期) でハイトマップをパブリッシュするループ"""
        print("[INFO] Starting heightmap publish loop at 10Hz. Press Ctrl+C to stop.")
        
        publish_dt = 0.1
        try:
            while True:
                start_time = time.time()
                
                with self.lock:
                    if self.new_data_available:
                        hmap = self.latest_heightmap_1d.copy()
                        self.new_data_available = False
                        
                        # PointCloud2 メッセージを作成して送信
                        msg = self.create_pointcloud2_msg(hmap)
                        self.hmap_pub.Write(msg)
                        
                elapsed = time.time() - start_time
                sleep_time = max(0.0, publish_dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("[INFO] Exiting publisher...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Heightmap Publisher to DDS rt/height_scan")
    parser.add_argument("--interface", type=str, required=True, help="DDS通信に使うネットワークインターフェース名 (例: eth0)")
    args = parser.parse_args()

    # DDS のトピック定義
    LIDAR_TOPIC = "rt/utlidar/cloud_deskewed"
    STATE_TOPIC = "rt/lowstate"
    PUBLISH_TOPIC = "rt/height_scan_processed" # C++の kHeightScanTopic と完全一致

    publisher = RealtimeHeightmapPublisher(
        interface=args.interface,
        lidar_topic=LIDAR_TOPIC,
        state_topic=STATE_TOPIC,
        publish_topic=PUBLISH_TOPIC
    )

    publisher.run_publish_loop()
