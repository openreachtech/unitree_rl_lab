# python deploy_real_milestone1.py --interface eth0

import sys
import os
import time
import argparse
import numpy as np
import torch
import yaml
import open3d as o3d
import scipy.spatial.transform as transform
import threading

# パスの解決と共通モジュールの追加
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
HEIGHTMAP_SRC_DIR = os.path.join(DEPLOY_DIR, "unitree_go2_locomotion_heightmap")
sys.path.append(HEIGHTMAP_SRC_DIR)

from lidar_processor import HeightmapProcessor

# Unitree SDK2 imports
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
except ImportError:
    try:
        from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointCloud2_ as PointCloud2_
    except ImportError:
        print("エラー: PointCloud2_ のインポートに失敗しました。unitree_sdk2pyのIDLパスを確認してください。")
        exit(1)

class RealLidarHeightmapViewer:
    def __init__(self, interface: str, topic: str):
        self.interface = interface
        self.topic = topic
        
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
        
        # グリッドのXY基準座標系
        self.grid_x_local = []
        self.grid_y_local = []
        x_range = self.spec_cfg['grid']['x_range']
        y_range = self.spec_cfg['grid']['y_range']
        res = self.spec_cfg['grid']['resolution']
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid_x_local.append(x_range[0] + i * res)
                self.grid_y_local.append(y_range[0] + j * res)
        self.grid_x_local = np.array(self.grid_x_local)
        self.grid_y_local = np.array(self.grid_y_local)

        # 3. データ送受信用変数 (スレッドセーフ化)
        self.lock = threading.Lock()
        self.frame_buffer = []  # (timestamp, points_base) のリスト
        self.latest_heightmap_2d = np.full((self.nx, self.ny), self.processor.unknown_fill)
        self.latest_accumulated_points = np.zeros((0, 3))
        self.new_data_available = False

        # 4. DDSサブスクライバの設定
        print(f"[INFO] Initializing DDS Subscriber on topic: {self.topic}")
        ChannelFactoryInitialize(0, self.interface)
        self.sub = ChannelSubscriber(self.topic, PointCloud2_)
        self.sub.Init(self.MessageHandler, 10)

    def MessageHandler(self, msg: PointCloud2_):
        """LiDAR点群の受信コールバックスレッド"""
        raw_data = bytes(msg.data)
        point_step = msg.point_step
        num_points = len(raw_data) // point_step
        
        if num_points == 0:
            return
            
        current_time = time.time()
        
        # バイナリデータから x, y, z 座標をデコード
        points = []
        for i in range(num_points):
            offset = i * point_step
            # x, y, z (各4バイト float32) をデコード
            x, y, z = np.frombuffer(raw_data[offset:offset+12], dtype=np.float32)
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
                
        if len(points) == 0:
            return
            
        points = np.array(points)
        # LiDARローカル座標からロボットベース座標系へ変換
        points_base = (self.lidar_rot_rel @ points.T).T + self.trans_offset
        
        with self.lock:
            # 1. 新しいフレームを追加
            self.frame_buffer.append((current_time, points_base))
            
            # 2. window_sec 秒より古いフレームを削除
            self.frame_buffer = [f for f in self.frame_buffer if current_time - f[0] <= self.window_sec]
            
            if len(self.frame_buffer) == 0:
                return
                
            # 3. 蓄積されている全フレームの点群を連結
            accumulated_pts = np.vstack([f[1] for f in self.frame_buffer])
            self.latest_accumulated_points = accumulated_pts
            
            # 4. ハイトマップ変換
            points_base_t = torch.from_numpy(accumulated_pts).float().unsqueeze(0)
            root_pos_t = torch.zeros((1, 3), dtype=torch.float32)
            root_quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            
            heightmap_t = self.processor.process(
                pos_w=points_base_t,
                root_pos_w=root_pos_t,
                root_quat_w=root_quat_t,
                randomize=False
            )
            heightmap = heightmap_t.squeeze(0).numpy() # [num_cells]
            
            # 2Dグリッドにして保持
            self.latest_heightmap_2d = heightmap.reshape(self.nx, self.ny)
            self.new_data_available = True

    def run_visualization(self):
        """メインスレッドで動作するOpen3Dビジュアライザ"""
        print("[INFO] Starting Open3D Visualization window...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Go2 Realtime Lidar & Heightmap (Milestone 1)", width=1280, height=720)

        # 1. 座標軸とロボット中心
        robot_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        robot_center.paint_uniform_color([0.0, 0.0, 1.0])  # 青
        vis.add_geometry(robot_center)
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)

        # 2. 蓄積点群 (pcd)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 3. ハイトマップセル球体群 (187個)
        spheres = []
        for _ in range(self.num_cells):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            s.paint_uniform_color([0.0, 0.8, 0.0]) # 緑
            vis.add_geometry(s)
            spheres.append(s)

        # カメラビューの初期化設定
        ctr = vis.get_view_control()
        # 原点中心にカメラを引き戻す
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.2)  # 初期画角調整 (15m範囲が見えるくらいに設定)
        
        print("\n--- 操作ヘルプ ---")
        print("  マウス左ドラッグ: カメラ回転")
        print("  マウス右ドラッグ: パン (平行移動)")
        print("  スクロール: ズーム")
        print("  [Q] または [ESC]: ウィンドウを閉じて終了")
        print("------------------\n")

        try:
            while vis.poll_events():
                # 新しいデータがある場合にジオメトリを更新
                update_needed = False
                with self.lock:
                    if self.new_data_available:
                        pts = self.latest_accumulated_points.copy()
                        hmap_2d = self.latest_heightmap_2d.copy()
                        self.new_data_available = False
                        update_needed = True
                        
                if update_needed:
                    # A. 蓄積点群の更新
                    if len(pts) > 0:
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        # 赤色
                        pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(pts), 1)))
                    else:
                        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    vis.update_geometry(pcd)

                    # B. ハイトマップグリッド球体の更新
                    hmap_1d = hmap_2d.flatten()
                    valid_mask = hmap_1d != self.processor.unknown_fill
                    
                    for i in range(self.num_cells):
                        sphere = spheres[i]
                        # 一旦原点に戻す
                        sphere.vertices = o3d.utility.Vector3dVector(
                            np.asarray(sphere.vertices) - sphere.get_center()
                        )
                        
                        if valid_mask[i]:
                            # Z座標の補正 (ハイトマップの定義 base_z - terrain_z に従って変換)
                            # z_phys = - (hmap_val + offset)
                            new_pos = [self.grid_x_local[i], self.grid_y_local[i], -hmap_1d[i] - self.processor.offset]
                            sphere.translate(new_pos)
                            sphere.paint_uniform_color([0.0, 0.8, 0.0]) # 有効点は緑
                        else:
                            # 無効点は一時退避 (地下へ非表示にする)
                            sphere.translate([self.grid_x_local[i], self.grid_y_local[i], -2.0])
                            sphere.paint_uniform_color([0.3, 0.3, 0.3]) # 無効点はグレー
                            
                        vis.update_geometry(sphere)

                vis.update_renderer()
                time.sleep(0.02)  # 約 50 FPS
        finally:
            vis.destroy_window()
            print("[INFO] Visualizer window closed.")

def main():
    parser = argparse.ArgumentParser(description="Go2 Real Lidar & Heightmap Realtime Viewer (Milestone 1)")
    parser.add_argument('--interface', type=str, default='eth0', help='ネットワークインターフェース名 (例: eth0)')
    parser.add_argument('--topic', type=str, default='rt/utlidar/cloud_deskewed', help='対象のLiDAR点群トピック')
    args = parser.parse_args()

    viewer = RealLidarHeightmapViewer(interface=args.interface, topic=args.topic)
    viewer.run_visualization()

if __name__ == '__main__':
    main()
