# python deploy_real_milestone2.py --interface eth0 --no-gui

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
from unitree_sdk2py.utils.joystick import Joystick

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except ImportError:
    try:
        from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointCloud2_ as PointCloud2_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_
    except ImportError:
        print("エラー: IDLのインポートに失敗しました。unitree_sdk2pyのインストールとIDLパスを確認してください。")
        exit(1)

class RealGo2ObsEvaluator:
    def __init__(self, interface: str, lidar_topic: str, state_topic: str):
        self.interface = interface
        self.lidar_topic = lidar_topic
        self.state_topic = state_topic
        
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

        # 3. ロボット設定パラメータ
        # デフォルト立位姿勢 (12次元)
        self.default_joint_pos = np.array([
            0.0, 0.8, -1.5,  # FR (hip, thigh, calf)
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # RR
            0.0, 0.8, -1.5   # RL
        ], dtype=np.float32)

        # ジョイスティックデコーダの初期化
        self.joystick = Joystick()

        # 4. スレッドセーフな共有データ
        self.lock = threading.Lock()
        self.frame_buffer = []  # (timestamp, points_base)
        
        # 最新の観測データ
        self.latest_heightmap_1d = np.full((self.num_cells,), self.processor.unknown_fill, dtype=np.float32)
        self.latest_accumulated_points = np.zeros((0, 3))
        
        self.latest_gyro = np.zeros(3, dtype=np.float32)
        self.latest_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # [w, x, y, z]
        self.latest_q = self.default_joint_pos.copy()
        self.latest_dq = np.zeros(12, dtype=np.float32)
        self.latest_commands = np.zeros(3, dtype=np.float32) # [vx, vy, wz]
        self.last_action = np.zeros(12, dtype=np.float32) # マイルストーン3用のダミー
        
        self.new_lidar_available = False
        self.new_state_available = False

        # 5. DDS初期化とサブスクライバ登録
        print(f"[INFO] Initializing DDS Factory with interface: {self.interface}")
        ChannelFactoryInitialize(0, self.interface)
        
        print(f"[INFO] Subscribing to LiDAR topic: {self.lidar_topic}")
        self.lidar_sub = ChannelSubscriber(self.lidar_topic, PointCloud2_)
        self.lidar_sub.Init(self.LidarMessageHandler, 10)
        
        print(f"[INFO] Subscribing to LowState topic: {self.state_topic}")
        self.state_sub = ChannelSubscriber(self.state_topic, LowState_)
        self.state_sub.Init(self.LowStateMessageHandler, 10)

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
            self.frame_buffer = [f for f in self.frame_buffer if current_time - f[0] <= self.window_sec]
            
            if len(self.frame_buffer) == 0:
                return
                
            accumulated_pts = np.vstack([f[1] for f in self.frame_buffer])
            self.latest_accumulated_points = accumulated_pts
            
            # ハイトマップ生成
            points_base_t = torch.from_numpy(accumulated_pts).float().unsqueeze(0)
            root_pos_t = torch.zeros((1, 3), dtype=torch.float32)
            # 現在の姿勢クォータニオンを適用（より高精度に投影可能）
            q = self.latest_quat
            root_quat_t = torch.tensor([[q[0], q[1], q[2], q[3]]], dtype=torch.float32) # [w, x, y, z]
            
            heightmap_t = self.processor.process(
                pos_w=points_base_t,
                root_pos_w=root_pos_t,
                root_quat_w=root_quat_t,
                randomize=False
            )
            self.latest_heightmap_1d = heightmap_t.squeeze(0).numpy() # [187]
            self.new_lidar_available = True

    def LowStateMessageHandler(self, msg: LowState_):
        """ロボット関節・IMU・リモコン受信コールバック"""
        with self.lock:
            # 1. IMUジャイロ & クォータニオン
            self.latest_gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
            self.latest_quat = np.array(msg.imu_state.quaternion, dtype=np.float32) # [w, x, y, z]
            
            # 2. モーターの角度 (q) と速度 (dq)
            # Go2は関節0〜11にマッピングされている
            self.latest_q = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
            self.latest_dq = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)
            
            # 3. ジョイスティック操作データの抽出
            self.joystick.extract(msg.wireless_remote)
            
            # 目標速度マッピング (デッドゾーン付き)
            deadzone = 0.05
            vx = self.joystick.ly.data
            vy = -self.joystick.lx.data
            wz = -self.joystick.rx.data
            
            self.latest_commands[0] = 0.0 if abs(vx) < deadzone else vx
            self.latest_commands[1] = 0.0 if abs(vy) < deadzone else vy
            self.latest_commands[2] = 0.0 if abs(wz) < deadzone else wz
            
            self.new_state_available = True

    def build_observation(self):
        """232次元の観測ベクトルを結合して構築する"""
        with self.lock:
            # A. ジャイロ (3)
            gyro = self.latest_gyro.copy()
            
            # B. 投影重力 (3)
            q = self.latest_quat.copy() # [w, x, y, z]
            # scipyは [x, y, z, w] を要求するため順序を入れ替え
            q_xyzw = np.array([q[1], q[2], q[3], q[0]])
            r = transform.Rotation.from_quat(q_xyzw)
            # 重力ベクトル [0,0,-1] を機体フレームに回転投影
            projected_gravity = r.inv().apply(np.array([0.0, 0.0, -1.0]))
            
            # C. 指令速度コマンド (3)
            commands = self.latest_commands.copy()
            
            # D. 関節偏差 (12)
            joint_pos_rel = self.latest_q.copy() - self.default_joint_pos
            
            # E. 関節速度 (12)
            joint_vel_rel = self.latest_dq.copy()
            
            # F. 過去アクション (12)
            last_action = self.last_action.copy()
            
            # G. ハイトマップ (187)
            heightmap = self.latest_heightmap_1d.copy()
            
        # 全要素をフラットに結合
        obs_vector = np.concatenate([
            gyro,               # 0-2 (3)
            projected_gravity,  # 3-5 (3)
            commands,           # 6-8 (3)
            joint_pos_rel,      # 9-20 (12)
            joint_vel_rel,      # 21-32 (12)
            last_action,        # 33-44 (12)
            heightmap           # 45-231 (187)
        ]).astype(np.float32)
        
        return obs_vector, gyro, projected_gravity, commands, joint_pos_rel

    def run_no_gui(self):
        """GUIを表示せず、232次元の結合観測ベクトルをログ表示するモード"""
        print("[INFO] Starting in No-GUI mode. Logging observations. Press Ctrl+C to stop.")
        
        try:
            while True:
                obs_vector, gyro, proj_grav, cmds, j_pos = self.build_observation()
                
                # コンソールに綺麗に表示 (1秒おき)
                os.system('cls' if os.name == 'nt' else 'clear')
                print("==================================================")
                print("       Go2 Realtime Observation Debugger          ")
                print("==================================================")
                print(f"時間: {time.strftime('%H:%M:%S')}")
                print(f"結合観測ベクトルサイズ: {len(obs_vector)} 次元")
                print("--------------------------------------------------")
                print(f"1. ジャイロ (base_ang_vel) [3]:\n   {gyro}")
                print(f"2. 投影重力 (projected_gravity) [3]:\n   {proj_grav}")
                print(f"3. 速度コマンド (commands) [3] (ゲームパッド入力):\n   vx={cmds[0]:.3f}, vy={cmds[1]:.3f}, wz={cmds[2]:.3f}")
                print(f"4. 関節偏差 (joint_pos_rel) [12]:\n   {j_pos[:3]} (FR)\n   {j_pos[3:6]} (FL)")
                
                # ハイトマップの分析
                valid_cells = np.sum(obs_vector[45:] != self.processor.unknown_fill)
                print("--------------------------------------------------")
                print(f"5. ハイトマップ有効数: {valid_cells}/{self.num_cells} セル")
                
                # 全データ確認用
                print("--------------------------------------------------")
                print(f"観測ベクトル冒頭部分 (45次元 - ハイトマップ以外):\n{obs_vector[:45]}")
                print("==================================================")
                
                time.sleep(0.5)  # 2Hzで画面をリフレッシュ
        except KeyboardInterrupt:
            print("[INFO] Exiting...")

    def run_visualization(self):
        """マイルストーン1と同等のOpen3D表示を非同期で走らせる"""
        print("[INFO] Starting Open3D Visualization window...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Go2 Lidar & Observation Monitor (Milestone 2)", width=1280, height=720)

        # 座標軸
        robot_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        robot_center.paint_uniform_color([0.0, 0.0, 1.0])
        vis.add_geometry(robot_center)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)

        # 点群
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # ハイトマップ球体
        spheres = []
        for _ in range(self.num_cells):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            s.paint_uniform_color([0.0, 0.8, 0.0])
            vis.add_geometry(s)
            spheres.append(s)

        ctr = vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(0.2)

        last_print_time = time.time()

        try:
            while vis.poll_events():
                update_needed = False
                with self.lock:
                    if self.new_lidar_available:
                        pts = self.latest_accumulated_points.copy()
                        hmap_1d = self.latest_heightmap_1d.copy()
                        self.new_lidar_available = False
                        update_needed = True
                
                if update_needed:
                    # 点群更新
                    if len(pts) > 0:
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(pts), 1)))
                    else:
                        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    vis.update_geometry(pcd)

                    # ハイトマップ球体更新
                    valid_mask = hmap_1d != self.processor.unknown_fill
                    for i in range(self.num_cells):
                        sphere = spheres[i]
                        sphere.vertices = o3d.utility.Vector3dVector(
                            np.asarray(sphere.vertices) - sphere.get_center()
                        )
                        if valid_mask[i]:
                            new_pos = [self.grid_x_local[i], self.grid_y_local[i], -hmap_1d[i] - self.processor.offset]
                            sphere.translate(new_pos)
                            sphere.paint_uniform_color([0.0, 0.8, 0.0])
                        else:
                            sphere.translate([self.grid_x_local[i], self.grid_y_local[i], -2.0])
                            sphere.paint_uniform_color([0.3, 0.3, 0.3])
                        vis.update_geometry(sphere)

                # デバッグ観測値ログをターミナルに定期出力 (1秒ごと)
                if time.time() - last_print_time >= 1.0:
                    obs_vector, gyro, proj_grav, cmds, _ = self.build_observation()
                    valid_cells = np.sum(obs_vector[45:] != self.processor.unknown_fill)
                    print(f"[DDS-OK] Cmd(vx={cmds[0]:.2f}, vy={cmds[1]:.2f}, wz={cmds[2]:.2f}) | Grav_proj={proj_grav} | Lidar_valid={valid_cells}/187")
                    last_print_time = time.time()

                vis.update_renderer()
                time.sleep(0.02)
        finally:
            vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Go2 Observation Vector Evaluator (Milestone 2)")
    parser.add_argument('--interface', type=str, default='eth0', help='ネットワークインターフェース名 (例: eth0)')
    parser.add_argument('--lidar-topic', type=str, default='rt/utlidar/cloud_deskewed', help='対象のLiDARトピック')
    parser.add_argument('--state-topic', type=str, default='rt/lowstate', help='対象のLowStateトピック')
    parser.add_argument('--no-gui', action='store_true', help='GUIを表示せず、観測値ベクトルのみ出力')
    args = parser.parse_args()

    evaluator = RealGo2ObsEvaluator(
        interface=args.interface,
        lidar_topic=args.lidar_topic,
        state_topic=args.state_topic
    )

    if args.no_gui:
        evaluator.run_no_gui()
    else:
        evaluator.run_visualization()

if __name__ == '__main__':
    main()
