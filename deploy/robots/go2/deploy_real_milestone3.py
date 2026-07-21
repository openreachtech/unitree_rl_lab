# python deploy_real_milestone3.py --interface eth0 --policy path/to/policy.pt --no-gui
#
# L2 ボタン: 緊急停止 (stiffness = 0, damping = 0.5 の完全脱力)
# L1 ボタン: ポリシー制御の開始 (Soft Start)

import sys
import os
import time
import argparse
import numpy as np
import torch
import yaml
import scipy.spatial.transform as transform
import threading

# パスの解決と共通モジュールの追加
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
HEIGHTMAP_SRC_DIR = os.path.join(DEPLOY_DIR, "unitree_go2_locomotion_heightmap")
sys.path.append(HEIGHTMAP_SRC_DIR)

from lidar_processor import HeightmapProcessor

# Unitree SDK2 imports
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.utils.joystick import Joystick

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_, MotorCmd_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
except ImportError:
    try:
        from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__PointCloud2_ as PointCloud2_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_ as MotorCmd_
        unitree_go_msg_dds__LowCmd_ = LowCmd_
        LowCmd_ = LowCmd_
    except ImportError:
        print("エラー: IDLのインポートに失敗しました。unitree_sdk2pyのインストールとIDLパスを確認してください。")
        exit(1)

class RealGo2Controller:
    def __init__(self, interface: str, lidar_topic: str, state_topic: str, cmd_topic: str, policy_path: str, bypass_publish: bool = False):
        self.interface = interface
        self.lidar_topic = lidar_topic
        self.state_topic = state_topic
        self.cmd_topic = cmd_topic
        self.bypass_publish = bypass_publish
        
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

        # 2. ハイトマッププロセッサの初期化 (実機推論用にCPUで実行)
        self.processor = HeightmapProcessor(config_yaml_path=yaml_path, device="cpu")
        self.nx, self.ny = self.processor.nx, self.processor.ny
        self.num_cells = self.nx * self.ny

        # 3. ポリシーのロード
        print(f"[INFO] Loading TorchScript policy from: {policy_path}")
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # 4. ゲイン値およびスケール設定 (Sim設定に完全準拠)
        self.target_kp = 25.0
        self.target_kd = 0.5
        self.action_scale = 0.25
        self.gyro_scale = 0.2
        self.joint_vel_scale = 0.05

        self.default_joint_pos = np.array([
            0.0, 0.8, -1.5,  # FR (hip, thigh, calf)
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # RR
            0.0, 0.8, -1.5   # RL
        ], dtype=np.float32)

        # ジョイスティックデコーダの初期化
        self.joystick = Joystick()

        # 5. スレッドセーフな共有データ
        self.lock = threading.Lock()
        self.frame_buffer = []  
        
        # 観測用最新データ
        self.latest_heightmap_1d = np.full((self.num_cells,), self.processor.unknown_fill, dtype=np.float32)
        self.latest_accumulated_points = np.zeros((0, 3))
        self.latest_gyro = np.zeros(3, dtype=np.float32)
        self.latest_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) 
        self.latest_q = self.default_joint_pos.copy()
        self.latest_dq = np.zeros(12, dtype=np.float32)
        self.latest_commands = np.zeros(3, dtype=np.float32) 
        self.last_action = np.zeros(12, dtype=np.float32) 

        # 履歴バッファ (履歴長 = 3)
        self.history_len = 3
        self.q_history = [np.zeros(12, dtype=np.float32) for _ in range(self.history_len)]
        self.dq_history = [np.zeros(12, dtype=np.float32) for _ in range(self.history_len)]
        self.action_history = [np.zeros(12, dtype=np.float32) for _ in range(self.history_len)]

        # 状態管理フラグ
        self.control_mode = "STANDBY" # "STANDBY" | "SOFT_START" | "ACTIVE" | "EMERGENCY"
        self.mode_start_time = 0.0
        self.soft_start_duration = 2.0  # ゲイン立ち上げ時間 (秒)

        # 6. DDS初期化とパブリッシャ/サブスクライバ登録
        print(f"[INFO] Initializing DDS Factory with interface: {self.interface}")
        ChannelFactoryInitialize(0, self.interface)
        
        self.lidar_sub = ChannelSubscriber(self.lidar_topic, PointCloud2_)
        self.lidar_sub.Init(self.LidarMessageHandler, 10)
        
        self.state_sub = ChannelSubscriber(self.state_topic, LowState_)
        self.state_sub.Init(self.LowStateMessageHandler, 10)

        self.cmd_pub = ChannelPublisher(self.cmd_topic, LowCmd_)
        self.cmd_pub.Init()

    def LidarMessageHandler(self, msg: PointCloud2_):
        """LiDAR点群受信コールバック"""
        raw_data = bytes(msg.data)
        point_step = msg.point_step
        num_points = len(raw_data) // point_step
        
        if num_points == 0:
            return
            
        current_time = time.time()
        
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
            
            # ハイトマップ生成 (機体姿勢を反映)
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

    def LowStateMessageHandler(self, msg: LowState_):
        """ロボット関節・IMU・ジョイスティック受信コールバック"""
        with self.lock:
            # 1. IMUジャイロ & クォータニオン
            self.latest_gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
            self.latest_quat = np.array(msg.imu_state.quaternion, dtype=np.float32) # [w, x, y, z]
            
            # 2. モーターの角度 (q) と速度 (dq)
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

            # 4. 安全停止のトリガーチェック (L2ボタンで即座に緊急停止)
            if self.joystick.l2.data > 0.5:
                if self.control_mode != "EMERGENCY":
                    print("[WARNING] Emergency Stop triggered by Joystick L2 Button!")
                    self.control_mode = "EMERGENCY"

            # 5. 制御開始のトリガーチェック (L1ボタンでソフトスタート開始)
            elif self.joystick.l1.data > 0.5 and self.control_mode == "STANDBY":
                print("[INFO] Policy activation triggered! Initiating Soft Start...")
                self.control_mode = "SOFT_START"
                self.mode_start_time = time.time()

    def update_history(self, q_rel, dq, action):
        """観測履歴をスライド更新"""
        self.q_history.pop(-1)
        self.q_history.insert(0, q_rel.copy())
        
        self.dq_history.pop(-1)
        self.dq_history.insert(0, dq.copy())
        
        self.action_history.pop(-1)
        self.action_history.insert(0, action.copy())

    def build_observation(self, step_history: bool = False):
        """304次元の観測ベクトルをスケーリングを考慮して組み立てる"""
        with self.lock:
            # A. ジャイロ (3) * スケール 0.2
            gyro = self.latest_gyro.copy() * self.gyro_scale
            
            # B. 投影重力 (3) * スケール 1.0
            q = self.latest_quat.copy() 
            q_xyzw = np.array([q[1], q[2], q[3], q[0]])
            r = transform.Rotation.from_quat(q_xyzw)
            projected_gravity = r.inv().apply(np.array([0.0, 0.0, -1.0]))
            
            # C. 指令速度コマンド (3) * スケール 1.0
            commands = self.latest_commands.copy()
            
            # 関関節偏差・速度・前回アクション
            q_rel_current = self.latest_q.copy() - self.default_joint_pos
            # 関節速度はスケール 0.05 を掛ける
            dq_current = self.latest_dq.copy() * self.joint_vel_scale
            action_current = self.last_action.copy()
            
            if step_history:
                self.update_history(q_rel_current, dq_current, action_current)
                
            q_hist_flat = np.concatenate(self.q_history)
            dq_hist_flat = np.concatenate(self.dq_history)
            act_hist_flat = np.concatenate(self.action_history)
            
            # G. ハイトマップ (187)
            heightmap = self.latest_heightmap_1d.copy()
            
        # 全要素をフラットに結合 (117 + 187 = 304 次元)
        obs_vector = np.concatenate([
            gyro,               # 0-2 (3)
            projected_gravity,  # 3-5 (3)
            commands,           # 6-8 (3)
            q_hist_flat,        # 9-44 (36)
            dq_hist_flat,       # 45-80 (36)
            act_hist_flat,      # 81-116 (36)
            heightmap           # 117-303 (187)
        ]).astype(np.float32)
        
        return obs_vector

    def calculate_gains(self):
        """現在の制御モードに応じたPDゲインを計算 (ソフトスタート用)"""
        if self.control_mode == "STANDBY":
            return 0.0, 0.5  # 完全脱力 (ダンピングのみ)
        elif self.control_mode == "EMERGENCY":
            return 0.0, 0.5  # 緊急停止 (ダンピングのみで安全にへたり込ませる)
        elif self.control_mode == "ACTIVE":
            return self.target_kp, self.target_kd
        elif self.control_mode == "SOFT_START":
            elapsed = time.time() - self.mode_start_time
            if elapsed >= self.soft_start_duration:
                # ソフトスタート完了
                self.control_mode = "ACTIVE"
                print("[INFO] Soft Start complete. Policy control is now fully ACTIVE.")
                return self.target_kp, self.target_kd
            else:
                # 線形にゲインを上昇
                alpha = elapsed / self.soft_start_duration
                kp = alpha * self.target_kp
                kd = 0.5 + alpha * (self.target_kd - 0.5) # kdは0.5から目標値まで
                return kp, kd
        return 0.0, 0.5

    def check_tilt_safety(self):
        """姿勢角度(傾斜)の制限チェック"""
        q = self.latest_quat
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        r = transform.Rotation.from_quat(q_xyzw)
        euler = r.as_euler('xyz', degrees=True)
        roll, pitch = euler[0], euler[1]
        
        # 30度以上の傾きがあれば緊急停止
        if abs(roll) > 30.0 or abs(pitch) > 30.0:
            if self.control_mode in ["SOFT_START", "ACTIVE"]:
                print(f"[FATAL ERROR] Robot tilt angle exceeded limit! Roll: {roll:.1f}deg, Pitch: {pitch:.1f}deg. Terminating control...")
                self.control_mode = "EMERGENCY"

    def run_loop(self, no_gui: bool):
        """200Hzの主制御＆推論ループ"""
        print("[INFO] Starting controller loop at 200Hz.")
        
        # 200Hz (0.005秒) 制御周期
        control_dt = 0.005
        # 50Hz (0.02秒) ポリシー推論周期 ➡ 4制御サイクルに1回推論
        inference_decimation = 4
        cycle_count = 0

        # アクションの制限クリップ
        action_limit = 1.0

        try:
            while True:
                start_loop = time.time()

                # A. 姿勢傾斜セーフティの判定
                self.check_tilt_safety()

                # B. 50Hzでポリシーの推論実行
                if cycle_count % inference_decimation == 0:
                    # 観測値を組み立てて履歴をアップデート
                    obs_vector = self.build_observation(step_history=True)
                    
                    if self.control_mode in ["SOFT_START", "ACTIVE"]:
                        # テンソル化
                        obs_tensor = torch.from_numpy(obs_vector).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            # 推論実行
                            action_tensor = self.policy(obs_tensor)
                            action = action_tensor.squeeze(0).numpy()
                            # 異常な出力をクリップ
                            action = np.clip(action, -action_limit, action_limit)
                            
                        with self.lock:
                            self.last_action = action.copy()
                    else:
                        # 待機・緊急停止時は目標アクションを0(デフォルト立ち姿勢)にする
                        with self.lock:
                            self.last_action = np.zeros(12, dtype=np.float32)

                # C. 200Hzで実機指令（LowCmd）を作成してパブリッシュ
                kp, kd = self.calculate_gains()
                
                # 目標関節位置: default + action * action_scale (0.25)
                with self.lock:
                    target_q = self.default_joint_pos + self.last_action * self.action_scale

                # LowCmd IDLオブジェクトの構築 (ヘルパー関数を使用)
                cmd_msg = unitree_go_msg_dds__LowCmd_()
                cmd_msg.head = [0xFE, 0xEF]
                cmd_msg.level_flag = 0xFF # 低レベル制御モードを明示指定

                for i in range(12):
                    cmd_msg.motor_cmd[i].q = float(target_q[i])
                    cmd_msg.motor_cmd[i].dq = 0.0
                    cmd_msg.motor_cmd[i].tau = 0.0
                    cmd_msg.motor_cmd[i].kp = float(kp)
                    cmd_msg.motor_cmd[i].kd = float(kd)

                # コマンド送信 (デバッグ時はスキップ可能)
                if not self.bypass_publish:
                    self.cmd_pub.Write(cmd_msg)

                # デバッグ画面 (no-guiモード) の表示更新 (20サイクルに1回 ＝ 10Hz)
                if no_gui and cycle_count % 20 == 0:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("==================================================")
                    print("       Go2 Policy Deploy & Control Loop           ")
                    print("==================================================")
                    print(f"制御モード: \033[92m{self.control_mode}\033[0m")
                    print(f"適用ゲイン: Kp = {kp:.2f}, Kd = {kd:.2f} (目標: Kp=25.0, Kd=0.5)")
                    print("--------------------------------------------------")
                    print(f"入力コマンド: vx={self.latest_commands[0]:.2f}, vy={self.latest_commands[1]:.2f}, wz={self.latest_commands[2]:.2f}")
                    print(f"最新の関節目標 (target_q) [12]:\n   {target_q[:3]} (FR)\n   {target_q[3:6]} (FL)")
                    print(f"最新のポリシー出力 (action) [12]:\n   {self.last_action[:3]}")
                    print("==================================================")

                cycle_count += 1
                
                # 200Hzループの周期調整
                elapsed = time.time() - start_loop
                sleep_time = max(0.0, control_dt - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt. Safely shutting down and disabling motor torque...")
            # 安全のため、終了時はKp=0でトルクを完全に抜く
            cmd_msg = unitree_go_msg_dds__LowCmd_()
            for i in range(12):
                cmd_msg.motor_cmd[i].kp = 0.0
                cmd_msg.motor_cmd[i].kd = 0.5 # ダンピングだけ残してゆっくり倒れさせる
                cmd_msg.motor_cmd[i].q = 0.0
                cmd_msg.motor_cmd[i].dq = 0.0
                cmd_msg.motor_cmd[i].tau = 0.0
            if not self.bypass_publish:
                self.cmd_pub.Write(cmd_msg)
            time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Realtime Policy Deployment (Milestone 3)")
    parser.add_argument("--interface", type=str, required=True, help="DDS通信に使うネットワークインターフェース名 (例: eth0)")
    parser.add_argument("--policy", type=str, required=True, help="エクスポートした TorchScript ポリシーファイル (.pt) へのパス")
    parser.add_argument("--no-gui", action="store_true", default=True, help="GUIなしでコンソールログデバッグを表示する")
    parser.add_argument("--bypass-publish", action="store_true", help="DDSコマンドの送信(Write)をスキップして未接続環境でデバッグする")
    
    args = parser.parse_args()

    # DDS のトピック定義
    LIDAR_TOPIC = "rt/utlidar/cloud_deskewed"
    STATE_TOPIC = "rt/lowstate"
    CMD_TOPIC = "rt/lowcmd"

    controller = RealGo2Controller(
        interface=args.interface,
        lidar_topic=LIDAR_TOPIC,
        state_topic=STATE_TOPIC,
        cmd_topic=CMD_TOPIC,
        policy_path=args.policy,
        bypass_publish=args.bypass_publish
    )

    # 制御ループの実行
    controller.run_loop(no_gui=args.no_gui)
