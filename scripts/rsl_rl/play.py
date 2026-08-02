# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import select  # ===== 追加: stdin teleop =====
import sys     # ===== 追加: stdin teleop =====
import termios # ===== 追加: stdin teleop =====
import time
import torch
import tty     # ===== 追加: stdin teleop =====

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # commented out for v2.3.2
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    # --- 連続再生: 自動リセットを止める ---
    env_cfg.episode_length_s = 1.0e6  # 時間切れリセットを実質無効化
    for _name in list(vars(env_cfg.terminations).keys()):
        setattr(env_cfg.terminations, _name, None)  # time_out・転倒など全 term を無効化
    # ------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ===== stdin キーボード teleop セットアップ =====
    # ヘッドレス/WebRTC では Se2Keyboard が効かない(offscreen)ため、
    # この play.py を起動した SSH 端末の標準入力から直接キーを読む。
    # 操作: W/S=前後  A/D=左右  Q/E=旋回  Space=停止
    VX_STEP, VY_STEP, WZ_STEP = 0.2, 0.2, 0.3          # 1キーあたりの増分
    VX_MAX, VY_MAX, WZ_MAX = 1.0, 0.6, 1.5             # 各軸の上限(±)
    teleop_cmd = [0.0, 0.0, 0.0]                       # [vx, vy, wz]

    _stdin_is_tty = sys.stdin.isatty()
    _stdin_fd = sys.stdin.fileno() if _stdin_is_tty else None
    _old_term = termios.tcgetattr(_stdin_fd) if _stdin_is_tty else None
    if _stdin_is_tty:
        tty.setcbreak(_stdin_fd)  # 1文字ずつ即時取得(Ctrl+C は有効のまま)

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _read_keys_and_update():
        if not _stdin_is_tty:
            return
        updated = False
        # 端末に溜まったキーを全部処理
        while select.select([sys.stdin], [], [], 0.0)[0]:
            c = sys.stdin.read(1)
            if c in ("w", "W"):
                teleop_cmd[0] = _clamp(teleop_cmd[0] + VX_STEP, -VX_MAX, VX_MAX); updated = True
            elif c in ("s", "S"):
                teleop_cmd[0] = _clamp(teleop_cmd[0] - VX_STEP, -VX_MAX, VX_MAX); updated = True
            elif c in ("a", "A"):
                teleop_cmd[1] = _clamp(teleop_cmd[1] + VY_STEP, -VY_MAX, VY_MAX); updated = True
            elif c in ("d", "D"):
                teleop_cmd[1] = _clamp(teleop_cmd[1] - VY_STEP, -VY_MAX, VY_MAX); updated = True
            elif c in ("q", "Q"):
                teleop_cmd[2] = _clamp(teleop_cmd[2] + WZ_STEP, -WZ_MAX, WZ_MAX); updated = True
            elif c in ("e", "E"):
                teleop_cmd[2] = _clamp(teleop_cmd[2] - WZ_STEP, -WZ_MAX, WZ_MAX); updated = True
            elif c == " ":
                teleop_cmd[0] = teleop_cmd[1] = teleop_cmd[2] = 0.0; updated = True
            # それ以外(矢印キーのエスケープ等)は無視
        if updated:
            print(
                f"\r[teleop] vx={teleop_cmd[0]:+.2f}  vy={teleop_cmd[1]:+.2f}  wz={teleop_cmd[2]:+.2f}    ",
                end="", flush=True,
            )

    # ===== 追加: /cmd_vel 由来のファイルから速度コマンドを読む =====
    # cmd_vel_bridge.py (system python3.10) が /tmp/cmd_vel.txt に
    # "vx vy wz" を書く。ここを毎ステップ読み、teleop_cmd を上書きする。
    CMD_FILE = "/tmp/cmd_vel.txt"

    def _read_cmd_from_file():
        try:
            with open(CMD_FILE, "r") as f:
                parts = f.read().split()
            if len(parts) >= 3:
                teleop_cmd[0] = _clamp(float(parts[0]), -VX_MAX, VX_MAX)
                teleop_cmd[1] = _clamp(float(parts[1]), -VY_MAX, VY_MAX)
                teleop_cmd[2] = _clamp(float(parts[2]), -WZ_MAX, WZ_MAX)
        except (FileNotFoundError, ValueError, OSError):
            pass
        # ===== 切り分け用: 読んだ値を毎ステップ表示（確認後は削除可） =====
        print("[cmd]", teleop_cmd, flush=True)
        # ===============================================================
    # ============================================================

    # ===== 追加: base の位置・姿勢(world系, sim真値)をファイルに書く =====
    # 目的: 後段の system python3.10 プロセスが /tmp/odom.txt を読み、
    #       odom と TF(odom->base_link 等)を ROS2 に publish するための元データ。
    # 形式: "px py pz qx qy qz qw"（位置3 + クォータニオン4）を原子的に書く。
    # [未検証] env.unwrapped.scene["robot"] のキー名・.data.root_pos_w /
    #          root_quat_w がこのタスクで有効か、quat の並びが (w,x,y,z) か。
    #          取得に失敗しても本体を止めないよう try/except で保護する。
    ODOM_FILE = "/tmp/odom.txt"
    _pose_err_printed = [False]  # エラーメッセージを1回だけ出すためのフラグ

    def _write_base_pose_to_file():
        try:
            robot = env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0]      # (x, y, z) world系, env 0
            quat = robot.data.root_quat_w[0]    # IsaacLab は一般に (w, x, y, z) 順
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            # ROS 慣例に合わせ (qx, qy, qz, qw) の順で書く
            line = f"{px} {py} {pz} {qx} {qy} {qz} {qw}"
            tmp = ODOM_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write(line)
            os.replace(tmp, ODOM_FILE)  # 原子的に置き換え
        except Exception as e:
            if not _pose_err_printed[0]:
                print("[odom] base pose 取得に失敗:", repr(e), flush=True)
                _pose_err_printed[0] = True
    # ============================================================

    print("=" * 60)
    print(" 操作 (この端末にフォーカスして入力):")
    print("   W / S : 前進 / 後退")
    print("   A / D : 左 / 右 平行移動")
    print("   Q / E : 左 / 右 旋回")
    print("   Space : 停止(全0)")
    print("   Ctrl+C: 終了")
    print("=" * 60)

    # 手動操作のため velocity コマンドの自動生成系を無効化
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)  # 実質リサンプルしない
    cmd_term.cfg.heading_command = False                  # yaw を手動で持つ
    cmd_term.cfg.rel_standing_envs = 0.0                  # 立ち当番 env を作らない
    cmd_term.is_standing_env[:] = False                   # 既存の立ち当番マスクも解除
    cmd_term.time_left[:] = 1.0e9                         # 保留中のリサンプルも止める
    # ============================================

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # ===== /cmd_vel(ファイル経由)を読んで速度コマンドに反映 =====
            _read_cmd_from_file()
            cmd_term.vel_command_b[:] = torch.tensor(
                teleop_cmd, dtype=torch.float32, device=env.unwrapped.device
            )
            # ================================================

            # ===== 追加: base の位置・姿勢をファイルに書く =====
            _write_base_pose_to_file()
            # ================================================

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # 端末設定を必ず元に戻す(これを忘れるとシェルが壊れる)
        if _stdin_is_tty and _old_term is not None:
            termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _old_term)
            print()  # 改行

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    '''
    ============================================================
    操作 (この端末にフォーカスして入力):
    W / S : 前進 / 後退
    A / D : 左 / 右 平行移動
    Q / E : 左 / 右 旋回
    Space : 停止(全0)
    Ctrl+C: 終了
    ============================================================
    '''