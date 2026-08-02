# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.

WASD 手動操作を廃止し、ビューポートに置いた青球(/World/start)・赤球(/World/goal)で
スタートとゴールだけ指定する版。start 球を床に置くとロボットがそこに直立で出て、goal へ歩く。
倒れたら自動で start へ立て直す。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
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
import math
import os
import time
import torch

import omni.usd
from pxr import Gf, Usd, UsdGeom

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

    # =========================================================
    # スタート/ゴールだけ指定して自動歩行 (WASD不要)
    # =========================================================
    stage = omni.usd.get_context().get_stage()

    def make_marker(path, x, y, rgb, z=0.1):
        """視覚のみの球マーカー (コライダ無し → ロボットは衝突しない)。GUIでドラッグして位置指定。"""
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(0.2)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(x, y, z))

    make_marker("/World/start", 1.0, 1.0, (0.1, 0.6, 1.0))  # 青 = スタート
    make_marker("/World/goal", 2.83, 1.14, (1.0, 0.3, 0.1))          # 赤 = ゴール

    def world_xyz(path):
        """prim の world 座標 (x, y, z) を読む。"""
        prim = stage.GetPrimAtPath(path)
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return float(t[0]), float(t[1]), float(t[2])

    def go_to_goal(px, py, yaw, gx, gy, vx_max=0.8, wz_max=1.0, k_v=1.0, k_w=2.0, tol=0.2):
        """現在姿勢→ゴールから cmd_vel(vx, vy, wz) を作る。到達したら全0で停止。"""
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        if dist < tol:
            return 0.0, 0.0, 0.0
        ang = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi  # -pi..pi
        vx = max(min(k_v * dist, vx_max), 0.0) * max(math.cos(ang), 0.0)      # 向いてないほど前進を落とす
        wz = max(min(k_w * ang, wz_max), -wz_max)
        return vx, 0.0, wz

    robot = env.unwrapped.scene["robot"]
    stand_h = 0.33

    def teleport_to_start():
        """start 球の真上に、直立・適正高さ・既定の脚姿勢でロボットを置く。"""
        sx, sy, sz = world_xyz("/World/start")
        with torch.inference_mode():
            # base の姿勢
            root = robot.data.root_state_w.clone()
            root[0, 0] = sx
            root[0, 1] = sy
            root[0, 2] = sz + stand_h                              # 床(start球)の高さ + 立ち高さ
            root[0, 3:7] = robot.data.default_root_state[0, 3:7]   # 直立の向き
            root[0, 7:13] = 0.0                                    # 速度ゼロ
            robot.write_root_state_to_sim(root)
            # 脚の姿勢: これが無いと一度こけた脚配置のまま戻らず、置いてもまた倒れる
            jp = robot.data.default_joint_pos.clone()
            jv = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(jp, jv)

    def aim_camera_at_start():
        """ビューポートのカメラを start の方へ向ける(カメラ操作で手間取らないように)。"""
        sx, sy, sz = world_xyz("/World/start")
        env.unwrapped.sim.set_camera_view([sx + 3.0, sy + 3.0, sz + 2.0], [sx, sy, sz])

    # velocity コマンドの自動生成系を無効化 (手動で vel_command_b を上書きするため)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)  # 実質リサンプルしない
    cmd_term.cfg.heading_command = False                  # yaw を手動で持つ
    cmd_term.cfg.rel_standing_envs = 0.0                  # 立ち当番 env を作らない
    cmd_term.is_standing_env[:] = False                   # 既存の立ち当番マスクも解除
    cmd_term.time_left[:] = 1.0e9                         # 保留中のリサンプルも止める

    print("=" * 60)
    print(" 操作: ビューポートで2つの球をドラッグするだけ (WASD不要)")
    print("   青球 /World/start : 床の上に置くと、その真上にロボットが立つ")
    print("   赤球 /World/goal  : 床の上に置くと、そこへ歩く")
    print(" 倒れたら自動で start へ立て直す。start を動かすとそこへ再配置。Ctrl+C で終了。")
    print(" [dbg] の up=-1.00 が直立。0 に近づくほど倒れかけ、+1 で逆さ。")
    print("=" * 60)

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # start 球が動いたら再配置する判定用。起動時は配置せず、ユーザーが start を床に置いた時点で配置。
    last_start = world_xyz("/World/start")[:2]

    timestep = 0
    last_teleport_step = -10000      # 直近に立て直した step
    grace = 30                       # 立て直し後この step 数は転倒判定しない(立つ猶予)
    # ===== 起動時に start 位置へ配置 =====
    teleport_to_start()
    aim_camera_at_start()
    last_teleport_step = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # ===== 倒れたら自動で start へ立て直す =====
        up = robot.data.projected_gravity_b[0, 2].item()  # -1≈直立, 0≈横倒し, +1≈逆さ
        if up > -0.6 and (timestep - last_teleport_step) > grace:
            teleport_to_start()
            last_teleport_step = timestep

        # ===== start 球が動いたら、その位置へ再配置 + 視点合わせ =====
        sx, sy, _sz = world_xyz("/World/start")
        if abs(sx - last_start[0]) + abs(sy - last_start[1]) > 0.05:
            teleport_to_start()
            aim_camera_at_start()
            last_teleport_step = timestep
            last_start = (sx, sy)

        # ===== ゴールへ向かう速度指令を毎ステップ生成 =====
        px = robot.data.root_pos_w[0, 0].item()
        py = robot.data.root_pos_w[0, 1].item()
        q = robot.data.root_quat_w[0]  # (w, x, y, z)
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        gx, gy, _ = world_xyz("/World/goal")
        vx, vy, wz = go_to_goal(px, py, yaw, gx, gy)
        cmd_term.vel_command_b[:] = torch.tensor(
            [vx, vy, wz], dtype=torch.float32, device=env.unwrapped.device
        )

        if timestep % 20 == 0:
            d = math.hypot(gx - px, gy - py)
            pz = robot.data.root_pos_w[0, 2].item()
            lv = robot.data.root_lin_vel_b[0]                  # 実際の base 速度
            print(
                f"[dbg] pos=({px:+.2f},{py:+.2f},{pz:+.2f}) up={up:+.2f} "
                f"vel=({lv[0].item():+.2f},{lv[1].item():+.2f}) cmd=({vx:+.2f},{wz:+.2f}) d={d:.2f}",
                flush=True,
            )
        timestep += 1

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()