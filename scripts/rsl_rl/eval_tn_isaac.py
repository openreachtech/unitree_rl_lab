"""同じポリシーを「実機相当の機体」と「mujoco相当の機体」で走らせて比べる（2026-09-16）。

学習では T-N 曲線を DR している（`actuator_torque_speed_curve`、speed_scale 0.80-2.50）。
speed_scale=1.0 は X2=30 rad/s ＝ 実モーター GO-M8010-6 の無負荷回転数そのもの＝**実機相当**。
speed_scale=2.5 は X2=75 rad/s で、踏切の関節速度ピーク（膝で 37-51 rad/s）を全部飲み込む
＝**デレート実質ゼロ＝mujoco相当**。既定の (0.80, 2.50) は平均 1.65 倍甘い機体の分布で、
つまり学習ログの `unaided_*` は実機相当の性能ではない。

TN_SPEED_SCALE_MIN/MAX（と TORQUE 側）で帯を固定して、同じ ckpt を両端で測る。
Isaac 内の対照実験なので「Isaac の順位は mujoco で崩れる」問題を踏まない。
"""
import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Unitree-Go2-LongJump-v1")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--out", type=str, default=None, help="結果を書くファイル")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs   # play cfg は 1 に固定するので上書き
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = retrieve_file_path(args_cli.checkpoint)

    tn = env_cfg.events.actuator_torque_speed_curve.params
    print(f"\n[TN-EVAL] {args_cli.label}  speed_scale={tn['speed_scale_range']} "
          f"torque_scale={tn['torque_scale_range']}  envs={args_cli.num_envs}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd = env.unwrapped.command_manager.get_term("jump_command")
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # 跳躍ごとの結果を集める。last_jump_distance 等は次の跳躍まで値が残るので、
    # 「real_jump が立っている env の、その時点の値」を毎ステップ足して時間平均を取る
    # （学習ログの unaided_* と同じ集計）。
    acc = {k: 0.0 for k in ("d", "vx", "vz", "h", "air", "trig")}
    n_scored = 0
    n_real = 0
    n_step = 0
    with torch.inference_mode():
        for _i in range(args_cli.steps):
            if _i % 500 == 0:
                print(f"[TN-EVAL] step {_i}/{args_cli.steps} scored={n_scored}", flush=True)
            actions = policy(obs)
            out = env.step(actions)
            obs = out[0]
            scored = cmd.real_jump
            k = int(scored.sum().item())
            n_step += env.unwrapped.num_envs
            n_real += k
            if k:
                acc["d"] += float(cmd.last_jump_distance[scored].sum())
                acc["vx"] += float(cmd.liftoff_vel_x[scored].sum())
                acc["vz"] += float(cmd.liftoff_vel_z[scored].sum())
                acc["h"] += float(cmd.max_height_gain[scored].sum())
                acc["air"] += float(cmd.airborne_time[scored].sum())
                acc["trig"] += float(cmd.trigger_speed[scored].sum())
                n_scored += k

    lines = []
    def out(t):
        print(t, flush=True)
        lines.append(t)
    out(f"=== [TN-EVAL] {args_cli.label} ===")
    out(f"  speed_scale       {tn['speed_scale_range']}")
    out(f"  real_jump の割合  {100*n_real/max(n_step,1):.1f}%")
    if n_scored:
        s = lambda k: acc[k] / n_scored
        out(f"  跳躍距離          {s('d'):.3f} m")
        out(f"  離陸 v_x / v_z    {s('vx'):.3f} / {s('vz'):.3f}   (積 {s('vx')*s('vz'):.3f})")
        out(f"  最高到達高さ      {s('h'):.3f} m")
        out(f"  滞空              {s('air'):.3f} s")
        out(f"  トリガ時速度      {s('trig'):.3f} m/s")
    else:
        out("  real_jump がゼロ")
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
