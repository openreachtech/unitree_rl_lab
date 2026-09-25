"""Isaac 側の踏切を mujoco の判定とまったく同じ手順で測る（2026-09-16）。

同じ ckpt の離陸 v_z が Isaac 1.44 / mujoco 2.18 と 1.5 倍違う。これがシミュレータの差なのか
指標の定義の差なのかを切り分けるため、Isaac 側でも mujoco と同じ後処理をする:

  1. jump_command の立ち上がりを踏切のトリガとする
  2. その後の滞空区間のうち「滞空 >= 0.20 s かつ 上昇 >= 0.08 m」を満たす最初のものを跳躍とする
     （mujoco_jump_eval.py の real_jump ゲートと同じ）
  3. その離陸時点の胴体速度を v_x / v_z とし、直前の連続接地区間を踏切とする

JumpCommand が持つ latch（窓内で最も v_z の大きい離陸を採る）は使わない。mujoco 側は
「最初に条件を満たした滞空」を採っているので、そこを揃える。
"""
import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Unitree-Go2-LongJump-v1")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--sim_dt", type=float, default=None,
                    help="物理ステップを変える。制御周期 0.02 s は decimation で保つ。")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

AIR_S = 0.20
RISE_M = 0.08
TAKEOFF_MAX_S = 0.40


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.sim_dt:
        # step_dt = sim.dt * decimation を 0.02 s に保ったまま物理刻みだけ変える。
        env_cfg.sim.dt = args_cli.sim_dt
        env_cfg.decimation = int(round(0.02 / args_cli.sim_dt))
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.scene.contact_forces.update_period = env_cfg.sim.dt
        print(f"[TRACE] sim.dt={env_cfg.sim.dt} decimation={env_cfg.decimation}", flush=True)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cmd = u.command_manager.get_term("jump_command")
    robot = u.scene["robot"]
    dt = u.step_dt
    knee = [i for i, n in enumerate(robot.data.joint_names) if "calf" in n]
    print(f"[TRACE] dt={dt} knee joints={[robot.data.joint_names[i] for i in knee]}", flush=True)

    rec = {k: [] for k in ("air", "z", "vz", "vxy", "jc", "wmax", "wknee")}
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    with torch.inference_mode():
        for i in range(args_cli.steps):
            if i % 500 == 0:
                print(f"[TRACE] step {i}/{args_cli.steps}", flush=True)
            actions = policy(obs)
            obs = env.step(actions)[0]
            v = robot.data.root_lin_vel_w
            jv = robot.data.joint_vel
            rec["air"].append(cmd.airborne_now.clone().cpu().numpy())
            rec["z"].append(robot.data.root_pos_w[:, 2].clone().cpu().numpy())
            rec["vz"].append(v[:, 2].clone().cpu().numpy())
            rec["vxy"].append(torch.norm(v[:, :2], dim=-1).clone().cpu().numpy())
            rec["jc"].append(cmd.jump_command[:, 0].clone().cpu().numpy())
            rec["wmax"].append(jv.abs().max(dim=1).values.clone().cpu().numpy())
            rec["wknee"].append(jv[:, knee].abs().max(dim=1).values.clone().cpu().numpy())
    A = {k: np.array(v) for k, v in rec.items()}       # [T, E]
    T, E = A["air"].shape

    air_min = int(round(AIR_S / dt))
    jumps = []
    for e in range(E):
        air = A["air"][:, e].astype(bool)
        jc = A["jc"][:, e] > 0.5
        trig = np.where(jc[1:] & ~jc[:-1])[0] + 1
        for t0 in trig:
            t_end = min(T, t0 + int(round(2.2 / dt)))
            i = t0
            while i < t_end:
                if air[i]:
                    j = i
                    while j < t_end and air[j]:
                        j += 1
                    if (j - i) >= air_min and (A["z"][i:j, e].max() - A["z"][i, e]) >= RISE_M:
                        k0 = i - 1
                        while k0 > t0 and not air[k0 - 1] and (i - (k0 - 1)) * dt <= TAKEOFF_MAX_S:
                            k0 -= 1
                        if i - k0 >= 1:
                            jumps.append(dict(
                                vz=A["vz"][i, e], vx=A["vxy"][i, e],
                                air=(j - i) * dt, rise=A["z"][i:j, e].max() - A["z"][i, e],
                                stance=(i - k0) * dt,
                                wmax=A["wmax"][k0:i, e].max(), wknee=A["wknee"][k0:i, e].max(),
                            ))
                        break
                    i = j
                else:
                    i += 1

    lines = []
    def out(t):
        print(t, flush=True)
        lines.append(t)
    out(f"=== [TRACE] {args_cli.label} ===")
    out(f"  トリガ数に対する跳躍成立   {len(jumps)} 件")
    if jumps:
        g = lambda k: float(np.mean([j[k] for j in jumps]))
        out(f"  離陸 v_x / v_z             {g('vx'):.3f} / {g('vz'):.3f}  (積 {g('vx')*g('vz'):.3f})")
        out(f"  弾道の飛距離 2*vx*vz/g     {2*g('vx')*g('vz')/9.81:.3f} m")
        out(f"  滞空 / 上昇                {g('air'):.3f} s / {g('rise'):.3f} m")
        out(f"  踏切の接地時間             {1000*g('stance'):.0f} ms")
        out(f"  踏切中の関節速度ピーク     全体 {g('wmax'):.1f} rad/s  膝 {g('wknee'):.1f} rad/s")
    if args_cli.out:
        open(args_cli.out, "w").write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
