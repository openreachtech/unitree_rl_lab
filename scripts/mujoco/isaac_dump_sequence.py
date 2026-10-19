"""Isaac 側の基準データ: film_wallcross_sequence.py と同じ時刻表を回し、全 env の状態・観測・行動を npz に保存する。

wallcross_mujoco.py (MuJoCo 版) と突き合わせるための参照。film スクリプトとの違いは
録画しないこと・push を切れること (``--no_push``)・成功判定後 1.0 s で歩行に戻すこと (``--handover_s``)
・そして全ステップを保存すること。共有コードには触らない (env_cfg をこのスクリプト内で書き換えるだけ)。

    PYTHONPATH=source/unitree_rl_lab TMPDIR=$HOME/.cache/isaaclab_tmp \
    /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/mujoco/isaac_dump_sequence.py --headless \
        --num_envs 32 --out reproduce_logs/mujoco_wallcross/isaac_ref.npz --no_push
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll")
parser.add_argument("--walker_task", type=str, default="Go2-Multitask-Gallop-Phase2")
parser.add_argument("--walker", type=str,
                    default="logs/rsl_rl/go2_multitask_gallop_phase2/2026-09-26_02-43-47/model_2500.pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--row", type=int, default=53)
parser.add_argument("--back_gap", type=float, default=0.9)
parser.add_argument("--fixed_distance", type=float, default=2.4,
                    help="壁の内面までの初期距離をこれに揃える (<=0 なら film と同じく back_gap 基準)")
parser.add_argument("--walk_speed", type=float, default=0.6)
parser.add_argument("--approach_speed", type=float, default=0.6)
parser.add_argument("--walk_until", type=float, default=3.2)
parser.add_argument("--rise_at", type=float, default=3.2)
parser.add_argument("--approach_at", type=float, default=3.2)
parser.add_argument("--trigger_gap", type=float, default=0.5)
parser.add_argument("--handover_s", type=float, default=1.0)
parser.add_argument("--walk_after_s", type=float, default=3.0)
parser.add_argument("--stop_after_s", type=float, default=1.0)
parser.add_argument("--max_s", type=float, default=16.0)
parser.add_argument("--no_push", action="store_true")
parser.add_argument("--no_dr", action="store_true", help="friction/mass のランダム化も切る")
parser.add_argument("--fixed_delay", type=int, default=-1, help="actuator delay を固定 (物理ステップ数)。-1 = 学習どおり 0-6 ランダム")
parser.add_argument("--velocity_limit_sim", type=float, default=None, help="PhysX の関節最大速度を上書き (USD: hip/thigh 30.1, calf 15.7)")
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, help="articulation の position/velocity 反復回数 (既定 8 4)")
parser.add_argument("--hrange", type=float, nargs=2, default=None,
                    help="--row の代わりに、高さがこの範囲・厚さ < 0.1 m・half >= 1.45 の壁の区画を全行から探し、env を順番に割り当てる")
parser.add_argument("--map_noise_off", action="store_true", help="MID-360 地図のノイズを切る (MuJoCo 評価の既定と同じ)")
parser.add_argument("--sim_dt", type=float, default=0.0,
                    help=">0: 物理刻みをこれにする (制御 50 Hz は保つ: decimation = 0.02 / sim_dt)。--fixed_delay は 5 ms 単位のまま換算")
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    a = args_cli
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.curriculum.terrain_levels = None
    env_cfg.episode_length_s = a.max_s + 5.0
    hs = env_cfg.commands.handstand
    hs.pinned = False
    hs.trigger_time_range = (a.rise_at, a.rise_at)
    hs.hold_duration_range = (1.0e9, 1.0e9)
    hs.episode_probability = 1.0
    hs.initial_takeoff_speed_limit = 10.0
    wc = env_cfg.commands.wallcross
    wc.jump_episode_probability = 1.0
    wc.trigger_min_time_s = a.rise_at + 1.0
    wc.trigger_max_time_s = 1.0e9
    wc.trigger_gap_range = (a.trigger_gap, a.trigger_gap)
    wc.wall_timeout_s = 1.0e9
    env_cfg.terminations.wall_success_done = None
    vel = env_cfg.commands.base_velocity
    vel.heading_command = False
    vel.resampling_time_range = (1.0e9, 1.0e9)
    # the film starts from the default standing pose, not from the walker state bank
    env_cfg.events.reset_from_walk = None
    if a.no_push:
        env_cfg.events.push_robot = None
    if a.map_noise_off and hasattr(env_cfg.observations.policy, "mid360_map"):
        env_cfg.observations.policy.mid360_map.params["noise"] = None
    if a.no_dr:
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None

    if a.velocity_limit_sim is not None:
        if getattr(env_cfg.events, "joint_speed_limit", None) is not None:
            env_cfg.events.joint_speed_limit = None   # the NoCap/VL tasks rewrite the limits at reset; --velocity_limit_sim wins
        env_cfg.scene.robot.actuators["GO2HV"].velocity_limit_sim = a.velocity_limit_sim
    if a.solver_iters is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = a.solver_iters[0]
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = a.solver_iters[1]
    delay_scale = 1
    if a.sim_dt > 0:
        step = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.decimation = int(round(step / a.sim_dt))
        delay_scale = env_cfg.sim.dt / a.sim_dt
        env_cfg.sim.dt = a.sim_dt
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.scene.contact_forces.update_period = a.sim_dt
        act = env_cfg.scene.robot.actuators["GO2HV"]
        act.min_delay = int(round(act.min_delay * delay_scale))
        act.max_delay = int(round(act.max_delay * delay_scale))
        print(f"[isaac] sim.dt {a.sim_dt} decimation {env_cfg.decimation} delay steps {act.min_delay}-{act.max_delay}", flush=True)
    if a.fixed_delay >= 0:
        env_cfg.scene.robot.actuators["GO2HV"].min_delay = int(round(a.fixed_delay * delay_scale))
        env_cfg.scene.robot.actuators["GO2HV"].max_delay = int(round(a.fixed_delay * delay_scale))
    env = gym.make(a.task, cfg=env_cfg)
    terrain = env.unwrapped.scene.terrain
    if a.hrange is not None:
        from unitree_rl_lab.tasks.wallcross import terrains as wall_terrains
        tab = wall_terrains.WALL_TABLE
        ok = np.argwhere((tab[..., 1] >= a.hrange[0]) & (tab[..., 1] <= a.hrange[1]) & (tab[..., 2] < 0.1)
                         & (tab[..., 0] >= 1.45))
        if len(ok) == 0:
            raise SystemExit(f"no thin wall tile with height in {a.hrange}")
        print(f"[isaac] {len(ok)} tiles with thin walls {a.hrange}: "
              + ", ".join(f"({r},{c}) h {tab[r, c, 1]:.3f} t {tab[r, c, 2]:.3f}" for r, c in ok), flush=True)
        pick = ok[np.arange(env.unwrapped.num_envs) % len(ok)]
        terrain.terrain_levels[:] = torch.as_tensor(pick[:, 0], device=terrain.terrain_levels.device)
        terrain.terrain_types[:] = torch.as_tensor(pick[:, 1], device=terrain.terrain_types.device)
    else:
        terrain.terrain_levels[:] = a.row
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    agent_cfg = cli_args.parse_rsl_rl_cfg(a.task, a)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped
    dev, dt, n = inner.device, inner.step_dt, inner.num_envs

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(a.checkpoint)
    wall_policy = runner.get_inference_policy(device=dev)
    walk_args = copy.copy(a)
    walk_args.checkpoint = a.walker
    walk_cfg = cli_args.parse_rsl_rl_cfg(a.walker_task, walk_args)
    walk_runner = OnPolicyRunner(env, walk_cfg.to_dict(), log_dir=None, device=walk_cfg.device)
    walk_path = a.walker
    pol = walk_runner.alg.policy.state_dict()
    ck = torch.load(a.walker, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    if any(sd[k].shape != pol[k].shape for k in ("actor.0.weight", "critic.0.weight")):
        # height-map env: the blind walker gets zero weights on the trailing columns (same function)
        for k in ("actor.0.weight", "critic.0.weight"):
            w = torch.zeros_like(pol[k], device="cpu")
            w[:, : sd[k].shape[1]] = sd[k]
            sd[k] = w
        walk_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "walker_widened_for_dump.pt")
        torch.save(ck, walk_path)
        print(f"[isaac] walker widened to {pol['actor.0.weight'].shape[1]} actor inputs (zero weights)", flush=True)
    walk_runner.load(walk_path)
    walk_policy = walk_runner.get_inference_policy(device=dev)

    wall = inner.command_manager.get_term("wallcross")
    velocity = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    half, height, thick = wall.wall()
    print("[isaac] joint names:", robot.joint_names)
    print("[isaac] body names:", robot.body_names)

    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += inner.scene.env_origins
    if a.fixed_distance > 0:
        pose[:, 0] += 0.0 - (a.fixed_distance - half)     # inner face at +half → start at half - d
    else:
        pose[:, 0] -= half - a.back_gap
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))

    sensor = inner.scene.sensors["contact_forces"]
    foot_ids, _ = sensor.find_bodies(["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True)
    rec = {k: [] for k in ("obs", "act", "root_pos", "root_quat", "jpos", "jvel", "feet_f", "gap", "on_wall",
                           "triggered", "success", "done", "cmd", "angvel_b", "linvel_b", "body_f", "body_pos")}
    success_at = torch.full((n,), float("nan"), device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    obs = env.get_observations()
    for step in range(int(a.max_s / dt)):
        t = step * dt
        after = t - success_at
        on_wall_policy = (t >= a.rise_at) & ~(after >= a.handover_s)
        cmd = torch.zeros(n, device=dev)
        if t < a.walk_until:
            cmd[:] = a.walk_speed if t >= 0.5 else 0.0
        elif t >= a.approach_at:
            cmd[:] = a.approach_speed
        walking_on = after >= a.handover_s
        cmd = torch.where(walking_on & (after < a.handover_s + a.walk_after_s), a.walk_speed, cmd)
        cmd = torch.where(walking_on & (after >= a.handover_s + a.walk_after_s), 0.0, cmd)
        velocity.vel_command_b[:, 0] = cmd
        velocity.vel_command_b[:, 1:] = 0.0
        o = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        with torch.inference_mode():
            act = torch.where(on_wall_policy[:, None], wall_policy(obs), walk_policy(obs))
            rec["obs"].append(o.cpu().numpy().copy())
            rec["act"].append(act.cpu().numpy().copy())
            rec["on_wall"].append(on_wall_policy.cpu().numpy().copy())
            rec["cmd"].append(cmd.cpu().numpy().copy())
            obs, _, dones, _ = env.step(act)
        rec["root_pos"].append((robot.data.root_pos_w - inner.scene.env_origins).cpu().numpy().copy())
        rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy().copy())
        rec["jpos"].append(robot.data.joint_pos.cpu().numpy().copy())
        rec["jvel"].append(robot.data.joint_vel.cpu().numpy().copy())
        rec["angvel_b"].append(robot.data.root_ang_vel_b.cpu().numpy().copy())
        rec["linvel_b"].append(robot.data.root_lin_vel_b.cpu().numpy().copy())
        rec["feet_f"].append(torch.linalg.norm(sensor.data.net_forces_w[:, foot_ids], dim=-1).cpu().numpy().copy())
        rec["body_f"].append(torch.linalg.norm(sensor.data.net_forces_w, dim=-1).cpu().numpy().copy())
        rec["body_pos"].append((robot.data.body_pos_w - inner.scene.env_origins[:, None]).cpu().numpy().copy())
        rec["gap"].append(wall.base_gap.cpu().numpy().copy())
        rec["triggered"].append(wall.triggered.cpu().numpy().copy())
        rec["success"].append(wall.success.cpu().numpy().copy())
        rec["done"].append(dones.bool().cpu().numpy().copy())
        new_succ = wall.success & torch.isnan(success_at)
        success_at[new_succ] = t
        alive &= ~dones.bool()
        if step % 50 == 0:
            print(f"[isaac] step {step} t {t:.2f} alive {int(alive.sum())} triggered {int(wall.triggered.sum())} "
                  f"success {int(wall.success.sum())}", flush=True)

    out = {k: np.stack(v) for k, v in rec.items()}
    out["half"], out["height"], out["thick"] = (x.cpu().numpy() for x in (half, height, thick))
    out["dt"] = np.array(dt)
    out["sensor_body_names"] = np.array(sensor.body_names)
    out["body_names"] = np.array(robot.body_names)
    out["joint_names"] = np.array(robot.joint_names)
    out["default_joint_pos"] = robot.data.default_joint_pos[0].cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"[isaac] wrote {a.out}", flush=True)
    s = np.isfinite(success_at.cpu().numpy())
    print(f"[isaac] crossed {s.sum()}/{n}, alive at end {int(alive.sum())}/{n}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
