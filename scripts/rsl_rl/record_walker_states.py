"""Record the walking expert's states, to start wall-crossing episodes from a real walk.

Local helper. The crossing is only asked for when the robot is to go over a wall: it walks on four
legs (walking expert), the stance is switched on, and the crossing expert takes over from whatever
state the walk is in. Training so far started every episode standing still, so this records that
hand-over state instead: the walker runs on the wall terrain at the approach speeds, and every
``--every_s`` after ``--from_s`` the state of every robot still up is stored, heading removed.

Saved (torch.save, dict of CPU tensors): joint_pos, joint_vel (N, 12), root_z (N,) above the tile
ground, root_quat_rp (N, 4) roll/pitch only, root_lin_vel_b, root_ang_vel_b (N, 3), cmd (N,).

    python scripts/rsl_rl/record_walker_states.py --walker <model_2500.pt> --out <bank.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-Paced")
parser.add_argument("--walker_task", type=str, default="Go2-Multitask-Gallop-Phase2")
parser.add_argument("--walker", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--speed_range", type=float, nargs=2, default=(0.5, 0.8))
parser.add_argument("--from_s", type=float, default=1.0)
parser.add_argument("--until_s", type=float, default=3.0)
parser.add_argument("--every_s", type=float, default=0.1)
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    a = args_cli
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.curriculum.terrain_levels = None
    env_cfg.scene.terrain.max_init_terrain_level = 0          # low walls; the walk never reaches them anyway
    env_cfg.episode_length_s = a.until_s + 5.0
    hs = env_cfg.commands.handstand
    hs.pinned = False
    hs.trigger_time_range = (1.0e9, 1.0e9)                     # never: four legs throughout
    wc = env_cfg.commands.wallcross
    wc.jump_episode_probability = 1.0
    wc.trigger_min_time_s = 1.0e9                              # never fires
    vel = env_cfg.commands.base_velocity
    vel.heading_command = False
    vel.resampling_time_range = (1.0e9, 1.0e9)

    env = gym.make(a.task, cfg=env_cfg)
    walk_cfg = cli_args.parse_rsl_rl_cfg(a.walker_task, a)
    env = RslRlVecEnvWrapper(env, clip_actions=walk_cfg.clip_actions)
    inner = env.unwrapped
    dev, dt, n = inner.device, inner.step_dt, inner.num_envs
    runner = OnPolicyRunner(env, walk_cfg.to_dict(), log_dir=None, device=walk_cfg.device)
    runner.load(a.walker)
    policy = runner.get_inference_policy(device=dev)

    wall = inner.command_manager.get_term("wallcross")
    velocity = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    # Back towards the wall behind, facing +x, so the walk has room (same placement as the film script).
    half, _, _ = wall.wall()
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += inner.scene.env_origins
    pose[:, 0] -= half - 0.9
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))

    speed = torch.empty(n, device=dev).uniform_(*a.speed_range)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    bank = {k: [] for k in ("joint_pos", "joint_vel", "root_z", "root_quat_rp", "root_lin_vel_b", "root_ang_vel_b", "cmd")}
    every = max(1, round(a.every_s / dt))
    obs = env.get_observations()
    for step in range(int(a.until_s / dt)):
        t = (step + 1) * dt
        velocity.vel_command_b[:, 0] = torch.where(torch.tensor(t >= 0.3, device=dev), speed, torch.zeros_like(speed))
        velocity.vel_command_b[:, 1:] = 0.0
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        alive &= ~dones.bool()
        if t >= a.from_s and step % every == 0:
            ids = alive.nonzero(as_tuple=False).squeeze(-1)
            q = robot.data.root_quat_w[ids]
            yaw = math_utils.yaw_quat(q)
            bank["joint_pos"].append(robot.data.joint_pos[ids].clone())
            bank["joint_vel"].append(robot.data.joint_vel[ids].clone())
            bank["root_z"].append(robot.data.root_pos_w[ids, 2] - inner.scene.env_origins[ids, 2])
            bank["root_quat_rp"].append(math_utils.quat_mul(math_utils.quat_inv(yaw), q))
            bank["root_lin_vel_b"].append(robot.data.root_lin_vel_b[ids].clone())
            bank["root_ang_vel_b"].append(robot.data.root_ang_vel_b[ids].clone())
            bank["cmd"].append(speed[ids].clone())
    out = {k: torch.cat(v).cpu() for k, v in bank.items()}
    torch.save(out, a.out)
    vx = out["root_lin_vel_b"][:, 0]
    print(f"[bank] {out['cmd'].shape[0]} states from {int(alive.sum())}/{n} robots still up; "
          f"vx p10/50/90 {vx.quantile(0.1):.2f}/{vx.median():.2f}/{vx.quantile(0.9):.2f} m/s, "
          f"z p50 {out['root_z'].median():.3f} m -> {a.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
