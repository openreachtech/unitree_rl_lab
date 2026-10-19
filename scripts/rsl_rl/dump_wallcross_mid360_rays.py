"""Dump what the Isaac MID-360 sensor and map produce, step by step, for the MuJoCo cross-check.

Local helper for ``scripts/mujoco/mid360_selftest.py --isaac <npz>``. Runs the wall policy on the Mid360
task (noise off) and records, per control step and env: the scan-window index, the base pose, the world
ray origins / directions / hits of the sensor and the 609-column map the policy saw.

    python scripts/rsl_rl/dump_wallcross_mid360_rays.py --checkpoint <ckpt> --headless --device cpu \
        --out reproduce_logs/mid360/isaac_rays.npz
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--row", type=int, default=53)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    env_cfg.curriculum.terrain_levels = None
    env_cfg.observations.policy.mid360_map.params["noise"] = None
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(a.task, cfg=env_cfg)
    terrain = env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = a.row
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    agent_cfg = cli_args.parse_rsl_rl_cfg(a.task, a)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(a.checkpoint)
    policy = runner.get_inference_policy(device=inner.device)
    env.reset()
    robot = inner.scene["robot"]
    sensor = inner.scene.sensors["mid360_scanner"]
    rec = {k: [] for k in ("window", "root_pos", "root_quat", "starts", "dirs", "hits", "map", "done", "sensor_pos", "sensor_quat")}

    def grab(obs, done):
        mp = (obs["policy"] if not torch.is_tensor(obs) else obs)[:, -609:]
        rec["window"].append(int(sensor._window_index))
        rec["root_pos"].append(robot.data.root_pos_w.cpu().numpy().copy())
        rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy().copy())
        rec["starts"].append(sensor._ray_starts_w.cpu().numpy().copy())
        rec["dirs"].append(sensor._ray_directions_w.cpu().numpy().copy())
        rec["hits"].append(sensor.data.ray_hits_w.cpu().numpy().copy())
        rec["sensor_pos"].append(sensor.data.pos_w.cpu().numpy().copy())
        rec["sensor_quat"].append(sensor.data.quat_w.cpu().numpy().copy())
        rec["map"].append(mp.cpu().numpy().copy())
        rec["done"].append(done.cpu().numpy().copy())

    obs = env.get_observations()
    grab(obs, torch.ones(inner.num_envs, dtype=torch.bool))
    for _ in range(a.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        grab(obs, dones.bool())
    out = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))
    print("windows:", out["window"][:12].tolist(), "...")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
