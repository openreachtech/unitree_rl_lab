"""Check what the MID-360 map shows: at the jump cue (hind stance) and 1 s before (rising / four legs).

Local helper. Runs the wall policy on the Mid360 task, and for each env records the map at the cue and
one second earlier. Prints, per forward row of the map (central 5 columns, |y| <= 0.1 m), the highest
height above the ground, next to the true wall position (base gap) and height.

    python scripts/rsl_rl/probe_wallcross_mid360.py --checkpoint <ckpt> --row 53 --headless --device cpu
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-FromWalkLandAll-Mid360")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--row", type=int, default=53)
parser.add_argument("--max_s", type=float, default=8.0)
parser.add_argument("--no_noise", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.wallcross.mid360 import MAP_OFFSET  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    a = args_cli
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    env_cfg.curriculum.terrain_levels = None
    if a.no_noise:
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
    wall = inner.command_manager.get_term("wallcross")
    robot = inner.scene["robot"]
    n, dt = inner.num_envs, inner.step_dt
    half, height, thick = wall.wall()
    hist = []          # (map heights above ground, base gap, pitch) per step
    cue_step = torch.full((n,), -1, dtype=torch.long)
    obs = env.get_observations()
    for step in range(int(a.max_s / dt)):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        mp = (obs["policy"] if not torch.is_tensor(obs) else obs)[:, -609:]
        base_z = robot.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]
        above = base_z.unsqueeze(-1) - mp - MAP_OFFSET
        q = robot.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        bxz = 2 * (x * z - w * y)
        hist.append((above.cpu(), wall.base_gap.cpu(), bxz.cpu(), base_z.cpu()))
        new = wall.triggered.cpu() & (cue_step < 0)
        cue_step[new] = step
    xs = torch.linspace(-0.7, 0.7, 29)
    for e in range(n):
        s = int(cue_step[e])
        if s < 0:
            print(f"env {e}: no cue")
            continue
        print(f"\nenv {e}: wall height {float(height[e]):.2f} thickness {float(thick[e]):.3f}")
        for label, k in (("cue (stance)", s), ("1 s before", max(0, s - 50))):
            ab, gap, bxz, bz = (t[e] for t in hist[k])
            g = ab.view(29, 21)[:, 8:13].max(dim=1).values
            row = " ".join(f"{float(v):.2f}" for v in g[14:])
            print(f"  {label:13s} base gap {float(gap):.2f} m, base z {float(bz):.2f}, body-x up {float(bxz):+.2f}")
            print(f"     ahead x=0.00..0.70: {row}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
