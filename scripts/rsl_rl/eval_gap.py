"""Gap-only evaluation of a G1-FPM checkpoint: how often does it fall, and under which command?

Runs the policy headless on the concentric-gap terrain only (difficulty in --difficulty) and,
per command class (forward / backward / lateral / slow, and turning or not), counts episode
endings (fall into hole, bad orientation, time out) together with exposure time, so the rates
are per minute of walking under that command.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-FPM")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--difficulty", type=float, nargs=2, default=(0.2, 0.3))
parser.add_argument("--out", required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, entry_point_key="play_env_cfg_entry_point")
gen = env_cfg.scene.terrain.terrain_generator
gen.sub_terrains = {"gap": gen.sub_terrains["gap"]}
gen.num_cols, gen.num_rows = 8, 1
gen.difficulty_range = tuple(args.difficulty)
env_cfg.scene.terrain.max_init_terrain_level = 0
env_cfg.terminations.fell_into_hole.params["terrain_types"] = (0, 1, 2, 3, 4, 5, 6, 7)
p = env_cfg.observations.foothold_patches.foothold_patches.params
p["debug_vis"] = False
p["map_params"]["debug_vis"] = False

agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
ckpt = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)

env = gym.make(args.task, cfg=env_cfg)
env.unwrapped.fpm_load_dir = os.path.dirname(ckpt)
env.unwrapped.fpm_load_iteration = int(re.search(r"model_(\d+)\.pt$", ckpt).group(1))
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)
net = runner.alg.policy
u = env.unwrapped
tm = u.termination_manager

DIRS = ["forward", "backward", "lateral", "slow"]
TURN = ["straight", "turning"]


def classify(cmd):
    v = cmd[:, :2].norm(dim=1)
    ang = torch.atan2(cmd[:, 1], cmd[:, 0]).abs()
    d = torch.full_like(v, 2, dtype=torch.long)  # lateral
    d[ang < math.radians(45)] = 0
    d[ang > math.radians(135)] = 1
    d[v < 0.2] = 3
    t = (cmd[:, 2].abs() > 0.5).long()
    return d * 2 + t


n_cls = len(DIRS) * len(TURN)
exposure = torch.zeros(n_cls, device=u.device)
ends = {k: torch.zeros(n_cls, device=u.device) for k in ("fell_into_hole", "bad_orientation", "time_out")}

obs = env.get_observations()
for step in range(args.steps):
    cmd = u.command_manager.get_command("base_velocity").clone()
    cls = classify(cmd)
    exposure += torch.bincount(cls, minlength=n_cls).float()
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
        net.reset(dones)
    for k in ends:
        hit = tm.get_term(k) & dones.bool()
        if hit.any():
            ends[k] += torch.bincount(cls[hit], minlength=n_cls).float()

minutes = exposure * u.step_dt / 60.0
with open(args.out, "w") as f:
    f.write(f"checkpoint {ckpt}\ndifficulty {args.difficulty}, gap width "
            f"{gen.sub_terrains['gap'].gap_width_range}\n")
    f.write(f"{'command':<20}{'minutes':>9}{'share':>7}{'hole/min':>10}{'flip/min':>10}{'falls':>7}\n")
    for i in range(n_cls):
        name = f"{DIRS[i // 2]}/{TURN[i % 2]}"
        m = float(minutes[i])
        h, b = float(ends["fell_into_hole"][i]), float(ends["bad_orientation"][i])
        f.write(f"{name:<20}{m:>9.1f}{m / float(minutes.sum()):>7.0%}{h / max(m, 1e-6):>10.3f}"
                f"{b / max(m, 1e-6):>10.3f}{int(h + b):>7}\n")
    tot_m = float(minutes.sum())
    tot_f = float(ends["fell_into_hole"].sum() + ends["bad_orientation"].sum())
    f.write(f"total: {tot_m:.0f} min, {int(tot_f)} falls, {tot_f / tot_m:.3f} falls/min, "
            f"time outs {int(ends['time_out'].sum())}\n")
env.close()
app.close()
