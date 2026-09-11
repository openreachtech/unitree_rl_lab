# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, noise-controlled wall-crossing evaluation for a Go2W Phase5 checkpoint.

Runs the task's *Play* environment (wall-only, one pinned wall height per column) with
the checkpoint's policy for a fixed number of completed episodes per column and reports,
per wall height: arrival rate (goal reached as the command term defines it), crossing
rate (base got past the wall ring's far face at any point), and the termination mix.

Why this exists (sandbox/SUMMARY.md, 2026-09-09..11): training statistics are collected under
the stochastic policy (mean_noise_std ~1.5 x action scale 0.25 = ~0.38 rad of joint-
target noise per step), while Play and MuJoCo run the deterministic mean. The two have
disagreed on whether the policy crosses 0.60 m. ``--stochastic`` samples actions from
the policy's distribution instead of taking the mean, so the same checkpoint can be
measured both ways under otherwise identical conditions.

Differences from Play, for a clean measurement: ``rel_standing_envs`` is forced to 0
(no zero-command episodes), and the interval ``push_robot`` event is disabled unless
``--keep-push`` is given. Observation noise stays as the task configures it.

Example:
    python scripts/rsl_rl/eval_wall.py --task Go2w-v1-Phase5 --headless \\
        --num_envs 128 --episodes 100
    python scripts/rsl_rl/eval_wall.py --task Go2w-v1-Phase5 --headless \\
        --num_envs 128 --episodes 100 --stochastic
"""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate wall-crossing rate of an RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments (spread over the Play columns).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--episodes", type=int, default=100, help="Completed episodes required per wall height.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions instead of using the mean.")
parser.add_argument("--keep-push", action="store_true", default=False, help="Keep the interval push_robot event.")
parser.add_argument("--max_steps", type=int, default=20000, help="Safety cap on environment steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Measurement hygiene (see module docstring).
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if not args_cli.keep_push and getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    # No RSI in the eval even if a try's Play cfg ever enables it.
    if getattr(env_cfg.events, "rsi_climb_keyframe", None) is not None:
        env_cfg.events.rsi_climb_keyframe = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    policy_nn.eval()
    infer = runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.stochastic:
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

        def act(obs):
            if normalizer is not None:
                obs = normalizer(obs)
            return policy_nn.act(obs)
    else:
        act = infer

    uenv = env.unwrapped
    dev = uenv.device
    terrain = uenv.scene.terrain
    sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    num_cols = terrain.cfg.terrain_generator.num_cols
    col_of_env = terrain.terrain_types.clone()  # column index per env (fixed for the run)
    # Column -> sub-terrain name, replicating TerrainGenerator's proportion-based assignment.
    props = torch.tensor([terrain.cfg.terrain_generator.sub_terrains[n].proportion for n in sub_names])
    cums = torch.cumsum(props / props.sum(), dim=0)
    col_name = [sub_names[int(torch.nonzero(c / num_cols + 0.001 < cums)[0])] for c in range(num_cols)]
    heights = {}
    for c, n in enumerate(col_name):
        r = getattr(terrain.cfg.terrain_generator.sub_terrains[n], "wall_height_range", None)
        heights[c] = r[0] if r is not None else float("nan")

    cmd = uenv.command_manager.get_term("base_velocity")
    arrival_radius = cmd.cfg.arrival_radius
    far_face = 1.45  # thin_wall ring centreline 1.25 + half thickness 0.20 (Phase5 tile)

    ever_arrived = torch.zeros(uenv.num_envs, dtype=torch.bool, device=dev)
    last_dist = torch.full((uenv.num_envs,), float("inf"), device=dev)  # goal distance on the env's last live step
    max_r = torch.zeros(uenv.num_envs, device=dev)
    term_names = [n for n in uenv.termination_manager.active_terms]
    # "arrived": within arrival_radius of the goal at any point in the episode (a crossing
    # followed by anything). "held": within arrival_radius on the episode's final step --
    # the condition mdp.goal_arrival_reward actually pays for, i.e. arrive *and* stay.
    stats = {c: {"episodes": 0, "arrived": 0, "held": 0, "crossed": 0, **{t: 0 for t in term_names}} for c in range(num_cols)}

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    steps = 0
    with torch.inference_mode():
        while simulation_app.is_running() and steps < args_cli.max_steps:
            actions = act(obs)
            obs, _, dones, _ = env.step(actions)
            steps += 1

            pos = uenv.scene["robot"].data.root_pos_w[:, :2]
            dist_goal = torch.norm(cmd.goal_pos_w - pos, dim=-1)
            r = torch.norm(pos - uenv.scene.env_origins[:, :2], dim=-1)
            # dones envs have already been reset by env.step; their pre-reset state was
            # captured in the previous iteration's ever_arrived / max_r, so read those.
            done_ids = torch.nonzero(dones.bool(), as_tuple=False).flatten()
            for i in done_ids.tolist():
                c = int(col_of_env[i])
                s = stats[c]
                s["episodes"] += 1
                s["arrived"] += int(ever_arrived[i])
                s["held"] += int(last_dist[i] < arrival_radius)
                s["crossed"] += int(max_r[i] > far_face)
                for t in term_names:
                    s[t] += int(uenv.termination_manager.get_term(t)[i])
            # reset trackers for the envs that just restarted, then update the rest
            ever_arrived[done_ids] = False
            max_r[done_ids] = 0.0
            last_dist[done_ids] = float("inf")
            live = ~dones.bool()
            ever_arrived |= live & (dist_goal < arrival_radius)
            last_dist = torch.where(live, dist_goal, last_dist)
            max_r = torch.where(live, torch.maximum(max_r, r), max_r)

            if all(stats[c]["episodes"] >= args_cli.episodes for c in range(num_cols)):
                break

    mode = "stochastic" if args_cli.stochastic else "deterministic"
    lines = [
        f"# eval_wall: {args_cli.task} ({mode})",
        f"checkpoint: {resume_path}",
        f"num_envs: {uenv.num_envs}, steps: {steps}, push_robot: {'on' if args_cli.keep_push else 'off'}, rel_standing_envs: 0",
        "",
        "| wall | episodes | arrived (ever) | held (at end) | crossed (r > 1.45) | " + " | ".join(term_names) + " |",
        "|---|---|---|---|---|" + "---|" * len(term_names),
    ]
    for c in range(num_cols):
        s = stats[c]
        n = max(s["episodes"], 1)
        lines.append(
            f"| {heights[c]:.2f} m | {s['episodes']} | {s['arrived'] / n:.2f} | {s['held'] / n:.2f} | {s['crossed'] / n:.2f} | "
            + " | ".join(f"{s[t] / n:.2f}" for t in term_names)
            + " |"
        )
    report = "\n".join(lines)
    print("\n" + report + "\n")
    tag = mode + ("_push" if args_cli.keep_push else "")
    out = os.path.join(os.path.dirname(resume_path), f"eval_wall_{tag}_{os.path.basename(resume_path).replace('.pt', '')}.md")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"[INFO] written: {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
