# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the terrain encoder behind a frozen blind walking policy.

This is supervised learning, not RL, so it does not go through ``train.py``. A
finished Blind-GRU policy drives the robot across its own phase's terrain; the
encoder watches the grid the LiDAR fan produces and learns to reconstruct the
terrain that is really there, supervised by the clean top-down scan. No gradient
reaches the policy, and the policy never reads the encoder.

Because the policy is frozen *and* blind, the distribution of states the robot
visits does not depend on the encoder at all. That is what makes this simpler than
the reference method: Miki et al. need DAgger because their student's own actions
move the distribution, and here nothing does. The same rollout can also be reused
for several gradient steps without going stale, which matters -- the simulator
costs several times what the backward pass does.

    python scripts/rsl_rl/train_terrain_encoder.py \
        --task Go2-Terrain-Encoder-Phase2 \
        --policy_checkpoint logs/rsl_rl/go2_blind_gru_phase2/2026-08-23_23-35-29/model_6497.pt \
        --num_envs 512 --max_iterations 3000 --headless

Phases chain the way the walking policy's did, by resuming the encoder:

    --resume logs/terrain_encoder/<phase2 run>/encoder_3000.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the terrain encoder with a frozen policy.")
parser.add_argument("--task", type=str, required=True, help="Terrain-encoder task id.")
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    required=True,
    help="Frozen walking policy to drive the robot. Use the phase's own checkpoint.",
)
parser.add_argument("--resume", type=str, default=None, help="Encoder checkpoint to continue from.")
parser.add_argument("--num_envs", type=int, default=512, help="Environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=3000, help="Rollout/train iterations.")
parser.add_argument("--tbptt", type=int, default=None, help="Truncation window; defaults to the rollout length.")
parser.add_argument("--epochs", type=int, default=2, help="Gradient passes over each rollout.")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Adam learning rate.")
parser.add_argument(
    "--body_weight",
    type=float,
    default=1.0,
    help="Loss weight on the never-measurable body-footprint cells, relative to 1.0 elsewhere.",
)
parser.add_argument(
    "--arch",
    choices=("belief", "convgru"),
    default="belief",
    help=(
        "Which encoder to train. 'belief' is the adopted default; 'convgru' is kept "
        "because it wins on walls. See sandbox/TERRAIN_ENCODER.md."
    ),
)
parser.add_argument(
    "--belief_latent",
    type=int,
    default=96,
    help="l_e width for --arch belief. 96 is the paper's total; config.yaml uses 24.",
)
parser.add_argument("--hidden_channels", type=int, default=16)
parser.add_argument("--extero_channels", type=int, default=16)
parser.add_argument("--save_interval", type=int, default=100, help="Iterations between checkpoints.")
parser.add_argument("--log_interval", type=int, default=10, help="Iterations between console lines.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.models.modules.runners import UnitreeOnPolicyRunner
from unitree_rl_lab.assets.models.terrain_encoder import gaussian_nll_loss
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import GO2_HEIGHT_SCAN_OFFSET
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_terrain_encoder import (
    build_terrain_encoder,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


@torch.no_grad()
def collect(env, policy, steps: int) -> dict[str, torch.Tensor]:
    """Roll the frozen policy forward and record what the encoder needs.

    ``done[t]`` belongs with the transition out of step ``t``, so the observation at
    ``t + 1`` is already post-reset. The training pass zeroes the recurrent state on
    that boundary -- terrain remembered across a reset is memory of another patch of
    ground.
    """
    obs = env.get_observations()
    out = {k: [] for k in ("lidar", "proprio", "target", "unobserved", "done")}
    for _ in range(steps):
        out["lidar"].append(obs["lidar_map"].clone())
        out["proprio"].append(obs["policy"].clone())
        out["target"].append(obs["terrain_target"].clone())
        out["unobserved"].append(env.unwrapped.lidar_map_unobserved_cells.clone())
        obs, _, done, _ = env.step(policy(obs))
        out["done"].append(done.bool().clone())
    return {k: torch.stack(v) for k, v in out.items()}


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Straight from the registry rather than through cli_args: that helper folds in
    # rsl_rl's own flag semantics, where --resume is a boolean paired with --load_run,
    # and here --resume names an encoder checkpoint.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    window = args_cli.tbptt or agent_cfg.num_steps_per_env
    device = torch.device(agent_cfg.device)

    log_dir = os.path.join(
        os.path.abspath("logs"), "terrain_encoder", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "args.yaml"), vars(args_cli))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Logging to {log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # The walking policy: loaded, then left alone. eval() so its own GRU and any
    # observation normaliser stop updating, and no_grad throughout collection.
    runner = UnitreeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.policy_checkpoint)
    policy = runner.get_inference_policy(device=device)
    for p in runner.alg.policy.parameters():
        p.requires_grad_(False)
    runner.alg.policy.eval()
    print(f"[INFO] Frozen policy from {args_cli.policy_checkpoint}", flush=True)

    encoder = build_terrain_encoder(
        device,
        extero_channels=args_cli.extero_channels,
        hidden_channels=args_cli.hidden_channels,
        arch=args_cli.arch,
        belief_latent=args_cli.belief_latent,
    )
    start_iter = 0
    if args_cli.resume:
        state = torch.load(args_cli.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        start_iter = state.get("iteration", 0)
        print(f"[INFO] Encoder resumed from {args_cli.resume} at iteration {start_iter}", flush=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args_cli.lr)
    if args_cli.resume and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    n_param = sum(p.numel() for p in encoder.parameters())
    print_dict(
        {
            "task": args_cli.task,
            "architecture": args_cli.arch,
            "envs": env.num_envs,
            "grid": encoder.grid_shape,
            "observation cells": int(encoder.keep_index.numel()),
            "body cells": int(encoder.body_mask(device).sum()),
            "tbptt window": window,
            "epochs per rollout": args_cli.epochs,
            "encoder parameters": n_param,
            "hidden state floats per env": encoder.hidden_channels * encoder.grid_shape[0] * encoder.grid_shape[1],
        },
        nesting=1,
    )

    body = encoder.body_mask(device)
    weight = torch.ones(encoder.grid_shape, device=device)
    weight[body] = args_cli.body_weight

    hidden = encoder.init_hidden(env.num_envs, device)
    t_start = time.time()

    for it in range(start_iter, start_iter + args_cli.max_iterations):
        t_collect = time.time()
        batch = collect(env, policy, window)
        collect_s = time.time() - t_collect

        height = torch.stack([encoder.scatter_observation(x) for x in batch["lidar"]])
        target = batch["target"].view(window, env.num_envs, *encoder.grid_shape)
        unobserved = batch["unobserved"].view(window, env.num_envs, *encoder.grid_shape)

        t_train = time.time()
        for epoch in range(args_cli.epochs):
            state_h = hidden
            total = 0.0
            sq_err, counts = torch.zeros(3, device=device), torch.zeros(3, device=device)
            for t in range(window):
                if t > 0:
                    state_h = encoder.mask_hidden(state_h, batch["done"][t - 1])
                mean, log_std, state_h = encoder(height[t], batch["proprio"][t], state_h)
                total = total + gaussian_nll_loss(mean, log_std, target[t], weight)
                if epoch == args_cli.epochs - 1:
                    with torch.no_grad():
                        err = (mean - target[t]).pow(2)
                        # measured / in-band but unmeasured / never measurable
                        seen = ~unobserved[t] & ~body
                        blind = unobserved[t] & ~body
                        for i, m in enumerate((seen, blind, body.expand_as(err))):
                            sq_err[i] += err[m].sum()
                            counts[i] += m.sum()
            loss = total / window
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
        train_s = time.time() - t_train

        # Carry the state into the next window, cutting the graph and any episode
        # that ended on the last step.
        hidden = encoder.mask_hidden(state_h.detach(), batch["done"][-1])

        rmse = (sq_err / counts.clamp_min(1.0)).sqrt()
        writer.add_scalar("Loss/nll", loss.item(), it)
        writer.add_scalar("RMSE/measured", rmse[0].item(), it)
        writer.add_scalar("RMSE/occluded", rmse[1].item(), it)
        writer.add_scalar("RMSE/body", rmse[2].item(), it)
        writer.add_scalar("Timing/collect_s", collect_s, it)
        writer.add_scalar("Timing/train_s", train_s, it)
        writer.add_scalar("Lidar/unobserved_rate", float(getattr(env.unwrapped, "lidar_map_unobserved_rate", 0.0)), it)
        # What the robot is actually doing, so a run that quietly crawls or sits on the
        # easy rows shows up as a number rather than as a disappointing encoder.
        terrain = env.unwrapped.scene.terrain
        levels = getattr(terrain, "terrain_levels", None)
        if levels is not None:
            writer.add_scalar("Env/terrain_level_mean", levels.float().mean().item(), it)
            writer.add_scalar("Env/terrain_level_max", levels.max().item(), it)
        speed = env.unwrapped.command_manager.get_command("base_velocity")[:, :2].norm(dim=-1)
        writer.add_scalar("Env/commanded_speed_mean", speed.mean().item(), it)
        writer.add_scalar("Env/commanded_speed_max", speed.max().item(), it)
        writer.add_scalar("Env/episode_length_mean", env.unwrapped.episode_length_buf.float().mean().item(), it)

        if it % args_cli.log_interval == 0:
            print(
                f"it {it:6d}  nll {loss.item():8.4f}  "
                f"RMSE m/o/b {rmse[0] * 100:5.2f} / {rmse[1] * 100:5.2f} / {rmse[2] * 100:5.2f} cm  "
                f"cmd {speed.mean():.2f} m/s  lvl {levels.float().mean():.1f}  "
                f"ep {env.unwrapped.episode_length_buf.float().mean():5.0f}  "
                f"sim {collect_s:5.2f}s train {train_s:5.2f}s  "
                f"elapsed {(time.time() - t_start) / 60:6.1f} min",
                flush=True,
            )
        if (it + 1) % args_cli.save_interval == 0 or it + 1 == start_iter + args_cli.max_iterations:
            path = os.path.join(log_dir, f"encoder_{it + 1}.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": it + 1,
                    "grid_shape": encoder.grid_shape,
                    "hidden_channels": encoder.hidden_channels,
                    "arch": args_cli.arch,
                    "belief_latent": args_cli.belief_latent,
                    "extero_channels": args_cli.extero_channels,
                    "height_scale": encoder.height_scale,
                    "height_offset": encoder.height_offset,
                    "height_scan_offset": GO2_HEIGHT_SCAN_OFFSET,
                },
                path,
            )
            print(f"[INFO] saved {path}", flush=True)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
