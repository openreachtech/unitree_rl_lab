"""Measure a walking expert against fixed velocity commands: what it delivers, and how often it falls.

Local helper (not part of the branch). ``measure_run_speed.py`` answers the same first question for
forward speed, but it reads ``actuators["GO2HV"]`` and so cannot run on a robot whose actuators are
split into groups (Anaguma). This one only needs the root state and the termination manager, so it
works for any robot, and it also sweeps backward / lateral / turning commands.

Every command is an independent trial: all environments are reset, the command is pinned, the first
``--settle_s`` seconds are discarded and the next ``--measure_s`` are averaged. A fall is any
termination other than the time-out; a fallen environment is reset by the env and keeps counting, so
``fall/env`` is falls per environment over the measuring window.

Several checkpoints are evaluated in one process (the env is built once).

    python scripts/rsl_rl/measure_walk_cmds.py --task Anaguma-Multitask-Gallop-Phase2 --headless \
        --checkpoints logs/rsl_rl/.../model_2400.pt,logs/rsl_rl/.../model_2500.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint paths.")
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--measure_s", type=float, default=5.0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.tasks.locomotion.mdp.rewards import _local_stride_phase  # noqa: E402
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_run import (  # noqa: E402
    GAIT_BOUND,
    GAIT_GALLOP_ROTARY,
    GAIT_GALLOP_TRANSVERSE,
    GAIT_PACE,
    GAIT_PRONK,
    GAIT_TROT,
)

# Gait classification copied from measure_run_speed.py (which cannot be imported: it parses its own
# arguments at import time). Offsets are the stride phases of FR, RL, RR relative to FL.
CANONICAL_GAITS = {
    "pronk": GAIT_PRONK,
    "trot": GAIT_TROT,
    "pace": GAIT_PACE,
    "bound": GAIT_BOUND,
    "gallop-rotary": GAIT_GALLOP_ROTARY,
    "gallop-transverse": GAIT_GALLOP_TRANSVERSE,
}
GAIT_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def circular_mean(x: torch.Tensor) -> torch.Tensor:
    t = x * 2.0 * torch.pi
    return (torch.atan2(t.sin().mean(), t.cos().mean()) / (2.0 * torch.pi)) % 1.0


def circular_dist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def classify_gait(offsets) -> tuple[str, float]:
    scored = {n: sum(circular_dist(o, c) for o, c in zip(offsets, g)) for n, g in CANONICAL_GAITS.items()}
    best = min(scored, key=scored.get)
    return best, scored[best]

# (label, vx, vy, wz)
COMMANDS = [
    ("stand", 0.0, 0.0, 0.0),
    ("fwd 0.5", 0.5, 0.0, 0.0),
    ("fwd 1.0", 1.0, 0.0, 0.0),
    ("fwd 1.5", 1.5, 0.0, 0.0),
    ("fwd 2.0", 2.0, 0.0, 0.0),
    ("fwd 2.5", 2.5, 0.0, 0.0),
    ("fwd 3.0", 3.0, 0.0, 0.0),
    ("fwd 3.5", 3.5, 0.0, 0.0),
    ("back 0.5", -0.5, 0.0, 0.0),
    ("back 1.0", -1.0, 0.0, 0.0),
    ("left 0.5", 0.0, 0.5, 0.0),
    ("right 0.5", 0.0, -0.5, 0.0),
    ("left 1.0", 0.0, 1.0, 0.0),
    ("turn L 1.0", 0.0, 0.0, 1.0),
    ("turn R 1.0", 0.0, 0.0, -1.0),
]


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point"
    )
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(env_cfg.commands.base_velocity, "heading_command"):
        env_cfg.commands.base_velocity.heading_command = False
    assist = getattr(env_cfg.commands, "tow_assist", None)
    if assist is not None and assist.initial_assist_scale != 0.0:
        raise SystemExit("tow_assist is on in the play cfg; the measurement would include the tow.")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    asset = unwrapped.scene["robot"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    all_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    dt = unwrapped.step_dt
    settle_steps, measure_steps = int(args_cli.settle_s / dt), int(args_cli.measure_s / dt)
    from isaaclab.managers import SceneEntityCfg

    contact = unwrapped.scene.sensors["contact_forces"]
    foot_cfg = SceneEntityCfg("contact_forces", body_names=GAIT_FEET, preserve_order=True)
    foot_cfg.resolve(unwrapped.scene)
    foot_ids = foot_cfg.body_ids
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    for ckpt in args_cli.checkpoints.split(","):
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=unwrapped.device)
        print(f"\n=== {ckpt}", flush=True)
        print(f"{'command':<11} {'vx':>6} {'vy':>6} {'wz':>6} | {'got vx':>6} {'got vy':>6} {'got wz':>6} | {'fall/env':>8} | {'gait':<17} {'offsets FR/RL/RR':<16} {'flight%':>7} {'Hz':>5}")
        for label, vx, vy, wz in COMMANDS:
            with torch.inference_mode():
                obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            r = command_term.cfg.ranges
            r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
            command_term._resample_command(all_ids)
            acc = torch.zeros(3, device=unwrapped.device)
            falls = 0
            phases, flight, stride = [], 0.0, 0.0
            for step in range(settle_steps + measure_steps):
                with torch.inference_mode():
                    obs, _, dones, extras = env.step(policy(obs))
                command_term._resample_command(all_ids)
                if step >= settle_steps:
                    acc[0] += asset.data.root_lin_vel_b[:, 0].mean()
                    acc[1] += asset.data.root_lin_vel_b[:, 1].mean()
                    acc[2] += asset.data.root_ang_vel_b[:, 2].mean()
                    timeouts = extras.get("time_outs", torch.zeros_like(dones))
                    falls += int((dones.bool() & ~timeouts.bool()).sum())
                    cc = contact.data.current_contact_time[:, foot_ids]
                    ca = contact.data.current_air_time[:, foot_ids]
                    lc = contact.data.last_contact_time[:, foot_ids]
                    la = contact.data.last_air_time[:, foot_ids]
                    phases.append(_local_stride_phase(cc, ca, cc > 0.0, lc, la))
                    flight += float((cc <= 0.0).all(dim=1).float().mean())
                    stride += float((lc + la).clamp(min=1e-3).mean())
            got = (acc / measure_steps).tolist()
            ph = torch.cat(phases, dim=0)
            offs = tuple(float(circular_mean((ph[:, i] - ph[:, 0]) % 1.0)) for i in (1, 2, 3))
            gait, dist = classify_gait(offs)
            gait_txt = f"{gait:<17} {offs[0]:.2f}/{offs[1]:.2f}/{offs[2]:.2f}   {100 * flight / measure_steps:>7.1f} {measure_steps / stride:>5.2f}"
            print(
                f"{label:<11} {vx:>6.2f} {vy:>6.2f} {wz:>6.2f} | {got[0]:>6.2f} {got[1]:>6.2f} {got[2]:>6.2f} |"
                f" {falls / unwrapped.num_envs:>8.2f} | {gait_txt}",
                flush=True,
            )
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
