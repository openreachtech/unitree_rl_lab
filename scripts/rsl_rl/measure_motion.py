"""Per-attempt, per-environment measurement of an acrobatic motion.

Training logs report metrics as means over every environment, which repeatedly hid the thing that
mattered: a mean over 4096 envs where only a fraction are mid-flip is diluted by the rest, and a
per-motion conditional mean says nothing about the spread. This walks the attempts individually and
prints the distribution, so "it rotates 0.63 turns" can be distinguished from "half of them rotate
1.0 and half rotate 0.2".

    python scripts/rsl_rl/measure_motion.py --task Go2-Multitask-Jump-Inspect-SideflipRight \
        --experiment_name go2_multitask_jump_try_1 --num_envs 64 --steps 600
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--assist-scale", type=float, default=None,
                    help="Override initial_assist_scale. The Go2-Multitask-Jump-Inspect-* tasks hold\n"
                         "it at 1.0 to show the launch mechanics; 0.0 measures the policy unaided,\n"
                         "which is the condition the merged environment actually runs in.")
parser.add_argument("--random-policy", action="store_true",
                    help="Skip loading the checkpoint. Isolates what the environment does from what a\ntrained policy does -- every checkpoint in this project descends from the same standing phase, so\nagreement between two of them is not independent evidence.")
parser.add_argument("--window", type=int, default=100, help="Steps after a trigger that count as one attempt.")
AppLauncher.add_app_launcher_args(parser)
import cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

TWO_PI = 2.0 * 3.141592653589793


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    jump_cfg = env_cfg.commands.jump
    if args_cli.assist_scale is not None:
        jump_cfg.initial_assist_scale = args_cli.assist_scale
        jump_cfg.state_file = None  # else a saved curriculum overwrites the value being asked for
        print(f"[INFO] initial_assist_scale = {args_cli.assist_scale}")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = None
    if args_cli.random_policy:
        pass
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.random_policy:
        print("[INFO] random policy: checkpoint NOT loaded")
    else:
        runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term("jump")

    obs = env.get_observations()
    if isinstance(obs, tuple):  # rsl-rl 2.3 returns (obs, extras)
        obs = obs[0]
    prev_enabled = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    # One live record per env; a rising edge on `enabled` opens a new one, and it is banked once the
    # attempt's window has elapsed. Sampling at a fixed offset from the trigger rather than at reset
    # is what makes every attempt comparable -- `success` cannot be true before the landing check,
    # so a snapshot taken too early reads 0 for a flip that lands perfectly a few steps later.
    step_dt = env.unwrapped.step_dt
    live: dict[int, dict] = {}
    done: list[dict] = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        enabled = command.enabled.clone()
        rising = enabled & ~prev_enabled
        for i in rising.nonzero(as_tuple=False).flatten().tolist():
            live[i] = {"start": step, "roll": 0.0, "pitch": 0.0, "height": 0.0,
                       "success": False, "t_success": None}
        roll = (command.accumulated_roll / TWO_PI).tolist()
        pitch = (command.accumulated_pitch / TWO_PI).tolist()
        height = command.max_height.tolist()
        success = command.success.tolist()
        for i, rec in list(live.items()):
            if abs(roll[i]) > abs(rec["roll"]):
                rec["roll"] = roll[i]
            if abs(pitch[i]) > abs(rec["pitch"]):
                rec["pitch"] = pitch[i]
            rec["height"] = max(rec["height"], height[i])
            if success[i] and rec["t_success"] is None:
                # Steps from the trigger to the first moment the attempt counts as landed. This is
                # what decides whether a given command_duration_s covers the motion: the `enabled`
                # flag the policy reads goes low at that duration, so a window shorter than this
                # hands the robot back mid-flip.
                rec["t_success"] = (step - rec["start"]) * step_dt
            rec["success"] = rec["success"] or bool(success[i])
            if step - rec["start"] >= args_cli.window:
                done.append(rec)
                del live[i]
        prev_enabled = enabled

    env.close()
    if not done:
        print("no completed attempts -- raise --steps")
        return

    def col(key):
        return sorted(r[key] for r in done)

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))]

    n = len(done)
    print(f"\nattempts: {n}   success: {sum(r['success'] for r in done) / n:.3f}\n")
    print(f"{'':>10}{'p10':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}{'mean':>9}")
    print("-" * 64)
    for key in ("roll", "pitch", "height"):
        v = col(key)
        mean = sum(v) / n
        print(f"{key:>10}{pct(v,.1):9.3f}{pct(v,.25):9.3f}{pct(v,.5):9.3f}"
              f"{pct(v,.75):9.3f}{pct(v,.9):9.3f}{mean:9.3f}")
    times = sorted(r["t_success"] for r in done if r["t_success"] is not None)
    if times:
        print(f"\ntime from trigger to landing, over the {len(times)} attempts that landed:")
        print(f"    p10={pct(times,.1):.2f}s  median={pct(times,.5):.2f}s  "
              f"p90={pct(times,.9):.2f}s  max={times[-1]:.2f}s")
    print(f"\ntarget roll {command.target_roll_turns.mean().item():+.3f} turns, "
          f"pitch {command.target_pitch_turns.mean().item():+.3f} turns, "
          f"tolerance {command.cfg.rotation_tolerance_rad / TWO_PI:.3f} turns")


if __name__ == "__main__":
    main()
    simulation_app.close()
