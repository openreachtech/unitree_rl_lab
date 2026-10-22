"""Knock every environment down at once, repeatedly, and measure the fall and the get-up.

Training logs only give the fraction of time spent down, averaged over everything -- a policy that
never falls and one that falls and is up in a second look alike. This applies the task's own
``knockdown`` event to all environments at fixed intervals (the random interval events are switched
off) and reports, per knockdown: whether the robot fell, whether and how fast it was back on its
feet, and which expert drove while it was down.

    python scripts/rsl_rl/measure_recovery.py --task Unitree-Go2-Recovery-MoE-v0 --num_envs 256 \
        --checkpoint logs/rsl_rl/go2_recovery_moe/<run>/model_XXXX.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Unitree-Go2-Recovery-MoE-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--period_s", type=float, default=8.0, help="Time between knockdowns.")
parser.add_argument("--angular", type=float, nargs=2, default=None, help="Override knockdown angular_speed_range.")
parser.add_argument("--linear", type=float, nargs=2, default=None, help="Override knockdown linear_speed_range.")
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
from unitree_rl_lab.tasks.locomotion.robots.go2_recovery import mdp  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

SETTLE_S = 2.0
FELL_GZ = -0.5
"""Trunk past 60 degrees from level counts as a fall."""
STOOD = 0.9
STOOD_HOLD_S = 0.5
"""Back up = ``standing_weight`` above 0.9 for half a second without interruption."""


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point"
    )
    knock = dict(env_cfg.events.knockdown.params)
    env_cfg.events.knockdown = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_pose.params["standing_prob"] = 1.0
    env_cfg.episode_length_s = SETTLE_S + args_cli.rounds * args_cli.period_s + 1.0
    if args_cli.angular:
        knock["angular_speed_range"] = tuple(args_cli.angular)
    if args_cli.linear:
        knock["linear_speed_range"] = tuple(args_cli.linear)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    gating = runner.alg.policy.actor.gating

    base = env.unwrapped
    robot = base.scene["robot"]
    dt = base.step_dt
    n = base.num_envs
    all_ids = torch.arange(n, device=base.device)
    hold_steps = int(STOOD_HOLD_S / dt)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    def step(o):
        with torch.inference_mode():
            actions = policy(o)
            weights = gating(runner.alg.policy.get_actor_obs(o))
        o, _, _, _ = env.step(actions)
        return o, weights

    for _ in range(int(SETTLE_S / dt)):
        obs, _ = step(obs)

    results = []
    for round_index in range(args_cli.rounds):
        mdp.knockdown(base, all_ids, **knock)
        fell = torch.zeros(n, dtype=torch.bool, device=base.device)
        up_at = torch.full((n,), -1.0, device=base.device)
        streak = torch.zeros(n, dtype=torch.long, device=base.device)
        getup_weight_down = torch.zeros(n, device=base.device)
        down_steps = torch.zeros(n, device=base.device)
        for t in range(int(args_cli.period_s / dt)):
            obs, weights = step(obs)
            gz = robot.data.projected_gravity_b[:, 2]
            fell |= gz > FELL_GZ
            standing = mdp.standing_weight(base) > STOOD
            streak = torch.where(standing, streak + 1, torch.zeros_like(streak))
            newly_up = fell & (up_at < 0) & (streak >= hold_steps)
            up_at = torch.where(newly_up, torch.full_like(up_at, (t + 1 - hold_steps) * dt), up_at)
            down = fell & (up_at < 0)
            getup_weight_down += weights[:, 1] * down
            down_steps += down.float()
        results.append((fell, up_at, getup_weight_down, down_steps))

        recovered = fell & (up_at >= 0)
        times = up_at[recovered]
        print(f"\n[round {round_index}] knockdown {knock}")
        print(f"  fell:        {fell.float().mean():.3f} ({int(fell.sum())}/{n})")
        if fell.any():
            print(f"  recovered:   {recovered.sum() / fell.sum():.3f} of falls within {args_cli.period_s:.1f} s")
        if times.numel():
            q = torch.quantile(times, torch.tensor([0.5, 0.9], device=times.device))
            print(f"  time to up:  median {q[0]:.2f} s, p90 {q[1]:.2f} s")
        if down_steps.sum() > 0:
            print(f"  get-up expert weight while down: {getup_weight_down.sum() / down_steps.sum():.3f}")

    fell = torch.stack([r[0] for r in results])
    up = torch.stack([r[1] for r in results]) >= 0
    print(
        f"\n[total] fell {fell.float().mean():.3f}, recovered {(fell & up).sum() / fell.sum().clamp(min=1):.3f} of falls"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
