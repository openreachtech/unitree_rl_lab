"""Measure how well a trained policy tracks *low* velocity commands.

The training scalars cannot answer this. ``Metrics/base_velocity/error_vel_xy`` is
averaged over whatever the command curriculum happened to be sampling, which is dominated
by the fastest commands in range, and ``track_lin_vel_xy`` is an exponential kernel --
at 0.05 m/s commanded, standing perfectly still already scores 0.99 of the maximum. A
policy can look flawless on both while never moving at all below 0.2 m/s.

So this sweeps one commanded direction at a time, holds it, and reports what the base
actually did. Six patterns (forward, backward, left, right, turn left, turn right) times
a list of speeds; each cell is an independent trial with a fresh reset, so a trial that
stumbles does not carry its state into the next one.

Only environments that ran the whole measurement window without terminating are averaged
-- a reset drops the robot back to a standstill, and averaging that in would report the
reset rather than the tracking. The share that stayed up is reported as ``clean%``.

Commands are resampled to a single fixed value with ``rel_standing_envs = 0``, and
``push_robot`` is disabled, so nothing else is moving the robot during a trial.
Observation noise is left exactly as training had it.

Yaw rows take their number as rad/s (the command's own unit), so the ``0.2`` row is
0.2 m/s for the four linear patterns and 0.2 rad/s for the two turns.

Example
-------
    python scripts/rsl_rl/measure_low_speed_tracking.py \\
        --task Go2-Blind-GRU-Phase1 --num_envs 64 --headless \\
        --checkpoint logs/rsl_rl/go2_blind_gru_phase1/<run>/model_999.pt \\
        --json /tmp/go2_lowspeed.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Low-speed command tracking sweep.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model_*.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speeds", type=str, default="0.2,0.15,0.1,0.05", help="Comma-separated commanded values.")
parser.add_argument("--settle_s", type=float, default=3.0, help="Seconds discarded per trial before measuring.")
parser.add_argument("--measure_s", type=float, default=5.0, help="Seconds averaged per trial.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--json", type=str, default=None, help="Optional path to write the raw rows as JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# The two repos this script lives in name their task package and their runner subclass
# differently; everything below is identical, so resolve both here rather than forking
# the file.
try:
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.assets.models.modules.runners import UnitreeOnPolicyRunner as Runner
except ImportError:
    import tsubame_isaac.tasks  # noqa: F401
    from rsl_rl.runners import OnPolicyRunner as Runner


# (label, (vx, vy, wz) unit direction). The command's magnitude comes from the swept value.
PATTERNS = [
    ("forward", (1.0, 0.0, 0.0)),
    ("backward", (-1.0, 0.0, 0.0)),
    ("left", (0.0, 1.0, 0.0)),
    ("right", (0.0, -1.0, 0.0)),
    ("turn_left", (0.0, 0.0, 1.0)),
    ("turn_right", (0.0, 0.0, -1.0)),
]


def main() -> None:
    speeds = [float(s) for s in args_cli.speeds.split(",")]

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Every env must carry the swept command, held for the whole trial.
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.debug_vis = False
    # The command curriculum would widen the range underneath the sweep.
    if getattr(env_cfg.curriculum, "lin_vel_cmd_levels", None) is not None:
        env_cfg.curriculum.lin_vel_cmd_levels = None
    if getattr(env_cfg.curriculum, "ang_vel_cmd_levels", None) is not None:
        env_cfg.curriculum.ang_vel_cmd_levels = None
    # Nothing but the policy should be moving the robot while it is being measured.
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None

    # The registry's own runner cfg, with none of train.py's CLI overrides: this script
    # only ever reads the network shape and the checkpoint path it was handed.
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = Runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    device = unwrapped.device
    all_ids = torch.arange(unwrapped.num_envs, device=device)

    dt = unwrapped.step_dt
    settle_steps = max(1, int(round(args_cli.settle_s / dt)))
    measure_steps = max(1, int(round(args_cli.measure_s / dt)))

    def set_command(vx: float, vy: float, wz: float) -> None:
        command_term.cfg.ranges.lin_vel_x = (vx, vx)
        command_term.cfg.ranges.lin_vel_y = (vy, vy)
        command_term.cfg.ranges.ang_vel_z = (wz, wz)
        command_term._resample_command(all_ids)

    rows = []
    with torch.inference_mode():
        for label, (dx, dy, dwz) in PATTERNS:
            for value in speeds:
                vx, vy, wz = dx * value, dy * value, dwz * value

                # Independent trial: reset everything, including the GRU's hidden state.
                obs, _ = env.reset()
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset()
                set_command(vx, vy, wz)

                for _ in range(settle_steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    if hasattr(policy_nn, "reset"):
                        policy_nn.reset(dones)

                sum_vx = torch.zeros(unwrapped.num_envs, device=device)
                sum_vy = torch.zeros(unwrapped.num_envs, device=device)
                sum_wz = torch.zeros(unwrapped.num_envs, device=device)
                clean = torch.ones(unwrapped.num_envs, dtype=torch.bool, device=device)

                for _ in range(measure_steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    if hasattr(policy_nn, "reset"):
                        policy_nn.reset(dones)
                    sum_vx += robot.data.root_lin_vel_b[:, 0]
                    sum_vy += robot.data.root_lin_vel_b[:, 1]
                    sum_wz += robot.data.root_ang_vel_b[:, 2]
                    clean &= dones.bool().logical_not()

                n_clean = int(clean.sum().item())
                sel = clean if n_clean > 0 else torch.ones_like(clean)
                mean_vx = (sum_vx[sel] / measure_steps).mean().item()
                mean_vy = (sum_vy[sel] / measure_steps).mean().item()
                mean_wz = (sum_wz[sel] / measure_steps).mean().item()
                std_axis = (
                    (sum_wz[sel] / measure_steps).std().item()
                    if dwz != 0.0
                    else ((sum_vx if dx != 0.0 else sum_vy)[sel] / measure_steps).std().item()
                )

                achieved = mean_wz if dwz != 0.0 else (mean_vx if dx != 0.0 else mean_vy)
                commanded = wz if dwz != 0.0 else (vx if dx != 0.0 else vy)
                rows.append(
                    dict(
                        pattern=label,
                        commanded=commanded,
                        achieved=achieved,
                        error=achieved - commanded,
                        ratio=achieved / commanded if commanded != 0.0 else float("nan"),
                        std=std_axis,
                        vx=mean_vx,
                        vy=mean_vy,
                        wz=mean_wz,
                        clean_frac=n_clean / unwrapped.num_envs,
                    )
                )
                print(f"  {label:<10} cmd {commanded:+.2f} -> {achieved:+.3f} (clean {n_clean}/{unwrapped.num_envs})")

    print(f"\n=== {args_cli.task} | {args_cli.checkpoint} ===")
    print(f"{args_cli.num_envs} envs, {args_cli.settle_s:.0f}s settle + {args_cli.measure_s:.0f}s measured per cell\n")
    header = f"{'pattern':<11}{'cmd':>8}{'achieved':>10}{'error':>9}{'ratio':>8}{'std':>8}{'vx':>8}{'vy':>8}{'wz':>8}{'clean%':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['pattern']:<11}{row['commanded']:>8.2f}{row['achieved']:>10.3f}{row['error']:>9.3f}"
            f"{row['ratio']:>8.2f}{row['std']:>8.3f}{row['vx']:>8.3f}{row['vy']:>8.3f}{row['wz']:>8.3f}"
            f"{100 * row['clean_frac']:>8.0f}"
        )
    print("\ncmd/achieved/error/std are m/s, except the two turn rows, which are rad/s.")
    print("ratio = achieved / commanded. vx/vy/wz are the full body-frame velocity, so the")
    print("two columns the command did not ask for show drift.")
    print("clean% = share of envs that never terminated inside the measurement window.")

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(
                {
                    "task": args_cli.task,
                    "checkpoint": args_cli.checkpoint,
                    "num_envs": args_cli.num_envs,
                    "settle_s": args_cli.settle_s,
                    "measure_s": args_cli.measure_s,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\n[wrote] {args_cli.json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
