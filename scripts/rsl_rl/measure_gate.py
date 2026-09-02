"""Which expert actually drives, under a commanded gait held fixed.

The merged policy is a soft mixture of three experts -- locomotion, acrobatics, and a transition
expert that starts from random weights and has to earn its weight from the gate. Training logs the
gate split by the jump command's ``enabled`` flag, which answers "who drives at a take-off" but
averages over whatever gait the sampled command happened to be: a gallop and a backward walk land
in the same bucket. This holds one velocity command fixed for the whole run, so the answer is about
a named gait rather than a mixture of all of them.

The velocity command is written directly into the command term every step (its own resampling is
disabled), and every reading is cross-checked against the observation the policy actually saw, so a
step where anything overwrote the command is dropped rather than averaged in.

    # gallop -- above the take-off ceiling, so no move ever fires
    python scripts/rsl_rl/measure_gate.py --task Go2-Multitask --vx 3.0 --steps 400

    # backward walk, with the backflips it triggers
    python scripts/rsl_rl/measure_gate.py --task Go2-Multitask --vx -0.8 --steps 800
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--vx", type=float, default=0.0, help="Commanded forward velocity, m/s (body +x).")
parser.add_argument("--vy", type=float, default=0.0, help="Commanded lateral velocity, m/s (body +y = left).")
parser.add_argument("--wz", type=float, default=0.0, help="Commanded yaw rate, rad/s.")
parser.add_argument("--takeoff-limit", type=float, default=None,
                    help="Override the take-off speed ceiling. Needed only to force a move at a speed the\n"
                         "deployed policy would never be offered one at.")
parser.add_argument("--warmup", type=int, default=50, help="Steps discarded before measuring, so the gait settles.")
parser.add_argument("--follow", type=float, default=2.0,
                    help="Seconds after a trigger to keep following the move. Has to reach past the command\n"
                         "window, or the handover back to locomotion falls outside every bucket: this\n"
                         "environment sets rearm_after_s equal to command_duration_s, so the command's own\n"
                         "in_motion flag ends at the same instant the flag the policy reads goes low.")
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
from unitree_rl_lab.tasks.multitask.obs_spec import POLICY_UNIFIED, block_offsets  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

EXPERT_NAMES = ("locomotion", "acrobatics", "transition")


def summarize(label: str, weights: torch.Tensor, total: int) -> None:
    """One line per bucket: how much of the run it is, and who drives during it."""
    count = weights.shape[0]
    if count == 0:
        print(f"  {label:<22} {0.0:>6.1%} of steps   (never occurred)")
        return
    mean = weights.mean(dim=0)
    parts = "  ".join(f"{name} {mean[i]:.3f}" for i, name in enumerate(EXPERT_NAMES))
    print(f"  {label:<22} {count / total:>6.1%} of steps   {parts}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    # The command has to stay where it is put: resampling, the standing-environment fraction, and
    # heading control all rewrite vel_command_b inside the term's own update.
    velocity_cfg = env_cfg.commands.base_velocity
    velocity_cfg.resampling_time_range = (1.0e9, 1.0e9)
    velocity_cfg.rel_standing_envs = 0.0
    velocity_cfg.heading_command = False
    if args_cli.takeoff_limit is not None:
        env_cfg.commands.jump.initial_takeoff_speed_limit = args_cli.takeoff_limit
        env_cfg.commands.jump.state_file = None

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
    policy_module = runner.alg.policy
    gating = policy_module.actor.gating
    inference = runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    command = env.unwrapped.command_manager.get_term("jump")
    velocity = env.unwrapped.command_manager.get_term("base_velocity")
    held = torch.tensor([args_cli.vx, args_cli.vy, args_cli.wz], device=device)
    velocity.vel_command_b[:] = held

    step_dt = env.unwrapped.step_dt
    command_offset = block_offsets(POLICY_UNIFIED)["velocity_commands"]

    buckets: dict[str, list[torch.Tensor]] = {"locomotion": [], "window": [], "after": []}
    # Gate weight against time since the trigger, so the handover can be read as a shape rather
    # than as three averages.
    profile: dict[int, list[torch.Tensor]] = {}
    triggers = 0
    dropped = 0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    num_envs = env.unwrapped.num_envs
    prev_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Followed from the rising edge rather than read off the command, because the command's own
    # in_motion flag expires with the window here and would leave the recovery unmeasured.
    follow_steps = int(round(args_cli.follow / step_dt))
    last_trigger = torch.full((num_envs,), -10**9, dtype=torch.long, device=device)

    for step in range(args_cli.warmup + args_cli.steps):
        actor_obs = policy_module.get_actor_obs(obs)
        with torch.inference_mode():
            weights = gating(actor_obs)

        # Only count environments whose observation carried the command being asked about. A reset
        # resamples the command before the observation is built, so those steps describe a
        # different gait and would quietly bias the averages.
        seen = actor_obs[:, command_offset : command_offset + 3]
        valid = (seen - held).abs().max(dim=-1).values < 1e-4
        enabled = command.enabled
        rising = enabled & ~prev_enabled
        # An episode that ended mid-move restarts the robot on its feet, so the steps that follow
        # are ordinary running and must not be charged to the move that was cut short.
        just_reset = env.unwrapped.episode_length_buf <= 1
        last_trigger = torch.where(just_reset, torch.full_like(last_trigger, -10**9), last_trigger)
        last_trigger = torch.where(rising, torch.full_like(last_trigger, step), last_trigger)
        since = step - last_trigger
        following = since < follow_steps

        if step >= args_cli.warmup:
            dropped += int((~valid).sum().item())
            for name, mask in (
                ("window", valid & enabled),
                ("after", valid & ~enabled & following),
                ("locomotion", valid & ~enabled & ~following),
            ):
                if mask.any():
                    buckets[name].append(weights[mask].cpu())
            moving = valid & following
            for i, bin_index in zip(
                moving.nonzero(as_tuple=False).flatten().tolist(),
                since[moving].tolist(),
            ):
                profile.setdefault(bin_index, []).append(weights[i].cpu())
            triggers += int((rising & valid).sum().item())

        prev_enabled = enabled.clone()
        with torch.inference_mode():
            obs, _, _, _ = env.step(inference(obs))
        velocity.vel_command_b[:] = held

    env.close()

    merged = {
        name: torch.cat(chunks) if chunks else torch.empty(0, len(EXPERT_NAMES))
        for name, chunks in buckets.items()
    }
    total = sum(w.shape[0] for w in merged.values())
    if total == 0:
        print("no valid steps -- the commanded velocity never survived into the observation")
        return

    print()
    print(f"command: vx={args_cli.vx:+.2f} vy={args_cli.vy:+.2f} wz={args_cli.wz:+.2f} m/s"
          f"   |v|={(args_cli.vx**2 + args_cli.vy**2) ** 0.5:.2f}")
    print(f"take-off ceiling: {command.takeoff_speed_limit:.2f} m/s   moves triggered: {triggers}")
    print(f"env-steps measured: {total}   dropped (command not in observation): {dropped}")
    print()
    print("  bucket                  share            mean gate weight")
    summarize("running", merged["locomotion"], total)
    summarize("move (command on)", merged["window"], total)
    summarize(f"after window (<{args_cli.follow:.1f}s)", merged["after"], total)

    if profile:
        print()
        print("  time since trigger    n     " + "  ".join(f"{n:>11}" for n in EXPERT_NAMES))
        bin_width = max(1, int(round(0.1 / step_dt)))
        grouped: dict[int, list[torch.Tensor]] = {}
        for bin_index, values in profile.items():
            grouped.setdefault(bin_index // bin_width, []).extend(values)
        for group in sorted(grouped):
            stacked = torch.stack(grouped[group])
            mean = stacked.mean(dim=0)
            t0 = group * bin_width * step_dt
            print(f"  {t0:>6.2f}-{t0 + bin_width * step_dt:.2f} s   {stacked.shape[0]:>6}   "
                  + "  ".join(f"{mean[i]:>11.3f}" for i in range(len(EXPERT_NAMES))))


if __name__ == "__main__":
    main()
    simulation_app.close()
