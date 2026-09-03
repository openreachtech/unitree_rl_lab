"""Per-environment measurement of a bipedal stance, in the quantities its history is written in.

Training logs the stance as population means sampled at one instant per iteration, which is the
wrong shape for the question actually being asked. Two ways it misleads, both already paid for in
this repository:

*Mean against peak.* ``feat/biped``'s stance tables report the **peak** base height reached over a
short play run. Reading a training-time population mean against those numbers compares a settled
average under domain randomisation, added mass and pushes to an undisturbed best case -- which is
how a working handstand got diagnosed as a 24 cm shortfall.

*A completion flag too coarse to measure anything.* ``handstand/success`` reads 1.0 anywhere past
its gate threshold, so a shallow pike and a full handstand log identically. It sat at 1.000 from
iteration 205 to 2000 and reported nothing in between.

This walks the environments individually and prints distributions, in the same quantities and the
same conditions as the tables it needs to be compared against.

    python scripts/rsl_rl/measure_stance.py --task Go2-Multitask-Handstand --num_envs 32 --steps 1000
    python scripts/rsl_rl/measure_stance.py --task Go2-Multitask-Handstand --delay 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-Handstand")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--delay", type=int, default=None,
                    help="Pin the actuation delay to this many physics steps (5 ms each) instead of\n"
                         "drawing it per environment. The stance's hardware failure was reproduced in\n"
                         "simulation by delay alone, so a sweep over it is the acceptance test.")
parser.add_argument("--settle-fraction", type=float, default=0.25,
                    help="Trailing fraction of each environment's run used for the 'settled' figures.")
AppLauncher.add_app_launcher_args(parser)
import cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def quantiles(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "no data"
    q = torch.tensor([0.1, 0.5, 0.9], device=values.device)
    p10, p50, p90 = torch.quantile(values.float(), q).tolist()
    return f"p10 {p10:7.3f}   median {p50:7.3f}   p90 {p90:7.3f}   min {values.min():7.3f}   max {values.max():7.3f}"


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.delay is not None:
        actuator = env_cfg.scene.robot.actuators["GO2HV"]
        actuator.min_delay = args_cli.delay
        actuator.max_delay = args_cli.delay

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

    inner = env.unwrapped
    command = inner.command_manager.get_term("handstand")
    robot = inner.scene["robot"]
    device = inner.device

    stance_hip_ids, stance_hip_names = robot.find_bodies(["FR_hip", "FL_hip"])
    # The lowest point of the trunk in this stance, and the one an operator actually watches: the
    # body pitches nose-down, so the head leads everything else toward the floor. `front_hip_height`
    # regulates the shoulders, which sit behind it and higher.
    head_ids, head_names = robot.find_bodies(["Head_.*", "base"])
    # The joints carrying the robot's weight in this stance, by group. The thigh joint is the one
    # `feat/biped`'s hardware trace found pinned at the actuator ceiling.
    joint_groups = {
        name: robot.find_joints(pattern)[0]
        for name, pattern in (
            ("stance_hip", ["F[RL]_hip_joint"]),
            ("stance_thigh", ["F[RL]_thigh_joint"]),
            ("stance_calf", ["F[RL]_calf_joint"]),
        )
    }
    print(f"[INFO] stance hips: {stance_hip_names}   front bodies: {head_names}   joint groups: "
          + ", ".join(f"{k}={len(v)}" for k, v in joint_groups.items()))

    steps = args_cli.steps
    num_envs = inner.num_envs
    pitch = torch.zeros(steps, num_envs, device=device)
    roll = torch.zeros(steps, num_envs, device=device)
    base_z = torch.zeros(steps, num_envs, device=device)
    hip_z = torch.zeros(steps, num_envs, device=device)
    head_z = torch.zeros(steps, num_envs, device=device)
    torque = {name: torch.zeros(steps, num_envs, device=device) for name in joint_groups}
    # How close the stance legs run to their travel limits. `dof_pos_limits` carries weight -10 here
    # (the merged set's value; the validated bipedal recipe used -1), and a barrier that works reads
    # as zero penalty -- so its cost cannot be seen in the reward log at all. Measuring the margin
    # directly is the only way to tell "not binding" from "binding so hard nothing goes near it".
    margin = {name: torch.zeros(steps, num_envs, device=device) for name in joint_groups}
    position = {name: torch.zeros(steps, num_envs, device=device) for name in joint_groups}
    alive = torch.zeros(steps, num_envs, dtype=torch.bool, device=device)
    ended = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    for step in range(steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        # Everything after an environment's first reset belongs to a different attempt, so the
        # window closes there rather than averaging two starts together.
        alive[step] = ~ended
        pitch[step] = torch.rad2deg(torch.asin(command.pitch_alignment.clamp(-1.0, 1.0)))
        roll[step] = torch.rad2deg(torch.asin(command.roll_error.clamp(-1.0, 1.0))).abs()
        base_z[step] = robot.data.root_pos_w[:, 2]
        hip_z[step] = robot.data.body_pos_w[:, stance_hip_ids, 2].mean(dim=-1)
        head_z[step] = robot.data.body_pos_w[:, head_ids, 2].amin(dim=-1)
        applied = robot.data.applied_torque
        limits = robot.data.soft_joint_pos_limits
        for name, ids in joint_groups.items():
            torque[name][step] = applied[:, ids].abs().amax(dim=-1)
            q = robot.data.joint_pos[:, ids]
            lower, upper = limits[:, ids, 0], limits[:, ids, 1]
            margin[name][step] = torch.minimum(q - lower, upper - q).amin(dim=-1)
            position[name][step] = q.mean(dim=-1)
        ended |= dones.bool()

    env.close()

    counts = alive.sum(dim=0)
    live = counts > 0
    settle_from = (counts.float() * (1.0 - args_cli.settle_fraction)).long()
    index = torch.arange(steps, device=device).unsqueeze(-1)
    settled_mask = alive & (index >= settle_from.unsqueeze(0))

    def peak(series: torch.Tensor) -> torch.Tensor:
        return torch.where(alive, series, torch.full_like(series, -1e9)).amax(dim=0)[live]

    def trough(series: torch.Tensor) -> torch.Tensor:
        return torch.where(alive, series, torch.full_like(series, 1e9)).amin(dim=0)[live]

    def settled(series: torch.Tensor) -> torch.Tensor:
        weights = settled_mask.float()
        return ((series * weights).sum(dim=0) / weights.sum(dim=0).clamp(min=1.0))[live]

    step_dt = inner.step_dt
    print()
    print(f"task {args_cli.task}   envs {num_envs}   steps {steps} ({steps * step_dt:.1f} s)"
          f"   delay {'per-env 0-6' if args_cli.delay is None else args_cli.delay} steps")
    print(f"fell before the end: {int((~live).sum() + (counts < steps).sum() - (~live).sum())} / {num_envs}"
          f"   (episodes that reset early)")
    print()
    print("  quantity                        distribution across environments")
    print(f"  sagittal pitch, peak    (deg)   {quantiles(peak(pitch))}")
    print(f"  sagittal pitch, settled (deg)   {quantiles(settled(pitch))}")
    print(f"  lateral lean, peak      (deg)   {quantiles(peak(roll))}")
    print(f"  base height, peak         (m)   {quantiles(peak(base_z))}")
    print(f"  base height, settled      (m)   {quantiles(settled(base_z))}")
    print(f"  stance hip height, settled (m)  {quantiles(settled(hip_z))}")
    print(f"  head/trunk clearance, min (m)   {quantiles(trough(head_z))}")
    print(f"  head/trunk clearance, settl(m)  {quantiles(settled(head_z))}")
    for name in joint_groups:
        print(f"  |tau| {name:<14} peak (N*m)  {quantiles(peak(torque[name]))}")
        print(f"  |tau| {name:<14} settl(N*m)  {quantiles(settled(torque[name]))}")
    print()
    print("  joint travel (rad). 'margin' is the distance to the nearest soft limit; small means the")
    print("  dof_pos_limits barrier is active there, which a zero penalty in the reward log cannot show.")
    for name in joint_groups:
        closest = torch.where(alive, margin[name], torch.full_like(margin[name], 1e9)).amin(dim=0)[live]
        print(f"  {name:<14} q settled        {quantiles(settled(position[name]))}")
        print(f"  {name:<14} margin, closest  {quantiles(closest)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
