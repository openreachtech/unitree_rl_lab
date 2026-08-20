"""Sweep commanded forward speed against a trained policy and report what it actually delivers.

Answers three questions the TensorBoard scalars cannot:

1. **How fast can it really run?**  ``Curriculum/lin_vel_cmd_levels`` says how far the commanded
   range was widened during training, but that ratchet climbs while the tow assist is still
   helping -- it cannot tell "ran at 5 m/s" from "was towed to 5 m/s". This runs with the assist
   off (the play cfgs zero it) and reports achieved speed per commanded speed, so the point where
   achieved stops following commanded is a measurement rather than a training artifact.

2. **Is the stride long enough, and does it ever leave the ground?**  ``stride`` is achieved
   speed over stride frequency, in metres per cycle, and ``flight%`` is the share of time with no
   foot loaded. Together they say which of the two factors of speed is out of room: frequency
   saturates near 4.1 Hz on Go2, so beyond that only stride length is left, and a stride much
   longer than the leg needs a suspension phase to exist at all.

3. **What gait did it choose?**  For a policy trained without a footfall prescription, this is the
   whole question. Per-foot stride phase is recovered from contact history using the *same*
   ``_local_stride_phase`` helper that ``paired_gait_reward`` grades with, then reduced to the
   ``(theta_FR, theta_RL, theta_RR)`` triple that ``velocity_env_cfg_run.py``'s canonical gait
   constants are written in, and matched against them.

4. **Is the joint-level PD the bottleneck?**  The standard prescription for a faster swing is a
   higher Kp (see e.g. the ANYmal / Mini Cheetah / Go1 high-speed literature, which schedules
   gains with speed). That is a testable claim, not a given: it predicts a large standing
   position error ``|q_target - q|`` at speed. Reported alongside the split between the two PD
   terms -- ``kp*|err|`` drives the joint, ``kd*|dq|`` is subtracted from it whenever the joint
   moves fast, since the action term commands zero joint velocity -- and how close the joints run
   to the torque-speed knee X1, past which no gain can buy a faster swing.

5. **Is the actuator actually maxed out?**  Reproduces ``UnitreeActuator._clip_effort``'s
   torque-speed curve to get each joint's instantaneous limit, and reports what fraction of that
   limit the policy is using. A policy leaving 40% of available torque on the table is speed-
   limited by its own gait, not by hardware.

Example
-------
    python scripts/rsl_rl/measure_run_speed.py \\
        --task Go2-Speed-Free-Try-9 --num_envs 64 --headless \\
        --speeds 1,2,3,4,5,6,7 \\
        --checkpoint logs/rsl_rl/go2_speed_free_try_9/<run>/model_2999.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure achieved speed, gait, and actuator usage.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speeds", type=str, default="1,2,3,4,5,6,7", help="Comma-separated commanded m/s.")
parser.add_argument("--settle_s", type=float, default=3.0, help="Seconds discarded per speed before measuring.")
parser.add_argument("--measure_s", type=float, default=4.0, help="Seconds averaged per speed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac tears the process down hard at exit; block-buffered stdout is discarded wholesale, which
# silently loses the entire summary this script exists to print.
sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

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

CANONICAL_GAITS = {
    "pronk": GAIT_PRONK,
    "trot": GAIT_TROT,
    "pace": GAIT_PACE,
    "bound": GAIT_BOUND,
    "gallop-rotary": GAIT_GALLOP_ROTARY,
    "gallop-transverse": GAIT_GALLOP_TRANSVERSE,
}

# Feet in [FL, FR, RL, RR] order -- GaitCommand's convention, which the canonical constants and
# _local_stride_phase both assume. preserve_order matters: the articulation's own body order is
# different, and resolving without it silently pairs each foot with the wrong phase.
GAIT_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]



def circular_mean(angles_frac: torch.Tensor) -> torch.Tensor:
    """Mean of values living on a circle of circumference 1. A plain mean is wrong here: two
    samples at 0.99 and 0.01 are 0.02 apart, not 0.98, and would average to 0.5 -- the exact
    opposite of the truth."""
    theta = angles_frac * 2.0 * torch.pi
    return (torch.atan2(theta.sin().mean(), theta.cos().mean()) / (2.0 * torch.pi)) % 1.0


def circular_dist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def classify_gait(offsets: tuple[float, float, float]) -> tuple[str, float]:
    """Nearest canonical gait, by summed circular distance over the three offsets."""
    scored = {
        name: sum(circular_dist(o, c) for o, c in zip(offsets, canon)) for name, canon in CANONICAL_GAITS.items()
    }
    best = min(scored, key=scored.get)
    return best, scored[best]


def actuator_utilization(actuator, applied_torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Fraction of the speed-dependent torque limit currently in use, per joint.

    Mirrors ``UnitreeActuator._clip_effort``: the ceiling is Y1 when torque and velocity point the
    same way (driving) and the higher Y2 when they oppose (braking), and it falls off linearly
    once past the knee at X1, reaching zero at the no-load speed X2.

    Y1/Y2/X1/X2 are read off the live actuator rather than hardcoded, because they are no longer
    uniform across joints: the corrected model (``UNITREE_GO2_CORRECTED_CFG``) gives the calf 39.22/45.43
    N*m against the hip's and thigh's 20.2/23.4. Hardcoding the motor's bare curve would report
    roughly double the true utilization on every knee -- exactly the joint this measurement is
    most interested in.
    """
    y1 = actuator._effort_y1
    y2 = actuator._effort_y2
    x1 = actuator._velocity_x1
    x2 = actuator._velocity_x2
    same_direction = (joint_vel * applied_torque) > 0
    max_effort = torch.where(same_direction, y1.expand_as(applied_torque), y2.expand_as(applied_torque))
    slope = -max_effort / (x2 - x1)
    decayed = (slope * (joint_vel.abs() - x1) + max_effort).clip(min=0.0)
    limit = torch.where(joint_vel.abs() < x1, max_effort, decayed)
    return applied_torque.abs() / limit.clamp(min=1e-3)


def pd_step_stats(actuator, asset, joint_ids: list[int]) -> dict[str, torch.Tensor]:
    """One step's worth of joint-level PD diagnostics, over ``joint_ids``.

    ``JointPositionAction`` writes a position target and no velocity target, so the joint torque
    before clipping is ``kp * (q_target - q) - kd * dq``. Both halves matter at speed and they
    fight each other: at the stock 25 / 0.5, a knee swinging at 13 rad/s has 6.5 N*m of damping
    torque subtracted -- a third of the hip/thigh envelope -- so the position term has to overcome
    the damping term before any of it accelerates the leg.

    ``derate`` is the fraction of joint-steps already past the torque-speed knee X1, where the
    available torque is decaying toward zero at X2. Where that fraction is large, the swing rate
    is limited by the motor curve and no PD gain can raise it.
    """
    q = asset.data.joint_pos[:, joint_ids]
    dq = asset.data.joint_vel[:, joint_ids]
    target = asset.data.joint_pos_target[:, joint_ids]
    kp = actuator.stiffness[:, joint_ids]
    kd = actuator.damping[:, joint_ids]
    x1 = actuator._velocity_x1[:, joint_ids]
    err = (target - q).abs()
    return {
        "pd_err": err.mean(),
        "pd_err_max": err.max(),
        "kp_term": (kp * err).mean(),
        "kd_term": (kd * dq.abs()).mean(),
        "dq": dq.abs().mean(),
        "dq_max": dq.abs().max(),
        "derate": (dq.abs() > x1).float().mean(),
    }


def main() -> None:
    speeds = [float(s) for s in args_cli.speeds.split(",")]

    # play_env_cfg_entry_point, not the training one: the play configs are where tow_assist is
    # zeroed, and measuring achieved speed with the assist still pushing would report the tow.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Every env must carry the swept command: rel_standing_envs would otherwise zero 10% of them
    # and drag the achieved-speed average down by a fixed, speed-independent amount.
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    assist = getattr(env_cfg.commands, "tow_assist", None)
    if assist is not None and assist.initial_assist_scale != 0.0:
        raise SystemExit(
            f"tow_assist.initial_assist_scale is {assist.initial_assist_scale}, not 0 -- achieved "
            "speed would include the assist force. Fix the task's play cfg before measuring."
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            f"logs/rsl_rl/{agent_cfg.experiment_name}", agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    asset = unwrapped.scene["robot"]
    actuator = asset.actuators["GO2HV"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    command_term = unwrapped.command_manager.get_term("base_velocity")

    from isaaclab.managers import SceneEntityCfg

    foot_cfg = SceneEntityCfg("contact_forces", body_names=GAIT_FEET, preserve_order=True)
    foot_cfg.resolve(unwrapped.scene)
    foot_ids = foot_cfg.body_ids

    # Hips barely move while running forward; averaging them in would dilute every PD number
    # below. Thigh and calf are what swing the leg.
    swing_ids, swing_names = asset.find_joints([".*_thigh_joint", ".*_calf_joint"])
    print(f"[INFO] PD diagnostics over {len(swing_ids)} joints: {swing_names}")

    dt = unwrapped.step_dt
    settle_steps = int(args_cli.settle_s / dt)
    measure_steps = int(args_cli.measure_s / dt)

    # rsl-rl 2.3 returns (obs, extras); later versions return obs alone. Same probe play.py uses.
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    rows = []

    for commanded in speeds:
        # Reset every environment before each commanded speed. Without this the speeds are not
        # independent trials: a robot that stumbled or gave up at 4.0 m/s carries that state into
        # the 5.0 and 5.5 rows, which both inflates the spread between repeated sweeps (measured:
        # 5.21 vs 4.64 m/s for the same checkpoint at the same command) and biases the high rows
        # downward -- it is why a 6.0 row could read slower than the 5.5 row above it.
        # inference_mode matters: the stepping loop below runs inside it, so the observation and
        # asset buffers are inference tensors by the second pass. Resetting outside the context
        # then raises "Inplace update to inference tensor outside InferenceMode".
        with torch.inference_mode():
            obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Pin the command by narrowing the sampling range to a point and forcing a resample --
        # the normal code path, rather than writing into the command buffer behind the term's
        # back where _update_command would overwrite it again next step.
        command_term.cfg.ranges.lin_vel_x = (commanded, commanded)
        command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
        all_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        command_term._resample_command(all_ids)

        for _ in range(settle_steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            command_term._resample_command(all_ids)

        speed_sum = torch.zeros((), device=unwrapped.device)
        util_sum = torch.zeros((), device=unwrapped.device)
        util_max = torch.zeros((), device=unwrapped.device)
        phase_acc = []
        stride_sum = torch.zeros((), device=unwrapped.device)
        upright_sum = torch.zeros((), device=unwrapped.device)
        pd_sums = {k: torch.zeros((), device=unwrapped.device) for k in ("pd_err", "kp_term", "kd_term", "dq", "derate")}
        # Fraction of time with no foot loaded. A trot has essentially none; a bound/gallop needs a
        # real suspension phase, so this is the number that says whether the gait actually changed
        # A trot has essentially none; a bound or gallop needs a real suspension phase, so this is
        # the number that says whether the gait actually changed.
        flight_sum = torch.zeros((), device=unwrapped.device)
        pd_maxes = {k: torch.zeros((), device=unwrapped.device) for k in ("pd_err_max", "dq_max")}

        for _ in range(measure_steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            command_term._resample_command(all_ids)

            speed_sum += asset.data.root_lin_vel_b[:, 0].mean()
            util = actuator_utilization(actuator, asset.data.applied_torque, asset.data.joint_vel)
            util_sum += util.mean()
            util_max = torch.maximum(util_max, util.max())
            # projected_gravity_b[2] is -1 upright, 0 on its side, +1 inverted.
            upright_sum += (asset.data.projected_gravity_b[:, 2] < -0.7).float().mean()

            pd = pd_step_stats(actuator, asset, swing_ids)
            for k in pd_sums:
                pd_sums[k] += pd[k]
            for k in pd_maxes:
                pd_maxes[k] = torch.maximum(pd_maxes[k], pd[k])

            current_contact = contact_sensor.data.current_contact_time[:, foot_ids]
            current_air = contact_sensor.data.current_air_time[:, foot_ids]
            flight_sum += (current_contact <= 0.0).all(dim=1).float().mean()
            last_contact = contact_sensor.data.last_contact_time[:, foot_ids]
            last_air = contact_sensor.data.last_air_time[:, foot_ids]
            phase = _local_stride_phase(
                current_contact, current_air, current_contact > 0.0, last_contact, last_air
            )
            phase_acc.append(phase)
            stride_sum += (last_contact + last_air).clamp(min=1e-3).mean()

        achieved = (speed_sum / measure_steps).item()
        phases = torch.cat(phase_acc, dim=0)  # (steps * envs, 4)
        offsets = tuple(circular_mean((phases[:, i] - phases[:, 0]) % 1.0).item() for i in (1, 2, 3))
        gait, distance = classify_gait(offsets)
        stride_s = (stride_sum / measure_steps).item()

        rows.append({
            "cmd": commanded,
            "achieved": achieved,
            "err": commanded - achieved,
            "upright": (upright_sum / measure_steps).item(),
            "util_mean": (util_sum / measure_steps).item(),
            "util_max": util_max.item(),
            "offsets": offsets,
            "gait": gait,
            "gait_dist": distance,
            "freq": 1.0 / stride_s if stride_s > 0 else 0.0,
            "stride_m": achieved / (1.0 / stride_s) if stride_s > 0 else 0.0,
            "flight": (flight_sum / measure_steps).item(),
            **{k: (v / measure_steps).item() for k, v in pd_sums.items()},
            **{k: v.item() for k, v in pd_maxes.items()},
        })
        print(f"  ... {commanded:.1f} m/s commanded -> {achieved:.2f} m/s achieved")

    print(f"\n=== {args_cli.task} ===")
    header = (
        f"{'cmd':>6}{'achieved':>10}{'error':>8}{'upright':>9}{'torque%':>9}{'peak%':>7}{'Hz':>6}"
        f"{'stride':>8}{'flight%':>8}  gait"
    )
    print(header)
    print("-" * (len(header) + 22))
    for r in rows:
        off = "(" + ", ".join(f"{o:.2f}" for o in r["offsets"]) + ")"
        print(
            f"{r['cmd']:>6.1f}{r['achieved']:>10.2f}{r['err']:>8.2f}{r['upright']:>9.2f}"
            f"{r['util_mean'] * 100:>9.1f}{r['util_max'] * 100:>7.0f}{r['freq']:>6.1f}"
            f"{r['stride_m']:>8.2f}{r['flight'] * 100:>8.1f}"
            f"  {r['gait']} d={r['gait_dist']:.2f} {off}"
        )

    print("\n--- joint-level PD (thigh + calf) ---")
    pd_header = (
        f"{'cmd':>6}{'|err| rad':>11}{'max':>7}{'kp*err Nm':>11}{'kd*dq Nm':>10}"
        f"{'|dq| r/s':>10}{'max':>7}{'>X1':>7}"
    )
    print(pd_header)
    print("-" * len(pd_header))
    for r in rows:
        print(
            f"{r['cmd']:>6.1f}{r['pd_err']:>11.3f}{r['pd_err_max']:>7.2f}{r['kp_term']:>11.2f}"
            f"{r['kd_term']:>10.2f}{r['dq']:>10.2f}{r['dq_max']:>7.1f}{r['derate'] * 100:>6.1f}%"
        )
    print("Each row is an independent trial: environments are reset before every commanded speed.")
    print("|err| is how far the joint sits from its commanded angle -- the quantity a higher Kp "
          "would shrink.\n>X1 is the share of joint-steps past the torque-speed knee, where "
          "available torque is already decaying.")

    tracked = [r for r in rows if r["err"] < 0.5 and r["upright"] > 0.9]
    if tracked:
        best = max(tracked, key=lambda r: r["cmd"])
        print(f"\nTracks up to {best['cmd']:.1f} m/s (error < 0.5 m/s, stays upright).")
        print(f"Fastest achieved overall: {max(r['achieved'] for r in rows):.2f} m/s.")
        print(f"At the limit it uses {best['util_mean'] * 100:.0f}% of available torque on average, "
              f"peaking at {best['util_max'] * 100:.0f}%.")
    else:
        print("\nNo commanded speed was tracked within 0.5 m/s while staying upright.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
