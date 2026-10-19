"""Watch a velocity-tracking policy walk each direction in turn.

Local helper, not part of feat/multi-task. ``play.py`` can only pin the forward command
(``--pin_vel_x``), so this cycles every environment through the same fixed sequence -- forward,
backward, left, right, turn left, turn right, stand -- holding each for ``--phase_s`` seconds and
printing the commanded and achieved base velocity (env mean) at the end of each phase. The
command arrows drawn by the velocity command (green = commanded, blue = actual) show the same
thing on screen.

    python scripts/rsl_rl/play_directions.py --task Go2-Multitask-Gallop-Phase2 \
        --checkpoint logs/rsl_rl/go2_multitask_gallop_phase2/<run>/model_2500.pt
    python scripts/rsl_rl/play_directions.py --task Go2-Multitask-Biped-Hind \
        --checkpoint logs/rsl_rl/go2_multitask_biped_hind/<run>/model_1499.pt --vx 0.6 --vy 0.4 --wz 0.8
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--phase_s", type=float, default=6.0, help="Seconds to hold each direction.")
parser.add_argument("--vx", type=float, default=1.0, help="Forward/backward speed (m/s).")
parser.add_argument("--vx_fast", type=float, default=None, help="Optional extra fast-forward phase (m/s).")
parser.add_argument("--vy", type=float, default=0.6, help="Lateral speed (m/s).")
parser.add_argument("--wz", type=float, default=1.0, help="Yaw rate (rad/s).")
parser.add_argument("--follow", type=int, default=None, metavar="ENV",
                    help="Chase-camera on this environment's robot (e.g. 0). The camera moves with it.")
parser.add_argument("--cam_dist", type=float, default=2.0, help="Chase-camera distance from the robot (m).")
parser.add_argument("--video", type=str, default=None, metavar="MP4",
                    help="Record the (chase-)camera view to this mp4 instead of relying on the GUI, with the\n"
                         "phase and command burned in. A still per phase is saved next to it.")
parser.add_argument("--loops", type=int, default=0, help="Stop after this many sequences (0 = until closed).")
parser.add_argument("--fast", dest="real_time", action="store_false", help="Do not throttle to real time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point"
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = retrieve_file_path(args_cli.checkpoint)

    # The curriculum manager still runs at play time; the velocity ratchet would otherwise rewrite
    # the ranges pinned below at every reset.
    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if args_cli.follow is not None:
        # Isaac Lab's viewer follows the robot's root in world axes: offset diagonally behind and
        # above, looking at the base.
        d = args_cli.cam_dist
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.follow
        env_cfg.viewer.eye = (-0.8 * d, -0.8 * d, 0.6 * d)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.2)

    if args_cli.video:
        env_cfg.viewer.resolution = (1280, 720)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] checkpoint: {resume_path}")

    inner = env.unwrapped
    term = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    all_ids = torch.arange(inner.num_envs, device=inner.device)

    phases = [("forward", args_cli.vx, 0.0, 0.0)]
    if args_cli.vx_fast is not None:
        phases.append(("forward fast", args_cli.vx_fast, 0.0, 0.0))
    phases += [
        ("backward", -args_cli.vx, 0.0, 0.0),
        ("left", 0.0, args_cli.vy, 0.0),
        ("right", 0.0, -args_cli.vy, 0.0),
        ("turn left", 0.0, 0.0, args_cli.wz),
        ("turn right", 0.0, 0.0, -args_cli.wz),
        ("stand", 0.0, 0.0, 0.0),
    ]

    def pin(vx, vy, wz):
        # Pinned ranges make every resample -- including the one on an episode reset -- draw this
        # exact command, so a robot that falls and respawns comes back on the same phase.
        term.cfg.ranges.lin_vel_x = (vx, vx)
        term.cfg.ranges.lin_vel_y = (vy, vy)
        term.cfg.ranges.ang_vel_z = (wz, wz)
        term._resample(all_ids)

    dt = inner.step_dt
    steps_per_phase = int(args_cli.phase_s / dt)
    writer = None
    if args_cli.video:
        import cv2
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video)), exist_ok=True)
        writer = imageio.get_writer(args_cli.video, fps=round(1.0 / dt), codec="libx264", quality=8,
                                    macro_block_size=8)
        still_prefix = os.path.splitext(args_cli.video)[0]
        watched = args_cli.follow if args_cli.follow is not None else 0

    def record(label, index):
        frame = inner.render()
        if frame is None:
            return None
        frame = frame[..., :3].copy()
        lin = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[watched]
        wz = robot.data.root_ang_vel_w[watched, 2]
        lines = [label, f"env {watched}: vx={lin[0]:+.2f} vy={lin[1]:+.2f} wz={wz:+.2f}"]
        for row, text in enumerate(lines):
            y = 40 + 36 * row
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.append_data(frame)
        return frame

    obs = env.get_observations()
    loop = 0
    while simulation_app.is_running():
        for name, vx, vy, wz in phases:
            pin(vx, vy, wz)
            print(f"\n=== {name}: command vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} ===", flush=True)
            achieved = []
            for step in range(steps_per_phase):
                start = time.time()
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                if writer is not None:
                    frame = record(f"{name}  cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}", step)
                    if frame is not None and step == steps_per_phase // 2 and loop == 0:
                        imageio.imwrite(f"{still_prefix}_{name.replace(' ', '_')}.png", frame)
                if step >= steps_per_phase // 2:
                    # Heading (yaw-only) frame and world yaw rate, as the tracking rewards use. The
                    # body frame is wrong for a bipedal stance: pitched ~85 deg, body x points up.
                    lin = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
                    achieved.append(torch.cat([lin[:, :2], robot.data.root_ang_vel_w[:, 2:3]], dim=1))
                if not simulation_app.is_running():
                    if writer is not None:
                        writer.close()
                    return
                sleep = dt - (time.time() - start)
                if args_cli.real_time and sleep > 0:
                    time.sleep(sleep)
            mean = torch.stack(achieved).mean(dim=(0, 1))
            print(f"    achieved (2nd half, env mean): vx={mean[0]:+.2f} vy={mean[1]:+.2f} wz={mean[2]:+.2f}",
                  flush=True)
        loop += 1
        if args_cli.loops and loop >= args_cli.loops:
            break
    if writer is not None:
        writer.close()
        print(f"[INFO] video written to {args_cli.video}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
