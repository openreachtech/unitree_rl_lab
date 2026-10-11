"""Record a bank of settled crouch states for Unitree-G1-23dof-StandUp-v0.

Each robot is placed in a randomized deep crouch (feet flat, torso leaning forward, arms
somewhere in front), held there by PD for a while, and kept only if it is still balanced
with both feet flat on the ground. The task resets from these states.

    python scripts/make_g1_crouch_bank.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record settled G1 crouch states.")
parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StandUp-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_rounds", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=75, help="policy steps (50 Hz) to hold the pose")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--usd_path", type=str, default=None, help="override the robot USD (UNITREE_MODEL_DIR unset)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.g1_getup.mdp import ACTION_NAME, CROUCH_BANK_PATH

# leg ranges around the crouches that balanced when held by PD (hip, knee, ankle in rad)
HIP_PITCH = (-2.4, -1.8)
KNEE = (2.0, 2.6)
ANKLE_PITCH = (-0.85, -0.7)
LEAN = (0.4, 0.9)  # pelvis forward pitch
SIDE_NOISE = 0.05  # left/right difference

# rough planar leg model, only to spawn just above the ground (the hold settles the rest)
SHANK, THIGH, SOLE, HIP_BELOW_PELVIS = 0.336, 0.33, 0.035, 0.103


def u(lo, hi, n, device):
    return torch.empty(n, device=device).uniform_(lo, hi)


def main():
    output = args_cli.output or CROUCH_BANK_PATH
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.usd_path:
        env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    # the bank does not exist yet; and no randomization of the robot while recording
    env_cfg.events.reset_pose = None
    for name in ("actuator_gains", "physics_material", "scale_link_mass", "add_payload", "torso_com"):
        setattr(env_cfg.events, name, None)
    env_cfg.terminations.bad_orientation = None
    env_cfg.terminations.base_vel_out = None
    env_cfg.terminations.joint_vel_out = None
    env_cfg.episode_length_s = 1e6
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    device, n = env.device, env.num_envs
    names = robot.joint_names
    jid = {name: i for i, name in enumerate(names)}
    feet_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
    scale = env.action_manager.get_term(ACTION_NAME).action_scale
    env.reset()

    root_poses, joint_positions = [], []
    for r in range(args_cli.num_rounds):
        hip, knee, ankle, lean = u(*HIP_PITCH, n, device), u(*KNEE, n, device), u(*ANKLE_PITCH, n, device), u(*LEAN, n, device)
        target = robot.data.default_joint_pos.clone()
        for side, sign in (("left", 1.0), ("right", -1.0)):
            target[:, jid[f"{side}_hip_pitch_joint"]] = hip + u(-SIDE_NOISE, SIDE_NOISE, n, device)
            target[:, jid[f"{side}_knee_joint"]] = knee + u(-SIDE_NOISE, SIDE_NOISE, n, device)
            target[:, jid[f"{side}_ankle_pitch_joint"]] = ankle + u(-SIDE_NOISE, SIDE_NOISE, n, device)
            target[:, jid[f"{side}_shoulder_pitch_joint"]] = u(-1.0, 0.5, n, device)
            target[:, jid[f"{side}_shoulder_roll_joint"]] = sign * u(0.0, 0.5, n, device)
            target[:, jid[f"{side}_elbow_joint"]] = u(0.3, 1.5, n, device)
        limits = robot.data.soft_joint_pos_limits
        target = target.clamp(limits[..., 0], limits[..., 1])

        thigh_angle = knee + ankle  # from vertical, with the shank leaning forward by -ankle
        z = SOLE + SHANK * torch.cos(-ankle) + THIGH * torch.cos(thigh_angle) + HIP_BELOW_PELVIS * torch.cos(lean)
        root = torch.zeros(n, 7, device=device)
        root[:, :3] = env.scene.env_origins
        root[:, 2] += z + 0.02
        root[:, 3] = torch.cos(lean / 2)
        root[:, 5] = torch.sin(lean / 2)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))

        for _ in range(args_cli.settle_steps):
            env.step((target - robot.data.joint_pos) / scale)

        feet_z = robot.data.body_pos_w[:, feet_ids, 2] - env.scene.env_origins[:, 2:3]
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        ok = (
            (feet_z.max(dim=1).values < 0.045)
            & (tilt < 1.2)
            & (torch.norm(robot.data.root_lin_vel_w, dim=1) < 0.1)
            & (robot.data.joint_vel.abs().max(dim=1).values < 0.5)
        )
        pose = robot.data.root_state_w[:, :7].clone()
        pose[:, :3] -= env.scene.env_origins
        root_poses.append(pose[ok].cpu())
        joint_positions.append(robot.data.joint_pos[ok].cpu())
        print(f"[INFO] round {r + 1}/{args_cli.num_rounds}: kept {int(ok.sum())}/{n}")

    bank = {"root_pose": torch.cat(root_poses), "joint_pos": torch.cat(joint_positions)}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(bank, output)
    h = bank["root_pose"][:, 2]
    print(f"[INFO] wrote {h.shape[0]} crouch states to {output}")
    print(f"[INFO] pelvis height min {h.min():.3f} mean {h.mean():.3f} max {h.max():.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
