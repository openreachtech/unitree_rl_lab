"""Record initial states for whole-body pose tracking from the locomotion policy (III-G).

Reproduces III-G of "Whole-Body End-Effector Pose Tracking": robots are spawned at the centre of
each sub-terrain, walk for ``--duration`` seconds under the trained locomotion policy with random
velocity commands (x, y in [-1, 1] m/s, yaw rate in [-1, 1] rad/s), with the scripted arm motion
running, and the states of the robots that did not fall and lean less than ``--max_tilt_deg`` are
recorded. Starting the pose-tracking policy from them makes the switch from locomotion smooth.

The terrain is the pose-tracking task's (``POSE_TRACKING_TERRAIN_CFG``, seeded): a recorded base
position is only valid on the exact same sub-terrain. Every round assigns the robots to all
sub-terrain levels evenly, until each sub-terrain has ``--samples_per_cell`` states.

    python scripts/tools/collect_init_states.py --headless
    -> datasets/aliengo_z1/init_states.pt  (git-ignored)
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Aliengoz1-Locomotion-Phase2", help="Locomotion task the policy was trained on.")
parser.add_argument("--checkpoint", default=None, help="Policy checkpoint (default: latest of --task).")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--duration", type=float, default=4.0, help="Walking time before recording [s].")
parser.add_argument("--max_tilt_deg", type=float, default=55.0, help="Max angle between gravity and base -z.")
parser.add_argument("--samples_per_cell", type=int, default=200, help="States to collect per sub-terrain.")
parser.add_argument("--max_rounds", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default=None, help="Default: the pose-tracking task's INIT_STATE_DATASET.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.models.modules.runners import UnitreeOnPolicyRunner
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.aliengo_z1.pose_tracking_env_cfg import (
    INIT_STATE_DATASET,
    POSE_TRACKING_TERRAIN_CFG,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
from checkpoint_utils import latest_checkpoint_name, latest_run_dir, log_root_from_task  # noqa: E402

sys.path.pop(0)


def make_env_cfg():
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    # the pose-tracking terrain, laid out by difficulty; levels are assigned by hand below
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = POSE_TRACKING_TERRAIN_CFG.replace(curriculum=True)
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    # spawn at the sub-terrain centre with a random heading, no pushes, no time-outs during the walk
    cfg.events.reset_base.params["pose_range"] = {"yaw": (-math.pi, math.pi)}
    cfg.events.push_robot = None
    cfg.episode_length_s = 10.0 * args_cli.duration
    # one constant random velocity command for the whole walk
    cfg.commands.base_velocity.resampling_time_range = (1e3, 1e3)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
    )
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = make_env_cfg()
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device or agent_cfg.device

    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        run_dir = latest_run_dir(log_root_from_task(args_cli.task))
        checkpoint = os.path.join(run_dir, latest_checkpoint_name(run_dir))
    print(f"[INFO] policy: {checkpoint}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = UnitreeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    terrain = base_env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    num_steps = int(round(args_cli.duration / base_env.step_dt))
    cos_max_tilt = math.cos(math.radians(args_cli.max_tilt_deg))
    env_ids = torch.arange(env.num_envs, device=base_env.device)

    records = {k: [] for k in ("terrain_level", "terrain_type", "root_pos_rel", "root_quat", "joint_pos")}
    counts = torch.zeros(rows, cols, dtype=torch.long)
    # everything under inference mode: tensors the env updates in place are created inside it
    for rnd in range(args_cli.max_rounds):
        # every level equally often within each type column (types are fixed per env by the importer)
        terrain.terrain_levels[:] = (env_ids + rnd) % rows
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        obs, _ = env.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=base_env.device)
        for _ in range(num_steps):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()

        # upright: -z of the base within max_tilt of gravity, i.e. projected gravity z <= -cos(max_tilt)
        upright = robot.data.projected_gravity_b[:, 2] <= -cos_max_tilt
        ok = alive & upright
        levels, types = terrain.terrain_levels[ok], terrain.terrain_types[ok]
        records["terrain_level"].append(levels.cpu())
        records["terrain_type"].append(types.cpu())
        records["root_pos_rel"].append((robot.data.root_link_pos_w[ok] - terrain.env_origins[ok]).cpu())
        records["root_quat"].append(robot.data.root_link_quat_w[ok].cpu())
        records["joint_pos"].append(robot.data.joint_pos[ok].cpu())
        counts.index_put_((levels.cpu(), types.cpu()), torch.ones_like(levels.cpu()), accumulate=True)
        print(
            f"[INFO] round {rnd}: kept {int(ok.sum())}/{env.num_envs} (fell {int((~alive).sum())},"
            f" tilted {int((alive & ~upright).sum())}); min per cell {int(counts.min())}"
        )
        if counts.min() >= args_cli.samples_per_cell:
            break

    data = {k: torch.cat(v) for k, v in records.items()}
    data["joint_names"] = robot.joint_names
    data["meta"] = {
        "num_rows": rows,
        "num_cols": cols,
        "seed": POSE_TRACKING_TERRAIN_CFG.seed,
        "sub_terrains": list(POSE_TRACKING_TERRAIN_CFG.sub_terrains.keys()),
        "checkpoint": checkpoint,
        "duration": args_cli.duration,
        "max_tilt_deg": args_cli.max_tilt_deg,
    }
    output = os.path.abspath(args_cli.output or INIT_STATE_DATASET)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(data, output)
    print(f"[INFO] samples per sub-terrain (rows = levels, cols = {data['meta']['sub_terrains']}):\n{counts}")
    print(f"[INFO] saved {len(data['root_quat'])} states to {output}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    # simulation_app.close() can hang after the work is done; the dataset is already saved.
    sys.stdout.flush()
    os._exit(0)
