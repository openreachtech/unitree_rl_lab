"""Crossing rate over circular walls at fixed heights, for a Go2w wall task checkpoint.

The curriculum level does not rank checkpoints -- a ratchet can sit high while the policy crosses
little, and the reverse. This puts every robot in front of a wall of known height and counts.

One row of tiles per height, robots spread evenly over them, all driven straight forward at a
fixed speed for one episode. A robot has *crossed* when its base is ``--past`` m beyond the first
ring's outer face; it has *fallen* if its episode ended (any termination but the timeout) before
that. Each robot counts once, on its first episode. On the wall-jump tasks the task's own success
(over the wall *and* standing on four wheels) is reported too.

    python scripts/rsl_rl/measure_wall_crossing.py --task Go2w-Biped-WallJump-Drive \\
        --checkpoint logs/rsl_rl/go2w_biped_walljump_drive/<run>/model_X.pt \\
        --thickness 0.25 --heights 0.40 0.60 0.70 0.80 1.00 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--wall", choices=("wall", "thin"), default="wall")
parser.add_argument("--heights", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.55])
parser.add_argument("--thickness", type=float, default=None, help="Override the wall thickness, m.")
parser.add_argument("--speed", type=float, default=0.6, help="Forward command, m/s.")
parser.add_argument("--envs_per_height", type=int, default=64)
parser.add_argument(
    "--stochastic", action="store_true", help="Sample actions as in training instead of taking the mean."
)
parser.add_argument("--past", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.tasks.biped import mdp as wall_mdp  # noqa: E402
from unitree_rl_lab.tasks.biped import terrains  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

WALLS = {"wall": 0.05, "thin": 0.01}


def main():
    heights = tuple(args.heights)
    num_envs = args.envs_per_height * len(heights)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs, entry_point_key="play_env_cfg_entry_point")
    thickness = WALLS[args.wall]
    thickness = args.thickness or thickness
    walls = terrains.MeshCircularWallTerrainCfg(
        proportion=1.0, wall_thickness=thickness, row_heights=heights
    )
    cfg.scene.terrain.terrain_generator = terrains.WALL_TERRAIN_CFG.replace(
        num_rows=len(heights), num_cols=1, sub_terrains={"walls": walls}
    )
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.events.spread_over_rows = EventTerm(func=wall_mdp.spread_over_rows, mode="startup")
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(command, "column_ranges"):
        command.column_ranges = {}
    if hasattr(cfg.events, "push_robot"):
        cfg.events.push_robot = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args.stochastic:
        # The training-time action: the GRU carries its state across calls in act() as it does in
        # act_inference(), and the action is sampled from the policy's distribution.
        runner.alg.policy.eval()
        policy = runner.alg.policy.act

    u = env.unwrapped
    robot = u.scene["robot"]
    radius = terrains.ring_radii(walls, cfg.scene.terrain.terrain_generator.size)[0]
    target = radius + thickness / 2 + args.past
    crossed = torch.zeros(num_envs, dtype=torch.bool, device=u.device)
    fallen = torch.zeros_like(crossed)
    # Wall-jump tasks judge their own success (over the wall *and* standing on four wheels).
    jump = u.command_manager.get_term("walljump") if "walljump" in u.command_manager.active_terms else None
    succeeded = torch.zeros_like(crossed)
    finished = torch.zeros_like(crossed)

    obs = env.get_observations()
    for _ in range(int(u.max_episode_length) + 1):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
        r = torch.linalg.norm(robot.data.root_pos_w[:, :2] - u.scene.env_origins[:, :2], dim=1)
        crossed |= ~finished & (r > target)
        if jump is not None:
            succeeded |= ~finished & jump.success
        ended = dones.bool() & ~finished
        timeout = extras.get("time_outs", torch.zeros_like(ended)).bool()
        fallen |= ended & ~timeout & ~crossed
        finished |= ended
        if bool(finished.all()):
            break

    levels = u.scene.terrain.terrain_levels
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"[MW] {args.task} {args.wall} (t={thickness}) speed {args.speed} {mode}  {args.checkpoint}")
    for row, h in enumerate(heights):
        m = levels == row
        extra = f"  success {succeeded[m].float().mean() * 100:5.1f} %" if jump is not None else ""
        print(f"[MW]   {h * 100:5.1f} cm  cross {crossed[m].float().mean() * 100:5.1f} %  fall {fallen[m].float().mean() * 100:5.1f} %{extra}  (n={int(m.sum())})")


if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
