"""Film one continuous take: walk on four legs -> rise to the hind stance -> cross the wall -> walk on.

Local helper for presentation videos. Two policies share one environment (both are written on the
unified observation, same action scale), switched per environment on a fixed schedule:

* walker (``Go2-Multitask-Gallop-Phase2``) while on four legs, before the rise and after the landing;
* the wall crossing policy from the rise until shortly after the crossing succeeds.

The wall task is modified for the take only: the stance is not pinned but starts at ``--rise_at``
(the walker has stopped by then), the jump fires on distance alone (no time limit), the episode does
not end at success, and every robot is moved back to ``--back_gap`` from the wall behind it so there
is room to walk first.

    python scripts/rsl_rl/film_wallcross_sequence.py --checkpoint <wall.pt> --walker <walk.pt> \
        --row 53 --video out.mp4 --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-Paced")
parser.add_argument("--walker_task", type=str, default="Go2-Multitask-Gallop-Phase2")
parser.add_argument("--walker", type=str, required=True, help="Walking policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--row", type=int, default=53)
parser.add_argument("--env", type=int, default=None, help="Env to film (default: chosen from the wall table).")
parser.add_argument("--min_height", type=float, default=0.6)
parser.add_argument("--max_height", type=float, default=0.72)
parser.add_argument("--back_gap", type=float, default=0.9, help="Start this far from the wall behind the robot.")
parser.add_argument("--walk_speed", type=float, default=0.6)
parser.add_argument("--approach_speed", type=float, default=0.6)
parser.add_argument("--walk_until", type=float, default=2.5, help="Four-legged walk ends (s).")
parser.add_argument("--rise_at", type=float, default=3.2, help="Stance starts; policy switches to the wall one (s).")
parser.add_argument("--approach_at", type=float, default=3.2,
                    help="Forward command in the stance starts (s). Training has it on from the first step of the rise.")
parser.add_argument("--handover_s", type=float, default=0.5, help="Keep the wall policy this long after success.")
parser.add_argument("--walk_after_s", type=float, default=3.0)
parser.add_argument("--stop_after_s", type=float, default=1.0)
parser.add_argument("--max_s", type=float, default=25.0)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--moe", type=str, default=None,
                    help="One policy for the whole take: the MoE npz of scripts/moe/train_gate.py (walking + wall experts "
                         "and the gate). The schedule still sets the commands; the policy is no longer switched.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.wallcross import terrains  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.tasks.multitask.obs_spec import CRITIC_UNIFIED, POLICY_UNIFIED  # noqa: E402

POLICY_UNIFIED_DIM = sum(b.width for b in POLICY_UNIFIED)
CRITIC_UNIFIED_DIM = sum(b.width for b in CRITIC_UNIFIED)


def main():
    a = args_cli
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.curriculum.terrain_levels = None
    env_cfg.episode_length_s = a.max_s + 5.0
    hs = env_cfg.commands.handstand
    hs.pinned = False
    hs.trigger_time_range = (a.rise_at, a.rise_at)
    hs.hold_duration_range = (1.0e9, 1.0e9)
    hs.episode_probability = 1.0
    hs.initial_takeoff_speed_limit = 10.0   # the gate would defer the rise under the approach command
    wc = env_cfg.commands.wallcross
    wc.jump_episode_probability = 1.0
    wc.trigger_min_time_s = a.rise_at + 1.0   # training: 1.0 s after the (pinned) stance starts
    wc.trigger_max_time_s = 1.0e9
    wc.wall_timeout_s = 1.0e9
    env_cfg.terminations.wall_success_done = None
    vel = env_cfg.commands.base_velocity
    vel.heading_command = False
    vel.resampling_time_range = (1.0e9, 1.0e9)
    if a.video:
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-0.6, -2.6, 1.4)
        env_cfg.viewer.lookat = (0.5, 0.0, 0.35)

    env = gym.make(a.task, cfg=env_cfg, render_mode="rgb_array" if a.video else None)
    terrain = env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = a.row
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    agent_cfg = cli_args.parse_rsl_rl_cfg(a.task, a)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped
    dev, dt, n = inner.device, inner.step_dt, inner.num_envs

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(a.checkpoint)
    wall_policy = runner.get_inference_policy(device=dev)
    moe = None
    if a.moe:
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location("moe_export", os.path.join(os.path.dirname(__file__), "..", "moe", "export_moe_onnx.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        moe = mod.MoEPolicy(np.load(a.moe)).to(dev).eval()
        print(f"[film] one policy (MoE) from {a.moe}")
    # Height-map wall tasks append their map after the unified layout (actor and critic); the walker
    # is built and run on the unified part only.
    def walker_obs(o):
        if o["policy"].shape[-1] == POLICY_UNIFIED_DIM:
            return o
        o = o.clone()
        o["policy"] = o["policy"][..., :POLICY_UNIFIED_DIM]
        o["critic"] = o["critic"][..., :CRITIC_UNIFIED_DIM]
        return o

    class _WalkerView:
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            return getattr(self._base, name)

        def get_observations(self):
            return walker_obs(self._base.get_observations())

    walk_args = copy.copy(a)
    walk_args.checkpoint = a.walker
    walk_cfg = cli_args.parse_rsl_rl_cfg(a.walker_task, walk_args)
    walk_runner = OnPolicyRunner(_WalkerView(env), walk_cfg.to_dict(), log_dir=None, device=walk_cfg.device)
    walk_runner.load(a.walker)
    walk_policy = walk_runner.get_inference_policy(device=dev)

    wall = inner.command_manager.get_term("wallcross")
    velocity = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    half, height, thick = wall.wall()
    for e in range(n):
        print(f"[wall] env {e:2d} half {half[e]:.2f} height {height[e]:.3f} thickness {thick[e]:.3f}")
    if a.env is None:
        ok = (height >= a.min_height) & (height <= a.max_height) & (thick < 0.1)
        if not torch.any(ok):
            raise SystemExit("no wall in this row matches --min_height/--max_height and thickness < 0.1")
        film = int(torch.where(ok, half, torch.full_like(half, -1.0)).argmax())
    else:
        film = a.env
    print(f"[film] env {film}: half {half[film]:.2f} m, height {height[film]:.3f} m, thickness {thick[film]:.3f} m")
    if a.video:
        inner.viewport_camera_controller.set_view_env_index(film)

    # Move every robot back towards the wall behind it (tile -x side), facing +x, so it can walk first.
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += inner.scene.env_origins
    pose[:, 0] -= half - a.back_gap
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))

    writer = None
    if a.video:
        import imageio

        writer = imageio.get_writer(a.video, fps=round(1 / dt), codec="libx264", quality=8, macro_block_size=8)

    success_at = torch.full((n,), float("nan"), device=dev)
    triggered_at = torch.full((n,), float("nan"), device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell_at = torch.full((n,), float("nan"), device=dev)
    obs = env.get_observations()
    events = []
    for step in range(int(a.max_s / dt)):
        t = step * dt
        after = t - success_at                        # nan until success
        on_wall_policy = (t >= a.rise_at) & ~(after >= a.handover_s)
        cmd = torch.zeros(n, device=dev)
        if t < a.walk_until:
            cmd[:] = a.walk_speed if t >= 0.5 else 0.0
        elif t >= a.approach_at:
            cmd[:] = a.approach_speed
        walking_on = after >= a.handover_s
        cmd = torch.where(walking_on & (after < a.handover_s + a.walk_after_s), a.walk_speed, cmd)
        cmd = torch.where(walking_on & (after >= a.handover_s + a.walk_after_s), 0.0, cmd)
        velocity.vel_command_b[:, 0] = cmd
        velocity.vel_command_b[:, 1:] = 0.0

        with torch.inference_mode():
            if moe is not None:
                act = moe(obs["policy"])
            else:
                act = torch.where(on_wall_policy[:, None], wall_policy(obs), walk_policy(walker_obs(obs)))
            obs, _, dones, _ = env.step(act)
        if writer is not None:
            writer.append_data(inner.render())
        dones = dones.bool()
        new_trig = wall.triggered & torch.isnan(triggered_at)
        triggered_at[new_trig] = t
        new_succ = wall.success & torch.isnan(success_at)
        success_at[new_succ] = t
        if new_trig[film]:
            events.append(("jump", t))
        if new_succ[film]:
            events.append(("success", t))
        if dones[film] and alive[film]:
            tm = inner.termination_manager
            why = [k for k in tm.active_terms if bool(tm.get_term(k)[film])]
            events.append((f"episode ended ({'+'.join(why)})", t))
            print(f"[film] env {film} ended at t={t:.2f}s", flush=True)
        fell_at = torch.where(alive & dones & torch.isnan(fell_at), torch.full_like(fell_at, t), fell_at)
        alive &= ~dones
        if a.video and alive[film] and not torch.isnan(success_at[film]) and t - float(success_at[film]) >= (
            a.handover_s + a.walk_after_s + a.stop_after_s
        ):
            break
        if a.video and not alive[film] and t - [e for e in events if e[0].startswith("episode")][0][1] > 1.5:
            break

    if writer is not None:
        writer.close()
    print(f"[film] events of env {film}: {events}")
    s = torch.isfinite(success_at)
    print(f"[film] all envs: crossed {int(s.sum())}/{n}, still alive {int(alive.sum())}/{n}, "
          f"crossed and alive {int((s & alive).sum())}/{n}")
    fell = torch.isfinite(fell_at)
    stage = {
        "before_jump": fell & ~(fell_at >= triggered_at),
        "after_jump_before_success": fell & (fell_at >= triggered_at) & ~(fell_at >= success_at),
        "after_success_wall_policy": fell & (fell_at >= success_at) & (fell_at < success_at + a.handover_s),
        "after_handover_walker": fell & (fell_at >= success_at + a.handover_s),
    }
    trig = torch.isfinite(triggered_at)
    print("[film] alive, never jumped:", int((alive & ~trig).sum()), " alive, jumped but no success:", int((alive & trig & ~s).sum()),
          " envs:", [e for e in range(n) if bool(alive[e] & ~s[e])][:40])
    gap_now = wall.base_gap
    print("[film] gap of alive-no-success p10/50/90:", [round(float(x), 2) for x in torch.quantile(gap_now[alive & ~s], torch.tensor([0.1, 0.5, 0.9], device=dev))] if bool((alive & ~s).any()) else None)
    print("[film] falls by stage:", {k: int(v.sum()) for k, v in stage.items()})
    print("[film] crossed and still up:", [e for e in range(n) if bool(s[e] & alive[e])])
    print(f"[film] schedule: walk 0.5-{a.walk_until}s, stop, rise at {a.rise_at}s, approach from {a.approach_at}s, "
          f"walker again {a.handover_s}s after success for {a.walk_after_s}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
