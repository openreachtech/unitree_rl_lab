"""What the robot does in the steps after it is parked (no-jump episodes of the wall crossing task).

Local helper. Every episode is a no-jump one; each step of every environment is recorded (joint
positions and velocities, policy actions, applied torques, base pose and velocity, contact forces,
the command columns the policy sees), and the windows around the moment the velocity command is
zeroed ("parked") are averaged over all episodes, split by whether the episode fell. The bodies
that fire ``base_contact`` are recorded inside the termination term itself (the sensor buffers are
cleared by the reset that follows it).

    python scripts/rsl_rl/probe_park_fall.py --checkpoint <model.pt> --out <dir> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-ThinFirst-Paced")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--duration_s", type=float, default=20.0)
parser.add_argument("--pre_steps", type=int, default=15)
parser.add_argument("--post_steps", type=int, default=20)
parser.add_argument("--align", choices=["park", "trigger"], default="park",
                    help="trigger: jump episodes only, windows around the jump cue instead (for comparison).")
parser.add_argument("--out", type=str, default="reproduce_logs/wallcross_env/park_probe")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.multitask.obs_spec import POLICY_UNIFIED, block_offsets  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.commands.wallcross.jump_episode_probability = 1.0 if args_cli.align == "trigger" else 0.0
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=inner.device)

    wall = inner.command_manager.get_term("wallcross")
    robot = inner.scene["robot"]
    sensor = inner.scene.sensors[wall.cfg.contact_sensor_name]
    tm = inner.termination_manager
    n, dev, dt = inner.num_envs, inner.device, inner.step_dt
    off = block_offsets(POLICY_UNIFIED)
    joint_names = list(robot.data.joint_names)
    body_names = list(sensor.body_names)

    # Which bodies fire base_contact: wrap the term so the forces are read before the reset.
    bc_i = tm.active_terms.index("base_contact")
    bc_cfg = tm._term_cfgs[bc_i]
    bc_ids = bc_cfg.params["sensor_cfg"].body_ids
    bc_names = [body_names[i] for i in bc_ids]
    fired_bodies = torch.zeros(n, len(bc_ids), dtype=torch.bool, device=dev)
    orig = bc_cfg.func

    def wrapped(env_, **params):
        out = orig(env_, **params)
        f = torch.linalg.norm(sensor.data.net_forces_w_history[:, :, bc_ids], dim=-1).amax(dim=1)
        fired_bodies[out] = (f > params.get("threshold", 1.0))[out]
        return out

    wrapped.reset = orig.reset          # class terms are reset by the manager
    bc_cfg.func = wrapped

    T = int(args_cli.duration_s / dt)
    feats = {
        "q": 12, "qd": 12, "act": 12, "tau": 12,
        "z": 1, "grav": 3, "angvel_b": 3, "linvel_b": 3,
        "obs_vel_cmd": 3, "obs_hs_cmd": 2, "obs_jump": 1,
        "force": len(body_names), "parked": 1, "alive_done": 1, "timeout": 1,
    }
    H = {k: torch.zeros(T, n, d, device=dev) for k, d in feats.items()}
    term_bodies = {}                                   # (env, step) -> list of body names
    default_q = robot.data.default_joint_pos
    lim = robot.data.soft_joint_pos_limits              # (n, 12, 2)
    obs = env.get_observations()
    for s in range(T):
        with torch.inference_mode():
            act = policy(obs)
            # State the action was chosen from (before the step).
            H["q"][s] = robot.data.joint_pos - default_q
            H["qd"][s] = robot.data.joint_vel
            H["act"][s] = act
            H["z"][s, :, 0] = robot.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]
            H["grav"][s] = robot.data.projected_gravity_b
            H["angvel_b"][s] = robot.data.root_ang_vel_b
            H["linvel_b"][s] = robot.data.root_lin_vel_b
            pol = obs["policy"] if not torch.is_tensor(obs) else obs
            H["obs_vel_cmd"][s] = pol[:, off["velocity_commands"]:off["velocity_commands"] + 3]
            H["obs_hs_cmd"][s] = pol[:, off["handstand_command"]:off["handstand_command"] + 2]
            H["obs_jump"][s, :, 0] = pol[:, off["jump_command"]]
            H["force"][s] = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
            H["parked"][s, :, 0] = (wall.triggered if args_cli.align == "trigger" else wall.parked).float()
            obs, _, dones, _ = env.step(act)
            H["tau"][s] = robot.data.applied_torque
            H["alive_done"][s, :, 0] = dones.float()
            H["timeout"][s, :, 0] = tm.time_outs.float()
            if dones.any():
                for e in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if fired_bodies[e].any():
                        term_bodies[(e, s)] = [bc_names[i] for i in fired_bodies[e].nonzero().flatten().tolist()]
                fired_bodies[dones.bool()] = False

    # Segment episodes, find the park step, collect windows.
    P, Q = args_cli.pre_steps, args_cli.post_steps
    H = {k: v.cpu() for k, v in H.items()}
    done = H["alive_done"][:, :, 0].bool()
    timeout = H["timeout"][:, :, 0].bool()
    parked = H["parked"][:, :, 0].bool()
    win = {"fell": [], "survived": []}
    bodies = Counter()
    fall_after_park = []
    for e in range(n):
        start = 0
        ends = done[:, e].nonzero().flatten().tolist() + [None]
        for end in ends:
            stop = T if end is None else end + 1
            seg = parked[start:stop, e]
            if seg.any():
                p = start + int(seg.nonzero()[0])
                fell = end is not None and not bool(timeout[end, e])
                if p - P >= start and (fell or p + Q < stop):
                    rows = torch.arange(p - P, p + Q + 1)
                    alive = rows <= (end if end is not None else T - 1)
                    rows = rows.clamp(max=T - 1)
                    w = {k: v[rows, e].clone() for k, v in H.items()}
                    w["alive"] = alive.float()
                    win["fell" if fell else "survived"].append(w)
                if fell:
                    bodies["+".join(term_bodies.get((e, end), ["not_base_contact"]))] += 1
                    fall_after_park.append((end - p) * dt)
            start = stop

    def mean_window(ws, key, cols=None):
        if not ws:
            return None
        x = torch.stack([w[key] for w in ws])          # (E, W, d)
        a = torch.stack([w["alive"] for w in ws])[..., None]
        m = (x * a).sum(0) / a.sum(0).clamp(min=1)
        return m if cols is None else m[:, cols]

    offsets = list(range(-P, Q + 1))
    summary = {
        "checkpoint": args_cli.checkpoint, "joint_names": joint_names, "body_names": body_names,
        "episodes_with_window": {k: len(v) for k, v in win.items()},
        "base_contact_bodies": dict(bodies.most_common()),
        "fall_after_park_s_p10_50_90": (torch.tensor(fall_after_park).quantile(
            torch.tensor([0.1, 0.5, 0.9])).tolist() if fall_after_park else None),
        "soft_limits": lim[0].cpu().tolist(), "default_q": default_q[0].cpu().tolist(),
        "offsets": offsets, "mean": {},
    }
    for k in win:
        if not win[k]:
            continue
        summary["mean"][k] = {key: mean_window(win[k], key).tolist() for key in feats if key not in ("alive_done", "timeout")}
        summary["mean"][k]["alive_frac"] = torch.stack([w["alive"] for w in win[k]]).mean(0).tolist()
    with open(os.path.join(args_cli.out, "park_probe.json"), "w") as f:
        json.dump(summary, f)
    torch.save(win, os.path.join(args_cli.out, "windows.pt"))
    print(json.dumps({k: summary[k] for k in ("episodes_with_window", "base_contact_bodies",
                                              "fall_after_park_s_p10_50_90")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
