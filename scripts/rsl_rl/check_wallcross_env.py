"""Check the wall crossing environment before training on it.

Local helper. Runs a policy (the hind-stance expert by default -- it cannot cross, so this is the
"hold still at the wall" case of the reward budget) and reports:

* **budget** -- mean episode return by outcome (crossed / fell / timed out at the wall / no jump),
  with the per-term breakdown in points, to compare with the design estimate;
* **cue** -- what the policy observes around the jump: ``jump_command`` on for 0.5 s with the
  intended ``target_height``, the stance columns dropping to zero in the same step;
* **gates** -- that the stance rewards and ``undesired_contacts`` fall silent after the jump.

    python scripts/rsl_rl/check_wallcross_env.py --checkpoint <model_1499.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--duration_s", type=float, default=45.0)
parser.add_argument("--levels", choices=["start", "spread"], default="start",
                    help="start: every env on row 0, as training begins. spread: all rows.")
parser.add_argument("--row", type=int, default=None, help="Put every env on this terrain row (overrides --levels).")
parser.add_argument("--video", type=str, default=None, help="Chase-camera mp4 of env 0.")
parser.add_argument("--jump_episode_probability", type=float, default=None,
                    help="Override the share of episodes that carry a jump (0 = only no-jump episodes).")
parser.add_argument("--trace", type=str, default=None, help="Per-step csv of env 0 (speed, command, gap, pose).")
parser.add_argument("--out", type=str, default="reproduce_logs/wallcross_env")
parser.add_argument("--gain_scale", type=float, default=None, help="Multiply every actuator's PD stiffness and damping (sim2sim probe).")
parser.add_argument("--sim_dt", type=float, default=None, help="Physics step [s]; decimation is set to keep the 50 Hz control rate.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.multitask.obs_spec import POLICY_UNIFIED, block_offsets  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

STANCE_TERMS = ["stance_pitch", "upright_balance", "stance_held", "track_lin_vel_xy", "track_ang_vel_z"]


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.scene.terrain.max_init_terrain_level = 0 if args_cli.levels == "start" else None
    # Budget is about the training distribution: keep the play env's curriculum from moving rows.
    env_cfg.curriculum.terrain_levels = None
    if args_cli.sim_dt is not None:
        control_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.decimation = round(control_dt / args_cli.sim_dt)
        env_cfg.sim.dt = args_cli.sim_dt
        env_cfg.sim.render_interval = env_cfg.decimation
        print(f"[sim] dt {env_cfg.sim.dt} decimation {env_cfg.decimation}", flush=True)
    if args_cli.jump_episode_probability is not None:
        env_cfg.commands.wallcross.jump_episode_probability = args_cli.jump_episode_probability
    if args_cli.video:
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-1.2, -2.2, 1.3)
        env_cfg.viewer.lookat = (0.6, 0.0, 0.3)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.row is not None:
        # Before the wrapper's reset, so the first episode already starts on the chosen row.
        terrain = env.unwrapped.scene.terrain
        terrain.terrain_levels[:] = args_cli.row
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.gain_scale is not None:
        for act in env.unwrapped.scene["robot"].actuators.values():
            act.stiffness *= args_cli.gain_scale
            act.damping *= args_cli.gain_scale
        print(f"[sim] actuator stiffness/damping x{args_cli.gain_scale}", flush=True)
    inner = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=inner.device)

    wall = inner.command_manager.get_term("wallcross")
    rm, tm = inner.reward_manager, inner.termination_manager
    names = rm.active_terms
    idx = {n: i for i, n in enumerate(names)}
    off = block_offsets(POLICY_UNIFIED)
    dt = inner.step_dt
    n = inner.num_envs
    dev = inner.device

    ep_terms = torch.zeros(n, len(names), device=dev)
    prev_triggered = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_jump_ep = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_success = torch.zeros(n, dtype=torch.bool, device=dev)
    per_episode = []                   # (row, wall height, wall thickness, outcome, kinematics) per finished episode

    # Jump kinematics, per env, from the trigger to the end of the episode. Heights are the base above
    # the tile's ground (env origin z; the tile is flat apart from the ring). "Airborne" = every body
    # of the contact sensor under the contact threshold; "hind air" = both hind feet off the ground.
    robot = inner.scene["robot"]
    sensor = inner.scene.sensors[wall.cfg.contact_sensor_name]
    print(f"[kin] contact sensor bodies {sensor.num_bodies} / robot bodies {robot.num_bodies}", flush=True)
    sensor_all = sensor
    sens_names = list(sensor.body_names)
    rob_ids_sorted, _ = robot.find_bodies(sens_names, preserve_order=True)
    sens2rob = list(range(len(sens_names)))
    snap, snap_touch_floor, snap_touch_wall = None, None, None
    fall_detail = []
    hind_ids = [wall.sensor_foot_ids[2], wall.sensor_foot_ids[3]]   # foot_body_names = FR, FL, RR, RL
    thr = wall.cfg.contact_threshold
    kin_names = ["z_trigger", "z_apex", "air_max_s", "hind_air_max_s", "z_success", "tilt_success_deg",
                 "feet_success", "z_min_after_success"]
    kin = {k: torch.full((n,), float("nan"), device=dev) for k in kin_names}
    air_run = torch.zeros(n, device=dev)
    hind_run = torch.zeros(n, device=dev)
    prev_succ_k = torch.zeros(n, dtype=torch.bool, device=dev)
    # Before the jump: episode clock, whether the stance was ever up, the fastest horizontal base speed,
    # and the gap to the wall -- what a fall before the jump is broken down by.
    pre = {k: torch.zeros(n, device=dev) for k in ("t", "ever_up", "speed_max", "gap")}
    # No-jump episodes: the moment the velocity command is zeroed ("parked") and the state at the end,
    # to tell "cannot stop" from other falls. gx = projected gravity on the body x axis (the stance
    # stands nose up: gx near -1 upright on the hind legs, near 0 on four legs).
    park_names = ["park_t", "park_speed", "park_gap", "last_speed", "last_gx", "last_z", "last_vgap"]
    park = {k: torch.full((n,), float("nan"), device=dev) for k in park_names}
    prev_parked = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_gap = wall.base_gap.clone()
    trace = open(args_cli.trace, "w") if args_cli.trace else None
    if trace:
        trace.write("t,jump_episode,parked,triggered,speed,cmd,gap,z,gx\n")

    def reset_kin(ids):
        for v in kin.values():
            v[ids] = float("nan")
        air_run[ids] = 0.0
        hind_run[ids] = 0.0
        prev_succ_k[ids] = False
        for v in pre.values():
            v[ids] = 0.0
        for v in park.values():
            v[ids] = float("nan")
    since_trigger = torch.full((n,), -1, dtype=torch.long, device=dev)
    outcomes = defaultdict(list)       # outcome -> list of per-term point vectors
    cue = defaultdict(list)            # step offset from trigger -> observed values
    gates = defaultdict(list)          # step offset -> stance-term and contact-term values
    writer = None
    if args_cli.video:
        import imageio

        writer = imageio.get_writer(args_cli.video, fps=round(1 / dt), codec="libx264", quality=8,
                                    macro_block_size=8)

    hs = inner.command_manager.get_term("handstand")
    prev_upright = torch.zeros(n, dtype=torch.bool, device=dev)
    upright_at_trigger = []
    trigger_times = []
    prev_row = inner.scene.terrain.terrain_levels.clone()
    _, prev_wall_h, prev_wall_t = (x.clone() for x in wall.wall())
    obs = env.get_observations()
    for step in range(int(args_cli.duration_s / dt)):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        pol = obs["policy"] if not torch.is_tensor(obs) else obs
        ep_terms += rm._step_reward * dt
        dones = dones.bool()

        # Jump kinematics (continuing environments only: the done ones already hold the next episode).
        z = robot.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]
        force = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
        airborne = (force < thr).all(dim=1)
        hind_air = (force[:, hind_ids] < thr).all(dim=1)
        after = wall.triggered & ~dones
        just = after & torch.isnan(kin["z_trigger"])
        kin["z_trigger"][just] = z[just]
        kin["z_apex"][after] = torch.fmax(kin["z_apex"][after], z[after])
        air_run = torch.where(after & airborne, air_run + dt, torch.zeros_like(air_run))
        hind_run = torch.where(after & hind_air, hind_run + dt, torch.zeros_like(hind_run))
        kin["air_max_s"][after] = torch.fmax(kin["air_max_s"][after], air_run[after])
        kin["hind_air_max_s"][after] = torch.fmax(kin["hind_air_max_s"][after], hind_run[after])
        succ_now = wall.success & ~prev_succ_k & ~dones
        if succ_now.any():
            feet_f = force[:, wall.sensor_foot_ids]
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
            kin["z_success"][succ_now] = z[succ_now]
            kin["tilt_success_deg"][succ_now] = tilt[succ_now]
            kin["feet_success"][succ_now] = (feet_f[succ_now] > thr).sum(dim=1).float()
        in_succ = wall.success & ~dones
        kin["z_min_after_success"][in_succ] = torch.fmin(kin["z_min_after_success"][in_succ], z[in_succ])
        prev_succ_k = wall.success & ~dones
        live_pre = ~wall.triggered & ~dones
        pre["t"][~dones] += dt
        pre["ever_up"][live_pre & hs.success] = 1.0
        speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
        pre["speed_max"][live_pre] = torch.maximum(pre["speed_max"][live_pre], speed[live_pre])
        pre["gap"][live_pre] = wall.base_gap[live_pre]
        cmd = torch.linalg.norm(inner.command_manager.get_command("base_velocity")[:, :2], dim=-1)
        gx = robot.data.projected_gravity_b[:, 0]
        new_park = wall.parked & ~prev_parked & ~dones
        park["park_t"][new_park] = pre["t"][new_park]
        park["park_speed"][new_park] = speed[new_park]
        park["park_gap"][new_park] = wall.base_gap[new_park]
        live_park = wall.parked & ~dones
        park["last_speed"][live_park] = speed[live_park]
        park["last_gx"][live_park] = gx[live_park]
        park["last_z"][live_park] = z[live_park]
        park["last_vgap"][live_park] = ((prev_gap - wall.base_gap) / dt)[live_park]   # + = toward the wall
        prev_parked = wall.parked & ~dones
        prev_gap = wall.base_gap.clone()
        if trace:
            trace.write(f"{float(pre['t'][0]):.3f},{int(wall.jump_episode[0])},{int(wall.parked[0])},"
                        f"{int(wall.triggered[0])},{float(speed[0]):.3f},{float(cmd[0]):.3f},"
                        f"{float(wall.base_gap[0]):.3f},{float(z[0]):.3f},{float(gx[0]):.3f}\n")

        # Cue and gates, aligned on the trigger (continuing environments only).
        fired = wall.triggered & ~prev_triggered & ~dones
        if fired.any():
            # The stance is released inside the step that fires, so read it from the step before.
            upright_at_trigger.append(prev_upright[fired].float().cpu())
            trigger_times.append(wall.trigger_time[fired].cpu())
        since_trigger[fired] = 0
        live = (since_trigger >= 0) & ~dones
        for k in range(0, 40):
            m = live & (since_trigger == k)
            if m.any() and k in (0, 1, 5, 10, 12, 20, 24, 25, 26, 30, 39) or (m.any() and k < 3):
                cue[k].append(torch.stack([
                    pol[m, off["jump_command"]], pol[m, off["jump_command"] + 1],
                    pol[m, off["jump_time"]], pol[m, off["handstand_command"]],
                    pol[m, off["handstand_command"] + 1],
                    (pol[m, off["jump_command"] + 1] - (wall.wall()[1][m] - 0.25).clamp(0.05, 1.0)).abs(),
                ], dim=1).cpu())
                gates[k].append(torch.stack(
                    [rm._step_reward[m, idx[t]] for t in STANCE_TERMS if t in idx]
                    + [rm._step_reward[m, idx["undesired_contacts"]]], dim=1).abs().cpu())
        since_trigger[live] += 1

        # Episodes that ended this step: classify with the state from the step before the reset.
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            last = tm._last_episode_dones[done_ids]
            tn = tm.active_terms
            for j, e in enumerate(done_ids.tolist()):
                reasons = [tn[i] for i in range(len(tn)) if bool(last[j, i])]
                if "wall_success_done" in reasons or bool(prev_success[e]):
                    key = "crossed"
                elif "base_contact" in reasons or "collapsed" in reasons:
                    key = "fell_after_jump" if bool(prev_triggered[e]) else "fell_before_jump"
                elif "wall_timeout" in reasons:
                    key = "held_at_wall_timeout"
                elif not bool(prev_jump_ep[e]):
                    key = "no_jump_episode"
                else:
                    key = "time_out_" + "+".join(reasons)
                outcomes[key].append(ep_terms[e].cpu())
                if key == "fell_after_jump" and snap is not None:
                    fall_detail.append({
                        "h": float(prev_wall_h[e]), "t": float(prev_wall_t[e]),
                        **{k: float(v[e]) for k, v in snap.items()},
                        "floor": [sens_names[i] for i in range(len(sens_names)) if bool(snap_touch_floor[e, i])],
                        "wall": [sens_names[i] for i in range(len(sens_names)) if bool(snap_touch_wall[e, i])],
                        "reason": "+".join(reasons)})
                per_episode.append((int(prev_row[e]), float(prev_wall_h[e]), float(prev_wall_t[e]), key,
                                    {**{k: float(v[e]) for k, v in kin.items()},
                                     **{"pre_" + k: float(v[e]) for k, v in pre.items()},
                                     **{k: float(v[e]) for k, v in park.items()},
                                     "reason": "+".join(reasons), "jump_episode": float(prev_jump_ep[e])}))
                if e == 0 and writer is not None:
                    print(f"[env0] t={(step + 1) * dt:6.2f}s row={int(prev_row[e])} wall h={float(prev_wall_h[e]):.3f} "
                          f"t={float(prev_wall_t[e]):.3f} -> {key} ({'+'.join(reasons)})", flush=True)
            ep_terms[done_ids] = 0.0
            since_trigger[done_ids] = -1
            reset_kin(done_ids)
        # Snapshot for the falls after the jump: the robot is already reset on the done step.
        cheb_b = wall.chebyshev(robot.data.body_pos_w)
        half_, h_, th_ = wall.wall()
        zb = robot.data.body_pos_w[..., 2] - inner.scene.env_origins[:, None, 2]
        on_wall_b = (cheb_b > (half_ - 0.15)[:, None]) & (cheb_b < (half_ + th_ + 0.15)[:, None]) & (zb >= 0.15)
        touch_b = torch.linalg.norm(sensor_all.data.net_forces_w, dim=-1) > 1.0
        snap = {
            "since_trig_s": (wall.elapsed - wall.trigger_time),
            "gap": wall.base_gap.clone(),
            "past_outer": (-wall.base_gap - th_),
            "parts_past": wall.parts_past.float(),
            "tilt_deg": torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))),
            "z": robot.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2],
            "vz": robot.data.root_lin_vel_w[:, 2].clone(),
            "z_max_minus_wall": wall.z_max - h_,
        }
        snap_touch_floor = (touch_b[:, sens2rob] & ~on_wall_b[:, rob_ids_sorted]).clone()
        snap_touch_wall = (touch_b[:, sens2rob] & on_wall_b[:, rob_ids_sorted]).clone()
        prev_triggered = wall.triggered.clone()
        prev_upright = hs.success.clone()
        prev_jump_ep = wall.jump_episode.clone()
        prev_success = wall.success.clone()
        prev_row = inner.scene.terrain.terrain_levels.clone()
        _, prev_wall_h, prev_wall_t = (x.clone() for x in wall.wall())
        if writer is not None:
            frame = inner.render()
            if frame is not None:
                writer.append_data(frame[..., :3])
    if writer is not None:
        writer.close()
    if trace:
        trace.close()

    summary = {"episodes": {k: len(v) for k, v in outcomes.items()}, "return_points": {}, "terms_points": {}}
    if fall_detail:
        import collections
        thin = [f for f in fall_detail if f["t"] < 0.1]
        def qd(xs):
            xs = torch.tensor(xs)
            return [round(float(x), 3) for x in torch.quantile(xs, torch.tensor([0.1, 0.5, 0.9]))] if len(xs) else None
        summary["fall_after_jump_thin"] = {
            "n": len(thin),
            **{k + "_p10_50_90": qd([f[k] for f in thin]) for k in snap},
            "base_side": {"before_wall": sum(f["gap"] > 0 for f in thin), "within_wall": sum(f["gap"] <= 0 and f["past_outer"] <= 0 for f in thin),
                          "past_wall": sum(f["past_outer"] > 0 for f in thin)},
            "floor_bodies": collections.Counter(b for f in thin for b in f["floor"]).most_common(8),
            "wall_bodies": collections.Counter(b for f in thin for b in f["wall"]).most_common(8),
            "reasons": collections.Counter(f["reason"] for f in thin).most_common(5),
        }
        with open(os.path.join(args_cli.out, "fall_after_jump.json"), "w") as fh:
            json.dump(fall_detail, fh)
    for k, v in outcomes.items():
        t = torch.stack(v).mean(0)
        summary["return_points"][k] = round(float(t.sum()), 2)
        summary["terms_points"][k] = {nm: round(float(x), 2) for nm, x in zip(names, t) if abs(float(x)) >= 0.05}
    if upright_at_trigger:
        u = torch.cat(upright_at_trigger)
        tt = torch.cat(trigger_times)
        summary["jumps_observed"] = int(u.numel())
        summary["hind_stance_held_at_trigger"] = round(float(u.mean()), 3)
        summary["trigger_time_s_p10_p50_p90"] = [round(float(x), 2) for x in torch.quantile(tt, torch.tensor([0.1, 0.5, 0.9]))]
    cue_names = ["jump_enabled", "jump_target_height", "jump_time", "handstand_enabled", "handstand_stance",
                 "target_err"]
    summary["cue_mean_by_step_after_trigger"] = {
        k: {nm: round(float(x), 3) for nm, x in zip(cue_names, torch.cat(v).mean(0))} for k, v in sorted(cue.items())
    }
    gate_names = [t for t in STANCE_TERMS if t in idx] + ["undesired_contacts"]
    summary["gate_abs_mean_by_step_after_trigger"] = {
        k: {nm: round(float(x), 4) for nm, x in zip(gate_names, torch.cat(v).mean(0))} for k, v in sorted(gates.items())
    }
    # Outcomes by the wall each episode faced: per row, and binned by height (thick stages) / thickness.
    def tally(eps):
        c = defaultdict(int)
        stood = 0
        for _, _, _, k, x in eps:
            c[k] += 1
            # "crossed" includes episodes that fell within success_end_s after the success judgement
            # (the success reward is already paid by then). Stood = ended by wall_success_done alone.
            stood += int(k == "crossed" and x["reason"] == "wall_success_done")
        jumped = len(eps) - c["no_jump_episode"] - c["fell_before_jump"]
        return {"episodes": len(eps), **dict(c), "crossed_and_stood": stood,
                "stood_of_jumped": round(stood / jumped, 3) if jumped else None,
                "crossed_of_jumped": round(c["crossed"] / jumped, 3) if jumped else None,
                "crossed_of_all": round(c["crossed"] / len(eps), 3) if eps else None}
    by_row = defaultdict(list)
    for ep in per_episode:
        by_row[ep[0]].append(ep)
    summary["by_row"] = {
        r: {"height_mean": round(sum(e[1] for e in v) / len(v), 3),
            "thickness_mean": round(sum(e[2] for e in v) / len(v), 3), **tally(v)}
        for r, v in sorted(by_row.items())
    }
    thick = [e for e in per_episode if e[2] >= 0.45]
    summary["thick_by_height"] = {
        f"{lo:.2f}-{lo + 0.1:.2f}": tally([e for e in thick if lo <= e[1] < lo + 0.1])
        for lo in [x / 10 for x in range(0, 9)] if any(lo <= e[1] < lo + 0.1 for e in thick)
    }
    # Thin walls (the steel-plate target) by height; the thin-first terrain has them from 0.3 m up.
    thin = [e for e in per_episode if e[2] < 0.1]
    summary["thin_by_height"] = {
        f"{lo:.2f}-{lo + 0.1:.2f}": tally([e for e in thin if lo <= e[1] < lo + 0.1])
        for lo in [x / 10 for x in range(0, 9)] if any(lo <= e[1] < lo + 0.1 for e in thin)
    }
    high = [e for e in per_episode if e[1] >= 0.6]
    summary["high_by_thickness"] = {
        f"{lo:.2f}-{lo + 0.1:.2f}": tally([e for e in high if lo <= e[2] < lo + 0.1])
        for lo in [x / 10 for x in range(0, 6)] if any(lo <= e[2] < lo + 0.1 for e in high)
    }
    # Kinematics of the jumps (episodes whose jump fired), by wall height; and the pose at success.
    def q(vals, qs=(0.1, 0.5, 0.9)):
        vals = torch.tensor([v for v in vals if v == v])
        return [round(float(x), 3) for x in torch.quantile(vals, torch.tensor(qs))] if vals.numel() else None

    def kin_stats(eps):
        eps = [e for e in eps if e[4]["z_trigger"] == e[4]["z_trigger"]]
        if not eps:
            return None
        k = [e[4] for e in eps]
        clear = [ki["z_apex"] - e[1] for ki, e in zip(k, eps)]
        return {
            "jumps": len(eps),
            "z_trigger_p10_50_90": q([x["z_trigger"] for x in k]),
            "z_apex_p10_50_90": q([x["z_apex"] for x in k]),
            "rise_p10_50_90": q([x["z_apex"] - x["z_trigger"] for x in k]),
            "apex_minus_wall_p10_50_90": q(clear),
            "air_max_s_p50_90": q([x["air_max_s"] for x in k], (0.5, 0.9)),
            "frac_airborne_ge_0.1s": round(sum(x["air_max_s"] >= 0.1 for x in k) / len(k), 3),
            "hind_air_max_s_p50_90": q([x["hind_air_max_s"] for x in k], (0.5, 0.9)),
            "frac_hind_air_ge_0.1s": round(sum(x["hind_air_max_s"] >= 0.1 for x in k) / len(k), 3),
        }
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.8)]
    summary["jump_kinematics_by_height"] = {}
    for lo, hi in bins:
        sel = [e for e in per_episode if lo <= e[1] < hi]
        entry = {"all": kin_stats(sel)}
        for outcome in ("crossed", "fell_after_jump", "held_at_wall_timeout"):
            entry[outcome] = kin_stats([e for e in sel if e[3] == outcome])
        summary["jump_kinematics_by_height"][f"{lo:.1f}-{hi:.1f}"] = entry
    crossed = [e[4] for e in per_episode if e[3] == "crossed" and e[4]["z_success"] == e[4]["z_success"]]
    if crossed:
        summary["success_pose"] = {
            "n": len(crossed),
            "z_at_success_p10_50_90": q([x["z_success"] for x in crossed]),
            "z_min_after_success_p10_50_90": q([x["z_min_after_success"] for x in crossed]),
            "tilt_deg_at_success_p10_50_90": q([x["tilt_success_deg"] for x in crossed]),
            "feet_in_contact_at_success_p10_50_90": q([x["feet_success"] for x in crossed]),
        }
    # Falls before the jump: when, whether the stance had ever been up, how fast, how far from the wall;
    # against the same quantities of the episodes whose jump fired.
    def pre_stats(eps):
        if not eps:
            return None
        k = [e[4] for e in eps]
        reasons = defaultdict(int)
        for x in k:
            reasons[x["reason"]] += 1
        return {
            "n": len(k),
            "t_s_p10_50_90": q([x["pre_t"] for x in k]),
            "ever_upright": round(sum(x["pre_ever_up"] for x in k) / len(k), 3),
            "speed_max_p10_50_90": q([x["pre_speed_max"] for x in k]),
            "gap_last_p10_50_90": q([x["pre_gap"] for x in k]),
            "reasons": dict(reasons),
            "jump_episode": round(sum(x["jump_episode"] for x in k) / len(k), 3),
        }
    summary["before_jump"] = {
        "fell_before_jump": pre_stats([e for e in per_episode if e[3] == "fell_before_jump"]),
        "jumped": pre_stats([e for e in per_episode if e[4]["z_trigger"] == e[4]["z_trigger"]]),
    }
    for flag, name in ((1.0, "fell_before_jump_in_jump_episode"), (0.0, "fell_in_no_jump_episode")):
        summary["before_jump"][name] = pre_stats(
            [e for e in per_episode if e[3] == "fell_before_jump" and e[4]["jump_episode"] == flag])
    summary["before_jump"]["no_jump_episodes_total"] = sum(e[4]["jump_episode"] == 0.0 for e in per_episode)
    def park_stats(eps):
        eps = [e[4] for e in eps if e[4]["park_t"] == e[4]["park_t"]]
        if not eps:
            return None
        return {
            "n": len(eps),
            "park_t_p10_50_90": q([x["park_t"] for x in eps]),
            "park_speed_p10_50_90": q([x["park_speed"] for x in eps]),
            "park_gap_p10_50_90": q([x["park_gap"] for x in eps]),
            "park_to_end_s_p10_50_90": q([x["pre_t"] - x["park_t"] for x in eps]),
            "travel_after_park_m_p10_50_90": q([x["park_gap"] - x["pre_gap"] for x in eps]),
            "last_speed_p10_50_90": q([x["last_speed"] for x in eps]),
            "last_vgap_p10_50_90": q([x["last_vgap"] for x in eps]),
            "last_gx_p10_50_90": q([x["last_gx"] for x in eps]),
            "last_z_p10_50_90": q([x["last_z"] for x in eps]),
        }
    nj = [e for e in per_episode if e[4]["jump_episode"] == 0.0]
    summary["no_jump_parked"] = {
        "fell": park_stats([e for e in nj if e[3] == "fell_before_jump"]),
        "survived": park_stats([e for e in nj if e[3] != "fell_before_jump"]),
        "fell_without_park": sum(e[3] == "fell_before_jump" and e[4]["park_t"] != e[4]["park_t"] for e in nj),
    }
    fb = [e for e in per_episode if e[3] == "fell_before_jump"]
    summary["before_jump"]["fell_t_hist_0.25s"] = {
        f"{lo / 4:.2f}": sum(lo / 4 <= e[4]["pre_t"] < (lo + 1) / 4 for e in fb) for lo in range(0, 16)
    }
    with open(os.path.join(args_cli.out, f"summary_{args_cli.levels}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
