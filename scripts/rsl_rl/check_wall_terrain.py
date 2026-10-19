"""Check the concentric-wall terrain before any reward is written on top of it.

Local helper for the wall crossing task. Three checks, all written to ``--out``:

1. **Shape.** The wall table the generator recorded is compared with the mesh it actually built
   (inner and outer faces, height), and plotted per row against the intended curriculum.
2. **walk** -- the hind-stance expert, unchanged, walks forward into the walls. The robot cannot
   cross, so this only looks for bodies going *through* a wall: a body that moves from inside the
   ring to outside it between two control steps while staying below the wall's top, or whose origin
   sits inside the wall's volume.
3. **throw** -- every robot is launched horizontally at its wall at one of ``--speeds`` with the
   joints held at their defaults, terminations off. A harsher version of the same question, at body
   speeds well above anything the task should produce.

    python scripts/rsl_rl/check_wall_terrain.py --task Go2-Multitask-WallCross-TerrainCheck \
        --checkpoint logs/rsl_rl/go2_multitask_biped_hind/<run>/model_1499.pt --mode walk --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Go2-Multitask-WallCross-TerrainCheck")
parser.add_argument("--mode", choices=["walk", "throw"], default="walk")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--duration_s", type=float, default=16.0)
parser.add_argument("--vx", type=float, default=0.6, help="walk: forward command (m/s).")
parser.add_argument("--speeds", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0], help="throw: m/s.")
parser.add_argument("--show_rows", type=int, nargs="+", default=[0, 9, 17, 23, 30, 40, 47],
                    help="Rows the first environments are moved to, for the stills (env k -> k-th row).")
parser.add_argument("--rows", type=int, nargs="+", default=None,
                    help="Put every environment (after the --show_rows ones) on these rows, cycling.")
parser.add_argument("--still_s", type=float, default=7.0, help="walk: time of the stills.")
parser.add_argument("--video", action="store_true", help="walk: chase-camera mp4 of env 0.")
parser.add_argument("--feet_first", action="store_true", help="throw: launch nose-up so the feet lead.")
parser.add_argument("--debug", type=int, nargs="+", default=None, help="throw: print these envs' first steps.")
parser.add_argument("--out", type=str, default="reproduce_logs/wall_terrain")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.wallcross import terrains  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def check_shape(terrain, out):
    """Compare the recorded wall table with the vertices of the mesh that was built."""
    table = terrains.WALL_TABLE
    mesh = terrains.TERRAIN_MESH
    origins = terrain.terrain_origins.cpu().numpy()
    rows, cols, _ = table.shape
    verts = mesh.vertices
    raised = verts[verts[:, 2] > 1e-4]
    worst = {"inner": 0.0, "outer": 0.0, "height": 0.0}
    for r in range(rows):
        for c in range(cols):
            half, height, thick = table[r, c]
            cx, cy = origins[r, c, :2]
            near = raised[(np.abs(raised[:, 0] - cx) < 3.9) & (np.abs(raised[:, 1] - cy) < 3.9)]
            d = np.abs(np.concatenate([near[:, 0] - cx, near[:, 1] - cy]))
            worst["inner"] = max(worst["inner"], abs(d.min() - half))
            worst["outer"] = max(worst["outer"], abs(d.max() - (half + thick)))
            worst["height"] = max(worst["height"], abs(near[:, 2].max() - height))
    cfg = terrains.MeshConcentricWallsTerrainCfg()
    diffs = (np.arange(rows) + 0.5) / rows
    front = np.array([terrains.frontier(d, cfg) for d in diffs])
    per_row = {
        "height_min": np.nanmin(table[..., 1], 1), "height_max": np.nanmax(table[..., 1], 1),
        "thick_min": np.nanmin(table[..., 2], 1), "thick_max": np.nanmax(table[..., 2], 1),
    }
    summary = {
        "rows": rows, "cols": cols, "stage_bounds": terrains.stage_bounds(cfg),
        "mesh_vs_table_max_abs_err_m": worst,
        "inner_half_size_range": [float(np.nanmin(table[..., 0])), float(np.nanmax(table[..., 0]))],
        "max_outer_face_m": float(np.nanmax(table[..., 0] + table[..., 2])),
        "frontier_step_per_row_m": {
            "height": float(np.max(np.abs(np.diff(front[:, 0])))),
            "thickness": float(np.max(np.abs(np.diff(front[:, 1])))),
        },
    }
    np.save(os.path.join(out, "wall_table.npy"), table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    r_idx = np.repeat(np.arange(rows), cols)
    for ax, k, name, fk in ((axes[0], 1, "height (m)", 0), (axes[1], 2, "thickness (m)", 1)):
        ax.scatter(r_idx + np.random.uniform(-0.25, 0.25, r_idx.size), table[..., k].ravel(), s=6, alpha=0.5,
                   label="tiles (20 per row)")
        ax.plot(np.arange(rows), front[:, fk], color="k", lw=1.5, label="frontier")
        for b in terrains.stage_bounds(cfg):
            ax.axvline(b * rows - 0.5, color="gray", ls="--", lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[0].set_title("Wall per tile vs. row (stage A | B | C)")
    axes[1].set_xlabel("row (terrain level)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curriculum_rows.png"), dpi=110)
    plt.close(fig)

    # Top view of a few rows, drawn from the table.
    show = [0, rows // 4, rows // 2, (3 * rows) // 4, rows - 1]
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.6 * len(show)))
    for ax, r in zip(axes, show):
        for c in range(min(cols, 8)):
            half, height, thick = table[r, c]
            x0 = c * 8.0
            ax.add_patch(Rectangle((x0 - 4, -4), 8, 8, fill=False, lw=0.5, color="gray"))
            outer = half + thick
            ax.add_patch(Rectangle((x0 - outer, -outer), 2 * outer, 2 * outer, color=plt.cm.viridis(height / 0.8)))
            ax.add_patch(Rectangle((x0 - half, -half), 2 * half, 2 * half, color="white"))
            ax.plot(x0, 0, "r+")
            ax.text(x0, -3.7, f"h{height:.2f} t{thick:.2f}", ha="center", fontsize=7)
        ax.set_xlim(-4, 8 * min(cols, 8) - 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(f"row {r}: first 8 tiles (colour = height, + = spawn)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "top_view.png"), dpi=100)
    plt.close(fig)
    return summary


def place_rows(inner, rows, cycle=False):
    """Move environment k onto row ``rows[k]`` (its own column) and reset those environments. With
    ``cycle``, every environment is placed, going round ``rows``."""
    terrain = inner.scene.terrain
    n = inner.num_envs if cycle else len(rows)
    ids = torch.arange(n, device=inner.device)
    terrain.terrain_levels[ids] = torch.tensor(rows, device=inner.device)[ids % len(rows)]
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    inner._reset_idx(ids)


class Tunnel:
    """Counts bodies going through a wall rather than over it."""

    def __init__(self, inner, table, body_ids, names):
        self.inner = inner
        self.robot = inner.scene["robot"]
        self.body_ids = body_ids
        self.names = names
        t = torch.as_tensor(table, device=inner.device, dtype=torch.float32)
        self.table = t
        self.prev = None
        n, b = inner.num_envs, len(body_ids)
        self.through = torch.zeros(n, b, dtype=torch.long, device=inner.device)
        self.inside = torch.zeros(n, b, dtype=torch.long, device=inner.device)
        self.over = torch.zeros(n, b, dtype=torch.long, device=inner.device)

    def wall(self):
        terrain = self.inner.scene.terrain
        w = self.table[terrain.terrain_levels, terrain.terrain_types]
        return w[:, 0:1], w[:, 1:2], w[:, 2:3]

    def local(self):
        pos = self.robot.data.body_pos_w[:, self.body_ids]
        rel = pos - self.inner.scene.env_origins[:, None, :]
        cheb = torch.maximum(rel[..., 0].abs(), rel[..., 1].abs())
        return cheb, pos[..., 2] - self.inner.scene.env_origins[:, None, 2]

    def update(self, reset):
        half, height, thick = self.wall()
        cheb, z = self.local()
        side = torch.where(cheb < half, 0, torch.where(cheb > half + thick, 2, 1))
        # Strictly below the top at both samples: a body that went over the wall must have been
        # above it at one of them unless it rose and fell 5+ cm within 20 ms.
        low = z < height
        self.inside += ((side == 1) & (z < height)).long()
        if self.prev is not None:
            p_side, p_low = self.prev
            crossed = (p_side == 0) & (side == 2) & low & p_low & ~reset[:, None]
            self.through += crossed.long()
            self.over += ((p_side == 0) & (side == 2) & ~(low & p_low) & ~reset[:, None]).long()
        self.prev = (side, low)


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-1.2, -2.2, 1.3)
    env_cfg.viewer.lookat = (0.6, 0.0, 0.3)
    if args_cli.mode == "throw":
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.collapsed = None
        env_cfg.events.push_robot = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped
    robot = inner.scene["robot"]

    summary = {"mode": args_cli.mode, "shape": check_shape(inner.scene.terrain, args_cli.out)}
    print(json.dumps(summary["shape"], indent=1), flush=True)

    names = ["base", ".*_hip", ".*_thigh", ".*_calf", ".*_foot"]
    body_ids, body_names = robot.find_bodies(names, preserve_order=True)
    tunnel = Tunnel(inner, terrains.WALL_TABLE, body_ids, body_names)
    dt = inner.step_dt
    steps = int(args_cli.duration_s / dt)

    import imageio

    if args_cli.mode == "walk":
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=inner.device)
        term = inner.command_manager.get_term("base_velocity")
        term.cfg.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        term.cfg.ranges.ang_vel_z = (0.0, 0.0)
        term._resample(torch.arange(inner.num_envs, device=inner.device))
        if args_cli.rows:
            place_rows(inner, args_cli.rows, cycle=True)
        place_rows(inner, args_cli.show_rows)
        obs = env.get_observations()
        writer = None
        if args_cli.video:
            writer = imageio.get_writer(os.path.join(args_cli.out, "walk_env0.mp4"), fps=round(1 / dt),
                                        codec="libx264", quality=8, macro_block_size=8)
        still_step = int(args_cli.still_s / dt)
        for step in range(steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            tunnel.update(dones.bool())
            if writer is not None:
                frame = inner.render()
                if frame is not None:
                    writer.append_data(frame[..., :3])
            if step == still_step:
                ctrl = inner.viewport_camera_controller
                for k, row in enumerate(args_cli.show_rows):
                    ctrl.set_view_env_index(k)
                    for _ in range(4):
                        inner.sim.render()
                    frame = inner.render()
                    if frame is not None:
                        _, h, t = (float(v) for v in terrains.WALL_TABLE[row, int(inner.scene.terrain.terrain_types[k])])
                        imageio.imwrite(os.path.join(args_cli.out, f"still_row{row:02d}_h{h:.2f}_t{t:.2f}.png"),
                                        frame[..., :3])
                ctrl.set_view_env_index(0)
        if writer is not None:
            writer.close()
    else:
        # Launch each robot at its own wall along +x, 0.5 m short of the inner face.
        n = inner.num_envs
        ids = torch.arange(n, device=inner.device)
        if args_cli.rows:
            place_rows(inner, args_cli.rows, cycle=True)
        n_rows = len(args_cli.rows) if args_cli.rows else 1
        speeds = torch.tensor(args_cli.speeds, device=inner.device)[(ids // n_rows) % len(args_cli.speeds)]
        half, height, thick = tunnel.wall()
        state = robot.data.default_root_state.clone()
        state[:, :3] = inner.scene.env_origins
        zeros = torch.zeros(n, device=inner.device)
        if args_cli.feet_first:
            # Nose up (pitch -90 deg): the legs point along +x, so the feet -- the smallest collision
            # shapes -- meet the wall first instead of the head.
            state[:, 0] += half[:, 0] - 0.6
            state[:, 2] += 0.55
            state[:, 3:7] = quat_from_euler_xyz(zeros, zeros - 0.5 * torch.pi, zeros)
        else:
            state[:, 0] += half[:, 0] - 0.5
            state[:, 2] += 0.35
            state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
        state[:, 7:] = 0.0
        state[:, 7] = speeds
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
        tunnel.update(torch.zeros(n, dtype=torch.bool, device=inner.device))
        zero = torch.zeros(n, env.num_actions, device=inner.device)
        steps = int(1.0 / dt)
        max_x = torch.full((n,), -1e9, device=inner.device)
        max_speed_after = torch.zeros(n, device=inner.device)
        for step in range(steps):
            if step == 15:
                # Positive control for the detector: env 0 is teleported to just past its wall, below
                # the top -- what a tunnelling body looks like. It must show up as "through".
                s0 = robot.data.root_state_w[:1].clone()
                s0[0, 0] = inner.scene.env_origins[0, 0] + half[0, 0] + thick[0, 0] + 0.4
                with torch.inference_mode():
                    robot.write_root_state_to_sim(s0, env_ids=torch.tensor([0], device=inner.device))
            with torch.inference_mode():
                env.step(zero)
            if step == 10:
                # 0.2 s in: a robot that hit the wall has lost its speed along x.
                max_speed_after = robot.data.root_lin_vel_w[:, 0].clone()
            tunnel.update(torch.zeros(n, dtype=torch.bool, device=inner.device))
            if args_cli.debug and step < 20:
                rx = robot.data.body_pos_w[:, body_ids, 0] - inner.scene.env_origins[:, None, 0] - half
                for e in args_cli.debug:
                    print(f"[dbg] step {step:2d} env {e} v0={speeds[e]:.0f} gap_base={rx[e, 0]:+.3f} "
                          f"gap_max={rx[e].max():+.3f} vx={robot.data.root_lin_vel_w[e, 0]:+.2f} "
                          f"z={robot.data.root_pos_w[e, 2] - inner.scene.env_origins[e, 2]:.2f}", flush=True)
            rel_x = robot.data.body_pos_w[:, body_ids, 0] - inner.scene.env_origins[:, None, 0]
            max_x = torch.maximum(max_x, rel_x.max(dim=1).values)
        thick_c = thick[:, 0]
        bins = [(0.0, 0.06), (0.06, 0.15), (0.15, 0.51)]
        table_rows = []
        for v in args_cli.speeds:
            for lo, hi in bins:
                m = (speeds == v) & (thick_c >= lo) & (thick_c < hi)
                if m.sum() == 0:
                    continue
                m[0] = False  # the positive control
                past = (max_x[m] > (half[m, 0] + thick_c[m] + 0.02)).float().mean().item()
                # Body origins stop short of the wall (the collision shapes extend past them), so
                # "hit the wall" is read from the speed: stopped or bounced back by 0.2 s.
                stopped = (max_speed_after[m] < 0.5 * v).float().mean().item()
                table_rows.append({
                    "speed": v, "thickness": f"{lo:.2f}-{hi:.2f}", "robots": int(m.sum()),
                    "any_body_through": int((tunnel.through[m].sum(1) > 0).sum()),
                    "any_body_over_top_in_one_step": int((tunnel.over[m].sum(1) > 0).sum()),
                    "body_origin_inside_wall": int((tunnel.inside[m].sum(1) > 0).sum()),
                    "reached_past_outer_face": round(past, 3),
                    "stopped_by_wall": round(stopped, 3),
                    "vx_at_0.2s_median": round(float(max_speed_after[m].median()), 2),
                })
        summary["throw"] = table_rows
        summary["positive_control_env0_through"] = int(tunnel.through[0].sum())

    per_body = tunnel.through.sum(0).cpu().tolist()
    inside = tunnel.inside.sum(0).cpu().tolist()
    thin = torch.as_tensor(terrains.WALL_TABLE[..., 2], device=inner.device)[
        inner.scene.terrain.terrain_levels, inner.scene.terrain.terrain_types] < 0.06
    summary["through_events_by_body"] = {n_: c for n_, c in zip(body_names, per_body) if c}
    summary["inside_steps_by_body"] = {n_: c for n_, c in zip(body_names, inside) if c}
    summary["envs_with_through"] = int((tunnel.through.sum(1) > 0).sum())
    summary["envs_on_thin_walls"] = int(thin.sum())
    summary["envs_with_through_on_thin"] = int(((tunnel.through.sum(1) > 0) & thin).sum())
    tag = args_cli.mode + ("_feet_first" if args_cli.feet_first else "")
    with open(os.path.join(args_cli.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "shape"}, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
