"""Standing-and-walking (SaW) benchmark for G1 velocity policies, in simulation.

Training rewards cannot compare policies trained with *different* reward functions,
so this scores them on task-level metrics instead, in the spirit of van Marum et al.,
"Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking":

  1. push      survival rate vs. force x duration x direction, while standing and walking
  2. track     velocity accuracy and drift for single-axis commands; energy per meter;
               touchdown impact; cadence and left/right gait asymmetry; action rate
  3. stop      foot activity while commanded to stand; walk -> stop distance and time

Every checkpoint runs in the *same* environment (``--env_task``), so the comparison is
fair even if the checkpoints were trained on different tasks. Only the actor is loaded
from each checkpoint (critic shapes may differ between tasks); the policy acts
deterministically (action mean). The env keeps its training-time observation noise,
startup domain randomization and action term (e.g. Phase1's delay / noise), but all
interval events (random pushes) and command curricula are disabled, and commands are
written per env, so each env runs one fixed trial condition.

Example
-------
    python scripts/rsl_rl/evaluate_saw.py --headless \\
        --policy phase1=logs/rsl_rl/g1_mlp_phase1/<run>/model_4997.pt \\
        --policy default=logs/rsl_rl/unitree_g1_29dof_velocity/<run>/model_4998.pt \\
        --json /tmp/saw.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SaW benchmark for G1 velocity policies.")
parser.add_argument(
    "--policy", action="append", required=True, help="label=path/to/model_*.pt (repeat to compare several)."
)
parser.add_argument("--env_task", type=str, default="G1-MLP-Phase1", help="Task whose env every policy runs in.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--json", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402

FEET = ".*_ankle_roll_link"
PUSH_BODY = "torso_link"
G = 9.81

# push grid
PUSH_STATES = [("stand", 0.0), ("walk", 0.5)]  # (label, forward command m/s)
PUSH_DIRS = [("+x", (1.0, 0.0)), ("-x", (-1.0, 0.0)), ("+y", (0.0, 1.0)), ("-y", (0.0, -1.0))]
PUSH_FORCES = [50.0, 100.0, 150.0, 200.0]  # N
PUSH_DURATIONS = [0.2, 0.5]  # s
PUSH_SETTLE_S, PUSH_RECOVER_S = 3.0, 3.0

# tracking patterns: (label, vx, vy, wz); up to the paper's command ranges
TRACK_PATTERNS = [
    ("fwd_0.5", 0.5, 0.0, 0.0),
    ("fwd_1.0", 1.0, 0.0, 0.0),
    ("fwd_1.5", 1.5, 0.0, 0.0),
    ("fwd_2.0", 2.0, 0.0, 0.0),
    ("back_0.5", -0.5, 0.0, 0.0),
    ("left_0.3", 0.0, 0.3, 0.0),
    ("right_0.3", 0.0, -0.3, 0.0),
    ("left_0.5", 0.0, 0.5, 0.0),
    ("right_0.5", 0.0, -0.5, 0.0),
    ("turnL_0.2", 0.0, 0.0, 0.2),
    ("turnR_0.2", 0.0, 0.0, -0.2),
    ("turnL_0.5", 0.0, 0.0, 0.5),
    ("turnR_0.5", 0.0, 0.0, -0.5),
    ("turnL_1.0", 0.0, 0.0, 1.0),
    ("turnR_1.0", 0.0, 0.0, -1.0),
    # turning while walking (ratio = yaw rate achieved / commanded)
    ("fwd0.5_turnL_0.1", 0.5, 0.0, 0.1),
    ("fwd0.5_turnL_0.3", 0.5, 0.0, 0.3),
    ("fwd0.5_turnR_0.3", 0.5, 0.0, -0.3),
    ("fwd0.5_turnL_0.5", 0.5, 0.0, 0.5),
    ("fwd0.5_turnL_1.0", 0.5, 0.0, 1.0),
]
TRACK_SETTLE_S, TRACK_MEASURE_S = 3.0, 8.0

# stop: half the envs stand the whole time, half walk then stop
STOP_WALK_V = 0.5
STOP_WALK_S, STOP_HOLD_S = 5.0, 5.0
STAND_SETTLE_S = 2.0


# ---------------------------------------------------------------------------
# policy loading (actor only)
# ---------------------------------------------------------------------------


def load_actor(path: str, device: str) -> nn.Module:
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    if any(k.startswith("actor_obs_normalizer") for k in sd):
        raise NotImplementedError("checkpoint uses actor obs normalization; not handled by this script")
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers: list[nn.Module] = []
    for i, li in enumerate(idx):
        w, b = sd[f"actor.{li}.weight"], sd[f"actor.{li}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(idx) - 1:
            layers.append(nn.ELU())  # BasePPORunnerCfg activation
    return nn.Sequential(*layers).to(device).eval()


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


def make_env():
    cfg = load_cfg_from_registry(args_cli.env_task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    # no random pushes: this script applies its own
    for name in list(vars(cfg.events)):
        term = getattr(cfg.events, name)
        if isinstance(term, EventTermCfg) and term.mode == "interval":
            setattr(cfg.events, name, None)
    # no curriculum widening ranges underneath, no resampling, nobody standing by chance
    for name in list(vars(cfg.curriculum)):
        if name.endswith("_cmd_levels"):
            setattr(cfg.curriculum, name, None)
    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.heading_command = False
    cmd.debug_vis = False
    # an env that falls and auto-resets gets a zero command (it is already counted as failed)
    cmd.ranges.lin_vel_x = cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0.0, 0.0)
    return gym.make(args_cli.env_task, cfg=cfg).unwrapped


class Bench:
    def __init__(self, env):
        self.env = env
        self.n = env.num_envs
        self.dev = env.device
        self.dt = env.step_dt
        self.robot = env.scene["robot"]
        self.contact = env.scene.sensors["contact_forces"]
        self.cmd = env.command_manager.get_term("base_velocity")
        self.feet_sensor_ids, names = self.contact.find_bodies(FEET)
        assert len(names) == 2 and names[0].startswith("left"), names
        self.feet_body_ids, _ = self.robot.find_bodies(FEET)
        self.push_body_ids, _ = self.robot.find_bodies(PUSH_BODY)
        self.weight = (self.robot.data.default_mass.sum(dim=1) * G).to(self.dev)  # (N,) nominal
        self.obs = None
        self.prev_action = None

    # -- control ----------------------------------------------------------
    def reset(self):
        self.obs, _ = self.env.reset()
        self.prev_action = torch.zeros(self.n, self.env.action_manager.total_action_dim, device=self.dev)

    def set_cmd(self, cmd: torch.Tensor):
        self.cmd.vel_command_b[:] = cmd
        if hasattr(self.cmd, "is_standing_env"):
            self.cmd.is_standing_env[:] = False
        self.obs = self.env.observation_manager.compute()  # so the next action already sees it

    def step(self, actor):
        action = actor(self.obs["policy"])
        self.action_rate = (action - self.prev_action).square().sum(dim=1)
        self.prev_action = action
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        return (terminated | truncated).bool()

    def push(self, force_w: torch.Tensor | None):
        forces = torch.zeros(self.n, 1, 3, device=self.dev)
        if force_w is not None:
            forces[:, 0, :2] = force_w
        self.robot.set_external_force_and_torque(
            forces, torch.zeros_like(forces), body_ids=self.push_body_ids, is_global=True
        )

    # -- per-step measurements -------------------------------------------
    def heading(self):
        return self.robot.data.heading_w

    def touchdowns(self):
        return self.contact.compute_first_contact(self.dt)[:, self.feet_sensor_ids]  # (N, 2)

    def feet_force(self):
        return self.contact.data.net_forces_w[:, self.feet_sensor_ids].norm(dim=-1)  # (N, 2)


def steps(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_push(b: Bench, actor):
    conds = [
        (s, sv, d, dv, f, t)
        for (s, sv) in PUSH_STATES
        for (d, dv) in PUSH_DIRS
        for f in PUSH_FORCES
        for t in PUSH_DURATIONS
    ]
    cid = torch.arange(b.n, device=b.dev) % len(conds)
    vx = torch.tensor([c[1] for c in conds], device=b.dev)[cid]
    dir_b = torch.tensor([c[3] for c in conds], device=b.dev)[cid]  # (N, 2) in heading frame
    mag = torch.tensor([c[4] for c in conds], device=b.dev)[cid]
    dur = torch.tensor([c[5] for c in conds], device=b.dev)[cid]

    b.reset()
    cmd = torch.zeros(b.n, 3, device=b.dev)
    cmd[:, 0] = vx
    b.set_cmd(cmd)
    alive = torch.ones(b.n, dtype=torch.bool, device=b.dev)
    for _ in range(steps(PUSH_SETTLE_S, b.dt)):
        alive &= ~b.step(actor)
    settled = alive.clone()

    # push direction fixed in world at push onset, from the robot's heading then
    h = b.heading()
    c, s = torch.cos(h), torch.sin(h)
    dir_w = torch.stack([c * dir_b[:, 0] - s * dir_b[:, 1], s * dir_b[:, 0] + c * dir_b[:, 1]], dim=1)
    force_w = dir_w * mag.unsqueeze(1)
    total = steps(max(PUSH_DURATIONS) + PUSH_RECOVER_S, b.dt)
    for k in range(total):
        active = (k * b.dt) < dur
        b.push(force_w * active.unsqueeze(1))
        alive &= ~b.step(actor)
    b.push(None)

    rows = []
    for i, (st, _, d, _, f, t) in enumerate(conds):
        m = (cid == i) & settled
        rows.append(dict(state=st, dir=d, force=f, duration=t, n=int(m.sum()),
                         success=float(alive[m].float().mean()) if m.any() else float("nan")))
    return dict(rows=rows, settle_fall=float(1 - settled.float().mean()))


def test_track(b: Bench, actor):
    pid = torch.arange(b.n, device=b.dev) % len(TRACK_PATTERNS)
    pat = torch.tensor([p[1:] for p in TRACK_PATTERNS], device=b.dev)[pid]  # (N, 3)

    b.reset()
    b.set_cmd(pat)
    alive = torch.ones(b.n, dtype=torch.bool, device=b.dev)
    for _ in range(steps(TRACK_SETTLE_S, b.dt)):
        alive &= ~b.step(actor)

    p0 = b.robot.data.root_pos_w[:, :2].clone()
    h0 = b.heading().clone()
    h_prev = h0.clone()
    yaw_int = torch.zeros(b.n, device=b.dev)
    v_sum = torch.zeros(b.n, 3, device=b.dev)
    work = torch.zeros(b.n, device=b.dev)
    arate = torch.zeros(b.n, device=b.dev)
    td_force = torch.zeros(b.n, device=b.dev)
    td_count = torch.zeros(b.n, 2, device=b.dev)
    air_sum = torch.zeros(b.n, 2, device=b.dev)
    T = steps(TRACK_MEASURE_S, b.dt)
    for _ in range(T):
        alive &= ~b.step(actor)
        rd = b.robot.data
        v_sum += torch.cat([rd.root_lin_vel_b[:, :2], rd.root_ang_vel_b[:, 2:3]], dim=1)
        h = b.heading()
        yaw_int += wrap(h - h_prev)
        h_prev = h
        work += (rd.applied_torque * rd.joint_vel).clamp(min=0.0).sum(dim=1) * b.dt
        arate += b.action_rate
        td = b.touchdowns().float()
        td_count += td
        td_force += (b.feet_force() * td).sum(dim=1) / b.weight
        air_sum += b.contact.data.last_air_time[:, b.feet_sensor_ids] * td

    disp = b.robot.data.root_pos_w[:, :2] - p0
    c, s = torch.cos(h0), torch.sin(h0)
    d_fwd = c * disp[:, 0] + s * disp[:, 1]  # displacement in the initial heading frame
    d_lat = -s * disp[:, 0] + c * disp[:, 1]
    v_mean = v_sum / T

    rows = []
    for i, (label, vx, vy, wz) in enumerate(TRACK_PATTERNS):
        m = (pid == i) & alive
        if not m.any():
            rows.append(dict(pattern=label, n=0))
            continue
        mean = lambda x: float(x[m].mean())  # noqa: E731
        if wz != 0.0:
            commanded, achieved = wz, mean(v_mean[:, 2])
            drift = mean(disp.norm(dim=1))  # turning in place: any translation is drift
            yaw_err = mean((yaw_int - wz * TRACK_MEASURE_S).abs())
        else:
            ax = 0 if vx != 0.0 else 1
            commanded, achieved = (vx if ax == 0 else vy), mean(v_mean[:, ax])
            drift = mean((d_lat if ax == 0 else d_fwd).abs())  # perpendicular to the command
            yaw_err = mean(yaw_int.abs())
        dist = disp.norm(dim=1).clamp(min=1e-3)
        tds = td_count.sum(dim=1).clamp(min=1)
        air_l = air_sum[:, 0] / td_count[:, 0].clamp(min=1)
        air_r = air_sum[:, 1] / td_count[:, 1].clamp(min=1)
        rows.append(dict(
            pattern=label, n=int(m.sum()), alive=float(alive[pid == i].float().mean()),
            commanded=commanded, achieved=achieved, ratio=achieved / commanded,
            drift_m=drift, yaw_err_rad=yaw_err,
            energy_J_per_m=mean(work / dist) if wz == 0.0 else float("nan"),
            power_W=mean(work / TRACK_MEASURE_S),
            touchdown_force_bw=mean(td_force / tds),
            cadence_steps_per_s=mean(td_count.sum(dim=1) / TRACK_MEASURE_S),
            air_time_s=mean((air_l + air_r) / 2),
            air_asym=mean((air_l - air_r).abs() / ((air_l + air_r) / 2).clamp(min=1e-3)),
            action_rate=mean(arate / T),
        ))
    return dict(rows=rows)


def test_stop(b: Bench, actor):
    walker = (torch.arange(b.n, device=b.dev) % 2) == 1
    b.reset()
    cmd = torch.zeros(b.n, 3, device=b.dev)
    cmd[walker, 0] = STOP_WALK_V
    b.set_cmd(cmd)
    alive = torch.ones(b.n, dtype=torch.bool, device=b.dev)

    # phase A: walkers walk, standers settle
    for _ in range(steps(STOP_WALK_S, b.dt)):
        alive &= ~b.step(actor)

    # phase B: everyone commanded to stand
    b.set_cmd(torch.zeros(b.n, 3, device=b.dev))
    p_stop = b.robot.data.root_pos_w[:, :2].clone()
    stop_time = torch.full((b.n,), float("nan"), device=b.dev)
    stop_dist = torch.full((b.n,), float("nan"), device=b.dev)
    foot_travel = torch.zeros(b.n, device=b.dev)
    td_late = torch.zeros(b.n, device=b.dev)
    feet_prev = b.robot.data.body_pos_w[:, b.feet_body_ids, :2].clone()
    T = steps(STOP_HOLD_S, b.dt)
    late = steps(STOP_HOLD_S - STAND_SETTLE_S, b.dt)  # last part: should be fully still
    for k in range(T):
        alive &= ~b.step(actor)
        rd = b.robot.data
        speed = rd.root_lin_vel_b[:, :2].norm(dim=1)
        newly = torch.isnan(stop_time) & (speed < 0.1)
        stop_time[newly] = (k + 1) * b.dt
        stop_dist[newly] = (rd.root_pos_w[newly, :2] - p_stop[newly]).norm(dim=1)
        feet = rd.body_pos_w[:, b.feet_body_ids, :2]
        if k >= T - late:
            foot_travel += (feet - feet_prev).norm(dim=-1).sum(dim=1)
            td_late += b.touchdowns().float().sum(dim=1)
        feet_prev = feet.clone()

    late_s = late * b.dt
    out = {}
    for label, m in [("stand", ~walker & alive), ("walk_to_stop", walker & alive)]:
        grp = ~walker if label == "stand" else walker
        stopped = m & ~torch.isnan(stop_time)
        out[label] = dict(
            n=int(m.sum()), alive=float(alive[grp].float().mean()),
            foot_travel_m_per_s=float((foot_travel[m] / late_s).mean()) if m.any() else float("nan"),
            touchdowns_per_s=float((td_late[m] / late_s).mean()) if m.any() else float("nan"),
            stopped_frac=float(stopped.sum() / m.sum().clamp(min=1)),
            stop_time_s=float(stop_time[stopped].mean()) if stopped.any() else float("nan"),
            stop_dist_m=float(stop_dist[stopped].mean()) if stopped.any() else float("nan"),
        )
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report(results: dict):
    labels = list(results)
    print("\n" + "=" * 100)
    print(f"SaW benchmark | env {args_cli.env_task} | {args_cli.num_envs} envs | seed {args_cli.seed}")
    print("=" * 100)

    print("\n[1] push survival rate (%), per state / direction; columns = force N @ duration s")
    cols = [(f, t) for t in PUSH_DURATIONS for f in PUSH_FORCES]
    head = "".join(f"{int(f)}@{t}".rjust(9) for f, t in cols)
    for st, _ in PUSH_STATES:
        print(f"\n  {st}")
        print(f"  {'policy':<10}{'dir':<5}{head}{'mean':>8}")
        for lab in labels:
            rows = results[lab]["push"]["rows"]
            for d, _ in PUSH_DIRS:
                vals = [next(r["success"] for r in rows if r["state"] == st and r["dir"] == d
                             and r["force"] == f and r["duration"] == t) for f, t in cols]
                line = "".join(f"{100 * v:9.0f}" for v in vals)
                print(f"  {lab:<10}{d:<5}{line}{100 * sum(vals) / len(vals):8.0f}")
    for lab in labels:
        allv = [r["success"] for r in results[lab]["push"]["rows"]]
        print(f"  overall {lab}: {100 * sum(allv) / len(allv):.1f}%  (fell before push: "
              f"{100 * results[lab]['push']['settle_fall']:.1f}%)")

    print("\n[2] tracking (measured over the last "
          f"{TRACK_MEASURE_S:.0f}s; m/s, rad/s for turns)")
    keys = [("alive", "alive%", 100, ".0f"), ("ratio", "ratio", 1, ".2f"), ("drift_m", "drift", 1, ".2f"),
            ("yaw_err_rad", "yawErr", 1, ".2f"), ("energy_J_per_m", "J/m", 1, ".0f"),
            ("power_W", "W", 1, ".0f"), ("touchdown_force_bw", "tdF/BW", 1, ".2f"),
            ("cadence_steps_per_s", "step/s", 1, ".2f"), ("air_time_s", "air_s", 1, ".2f"),
            ("air_asym", "airAsym", 1, ".2f"), ("action_rate", "actRate", 1, ".2f")]
    print(f"  {'pattern':<18}{'policy':<10}" + "".join(k[1].rjust(9) for k in keys))
    for i, p in enumerate(TRACK_PATTERNS):
        for lab in labels:
            r = results[lab]["track"]["rows"][i]
            if r.get("n", 0) == 0:
                print(f"  {p[0]:<18}{lab:<10}  (all fell)")
                continue
            print(f"  {p[0]:<18}{lab:<10}" + "".join(format(r[k] * s, f).rjust(9) for k, _, s, f in keys))
    print("  ratio=achieved/commanded; drift=displacement perpendicular to the command (turns: any"
          " translation); yawErr=|heading change - commanded|;\n  tdF/BW=mean foot force at touchdown"
          " / body weight; airAsym=|L-R|/mean air time; actRate=mean |a_t-a_{t-1}|^2")

    print(f"\n[3] stand / stop (stand cmd for {STOP_HOLD_S:.0f}s; activity over its last "
          f"{STOP_HOLD_S - STAND_SETTLE_S:.0f}s)")
    skeys = [("alive", "alive%", 100, ".0f"), ("foot_travel_m_per_s", "footTrav", 1, ".3f"),
             ("touchdowns_per_s", "td/s", 1, ".2f"), ("stopped_frac", "stopped%", 100, ".0f"),
             ("stop_time_s", "stop_s", 1, ".2f"), ("stop_dist_m", "stop_m", 1, ".2f")]
    print(f"  {'case':<14}{'policy':<10}" + "".join(k[1].rjust(10) for k in skeys))
    for case in ["stand", "walk_to_stop"]:
        for lab in labels:
            r = results[lab]["stop"][case]
            print(f"  {case:<14}{lab:<10}" + "".join(format(r[k] * s, f).rjust(10) for k, _, s, f in skeys))
    print(f"  footTrav=horizontal foot motion (m/s); td/s=touchdowns per s; stop_*: from the stand"
          f" command until base speed < 0.1 m/s (walkers start at {STOP_WALK_V} m/s)")


def main():
    env = make_env()
    b = Bench(env)
    results = {}
    with torch.inference_mode():
        for spec in args_cli.policy:
            label, path = spec.split("=", 1)
            actor = load_actor(path, env.device)
            print(f"[eval] {label}: {path}")
            results[label] = dict(path=path, push=test_push(b, actor), track=test_track(b, actor),
                                  stop=test_stop(b, actor))
    report(results)
    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(dict(env_task=args_cli.env_task, num_envs=args_cli.num_envs, seed=args_cli.seed,
                           results=results), f, indent=2)
        print(f"\n[wrote] {args_cli.json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
