#!/usr/bin/env python3
"""Plot a 1 kHz speed capture written by State_RLBase (speed_<n>_YYYYMMDD-HHMMSS.csv).

Answers the two questions the deploy-side capture was added for, over the window from the moment a
forward command is latched until the robot stops:

  1. HOW FAST DID IT ACTUALLY GO? ``v_fwd`` is MuJoCo ground truth (framelinvel on the imu site,
     published on SportModeState and rotated into the body frame by the controller), so it is
     directly comparable to Isaac Lab's ``root_lin_vel_b[0]`` and to measure_run_speed.py's
     achieved-speed column. On hardware the column stays 0 -- nothing publishes it.

  2. IS EACH LEG RUNNING OUT OF TORQUE? Utilisation is computed against the SPEED-DEPENDENT
     envelope, not the flat clamp: Isaac Lab's actuator model holds full torque to X1 = 13.5 rad/s
     and then falls linearly to zero at X2 = 30, and at 5 m/s roughly a quarter of thigh/calf steps
     sit inside that region -- so a joint can be at 100% of what the motor can currently deliver
     while nowhere near the 23.7 / 45.43 N*m nameplate. Both are drawn.

     The capture records the COMMANDED torque as well as the applied one, which is what separates
     "just barely short" from "nowhere near enough": a joint pinned at its limit looks the same
     whether the controller wanted 24 N*m or 200. ``tau_cmd - tau_app`` is what the clamp threw
     away, and if it is ~0 the leg is not asking for more than it gets -- more torque would buy
     nothing.

Usage:
    python scripts/plot_speed_capture.py speed_0_20260820-125530.csv [-o out.png]
    python scripts/plot_speed_capture.py speed_0_20260820-125530.csv --legs FL RL

The output PNG defaults to the CSV's name with a .png suffix, so captures and plots stay paired.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Flat clamps from the MuJoCo model (go2.xml: abduction/hip +/-23.7, knee +/-45.43), which the
# robot's firmware and MuJoCo both enforce.
TORQUE_LIMIT = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}

# Isaac Lab's corrected envelope (UNITREE_GO2_CORRECTED_CFG): full torque below X1, falling linearly
# to zero at X2. unitree_mujoco mirrors these in simulate/config.yaml's torque_speed_curves.
#
# Y1 vs Y2 IS NOT A DETAIL. The peak is direction-dependent: Y1 applies when the torque pushes the
# joint the way it is already moving (driving), Y2 -- the larger figure -- when it opposes the motion
# (braking), because a motor can absorb more than it can deliver. Both UnitreeActuator._clip_effort
# and unitree_sdk2_bridge.h::_clip_effort select on sign(joint_vel * effort), and the derate slope
# starts from whichever peak applies. Grading everything against Y1 reports a braking thigh pinned at
# its real 23.4 N*m limit as "116% utilised", which is how this script first read the MuJoCo capture.
ISAAC_ENVELOPE = {
    "hip": dict(y1=20.2, y2=23.4, x1=13.5, x2=30.0),
    "thigh": dict(y1=20.2, y2=23.4, x1=13.5, x2=30.0),
    "calf": dict(y1=39.22, y2=45.43, x1=13.5, x2=30.0),
}

LEGS = ["FL", "FR", "RL", "RR"]
KINDS = ["hip", "thigh", "calf"]
SATURATED = 0.95  # fraction of the instantaneous limit that counts as "used up"


def joint_kind(name: str) -> str:
    for kind in KINDS:
        if name.endswith(kind):
            return kind
    raise ValueError(f"cannot classify joint {name!r}")


def isaac_limit(kind: str, speed: float, torque: float) -> float:
    """The instantaneous torque ceiling, mirroring ``UnitreeActuator._clip_effort``."""
    e = ISAAC_ENVELOPE[kind]
    peak = e["y1"] if speed * torque > 0.0 else e["y2"]
    if abs(speed) <= e["x1"]:
        return peak
    return max(0.0, peak * (e["x2"] - abs(speed)) / (e["x2"] - e["x1"]))


def load(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    joints = [k[: -len("_tau_app")] for k in cols if k.endswith("_tau_app")]
    return cols, joints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--legs", nargs="*", default=LEGS, help="subset of legs to plot (default: all)")
    args = ap.parse_args()

    cols, joints = load(args.csv)
    t = cols["t"]
    legs = [leg for leg in LEGS if leg in args.legs]

    # ---- per-joint utilisation against the instantaneous envelope -------------------------------
    util = {}
    thrown_away = {}
    asked_over = {}
    for j in joints:
        kind = joint_kind(j)
        app = cols[f"{j}_tau_app"]
        cmd = cols[f"{j}_tau_cmd"]
        dq = cols[f"{j}_dq"]
        # The ceiling is evaluated against the COMMANDED torque's direction, which is what the clip
        # sees before it truncates; using the applied sign would beg the question after saturation.
        limits = [isaac_limit(kind, v, c) for v, c in zip(dq, cmd)]
        util[j] = [abs(a) / lim if lim > 1e-3 else float("inf") for a, lim in zip(app, limits)]
        thrown_away[j] = [abs(c) - abs(a) for c, a in zip(cmd, app)]
        # How often the controller asked for more than the motor could give at that instant. This,
        # not the peak of thrown_away, is what says whether a joint is torque-starved for real or
        # only in a single transient spike.
        asked_over[j] = [1.0 if abs(c) > lim + 1e-3 else 0.0 for c, lim in zip(cmd, limits)]

    # ---- text summary --------------------------------------------------------------------------
    run = [i for i, v in enumerate(cols["cmd_vx"]) if v > 0.1] or list(range(len(t)))
    v_peak = max(cols["v_fwd"])
    i_peak = cols["v_fwd"].index(v_peak)
    print(f"{args.csv.name}: {len(t)} samples, {t[0]:.2f}s .. {t[-1]:.2f}s")
    print(f"  peak forward speed   {v_peak:.2f} m/s at t = {t[i_peak]:+.2f}s "
          f"(commanded {cols['cmd_vx'][i_peak]:.2f})")
    print(f"  peak commanded       {max(cols['cmd_vx']):.2f} m/s")
    print(f"  body height          {min(cols['base_z']):.3f} .. {max(cols['base_z']):.3f} m")
    print()
    print(f"  {'joint':<10}{'peak |tau|':>11}{'peak util':>10}{'mean util':>10}"
          f"{'>=95% lim':>10}{'asked>lim':>10}{'max short':>11}{'mean short':>11}")
    for leg in legs:
        for kind in KINDS:
            j = f"{leg}_{kind}"
            if j not in util:
                continue
            u = [util[j][i] for i in run]
            sat = sum(1 for x in u if x >= SATURATED) / max(1, len(u))
            over = sum(asked_over[j][i] for i in run) / max(1, len(run))
            peak_tau = max(abs(cols[f"{j}_tau_app"][i]) for i in run)
            short = [thrown_away[j][i] for i in run]
            mean_short = sum(x for x in short if x > 0) / max(1, sum(1 for x in short if x > 0))
            print(f"  {j:<10}{peak_tau:>9.1f} Nm{max(u) * 100:>9.0f}%{sum(u) / len(u) * 100:>9.0f}%"
                  f"{sat * 100:>9.1f}%{over * 100:>9.1f}%{max(short):>8.1f} Nm{mean_short:>8.1f} Nm")
    print()
    print("  peak/mean util are against the INSTANTANEOUS ceiling (direction- and speed-dependent),")
    print("  so 100% means the motor had nothing left to give at that moment, not that it hit its")
    print("  nameplate. 'asked>lim' is the share of the run where the controller requested more than")
    print("  the ceiling; 'short' is |tau_cmd| - |tau_app|, i.e. what the clamp threw away. A joint")
    print("  with asked>lim ~0 would gain nothing from a bigger motor.")

    # ---- figure --------------------------------------------------------------------------------
    fig, axes = plt.subplots(2 + len(legs), 1, figsize=(11, 3 + 2.2 * (2 + len(legs))), sharex=True)

    ax = axes[0]
    ax.plot(t, cols["cmd_vx"], label="commanded $v_x$", color="0.55", lw=1.2, ls="--")
    ax.plot(t, cols["v_fwd"], label="achieved $v_x$ (body frame)", color="tab:blue", lw=1.4)
    ax.plot(t[i_peak], v_peak, "o", color="tab:red", ms=5)
    ax.annotate(f"{v_peak:.2f} m/s", (t[i_peak], v_peak), textcoords="offset points",
                xytext=(6, 4), color="tab:red", fontsize=9)
    ax.set_ylabel("m/s")
    ax.set_title(f"{args.csv.name} -- forward speed, command latched to stop")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, cols["base_z"], color="tab:green", lw=1.2, label="base height")
    ax.set_ylabel("m")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    for ax, leg in zip(axes[2:], legs):
        for kind, colour in zip(KINDS, ("tab:blue", "tab:orange", "tab:red")):
            j = f"{leg}_{kind}"
            if j not in util:
                continue
            ax.plot(t, [min(u, 2.0) * 100 for u in util[j]], lw=1.0, color=colour, label=kind)
        ax.axhline(100, color="k", lw=0.8, ls=":")
        ax.axhline(SATURATED * 100, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(f"{leg}\n% of limit")
        ax.set_ylim(0, 205)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time from command (s)")
    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
