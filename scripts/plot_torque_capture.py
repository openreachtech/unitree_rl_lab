#!/usr/bin/env python3
"""Plot a 1 kHz torque capture written by State_Flip (torque_<motion>_<n>.csv).

Answers two questions the deploy-side capture was added for:

  1. Does the motor saturate during the jump, and for how long?
  2. If it does, by how much -- i.e. would more torque actually buy more height?

(2) is why the capture records the commanded torque as well as the applied one. A
joint pinned flat at its limit looks identical whether the controller wanted 46 N*m
or 200 N*m; only tau_cmd distinguishes "just barely short" from "nowhere near enough".

Usage:
    python scripts/plot_torque_capture.py torque_jump_0.csv [-o out.png]
    python scripts/plot_torque_capture.py torque_jump_0.csv --joints FL_calf RL_calf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Torque clamps, from the MuJoCo model (go2.xml: abduction/hip -23.7..23.7, knee
# -45.43..45.43) and matching Go2's published joint spec. These are the flat limits the
# robot's firmware and MuJoCo both enforce. Isaac Lab additionally derates torque with
# joint speed, so a trace that stays under these lines can still have been saturated in
# training -- see the speed-derate note printed by --check-derate.
TORQUE_LIMIT = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}

# Isaac Lab's corrected-knee envelope (jump_max_height.py): full torque below X1, falling
# linearly to zero at X2.
ISAAC_ENVELOPE = {
    "hip": dict(y1=20.2, x1=13.5, x2=30.0),
    "thigh": dict(y1=20.2, x1=13.5, x2=30.0),
    "calf": dict(y1=39.22, x1=13.5, x2=30.0),
}


def joint_kind(name: str) -> str:
    for kind in ("hip", "thigh", "calf"):
        if name.endswith(kind):
            return kind
    raise ValueError(f"cannot classify joint {name!r}")


def load(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    joints = [k[: -len("_tau_app")] for k in cols if k.endswith("_tau_app")]
    return cols, joints


def isaac_limit(kind: str, speed: float) -> float:
    e = ISAAC_ENVELOPE[kind]
    if abs(speed) <= e["x1"]:
        return e["y1"]
    frac = (e["x2"] - abs(speed)) / (e["x2"] - e["x1"])
    return max(0.0, e["y1"] * frac)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--joints", nargs="*", default=None, help="subset to plot (default: all 12)")
    args = ap.parse_args()

    cols, joints = load(args.csv)
    if args.joints:
        joints = [j for j in joints if j in args.joints]
    t = cols["t"]

    by_kind = {k: [j for j in joints if joint_kind(j) == k] for k in ("hip", "thigh", "calf")}
    by_kind = {k: v for k, v in by_kind.items() if v}

    # Captures made before the attitude columns existed still plot; they just lose the
    # last panel.
    has_attitude = "roll_turns" in cols
    n_panels = len(by_kind) + 1 + int(has_attitude)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    util_ax = axes[len(by_kind)]

    # --- one panel per joint kind: applied torque solid, commanded dashed -------------
    for ax, (kind, names) in zip(axes, by_kind.items()):
        limit = TORQUE_LIMIT[kind]
        for name in names:
            (line,) = ax.plot(t, cols[f"{name}_tau_app"], lw=1.2, label=f"{name} applied")
            ax.plot(t, cols[f"{name}_tau_cmd"], lw=0.8, ls="--", alpha=0.55, color=line.get_color())
        ax.axhline(limit, color="crimson", ls=":", lw=1.2)
        ax.axhline(-limit, color="crimson", ls=":", lw=1.2)
        ax.text(t[0], limit, f" clamp {limit:g} N·m", color="crimson", va="bottom", fontsize=8)
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_ylabel(f"{kind}\ntorque [N·m]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=4, loc="upper right")

    # --- utilisation: |applied| / clamp, so 1.0 means pinned -------------------------
    ax = util_ax
    for name in joints:
        limit = TORQUE_LIMIT[joint_kind(name)]
        ax.plot(t, [abs(v) / limit for v in cols[f"{name}_tau_app"]], lw=1.0, label=name)
    ax.axhline(1.0, color="crimson", ls=":", lw=1.2)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel("utilisation\n|tau| / clamp")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=6, loc="upper right")
    axes[-1].set_xlabel("time from trigger [s]")

    # --- height, when the capture carries it (simulation only) -----------------------
    if "height_delta" in cols and max(cols["height_delta"]) > 0.005:
        h = cols["height_delta"]
        peak = max(h)
        peak_t = t[h.index(peak)]
        ax2 = util_ax.twinx()
        ax2.plot(t, h, color="0.25", lw=1.6, ls="-", alpha=0.8)
        ax2.set_ylabel("height above standing [m]")
        ax2.annotate(f"peak {peak:.3f} m @ {peak_t:.2f}s", xy=(peak_t, peak),
                     xytext=(peak_t + 0.12, peak), color="0.25", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="0.25", lw=0.8))

    # --- attitude, when the capture carries it ----------------------------------------
    # A flip that fails and one that succeeds have nearly the same torque trace; what
    # separates them is where the rotation stopped. roll_turns is the same quantity the
    # training side calls accumulated_roll, so it can be read against the task's target
    # directly, and grav_z says which way up the robot finished: -1 upright, 0 on its
    # side, +1 inverted.
    if has_attitude:
        att = axes[-1]
        for key, label, colour in (
            ("roll_turns", "roll (sideflip axis)", "tab:blue"),
            ("pitch_turns", "pitch (backflip axis)", "tab:orange"),
        ):
            if any(abs(v) > 1e-3 for v in cols[key]):
                att.plot(t, cols[key], lw=1.6, color=colour, label=label)
        final_roll = cols["roll_turns"][-1]
        for turn in sorted({round(final_roll)} | {-1.0, -2.0, 1.0, 2.0}):
            if abs(turn) <= max(2.5, abs(final_roll) + 0.5) and turn != 0:
                att.axhline(turn, color="0.75", ls=":", lw=0.9)
        att.axhline(0.0, color="0.4", lw=0.8)
        att.axvline(0.0, color="0.4", lw=0.8)
        att.set_ylabel("rotation [turns]")
        att.set_xlabel("time from trigger [s]")
        att.grid(alpha=0.25)
        att.legend(fontsize=8, loc="upper left")
        gz = att.twinx()
        gz.plot(t, cols["grav_z"], color="0.35", lw=1.2, alpha=0.8, label="grav_z")
        gz.axhline(-0.9, color="seagreen", ls="--", lw=1.0)
        gz.text(t[0], -0.9, " upright gate -0.9", color="seagreen", va="top", fontsize=8)
        gz.set_ylabel("grav_z  (-1 upright, +1 inverted)")
        gz.set_ylim(-1.15, 1.15)

    fig.suptitle(f"{args.csv.name} - dashed = commanded (pre-clamp), solid = applied", fontsize=11)
    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # --- numeric summary --------------------------------------------------------------
    print(f"\n{'joint':<10} {'peak|app|':>10} {'clamp':>8} {'use%':>6} {'sat ms':>7} "
          f"{'peak|cmd|':>10} {'over%':>7} {'peak|dq|':>9} {'isaac sat ms':>13}")
    dt = (t[-1] - t[0]) / max(1, len(t) - 1)
    for name in joints:
        kind = joint_kind(name)
        limit = TORQUE_LIMIT[kind]
        app = [abs(v) for v in cols[f"{name}_tau_app"]]
        cmd = [abs(v) for v in cols[f"{name}_tau_cmd"]]
        dq = [abs(v) for v in cols[f"{name}_dq"]]
        sat_ms = sum(1 for v in cmd if v >= limit) * dt * 1000.0
        # Same samples judged against Isaac Lab's speed-dependent envelope instead.
        isaac_ms = sum(1 for a, w in zip(app, dq) if a >= isaac_limit(kind, w) - 1e-6) * dt * 1000.0
        print(f"{name:<10} {max(app):>10.2f} {limit:>8.2f} {100*max(app)/limit:>5.0f}% "
              f"{sat_ms:>7.0f} {max(cmd):>10.2f} {100*max(cmd)/limit:>6.0f}% "
              f"{max(dq):>9.2f} {isaac_ms:>13.0f}")

    if "height_delta" in cols:
        peak = max(cols["height_delta"])
        if peak > 0.005:
            print(f"\npeak height above standing: {peak:.3f} m"
                  "   <- compare directly against Isaac Lab's Metrics/jump/max_height")
        else:
            print("\npeak height above standing: not recorded (nothing published body position;"
                  " expected on hardware, unexpected under unitree_mujoco)")

    if "roll_turns" in cols:
        roll, gz, tilt = cols["roll_turns"], cols["grav_z"], cols["tilt_deg"]
        peak_roll = max(roll, key=abs)
        print(f"\nrotation: peak {peak_roll:+.2f} turns, final {roll[-1]:+.2f} turns"
              f"   (roll rate peaked at {max(cols['wx'], key=abs):+.1f} rad/s)")
        print(f"attitude at rest: grav_z {gz[-1]:+.2f}, tilt {tilt[-1]:.0f} deg"
              f"   -- upright is grav_z <= -0.9")
        if gz[-1] > -0.9:
            short = abs(abs(peak_roll) - round(abs(peak_roll)))
            print(f"  -> did NOT finish upright. Rotation ended {short:.2f} turns off a whole"
                  " turn; falling short by even 0.1 lands the robot on its flank.")

    print("\n'over%' is peak COMMANDED torque as a share of the clamp. Near 100% means the")
    print("controller was only just short and more torque would help; far above means the")
    print("PD was demanding torque no motor of this class could deliver, so the limit is")
    print("elsewhere. 'isaac sat ms' judges the same samples against Isaac Lab's")
    print("speed-derated envelope -- if it is large while 'sat ms' is 0, the jump is")
    print("speed-limited, not torque-limited, and gearing rather than torque is the lever.")


if __name__ == "__main__":
    main()
