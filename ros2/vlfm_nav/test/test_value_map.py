"""Checks for the parts of the value map that fail silently.

Pure numpy, no ROS and no model, so it runs anywhere::

    python ros2/vlfm_nav/test/test_value_map.py

Every case here exists because the failure it guards against is invisible in a running
system. A mirrored fan still paints, still publishes, still renders -- the robot just
walks away from what it is looking for. Paint that leaks through a wall looks like a
perfectly good value map until the frontier choice is wrong. A degenerate score spread
stretched onto [0, 1] produces a confident-looking picture of rounding error. None of
these throw, log, or show up as a rate, which is why they are pinned down here.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlfm_nav.decision import FrontierSelector  # noqa: E402
from vlfm_nav.value_map import GridValueMap, ValueMap  # noqa: E402

OCC_RES = 0.05
_fails: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "   " + extra))
    if not cond:
        _fails.append(name)


def blank(n: int = 400):
    """20 x 20 m of known-free space, origin at (-10, -10)."""
    return np.zeros((n, n), dtype=np.int16), (-10.0, -10.0), OCC_RES


def wall_at(x_m: float, n: int = 400):
    occ, org, res = blank(n)
    c = int((x_m - org[0]) / res)
    occ[:, c:c + 2] = 100
    return occ, org, res


def vmap() -> ValueMap:
    return ValueMap(size_m=20.0, resolution=0.2, origin=(-10.0, -10.0))


# --------------------------------------------------------------------- handedness
# Image x grows rightward, bearing grows counter-clockwise (REP-103), so the LEFTMOST
# strip is the LARGEST bearing. Get this backwards and the whole map mirrors about the
# optical axis -- silently.
occ, org, res = blank()
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 0.0, 0.0, 0.0], occ, org, res)
check("leftmost strip paints +y (image left = CCW)",
      vm.value[vm._cy > 0.5].sum() > 10 * max(vm.value[vm._cy < -0.5].sum(), 1e-9))

vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [0.0, 0.0, 0.0, 1.0], occ, org, res)
check("rightmost strip paints -y",
      vm.value[vm._cy < -0.5].sum() > 10 * max(vm.value[vm._cy > 0.5].sum(), 1e-9))

# ---------------------------------------------------------------------- occlusion
vm = vmap()
wocc, worg, wres = wall_at(2.0)
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 1.0, 1.0, 1.0], wocc, worg, wres)
check("nothing painted beyond a wall", vm.conf[vm._cx > 2.3].sum() == 0.0,
      f"{vm.conf[vm._cx > 2.3].sum():.3f}")
check("in front of the wall IS painted",
      vm.conf[(vm._cx > 0.5) & (vm._cx < 1.8) & (np.abs(vm._cy) < 0.5)].sum() > 0.0)

uocc, uorg, ures = blank()
uocc[:, :] = -1
uocc[180:220, 180:260] = 0          # a small known-free pocket around the robot
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 1.0, 1.0, 1.0], uocc, uorg, ures)
check("unknown space stops the fan", vm.conf[vm._cx > 3.5].sum() == 0.0,
      f"{vm.conf[vm._cx > 3.5].sum():.3f}")

# --------------------------------------------------------------------- confidence
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 1.0, 1.0, 1.0], occ, org, res)
axis = vm.conf[(np.abs(vm._cy) < 0.15) & (vm._cx > 1.0) & (vm._cx < 3.0)].mean()
edge_mask = (np.abs(np.arctan2(vm._cy, vm._cx)) > 0.68) & (vm.conf > 0)
edge = vm.conf[edge_mask].mean() if edge_mask.any() else 0.0
check("confidence peaks on the optical axis", axis > 0.9 and edge < 0.3,
      f"axis={axis:.3f} edge={edge:.3f}")

# A head-on look must not be talked down by a later glance from the edge of the view.
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [0.0, 0.0, 0.0, 0.0], occ, org, res)
vm.paint(np.array([0.0, -2.0]), math.pi / 2 - 0.70, [1.0, 1.0, 1.0, 1.0], occ, org, res)
after = vm.value[(np.abs(vm._cy) < 0.15) & (vm._cx > 1.0) & (vm._cx < 2.0)].mean()
check("an edge-of-view glimpse does not overwrite a head-on look", after < 0.5,
      f"{after:.3f}")

# -------------------------------------------------------------------- normalising
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [0.9, 0.1, 0.1, 0.1], occ, org, res)
s = vm.score(np.array([[2.0, 1.2], [-5.0, -5.0]]))
check("too few observations -> everything neutral (0)", bool((np.abs(s) < 1e-9).all()),
      f"{s}")

vm = vmap()
for k in range(12):
    vm.paint(np.array([0.0, 0.0]), k * 0.5,
             [0.9 if k % 2 == 0 else 0.0, 0.0, 0.0, 0.0], occ, org, res)
s = vm.score(np.array([[1.5, 1.5], [-6.0, -6.0]]))
check("unseen point returns neutral 0.0", abs(s[1]) < 1e-9, f"{s[1]:.3f}")
check("seen point returns a real value once a spread exists", abs(s[0]) > 1e-9,
      f"{s[0]:.3f}")
check("score stays in [0,1]", bool(((s >= 0) & (s <= 1)).all()))

# Observed 2026-10-02 looking for a toilet: 66% of scores were exactly 0.000 and p95 was
# 0.016. Stretching that onto [0, 1] magnified noise 60x and lit up rooms the target was
# nowhere near.
vm = vmap()
for k in range(12):
    vm.paint(np.array([0.0, 0.0]), k * 0.5, [0.004, 0.0, 0.002, 0.0], occ, org, res)
sf = vm.score(np.array([[1.5, 1.5], [1.0, -1.0]]))
check("a degenerate spread is not stretched into signal", bool((np.abs(sf) < 1e-9).all()),
      f"{sf}")

# -------------------------------------------------------------------- publishing
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [0.9, 0.1, 0.1, 0.1], occ, org, res)
val, cnf = vm.as_occupancy()
check("unobserved cells publish as -1", int(val[0, 0]) == -1 and int(cnf[0, 0]) == -1)
check("observed cells publish in 0..100",
      bool((val[vm.conf > 0] >= 0).all() and (val[vm.conf > 0] <= 100).all()))

# ------------------------------------------------------- following a loop closure
vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 1.0, 1.0, 1.0], occ, org, res)
before_mass = vm.conf.sum()
cx_before = (vm._cx * vm.conf).sum() / before_mass
vm.transform(1.0, 0.0, 0.0)
cx_after = (vm._cx * vm.conf).sum() / vm.conf.sum()
check("rigid shift moves the paint by the right amount",
      abs((cx_after - cx_before) - 1.0) < 0.21, f"moved {cx_after - cx_before:.3f} m")
check("rigid shift keeps (most of) the mass", vm.conf.sum() > 0.95 * before_mass,
      f"{vm.conf.sum():.1f} vs {before_mass:.1f}")

vm = vmap()
vm.paint(np.array([0.0, 0.0]), 0.0, [1.0, 0.0, 0.0, 0.0], occ, org, res)   # paints +y
vm.transform(0.0, 0.0, -math.pi / 2)                                       # +y -> +x
check("rigid rotation carries the paint around",
      vm.conf[vm._cx > 0.5].sum() > 5 * max(vm.conf[vm._cy > 0.5].sum(), 1e-9),
      f"+x={vm.conf[vm._cx > 0.5].sum():.1f} +y={vm.conf[vm._cy > 0.5].sum():.1f}")

# ----------------------------------------------- producer -> wire -> consumer
# Consistently high on the leftmost strip, zero elsewhere: the spread comes from the
# other three strips, so the painted value is unambiguous rather than an average.
vm = vmap()
for _ in range(12):
    vm.paint(np.array([0.0, 0.0]), 0.0, [0.9, 0.0, 0.0, 0.0], occ, org, res)
val, cnf = vm.as_occupancy()
g = GridValueMap(radius_m=0.5)
g.set_value(val, vm.origin, vm.res)
g.set_confidence(cnf, vm.origin, vm.res)
left = g.score(np.array([[1.5, 1.5]]))[0]
right = g.score(np.array([[1.5, -1.5]]))[0]
check("consumer reproduces the painted side as high", left > 0.6, f"{left:.3f}")
check("painted side clearly beats the empty side", left > 3 * max(right, 0.01),
      f"{left:.3f} vs {right:.3f}")
check("unseen is neutral 0.0", g.score(np.array([[-7.0, -7.0]]))[0] == 0.0)
check("consumer output stays in [0,1]", 0.0 <= left <= 1.0 and 0.0 <= right <= 1.0)
check("no grids yet -> all zero, i.e. nearest-first",
      bool((GridValueMap().score(np.array([[1.0, 1.0], [3.0, 2.0]])) == 0.0).all()))

# ----------------------------------------------------- value against distance
# 0.25/m, so a full point of value buys 4 m of detour -- "the next room", not the whole
# flat (VLFM's own trade-off works out at ~10 m, most of kujiale's diagonal).
sel = FrontierSelector()
check("value 1.0 wins a 3.5 m detour",
      sel.choose(np.array([[2.0, 0.0], [5.5, 0.0]]), np.array([0.0, 1.0]),
                 np.array([0.0, 0.0]), 0.4) == 1)
check("value 1.0 does NOT win a 6 m detour",
      sel.choose(np.array([[2.0, 0.0], [8.0, 0.0]]), np.array([0.0, 1.0]),
                 np.array([0.0, 0.0]), 0.4) == 0)

print("\n" + ("ALL PASS" if not _fails else f"{len(_fails)} FAILED: {_fails}"))
sys.exit(1 if _fails else 0)
