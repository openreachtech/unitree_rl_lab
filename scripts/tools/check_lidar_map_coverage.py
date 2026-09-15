"""Predict LiDAR-fan coverage of the height-map grid, without launching Isaac Sim.

The LiDAR elevation map (``mdp/lidar_elevation_map.py``) builds the policy's height
grid by firing a static fan from the LiDAR mount and binning the returns into cells,
so a cell is only observed if at least one beam lands in it. This script answers,
analytically, how many beams land where -- which is the "ray density" question that
decides the fan's ``channels`` / ``horizontal_res``:

    * on flat ground there is no occlusion at all, so every unobserved in-FOV cell
      is purely a density shortfall. That is the number to size the fan against.
    * with a wall in front, the extra unobserved cells are the shadow -- the signal
      we actually want the sensor model to produce.

Run before touching Isaac; ray-plane / ray-box intersection is enough to get the
counts, and iterating here costs seconds instead of a sim launch.

    python3 scripts/tools/check_lidar_map_coverage.py
    python3 scripts/tools/check_lidar_map_coverage.py --channels 64 --h-res 2.0
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# --- Grid geometry (mirrors velocity_env_cfg_go2.py) ------------------------
RESOLUTION = 0.05
SIZE_X, SIZE_Y = 1.4, 1.0
BODY_HALF_X, BODY_HALF_Y = 0.30, 0.20

# --- LiDAR mount, base frame; z is measured from the ground at nominal stance
LIDAR_X, LIDAR_Y, LIDAR_Z = 0.1934, 0.0, 0.42

MAX_OBSTACLE_HEIGHT = 0.25
"""Tallest terrain feature the fan has to see the top of (walls/boxes cap at 25 cm)."""


def derive_fan(kept, in_fov, radius, h_fov):
    """Smallest fan that still lands a beam per cell, from the geometry alone.

    Two separate limits, worth keeping apart because they answer different questions:

    * the *span* of the vertical FOV is set by what must be in view at all -- flat
      ground at the nearest observed cell (steepest) through the top of the tallest
      obstacle at the farthest (shallowest). Widening the body exclusion pushes the
      nearest cell outward, which is what shrinks the span and so the channel count.
    * the *spacing* is set by how fast the landing point moves per degree, worst at
      the farthest cell. Reported for flat ground (what the coverage test below
      measures) and for an obstacle top (what reading a wall height needs), because
      the latter is several times stricter and is a deliberate compromise.
    """
    target = kept & in_fov
    r_min, r_max = radius[target].min(), radius[target].max()
    steepest = math.degrees(math.atan2(LIDAR_Z, r_min))
    shallowest = math.degrees(math.atan2(LIDAR_Z - MAX_OBSTACLE_HEIGHT, r_max))

    # |dr/dphi| = ((z-h)^2 + r^2) / (z-h), in metres per radian.
    def spacing_deg(h: float) -> float:
        return math.degrees(RESOLUTION * (LIDAR_Z - h) / ((LIDAR_Z - h) ** 2 + r_max**2))

    d_flat, d_top = spacing_deg(0.0), spacing_deg(MAX_OBSTACLE_HEIGHT)
    channels = math.ceil((steepest - shallowest) / d_flat) + 1
    h_res = math.degrees(RESOLUTION / r_max)
    print("  導出:")
    print(f"    観測セルの距離範囲   {r_min:.3f} .. {r_max:.3f} m")
    print(f"    必要な仰角範囲       -{steepest:.1f} .. -{shallowest:.1f} deg (幅 {steepest - shallowest:.1f})")
    print(f"    仰角刻み             平地基準 {d_flat:.2f} deg / {MAX_OBSTACLE_HEIGHT * 100:.0f}cm天面基準 {d_top:.2f} deg")
    print(f"    方位刻み             {h_res:.2f} deg")
    span = h_fov[1] - h_fov[0]
    n_h = math.ceil(span / h_res) + (0 if span >= 359.9 else 1)
    print(f"    => channels={channels}  h_res={h_res:.1f}  本数 {channels * n_h}")
    return channels, h_res


def fan_directions(channels: int, v_fov: tuple[float, float], h_fov: tuple[float, float], h_res: float):
    """Ray directions of ``patterns.lidar_pattern``, reproduced exactly."""
    v = np.linspace(v_fov[0], v_fov[1], channels)
    num_h = math.ceil((h_fov[1] - h_fov[0]) / h_res) + 1
    h = np.linspace(h_fov[0], h_fov[1], num_h)
    # A full turn puts the first and last azimuth on the same ray; lidar_pattern drops
    # the duplicate, so the count is one lower than the endpoint-inclusive linspace.
    if abs(abs(h_fov[0] - h_fov[1]) - 360.0) < 1e-6:
        h = h[:-1]
    vv, hh = np.meshgrid(np.deg2rad(v), np.deg2rad(h), indexing="ij")
    d = np.stack(
        [np.cos(vv) * np.cos(hh), np.cos(vv) * np.sin(hh), np.sin(vv)], axis=-1
    ).reshape(-1, 3)
    return d, len(v), len(h)


def trace(directions: np.ndarray, boxes: list[tuple[float, float, float, float, float]]):
    """First hit of each ray against the ground plane z=0 plus axis-aligned boxes.

    ``boxes`` entries are ``(x_min, x_max, y_min, y_max, height)``. Returns the hit
    points and a validity mask (rays that escape upward hit nothing).
    """
    origin = np.array([LIDAR_X, LIDAR_Y, LIDAR_Z])
    t_best = np.full(len(directions), np.inf)

    # Ground plane z = 0.
    dz = directions[:, 2]
    going_down = dz < -1.0e-9
    t_ground = np.where(going_down, -origin[2] / np.where(going_down, dz, -1.0), np.inf)
    t_best = np.minimum(t_best, t_ground)

    # Boxes, by the slab method.
    for x_min, x_max, y_min, y_max, height in boxes:
        lo = np.array([x_min, y_min, 0.0])
        hi = np.array([x_max, y_max, height])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / directions
            t2 = (hi - origin) / directions
        t_near = np.nanmax(np.minimum(t1, t2), axis=1)
        t_far = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 0.0)
        t_box = np.where(hit, np.maximum(t_near, 0.0), np.inf)
        t_best = np.minimum(t_best, t_box)

    valid = np.isfinite(t_best)
    points = origin + directions * t_best[:, None]
    return points, valid


def bin_to_cells(points: np.ndarray, valid: np.ndarray):
    """Nearest-grid-point binning, matching the term's ``round`` assignment."""
    num_x = round(SIZE_X / RESOLUTION) + 1
    num_y = round(SIZE_Y / RESOLUTION) + 1
    x0, y0 = -SIZE_X / 2, -SIZE_Y / 2

    ix = np.rint((points[:, 0] - x0) / RESOLUTION).astype(int)
    iy = np.rint((points[:, 1] - y0) / RESOLUTION).astype(int)
    inside = valid & (ix >= 0) & (ix < num_x) & (iy >= 0) & (iy < num_y)

    counts = np.zeros(num_x * num_y, dtype=int)
    np.add.at(counts, ix[inside] * num_y + iy[inside], 1)
    return counts.reshape(num_x, num_y), num_x, num_y


def cell_masks(num_x: int, num_y: int, h_fov: tuple[float, float] = (-90.0, 90.0)):
    """Kept cells (body footprint removed) and the azimuth wedge the fan covers.

    Cells outside the wedge can never receive a beam, so they are excluded from the
    reported rate -- otherwise a narrow fan would look like a density problem when it
    is really a field-of-view choice.
    """
    x = np.linspace(-SIZE_X / 2, SIZE_X / 2, num_x)
    y = np.linspace(-SIZE_Y / 2, SIZE_Y / 2, num_y)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    eps = RESOLUTION * 1.0e-4
    kept = ~((np.abs(gx) <= BODY_HALF_X + eps) & (np.abs(gy) <= BODY_HALF_Y + eps))
    if h_fov[1] - h_fov[0] >= 359.9:
        in_fov = np.ones_like(gx, dtype=bool)
    else:
        azimuth = np.degrees(np.arctan2(gy - LIDAR_Y, gx - LIDAR_X))
        in_fov = (azimuth >= h_fov[0]) & (azimuth <= h_fov[1])
    radius = np.hypot(gx - LIDAR_X, gy - LIDAR_Y)
    return kept, in_fov, radius


def report(name: str, counts, kept, in_fov, radius):
    target = kept & in_fov
    observed = counts > 0
    unknown = target & ~observed

    print(f"\n=== {name} ===")
    print(f"  観測対象セル (kept & 前方FOV内) : {target.sum()}")
    print(f"  うち未観測                      : {unknown.sum()}  ({100 * unknown.sum() / target.sum():.1f}%)")

    bands = [("近 r<0.30", radius < 0.30), ("中 0.30-0.50", (radius >= 0.30) & (radius < 0.50)), ("遠 r>=0.50", radius >= 0.50)]
    print("  距離帯別:")
    for label, band in bands:
        sel = target & band
        if sel.sum() == 0:
            continue
        hits = counts[sel]
        print(
            f"    {label:14s} セル {sel.sum():4d} / 未観測 {(hits == 0).sum():4d}"
            f" ({100 * (hits == 0).sum() / sel.sum():5.1f}%) / 平均ヒット {hits.mean():6.2f}"
            f" / 中央値 {int(np.median(hits)):3d}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=33)
    p.add_argument("--v-fov", type=float, nargs=2, default=(-70.0, -13.0))
    p.add_argument("--h-fov", type=float, nargs=2, default=(-90.0, 90.0))
    p.add_argument("--h-res", type=float, default=4.0)
    p.add_argument("--lidar-z", type=float, default=None, help="LiDAR height above ground (m)")
    p.add_argument("--body-half-x", type=float, default=None, help="body exclusion half-extent x (m)")
    p.add_argument("--body-half-y", type=float, default=None, help="body exclusion half-extent y (m)")
    p.add_argument("--auto-fov", action="store_true", help="derive channels/v-fov/h-res from geometry")
    p.add_argument("--wall-x", type=float, default=0.45, help="wall near face, base-frame x")
    p.add_argument("--wall-thickness", type=float, default=0.10)
    p.add_argument("--wall-height", type=float, default=0.25)
    args = p.parse_args()

    global LIDAR_Z, BODY_HALF_X, BODY_HALF_Y
    if args.lidar_z is not None:
        LIDAR_Z = args.lidar_z
    if args.body_half_x is not None:
        BODY_HALF_X = args.body_half_x
    if args.body_half_y is not None:
        BODY_HALF_Y = args.body_half_y
    print(f"LiDAR 地上高 {LIDAR_Z:.2f} m / 除外矩形 半径 x={BODY_HALF_X:.2f} y={BODY_HALF_Y:.2f} m")

    if args.auto_fov:
        num_x = round(SIZE_X / RESOLUTION) + 1
        num_y = round(SIZE_Y / RESOLUTION) + 1
        k, f, rad = cell_masks(num_x, num_y, tuple(args.h_fov))
        ch, hr = derive_fan(k, f, rad, tuple(args.h_fov))
        args.channels, args.h_res = ch, round(hr, 1)
        args.v_fov = (
            -math.degrees(math.atan2(LIDAR_Z, rad[k & f].min())),
            -math.degrees(math.atan2(LIDAR_Z - MAX_OBSTACLE_HEIGHT, rad[k & f].max())),
        )
        print(f"    観測セル数 {(k & f).sum()} / kept {k.sum()}")

    directions, n_v, n_h = fan_directions(args.channels, tuple(args.v_fov), tuple(args.h_fov), args.h_res)
    print(f"扇: 仰角 {n_v} x 方位 {n_h} = {len(directions)} 本 (現状の真下グリッドは 609 本)")
    print(f"    仰角範囲 {args.v_fov[0]}..{args.v_fov[1]} deg / 刻み {(args.v_fov[1] - args.v_fov[0]) / (n_v - 1):.2f} deg")
    print(f"    方位刻み {args.h_res} deg")

    counts_flat, num_x, num_y = bin_to_cells(*trace(directions, []))
    kept, in_fov, radius = cell_masks(num_x, num_y, tuple(args.h_fov))
    report("平地 (遮蔽ゼロ -- 未観測はすべて密度不足)", counts_flat, kept, in_fov, radius)

    wall = (args.wall_x, args.wall_x + args.wall_thickness, -1.0, 1.0, args.wall_height)
    counts_wall, _, _ = bin_to_cells(*trace(directions, [wall]))
    report(
        f"壁 (x={args.wall_x:.2f}m, 高さ{args.wall_height * 100:.0f}cm) -- 平地との差が影",
        counts_wall,
        kept,
        in_fov,
        radius,
    )

    target = kept & in_fov
    extra = ((counts_wall == 0) & (counts_flat > 0) & target).sum()
    print(f"\n  壁によって新たに未観測になったセル (= 影): {extra}")
    print(f"  前方FOV外で構造的に観測できないセル      : {(kept & ~in_fov).sum()} / {kept.sum()}")


if __name__ == "__main__":
    main()
