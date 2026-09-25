"""fukuoka_test.usdz に三角メッシュの衝突判定(Collider)を焼き込み、
原点中心・床z=0 に整えて fukuoka_collider.usd として書き出す（GUI不要・ヘッドレス）。

pip版 Isaac Sim (env_isaaclab) を使う。実行:
    cd /home/tanaka/isaacsim/unitree_rl_lab
    source /home/tanaka/isaacsim/env_isaaclab/bin/activate
    python realterrain/bake_collider.py

ポイント:
- 取り込んだメッシュは Instanceable（インスタンス）の中に隠れるので、まず解除してから Collider を付ける。
- ブロックは原点から離れて配置されているので、xy中心を原点・床(min z)を0に平行移動する（Go2は原点付近にスポーンするため）。
- 単位がmm由来で巨大なら --scale 0.001。中心寄せしたくなければ --no_center。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bake triangle-mesh colliders into a terrain USD/USDZ (headless).")
parser.add_argument("--input", default="/home/tanaka/ダウンロード/fukuoka_test.usdz")
parser.add_argument("--output", default="/home/tanaka/isaacsim/unitree_rl_lab/realterrain/assets/fukuoka_collider.usd")
parser.add_argument("--scale", type=float, default=1.0, help="全体スケール倍率（mm由来なら 0.001）")
parser.add_argument("--no_center", action="store_true", help="原点中心・床z=0への平行移動をしない")
parser.add_argument("--report", default="/tmp/claude-1005/-home-tanaka/425f94ab-e193-4829-8edd-da7fcc2f1bd6/scratchpad/bake_report.txt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def world_bbox(stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )
    r = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def main():
    report = []

    def log(s):
        report.append(str(s))
        print(s)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    terrain = UsdGeom.Xform.Define(stage, "/World/terrain")
    terrain.GetPrim().GetReferences().AddReference(args.input)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # 1) Instanceable を解除（メッシュを編集可能にする）
    for _ in range(10):
        changed = False
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                changed = True
        if not changed:
            break

    xf = UsdGeom.Xformable(terrain)
    xf.ClearXformOpOrder()
    if args.scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))

    # 2) 原点中心・床z=0 へ平行移動
    mn, mx = world_bbox(stage)
    log(f"raw  bbox min={tuple(round(v,3) for v in mn)} max={tuple(round(v,3) for v in mx)}")
    if not args.no_center:
        cx, cy, minz = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, mn[2]
        xf.AddTranslateOp().Set(Gf.Vec3d(-cx, -cy, -minz))
        log(f"center translate=({-cx:.3f}, {-cy:.3f}, {-minz:.3f})")

    # 3) 全メッシュに衝突判定（三角メッシュ＝実形状・静的地形）
    n = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.CreateApproximationAttr().Set("none")
            n += 1
    log(f"meshes with collider = {n}")

    mn2, mx2 = world_bbox(stage)
    log(f"final bbox min={tuple(round(v,3) for v in mn2)} max={tuple(round(v,3) for v in mx2)}")
    log(f"final size(縦横高)={tuple(round(mx2[i]-mn2[i],3) for i in range(3))}")

    flat = stage.Flatten()
    flat.Export(args.output)
    log(f"saved: {args.output}")

    with open(args.report, "w") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
