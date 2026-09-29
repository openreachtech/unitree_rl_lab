#!/usr/bin/env python3
"""Generate <scene>_nodoors.usda: the InteriorAgent scene with door leaves composed out.

Why a wrapper file instead of deactivating at spawn time: omni.physx parses prims the
moment the reference loads, and the dataset's door leaves carry their own RigidBodyAPI
(nested, which PhysX itself flags as "unpredicted results"). Deactivating them *after*
the load left ghost colliders -- invisible to rendering and to the merged raycast mesh,
but still solid: the robot pressed against thin air in every doorway (observed
2026-09-25). A sublayer wrapper authors ``active = false`` over each door prim, so the
composed stage never contains them and PhysX never parses them.

Usage (needs USD python -- run via the Isaac venv with the bundled pxr):
    EXT=$HOME/isaacsim/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*
    PYLIB=$(find ~/.local/share/uv/python/cpython-3.11* -name "libpython3.11.so.1.0" | head -1)
    PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$(dirname $PYLIB) \
        ~/isaacsim/env_isaaclab/bin/python scripts/tools/make_kujiale_nodoors.py \
        /path/to/kujiale_0003/kujiale_0003.usda

Output lands next to the input as <name>_nodoors.usda; velocity_env_cfg_kujiale.py
loads that.
"""

from __future__ import annotations

import re
import sys

from pxr import Sdf, Usd

DOOR_LEAF_RE = re.compile(r"^door_(handle_)?\d+$")


def main(scene_path: str) -> None:
    stage = Usd.Stage.Open(scene_path)
    doors = []
    it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in it:
        if DOOR_LEAF_RE.match(prim.GetName()):
            doors.append(prim.GetPath())
            it.PruneChildren()
    if not doors:
        sys.exit(f"no door prims found in {scene_path} -- wrong file?")

    out_path = scene_path.replace(".usda", "_nodoors.usda")
    layer = Sdf.Layer.CreateNew(out_path)
    layer.subLayerPaths.append("./" + scene_path.rsplit("/", 1)[-1])
    # author `active = false` overs -- the doors never enter composition downstream
    for path in doors:
        spec = Sdf.CreatePrimInLayer(layer, path)
        spec.specifier = Sdf.SpecifierOver
        spec.active = False
    # keep the original stage metadata (meters, up axis) on the wrapper
    src = Sdf.Layer.FindOrOpen(scene_path)
    layer.defaultPrim = src.defaultPrim
    if src.HasFramesPerSecond():
        layer.framesPerSecond = src.framesPerSecond
    layer.Save()
    print(f"wrote {out_path}: {len(doors)} door leaf/handle prims composed out")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
