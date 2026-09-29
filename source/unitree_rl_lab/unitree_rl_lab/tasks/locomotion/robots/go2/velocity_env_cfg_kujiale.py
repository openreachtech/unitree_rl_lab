"""Go2-Blind-GRU-Mid360-Explore inside an InteriorAgent (kujiale) apartment.

Same policy, sensor and nav-facing behavior as the four-room explore config; only
the world changes: instead of a generated terrain, a photorealistic 12-room
apartment (~16 x 14 m) from the InteriorAgent dataset is spawned as a static USD
scene. This is the M4-in-real-floorplans world, and the RGB world M5's VLM will
score.

The dataset ships clean geometry but NO physics: colliders are applied to every
mesh at spawn time (``_spawn_kujiale`` -> ``setStaticCollider``, triangle-mesh
approximation -- valid for static geometry). Door leaves are composed out of the
scene before load (the _nodoors wrapper) -- closed and unopenable they wall off half
the apartment, and post-load deactivation left ghost PhysX colliders.
Ground contact comes from a plain physics plane at z=0, coplanar with the scene's
floor meshes.

The MID-360 raycasts against the whole apartment via
``RollingLivoxSensorCfg.combine_scene_meshes``: the scene is hundreds of separate
mesh prims, which the stock RayCaster cannot see past (it takes the first mesh),
so the sensor merges the subtree into one warp mesh. Ceilings are included --
a real MID-360 indoors sees them too, and LIO registration is the better for it.
The critic's top-down ``height_scanner`` still points at the flat ground plane;
its values are meaningless here, which is fine -- play never reads the critic.

Spawn: living-room centroid from the dataset's rooms.json, standing start.

    python scripts/ros2/play_ros2.py --task Go2-Blind-GRU-Mid360-Kujiale \
        --checkpoint logs/rsl_rl/go2_blind_gru_phase4/<run>/model_7300.pt

The dataset location can be overridden with the INTERIOR_AGENT_DIR environment
variable (default: /home/tak/isaacsim/interior_agent_data).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_explore import (
    RobotEnvCfgMid360Explore,
)

INTERIOR_AGENT_DIR = os.environ.get("INTERIOR_AGENT_DIR", "/home/tak/isaacsim/interior_agent_data")
KUJIALE_0003_USD = os.path.join(INTERIOR_AGENT_DIR, "kujiale_0003", "kujiale_0003_nodoors.usda")
"""The _nodoors wrapper (scripts/tools/make_kujiale_nodoors.py): door leaves composed
out BEFORE load. Deactivating them after spawn left ghost PhysX colliders -- the doors
carry their own RigidBodyAPI and omni.physx parses them the moment the reference
loads; SetActive(False) afterwards removed them from rendering and raycast but not
reliably from physics (observed 2026-09-25: robot blocked by invisible doors)."""

# Living-room centroid from kujiale_0003/rooms.json; the largest open space.
KUJIALE_0003_SPAWN_XY = (-1.3, 0.8)


def _spawn_kujiale(prim_path, cfg, translation=None, orientation=None):
    """spawn_from_usd, then make the scene a static collider.

    Doorways are opened upstream: the loaded file is the _nodoors wrapper (see
    KUJIALE_0003_USD), so door leaves never reach composition or PhysX.

    RigidBodyAPI is stripped from the whole subtree FIRST. The dataset applies it to
    every asset (281 prims: walls, cabinets, pillows...), and physx's ``setCollider``
    silently downgrades approximation "none" (trimesh) to **convexHull** for any mesh
    under a rigid body -- the convex hull of a wall with a door hole is a wall WITHOUT
    the hole. That was the invisible barrier in every doorway (2026-09-25): physics
    collided with hulls while the LiDAR raycast saw the true meshes. A static scene
    needs no rigid bodies at all; with the API gone, "none" stays trimesh and the
    collision world matches the raycast world exactly.

    ``setStaticCollider`` then walks the subtree and applies collision to every mesh.
    """
    if not os.path.exists(cfg.usd_path):
        raise FileNotFoundError(
            f"{cfg.usd_path} not found. Generate it with"
            " scripts/tools/make_kujiale_nodoors.py (see its docstring)."
        )
    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    import omni.usd
    from omni.physx.scripts import utils as physx_utils
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)

    stripped = 0
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            stripped += 1
        if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            p.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    print(f"[kujiale] stripped RigidBodyAPI from {stripped} prims (static scene, trimesh collision).")

    physx_utils.setStaticCollider(root, approximationShape="none")
    return prim


@configclass
class KujialeUsdCfg(sim_utils.UsdFileCfg):
    func = _spawn_kujiale


@configclass
class RobotEnvCfgMid360Kujiale(RobotEnvCfgMid360Explore):
    def __post_init__(self):
        super().__post_init__()
        # flat physics plane instead of the generated four-room terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None

        self.scene.kujiale = AssetBaseCfg(
            prim_path="/World/kujiale",
            spawn=KujialeUsdCfg(usd_path=KUJIALE_0003_USD),
        )

        # raycast the apartment, not the plane
        self.scene.mid360_scanner.mesh_prim_paths = ["/World/kujiale"]
        self.scene.mid360_scanner.combine_scene_meshes = True

        # spawn in the living room; keep the reset jitter small so a resample
        # cannot relocate the robot into furniture
        x, y = KUJIALE_0003_SPAWN_XY
        self.scene.robot.init_state.pos = (x, y, self.scene.robot.init_state.pos[2])
        self.events.reset_base.params["pose_range"]["x"] = (-0.2, 0.2)
        self.events.reset_base.params["pose_range"]["y"] = (-0.2, 0.2)


@configclass
class RobotPlayEnvCfgMid360Kujiale(RobotEnvCfgMid360Kujiale):
    """Same config; registered separately so play.py's entry-point lookup works."""

    pass
