"""Make the robot's own body block its ray-cast sensors.

IsaacLab's ``RayCaster`` casts against exactly one static mesh -- ``_initialize_warp_meshes``
raises ``NotImplementedError`` past one prim path and bakes the world transform at init --
so with ``mesh_prim_paths=["/World/ground"]`` a ray fired at the ground passes straight
through the robot's own legs and trunk. That is not merely "the leg is invisible": the ray
still hits the terrain behind it and reports a confident, correct-looking range, so a height
map built from those returns is *measured* exactly where the hardware would have no data at
all. Its unobserved pattern is therefore systematically wrong, and wrong in a gait-correlated
way -- on the real Go2 the front legs sweep a nose-mounted downward field of view once per
step, and swing widest just as the robot lifts them over an obstacle.

This module adds a second, moving mesh made of the robot's own collision geometry, following
OmniPerception's dynamic-mesh RayCaster
(``LidarSensor/LidarSensor/example/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py``,
https://github.com/aCodeDog/OmniPerception): build one warp mesh holding every environment's
copy of the geometry, move its vertices with the bodies each step, ``refit()`` the BVH, cast
the same rays at it, and merge by distance.

Four things are done differently from upstream, three for cost or correctness and one because
a nose-mounted sensor exposes a case upstream never hits:

* **Poses come from the physics views, not from USD.** Upstream reads
  ``XFormPrim.get_world_poses()`` every step, a USD round-trip per body. A rigid-body view per
  body gives the same numbers straight off the GPU-resident physics buffers.
* **Vertices are stored in the body frame.** Upstream transforms mesh-local points into the
  *parent* frame with ``rel[:3, 3]`` as the translation, but USD composes row vectors
  (``p' = p @ M``, translation in ``M[3, :3]``), so that column is zero for an ordinary
  translation and the offset is silently dropped. See :func:`_usd_local_to_world`.
* **An occluded ray is dropped, not re-pointed at the robot.** Upstream keeps the nearer of
  the two hits, which is what you want when the dynamic object is the thing being sensed. Here
  the dynamic object is the sensor's own robot: a real driver filters self-returns out, so the
  ray becomes a miss and a held height map's cell keeps its previous value.
  ``occluder_mode="hit"`` restores upstream's behaviour.
* **Geometry enclosing the sensor is dropped.** See below -- without this the Go2's MID-360
  returns an entirely blank map.

The sensor inside the shell
---------------------------
A LiDAR is bolted into a housing, and on the Go2 the housing is modelled. ``radar_joint`` puts
the L1 at ``(0.28945, 0, -0.046825)`` in the base frame; ``Head_lower``'s collision sphere is
4.7 cm in radius centred at ``(0.293, 0, -0.06)``, i.e. **1.4 cm away**. The mount is inside
it. Warp's ``mesh_query_ray`` does not cull back faces, so every ray would leave through the
inside of that sphere within 6 cm and *the entire map would go unobserved* -- a failure that
looks like a broken sensor rather than a modelling choice.

``occluder_drop_geometry_containing_sensor`` (on by default) tests the mount point against
each candidate geometry at init and drops the ones that contain it, naming them as it goes.
Physically that is the right call: the sensor looks out through an aperture in the shell it
sits in, and the shell it sits in is the one part of the robot it can never range on. The
parts that do matter -- trunk, hips, legs -- are 10 cm and further away and are untouched.
``occluder_min_distance`` is the blunter fallback for a mount this test cannot resolve.

Cost, which is the point of the comparison this was written for: per step, one gather and
transform over every vertex of every environment's robot, one BVH refit over the same, and one
extra ``raycast_mesh``. For the Go2's collision geometry that is roughly 650 vertices and 1,200
triangles per robot, so 4,096 environments carry a ~5M-triangle mesh rebuilt 50 times a second.
``occluder_body_names`` and ``occluder_max_distance`` are the two knobs that buy it back.
"""

from __future__ import annotations

import numpy as np
import re
import torch
import warp as wp
from collections.abc import Sequence
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_quat, quat_apply
from isaaclab.utils.warp import raycast_mesh

from .lidar_sensor_cfg import LidarSensorCfg
from .rolling_livox_sensor import RollingLivoxSensor

_SUPPORTED_GPRIM_TYPES = ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
"""Geometry types read off the stage, matching OmniPerception's list minus ``Plane``: an
infinite plane is not something a robot link is made of."""


# ---------------------------------------------------------------------------
# USD geometry -> triangles
# ---------------------------------------------------------------------------
def _usd_local_to_world(prim) -> tuple[np.ndarray, np.ndarray]:
    """The prim's local-to-world transform as ``(A, t)`` with ``p_world = p_local @ A + t``.

    USD composes row vectors, so ``GfMatrix4d`` carries the translation in its last *row*.
    Reading it as a column-vector matrix -- ``M[:3, 3]`` -- yields zeros for a pure
    translation, which is how upstream loses every link offset.
    """
    from pxr import UsdGeom

    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0), dtype=np.float64)
    return m[:3, :3], m[3, :3]


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate a USD face-vertex list. Quads and n-gons both turn up in converted
    collision meshes, and warp only takes triangles."""
    tris = []
    at = 0
    for n in counts:
        face = indices[at : at + n]
        at += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _gprim_to_trimesh(prim, tessellation: int, sphere_subdivisions: int):
    """One USD geometry prim as a trimesh in its own local frame, or ``None``.

    Analytic prims are re-tessellated here rather than read, so their vertex count is a
    parameter of this module instead of whatever the asset happened to be authored with -- it
    gets multiplied by the environment count, so it is worth controlling.
    """
    import trimesh
    from pxr import UsdGeom

    type_name = prim.GetTypeName()

    def _to_axis(mesh, axis: str):
        """trimesh builds cylinders/capsules/cones about +z; USD names the axis."""
        if axis == "X":
            mesh.apply_transform(
                np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
            )
        elif axis == "Y":
            mesh.apply_transform(
                np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
            )
        return mesh

    if type_name == "Mesh":
        mesh_api = UsdGeom.Mesh(prim)
        points = mesh_api.GetPointsAttr().Get()
        counts = mesh_api.GetFaceVertexCountsAttr().Get()
        indices = mesh_api.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            return None
        return trimesh.Trimesh(
            vertices=np.asarray(points, dtype=np.float64),
            faces=_triangulate(np.asarray(counts, dtype=np.int64), np.asarray(indices, dtype=np.int64)),
            process=False,
        )
    if type_name == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        size = 2.0 if size is None else float(size)
        return trimesh.creation.box(extents=(size, size, size))
    if type_name == "Sphere":
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        return trimesh.creation.icosphere(subdivisions=sphere_subdivisions, radius=radius)
    if type_name in ("Cylinder", "Capsule", "Cone"):
        gprim = getattr(UsdGeom, type_name)(prim)
        radius = float(gprim.GetRadiusAttr().Get() or 1.0)
        height = float(gprim.GetHeightAttr().Get() or 1.0)
        axis = str(gprim.GetAxisAttr().Get() or "Z")
        if type_name == "Cylinder":
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=tessellation)
        elif type_name == "Capsule":
            mesh = trimesh.creation.capsule(radius=radius, height=height, count=(tessellation, tessellation))
            mesh.apply_translation((0.0, 0.0, -height / 2.0))  # trimesh puts its base at z=0
        else:
            mesh = trimesh.creation.cone(radius=radius, height=height, sections=tessellation)
            mesh.apply_translation((0.0, 0.0, -height / 2.0))
        return _to_axis(mesh, axis)
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@configclass
class RobotOccluderCfgMixin:
    """The occluder fields, mixed into whichever sensor config wants them."""

    occluder_enabled: bool = True
    """Whether the robot blocks its own rays at all.

    Off, the sensor behaves exactly like the plain one: no mesh is built, no BVH is refit and
    no second raycast happens, so the only thing left of this class is the config fields. That
    makes an A/B where nothing but self-occlusion differs a one-line change, and it is the
    escape hatch if the extra cost is ever not worth it -- measured on a Go2 at 4,096
    environments it is about +25% on a training iteration."""

    occluder_prim_path: str = "{ENV_REGEX_NS}/Robot"
    """Articulation root whose bodies occlude, ``{ENV_REGEX_NS}`` resolved against the sensor's
    own prim path. Use :attr:`occluder_enabled` to switch the mechanism off; this field only
    says *what* occludes."""

    occluder_body_names: list[str] = [".*"]
    """Regexes, matched in full against each rigid body's prim name. The default takes every
    body; narrowing it is the cheapest way to cut the per-step cost, and for a nose-mounted
    downward sensor ``["base", "F[LR]_.*"]`` keeps the parts that actually cast a shadow."""

    occluder_mode: Literal["drop", "hit"] = "drop"
    """What an occluded ray reports. ``"drop"`` makes it a miss, which is what a driver that
    filters self-returns does, and what leaves a held height map's cell held rather than
    measured. ``"hit"`` returns the point on the robot -- OmniPerception's behaviour, right
    when the moving geometry is the thing being sensed rather than the sensor's own body."""

    occluder_max_distance: float = 2.0
    """Metres. Rays are only tested against the occluder this far out. Self-occlusion happens
    within a body length, so the default is generous; it also keeps a shallow ray from finding
    a *neighbouring* environment's robot, which shares the mesh and stands 2.5 m away."""

    occluder_min_distance: float = 0.0
    """Metres. Occluder hits nearer than this are ignored -- a blind radius for the sensor's
    own housing. Prefer ``occluder_drop_geometry_containing_sensor``, which is exact and costs
    nothing at runtime; reach for this only when the mount sits flush against a shell rather
    than inside it, so no single geometry encloses it."""

    occluder_drop_geometry_containing_sensor: bool = True
    """Drop any geometry that encloses the sensor mount, naming it at init.

    Warp does not cull back faces, so a sensor inside a shell would range on the inside of that
    shell and block every ray. Dropping it is also the physically right answer: the housing a
    LiDAR looks out of is the one part of the robot it can never see. Turn off only when the
    mount is genuinely outside every body and the test is costing time at startup."""

    occluder_tessellation: int = 12
    """Radial segments for analytic cylinders/capsules/cones. Multiplied by bodies and by
    environments, so this is a real memory and refit-time knob, not a cosmetic one."""

    occluder_sphere_subdivisions: int = 1
    """Icosphere subdivisions for analytic spheres. 1 gives 42 vertices, 2 gives 162."""

    occluder_max_vertices_per_geom: int = 256
    """Authored ``Mesh`` collision geometry above this vertex count is replaced by its convex
    hull, and by its bounding box if the hull is still over. Guards against an asset whose
    "collision" mesh is really the visual one: a few thousand vertices per link is unremarkable
    on its own and catastrophic once multiplied by 4,096 environments."""

    occluder_use_visual_geometry: bool = False
    """Fall back to visual geometry for bodies with no collision prims. Off by default -- the
    collision approximation is what the physics actually collides with, and it is the cheaper
    and more predictable of the two."""


class RobotOccluderMixin:
    """Adds a moving robot-shaped mesh to a ``RayCaster``. Mix in *before* the sensor class.

    Requires the 2026-era IsaacLab ``RayCaster``, which stores the world-frame rays it cast in
    ``_ray_starts_w`` / ``_ray_directions_w``. Reusing those is what keeps this class from
    having to restate the ``ray_alignment`` transform and the drift handling, so there is
    nothing here to fall out of step with upstream.
    """

    cfg: RobotOccluderCfgMixin

    def __init__(self, cfg):
        super().__init__(cfg)
        self._occluder_mesh: wp.Mesh | None = None
        self._occluder_views: list = []
        self._occluder_indices = np.zeros(0, dtype=np.int32)

    """
    Properties.
    """

    @property
    def occluder_num_triangles(self) -> int:
        """Triangles in the occluder mesh, summed over every environment. 0 when off."""
        return 0 if self._occluder_mesh is None else len(self._occluder_indices) // 3

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        self._initialize_occluder()

    def _initialize_occluder(self):
        self._occluder_mesh = None
        self._occluder_views = []
        if not self.cfg.occluder_enabled or not self.cfg.occluder_prim_path:
            return

        env_ns = self._resolve_env_regex_ns()
        env0_ns = env_ns.replace("env_.*", "env_0")
        root_pattern = self.cfg.occluder_prim_path.replace("{ENV_REGEX_NS}", env_ns)
        root_env0 = root_pattern.replace(env_ns, env0_ns)

        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(root_env0)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(
                f"Occluder root prim not found at '{root_env0}' (from occluder_prim_path="
                f"'{self.cfg.occluder_prim_path}'). Point it at the articulation root, or set"
                " it to None to disable self-occlusion."
            )
        sensor_pos_w = self._sensor_mount_world_position(stage, env0_ns)

        # Bodies are collected from env_0 alone and replicated: every environment holds the
        # same articulation, so the geometry is identical and only the poses differ.
        patterns = [re.compile(p) for p in self.cfg.occluder_body_names]
        base_points: list[np.ndarray] = []
        base_faces: list[np.ndarray] = []
        body_patterns: list[str] = []

        for body_prim in self._iter_rigid_bodies(root_prim):
            if not any(p.fullmatch(body_prim.GetName()) for p in patterns):
                continue
            geometry = self._body_geometry(body_prim, sensor_pos_w)
            if geometry is None:
                continue
            points, faces = geometry
            base_points.append(points)
            base_faces.append(faces)
            body_patterns.append(body_prim.GetPath().pathString.replace(env0_ns, env_ns))

        if not body_patterns:
            raise RuntimeError(
                f"No occluding bodies matched under '{root_env0}' (occluder_body_names="
                f"{self.cfg.occluder_body_names}). Either the filter excludes everything or"
                " every body's geometry encloses the sensor."
            )

        # One physics view per body, each expanding over the environments in env order -- the
        # same ordering assumption the RayCaster makes for its own sensor view.
        self._occluder_views = [
            self._physics_sim_view.create_rigid_body_view(pattern.replace(".*", "*"))
            for pattern in body_patterns
        ]
        num_envs = self._occluder_views[0].count
        for pattern, view in zip(body_patterns, self._occluder_views):
            if view.count != num_envs:
                raise RuntimeError(
                    f"Occluder body '{pattern}' expanded to {view.count} instances, not the"
                    f" {num_envs} of the first body; the articulation is not uniform."
                )

        # Assemble the combined mesh body-major, environment-minor, matching the order
        # torch.cat over the views produces.
        all_points: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        instance_counts: list[int] = []
        offset = 0
        for points, faces in zip(base_points, base_faces):
            for _ in range(num_envs):
                all_points.append(points)
                all_faces.append(faces + offset)
                instance_counts.append(len(points))
                offset += len(points)

        points_np = np.concatenate(all_points).astype(np.float32)
        self._occluder_indices = np.concatenate(all_faces).astype(np.int32).reshape(-1)
        # Vertex -> instance gather, precomputed: a step then costs an index_select instead of
        # a repeat_interleave that has to re-read the counts every time.
        self._occluder_vertex_instance = torch.repeat_interleave(
            torch.arange(len(instance_counts), device=self._device),
            torch.tensor(instance_counts, device=self._device, dtype=torch.long),
        )
        self._occluder_base_points = torch.from_numpy(points_np).to(self._device)
        # The warp mesh aliases this buffer, so the per-step update writes in place and only
        # the BVH still has to be told -- no reallocation, no host round-trip.
        self._occluder_points_w = self._occluder_base_points.clone()

        self._write_occluder_points()
        self._occluder_mesh = wp.Mesh(
            points=wp.from_torch(self._occluder_points_w, dtype=wp.vec3),
            indices=wp.array(self._occluder_indices, dtype=wp.int32, device=self._device),
        )

        per_robot = sum(len(p) for p in base_points)
        print(
            f"[INFO] Robot occluder for '{self.cfg.prim_path}': {len(body_patterns)} bodies x"
            f" {num_envs} envs, {len(points_np)} vertices,"
            f" {len(self._occluder_indices) // 3} triangles ({per_robot} vertices per robot),"
            f" mode={self.cfg.occluder_mode}, range="
            f"{self.cfg.occluder_min_distance}..{self.cfg.occluder_max_distance} m."
        )

    def _resolve_env_regex_ns(self) -> str:
        """``/World/envs/env_.*`` recovered from the sensor's own resolved prim path.

        The scene substitutes ``{ENV_REGEX_NS}`` into ``prim_path`` when it spawns the sensor,
        but only into that field -- our own paths have to be resolved by hand.
        """
        match = re.match(r"^(.*/env_[^/]*)", self.cfg.prim_path)
        if match is None:
            raise RuntimeError(
                f"Cannot recover the environment namespace from prim_path '{self.cfg.prim_path}';"
                " the occluder needs it to find the robot in every environment."
            )
        return match.group(1)

    def _sensor_mount_world_position(self, stage, env0_ns: str) -> np.ndarray:
        """Where the sensor sits in env_0's rest pose, for the enclosure test."""
        prim_path = self.cfg.prim_path.replace(self._resolve_env_regex_ns(), env0_ns)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Sensor parent prim not found at '{prim_path}'.")
        rot, trans = _usd_local_to_world(prim)
        return np.asarray(self.cfg.offset.pos, dtype=np.float64) @ rot + trans

    @staticmethod
    def _children(prim) -> list:
        """Children, descending into instance proxies.

        IsaacLab spawns robot links as *instanceable* references, so each link Xform is an
        instance whose geometry lives in a shared prototype. ``GetChildren()`` stops at the
        instance boundary and reports the link as childless, which reads exactly like an
        asset with no collision geometry at all. Instance proxies are read-only, which is all
        this module needs: vertices and transforms both come back correctly through them.
        """
        from pxr import Usd

        return prim.GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))

    def _iter_rigid_bodies(self, root_prim) -> list:
        """Rigid bodies under ``root_prim``, without descending into one another.

        Fixed links merged away by the URDF import leave their collision geometry parented to
        the surviving body, so walking bodies rather than links keeps that geometry and still
        costs one pose lookup per moving part.
        """
        from pxr import UsdPhysics

        if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return [root_prim]
        bodies = []
        stack = list(self._children(root_prim))
        while stack:
            prim = stack.pop()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(prim)
                continue
            stack.extend(self._children(prim))
        return sorted(bodies, key=lambda p: p.GetPath().pathString)

    def _body_geometry(self, body_prim, sensor_pos_w: np.ndarray):
        """One body's geometry as ``(points, faces)`` in the body frame, or ``None``."""
        from pxr import UsdPhysics

        collision_geoms, visual_geoms = [], []
        stack = list(self._children(body_prim))
        while stack:
            prim = stack.pop()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue  # a nested body owns its own geometry
            if prim.GetTypeName() in _SUPPORTED_GPRIM_TYPES:
                (collision_geoms if prim.HasAPI(UsdPhysics.CollisionAPI) else visual_geoms).append(prim)
            stack.extend(self._children(prim))

        geoms = collision_geoms or (visual_geoms if self.cfg.occluder_use_visual_geometry else [])
        if not geoms:
            return None

        body_rot, body_trans = _usd_local_to_world(body_prim)
        body_rot_inv = np.linalg.inv(body_rot)

        points_parts: list[np.ndarray] = []
        faces_parts: list[np.ndarray] = []
        offset = 0
        for geom in geoms:
            mesh = _gprim_to_trimesh(
                geom, self.cfg.occluder_tessellation, self.cfg.occluder_sphere_subdivisions
            )
            if mesh is None or len(mesh.vertices) == 0:
                continue
            mesh = self._cap_vertices(mesh, geom)
            geom_rot, geom_trans = _usd_local_to_world(geom)
            verts_world = np.asarray(mesh.vertices) @ geom_rot + geom_trans
            if self._encloses_sensor(mesh, verts_world, sensor_pos_w, geom):
                continue
            points_parts.append((verts_world - body_trans) @ body_rot_inv)
            faces_parts.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
            offset += len(verts_world)

        if not points_parts:
            return None
        return np.concatenate(points_parts), np.concatenate(faces_parts)

    def _encloses_sensor(self, mesh, verts_world: np.ndarray, sensor_pos_w: np.ndarray, geom) -> bool:
        """Does this geometry contain the sensor mount? See the module docstring.

        The cheap bounding-box test comes first so the ray-parity test only runs for the one or
        two geometries that could plausibly enclose the mount.
        """
        if not self.cfg.occluder_drop_geometry_containing_sensor:
            return False
        low, high = verts_world.min(axis=0), verts_world.max(axis=0)
        if np.any(sensor_pos_w < low) or np.any(sensor_pos_w > high):
            return False
        import trimesh

        world = trimesh.Trimesh(vertices=verts_world, faces=mesh.faces, process=False)
        try:
            inside = bool(world.contains(sensor_pos_w.reshape(1, 3))[0])
        except Exception as error:  # noqa: BLE001 -- a non-watertight shell is not fatal here
            print(
                f"[WARN] Occluder: enclosure test failed for '{geom.GetPath()}' ({error});"
                " keeping it. If the map comes back entirely unobserved, this is why --"
                " set occluder_min_distance to clear the housing."
            )
            return False
        if inside:
            print(
                f"[INFO] Occluder: the sensor mount is inside '{geom.GetPath()}'; dropping it."
                " A LiDAR cannot range on the housing it looks out of, and warp does not cull"
                " back faces, so keeping it would block every ray."
            )
        return inside

    def _cap_vertices(self, mesh, geom):
        """Convex hull, then bounding box, until the geometry fits the per-geom budget."""
        import trimesh

        if len(mesh.vertices) <= self.cfg.occluder_max_vertices_per_geom:
            return mesh
        hull = mesh.convex_hull
        if len(hull.vertices) <= self.cfg.occluder_max_vertices_per_geom:
            print(
                f"[INFO] Occluder: '{geom.GetPath()}' has {len(mesh.vertices)} vertices;"
                f" using its convex hull ({len(hull.vertices)})."
            )
            return hull
        low, high = mesh.bounds
        box = trimesh.creation.box(extents=high - low)
        box.apply_translation((low + high) / 2.0)
        print(
            f"[INFO] Occluder: '{geom.GetPath()}' has {len(mesh.vertices)} vertices and a"
            f" {len(hull.vertices)}-vertex hull; using its bounding box."
        )
        return box

    def _write_occluder_points(self):
        """Move every vertex onto its body's current pose. Writes in place: the warp mesh's
        points alias this buffer, so only the BVH still has to be told."""
        transforms = torch.cat([view.get_transforms() for view in self._occluder_views], dim=0)
        idx = self._occluder_vertex_instance
        quat = convert_quat(transforms[:, 3:7], to="wxyz")
        torch.add(
            quat_apply(quat[idx], self._occluder_base_points),
            transforms[:, 0:3][idx],
            out=self._occluder_points_w,
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        if self._occluder_mesh is None:
            return

        self._write_occluder_points()
        self._occluder_mesh.refit()

        ray_starts_w = self._ray_starts_w[env_ids]
        ray_directions_w = self._ray_directions_w[env_ids]
        occluder_hits, occluder_dist, _, _ = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.occluder_max_distance,
            mesh=self._occluder_mesh,
            return_distance=True,
        )
        terrain_hits = self._data.ray_hits_w[env_ids]
        # A missed terrain ray is inf and stays inf through the norm, so it loses this
        # comparison to any occluder hit -- which is right: the body blocks a ray that would
        # otherwise have flown on. A missed *occluder* ray is inf for the same reason and wins.
        terrain_dist = torch.linalg.norm(terrain_hits - ray_starts_w, dim=-1)
        blocked = occluder_dist < terrain_dist
        if self.cfg.occluder_min_distance > 0.0:
            blocked &= occluder_dist >= self.cfg.occluder_min_distance

        if self.cfg.occluder_mode == "hit":
            self._data.ray_hits_w[env_ids] = torch.where(blocked.unsqueeze(-1), occluder_hits, terrain_hits)
        else:
            self._data.ray_hits_w[env_ids] = torch.where(
                blocked.unsqueeze(-1), torch.inf, terrain_hits
            )

        # LidarSensor derives .distances from the hits inside the super() call above, so it is
        # stale by now. Recompute rather than leave a buffer that disagrees with itself.
        distances = getattr(self._data, "distances", None)
        if distances is not None:
            hits = self._data.ray_hits_w[env_ids]
            sensor_pos = self._get_true_sensor_pos()[env_ids].unsqueeze(1)
            new_distances = torch.norm(hits - sensor_pos, dim=2)
            new_distances[torch.isinf(hits).any(dim=2)] = self.cfg.max_distance
            distances[env_ids] = new_distances


class OccludedRollingLivoxSensor(RobotOccluderMixin, RollingLivoxSensor):
    """The rolling-window MID-360 with the robot's own body blocking its rays.

    Mixin first: its ``_update_buffers_impl`` has to run *around* ``LidarSensor``'s, which is
    what lets it correct the hits the base class has just written.
    """

    cfg: OccludedRollingLivoxSensorCfg


@configclass
class OccludedRollingLivoxSensorCfg(RobotOccluderCfgMixin, LidarSensorCfg):
    """``RollingLivoxSensorCfg`` plus the occluder fields.

    Declared after the sensor so ``class_type`` can be a real default: ``configclass`` bakes
    field defaults into the generated ``__init__``, so patching the attribute afterwards would
    leave every instance holding a placeholder.
    """

    class_type: type = OccludedRollingLivoxSensor
