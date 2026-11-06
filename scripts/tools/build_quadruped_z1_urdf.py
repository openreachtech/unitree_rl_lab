"""Merge an official Unitree quadruped (Aliengo or B1) and Z1 (arm) descriptions into a single URDF.

The quadruped and Z1 models in ``unitree_ros`` are defined separately. This script combines them the
same way Unitree's own ``aliengoZ1_description`` does: the Z1 base (``link00``) is attached to the
quadruped ``trunk`` with a fixed joint (``trunk_arm_joint``). Optionally the Unitree gripper
(``gripperStator`` / ``gripperMover`` / ``jointGripper``) is added, which is missing from the
standalone ``z1.urdf`` and only exists in the xacro.

Isaac Sim specific adjustments:

* Fixed joints of the feet and the gripper stator get ``dont_collapse="true"`` so Isaac Sim keeps them
  as separate bodies when merging fixed joints.
* COLLADA meshes declaring a non-metre ``<unit>`` (aliengo ``calf.dae``) are normalized to metres and
  the factor is moved into the URDF mesh ``scale``.
* URDF ``<material>`` is dropped from mesh visuals so the meshes keep their own look (trunk texture).
* ``link01`` gets the collision cylinder the aliengoZ1 xacro has but ``z1.urdf`` lacks.

Output is a self-contained directory (URDF + copied meshes, relative mesh paths), so it can be fed
directly to Isaac Lab's ``scripts/tools/convert_urdf.py``::

    <out_dir>/
        <base>_z1.urdf
        meshes/<base>/*.dae
        meshes/z1/{visual,collision}/*

Usage::

    python scripts/tools/build_quadruped_z1_urdf.py \
        --unitree-ros ~/unitree/unitree_ros \
        --base aliengo  # or b1
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

# --- Z1 gripper (from z1_description/xacro/const.xacro and aliengoZ1_description/xacro/robot.xacro) ---
GRIPPER_STATOR = dict(
    mass=0.52603655,
    com=(0.04764427, -0.00035819, -0.00249162),
    inertia=dict(ixx=0.00038683, ixy=-0.00000359, ixz=0.00007662, iyy=0.00068614, iyz=0.00000209, izz=0.00066293),
)
GRIPPER_MOVER = dict(
    mass=0.27621302,
    com=(0.01320633, 0.00476708, 0.00380534),
    inertia=dict(ixx=0.00017716, ixy=0.00001683, ixz=-0.00001786, iyy=0.00026787, iyz=0.00000262, izz=0.00035728),
)
GRIPPER_LIMIT = dict(lower=-math.pi * 90 / 180, upper=0.0, effort=30.0, velocity=3.1415)

# Supported quadruped bases. Arm mount defaults follow aliengoZ1_description/xacro/const.xacro:
#   x = trunk_length / 2 - 0.07, z = top of trunk.
BASES = {
    # aliengo: trunk_length=0.647, trunk_height=0.112 -> official mount z = trunk_height / 2.
    "aliengo": dict(package="aliengo_description", urdf="urdf/aliengo.urdf", arm_x=0.647 / 2 - 0.07, arm_z=0.112 / 2),
    # b1: top surface of the trunk mesh (trunkb.dae) is flat at z = 0.0852 m.
    "b1": dict(package="b1_description", urdf="xacro/b1.urdf", arm_x=0.647 / 2 - 0.07, arm_z=0.0852),
}


def _fmt(*v: float) -> str:
    return " ".join(f"{x:.10g}" for x in v)


def _origin(parent: ET.Element, xyz=(0, 0, 0), rpy=(0, 0, 0)) -> ET.Element:
    return ET.SubElement(parent, "origin", xyz=_fmt(*xyz), rpy=_fmt(*rpy))


def _inertial(parent: ET.Element, spec: dict) -> None:
    inertial = ET.SubElement(parent, "inertial")
    _origin(inertial, spec["com"])
    ET.SubElement(inertial, "mass", value=_fmt(spec["mass"]))
    ET.SubElement(inertial, "inertia", {k: _fmt(v) for k, v in spec["inertia"].items()})


def _mesh_link(name: str, visual: str, collision: str, spec: dict) -> ET.Element:
    link = ET.Element("link", name=name)
    vis = ET.SubElement(link, "visual")
    _origin(vis)
    ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=visual, scale="1 1 1")
    col = ET.SubElement(link, "collision")
    _origin(col)
    ET.SubElement(ET.SubElement(col, "geometry"), "mesh", filename=collision, scale="1 1 1")
    _inertial(link, spec)
    return link


def _joint(name, jtype, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0)) -> ET.Element:
    joint = ET.Element("joint", name=name, type=jtype)
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    return joint


def gripper_elements() -> list[ET.Element]:
    z1 = "package://z1_description/meshes"
    stator = _mesh_link(
        "gripperStator", f"{z1}/visual/z1_GripperStator.dae", f"{z1}/collision/z1_GripperStator.STL", GRIPPER_STATOR
    )
    mover = _mesh_link(
        "gripperMover", f"{z1}/visual/z1_GripperMover.dae", f"{z1}/collision/z1_GripperMover.STL", GRIPPER_MOVER
    )
    j_stator = _joint("gripperStator", "fixed", "link06", "gripperStator", xyz=(0.051, 0, 0))
    j_grip = _joint("jointGripper", "revolute", "gripperStator", "gripperMover", xyz=(0.049, 0, 0))
    ET.SubElement(j_grip, "axis", xyz="0 1 0")
    ET.SubElement(j_grip, "dynamics", damping="1.0", friction="1.0")
    ET.SubElement(j_grip, "limit", {k: _fmt(v) for k, v in GRIPPER_LIMIT.items()})
    return [j_stator, stator, j_grip, mover]


def add_link01_collision(robot: ET.Element) -> None:
    """``z1.urdf`` has no collision on link01; aliengoZ1_description/xacro/robot.xacro does
    (cylinder, length 2 * motor_height, radius motor_diameter, at the link origin)."""
    link = robot.find("link[@name='link01']")
    if link is None or link.find("collision") is not None:
        return
    col = ET.Element("collision")
    _origin(col)
    ET.SubElement(ET.SubElement(col, "geometry"), "cylinder", length=_fmt(2 * 0.051), radius=_fmt(0.065 / 2))
    # URDF convention: visual, collision, inertial.
    link.insert(list(link).index(link.find("visual")) + 1, col)


def strip_mesh_visual_materials(robot: ET.Element) -> None:
    """Drop URDF ``<material>`` from mesh visuals so the meshes keep their own look.

    The Unitree URDFs tag every visual with a flat colour (aliengo: ``orange``). Isaac Sim's importer
    lets that override the mesh's own materials, which hides the trunk texture
    (``trunk_uv_base_final.png``) and the grey legs. Primitive visuals keep their colour.
    """
    for visual in robot.iter("visual"):
        if visual.find("geometry/mesh") is not None:
            for mat in visual.findall("material"):
                visual.remove(mat)
    used = {m.get("name") for v in robot.iter("visual") for m in v.findall("material")}
    for mat in robot.findall("material"):
        if mat.get("name") not in used:
            robot.remove(mat)


def strip_sim_tags(robot: ET.Element) -> None:
    """Remove Gazebo / ros_control specific tags that are irrelevant for Isaac Sim."""
    for child in list(robot):
        if child.tag in ("gazebo", "transmission"):
            robot.remove(child)


def rewrite_mesh_paths(robot: ET.Element, base: str) -> None:
    mapping = {
        f"package://{BASES[base]['package']}/meshes/": f"meshes/{base}/",
        "package://z1_description/meshes/": "meshes/z1/",
    }
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename", "")
        for src, dst in mapping.items():
            if fn.startswith(src):
                mesh.set("filename", dst + fn[len(src) :])
                break
        else:
            raise ValueError(f"Unexpected mesh path: {fn}")


def normalize_collada_units(robot: ET.Element, out_dir: str) -> None:
    """Bake non-metre COLLADA ``<unit>`` into the URDF mesh ``scale``.

    Some Unitree meshes (e.g. aliengo ``calf.dae``) declare ``<unit meter="0.0254">``. Loaders disagree
    on whether to honour it (trimesh ignores it, others apply it), so rewrite the copied file to
    ``meter="1"`` and multiply the URDF ``scale`` instead -- unambiguous for every loader.
    """
    unit_re = re.compile(r'<unit\b[^>]*\bmeter="([0-9.eE+-]+)"[^>]*?(?:/>|>\s*</unit>)')
    scales = {}
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        if not fn.lower().endswith(".dae"):
            continue
        if fn not in scales:
            path = os.path.join(out_dir, fn)
            with open(path) as f:
                text = f.read()
            m = unit_re.search(text)
            scales[fn] = float(m.group(1)) if m else 1.0
            if scales[fn] != 1.0:
                with open(path, "w") as f:
                    f.write(unit_re.sub('<unit name="meter" meter="1"/>', text, count=1))
                print(f"  normalized COLLADA unit of {fn}: meter={scales[fn]} -> scale")
        k = scales[fn]
        if k != 1.0:
            old = [float(x) for x in mesh.get("scale", "1 1 1").split()]
            mesh.set("scale", _fmt(*(x * k for x in old)))


def build(args: argparse.Namespace) -> str:
    robots = os.path.join(os.path.expanduser(args.unitree_ros), "robots")
    base = BASES[args.base]
    base_dir = os.path.join(robots, base["package"])
    z1_dir = os.path.join(robots, "z1_description")
    arm_x = base["arm_x"] if args.arm_x is None else args.arm_x
    arm_z = base["arm_z"] if args.arm_z is None else args.arm_z

    quad = ET.parse(os.path.join(base_dir, base["urdf"])).getroot()
    z1 = ET.parse(os.path.join(z1_dir, "xacro", "z1.urdf")).getroot()
    strip_sim_tags(quad)
    strip_sim_tags(z1)

    merged = ET.Element("robot", name=f"{args.base}_z1_description")

    # Materials first (the quadruped defines them), then quadruped links/joints.
    for el in quad:
        merged.append(el)

    # Mount Z1 on top of the quadruped trunk.
    merged.append(
        _joint(
            "trunk_arm_joint",
            "fixed",
            "trunk",
            "link00",
            xyz=(arm_x, args.arm_y, arm_z),
            rpy=(0.0, 0.0, args.arm_yaw),
        )
    )

    # Z1 links/joints, dropping its "world" anchor (base_static_joint).
    for el in z1:
        if el.tag == "link" and el.get("name") == "world":
            continue
        if el.tag == "joint" and el.get("name") == "base_static_joint":
            continue
        if el.tag == "material" and merged.find(f"material[@name='{el.get('name')}']") is not None:
            continue
        merged.append(el)

    add_link01_collision(merged)
    strip_mesh_visual_materials(merged)

    if not args.no_gripper:
        merged.extend(gripper_elements())

    rewrite_mesh_paths(merged, args.base)

    # Isaac Sim's URDF importer merges links joined by fixed joints (Isaac Lab's default
    # merge_fixed_joints=True). That is what we want for rotors / IMU / cameras / trunk->base /
    # link00->base, but the feet must stay separate bodies for contact sensing, and the gripper
    # stator is kept as the end-effector frame.
    for joint in merged.findall("joint"):
        name = joint.get("name")
        if joint.get("type") == "fixed" and (name.endswith("_foot_fixed") or name == "gripperStator"):
            joint.set("dont_collapse", "true")

    # Sanity checks: unique names and a single root.
    links = [l.get("name") for l in merged.findall("link")]
    joints = [j.get("name") for j in merged.findall("joint")]
    assert len(links) == len(set(links)), "duplicate link names"
    assert len(joints) == len(set(joints)), "duplicate joint names"
    children = {j.find("child").get("link") for j in merged.findall("joint")}
    roots = [l for l in links if l not in children]
    assert roots == ["base"], f"expected single root 'base', got {roots}"

    # Write output.
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mesh_out = os.path.join(out_dir, "meshes")
    for name, src in ((args.base, os.path.join(base_dir, "meshes")), ("z1", os.path.join(z1_dir, "meshes"))):
        dst = os.path.join(mesh_out, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    normalize_collada_units(merged, out_dir)

    ET.indent(merged, space="  ")
    urdf_path = os.path.join(out_dir, f"{args.base}_z1.urdf")
    with open(urdf_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write("<!-- Generated by unitree_rl_lab/scripts/tools/build_quadruped_z1_urdf.py. Do not edit by hand. -->\n")
        f.write(ET.tostring(merged, encoding="unicode"))
        f.write("\n")

    movable = [j.get("name") for j in merged.findall("joint") if j.get("type") != "fixed"]
    mass = sum(float(m.get("value")) for m in merged.iter("mass"))
    print(f"Wrote {urdf_path}")
    print(f"  links: {len(links)}, joints: {len(joints)}, actuated joints ({len(movable)}): {movable}")
    print(f"  total mass: {mass:.2f} kg")
    return urdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unitree-ros", default="~/unitree/unitree_ros", help="Path to the unitree_ros repository.")
    parser.add_argument("--base", choices=sorted(BASES), default="aliengo", help="Quadruped to mount the Z1 on.")
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default: ~/unitree/unitree_model/<Base>Z1/urdf)."
    )
    parser.add_argument(
        "--arm-x", type=float, default=None, help="Arm mount x in trunk frame [m] (default: trunk_length/2 - 0.07)."
    )
    parser.add_argument("--arm-y", type=float, default=0.0, help="Arm mount y in trunk frame [m].")
    parser.add_argument(
        "--arm-z", type=float, default=None, help="Arm mount z in trunk frame [m] (default: trunk top)."
    )
    parser.add_argument("--arm-yaw", type=float, default=0.0, help="Arm mount yaw [rad].")
    parser.add_argument("--no-gripper", action="store_true", help="Do not add the Unitree gripper.")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f"~/unitree/unitree_model/{ {'aliengo': 'Aliengo', 'b1': 'B1'}[args.base] }Z1/urdf"
    build(args)


if __name__ == "__main__":
    main()
