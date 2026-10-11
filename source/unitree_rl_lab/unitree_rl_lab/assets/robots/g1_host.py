"""Unitree G1 23dof as modelled in HoST (OpenRobotLab/HoST, MIT).

``g1_23dof_host/g1_23dof.urdf`` and the meshes it references are copied unchanged from
HoST ``legged_gym/resources/robots/g1`` (commit 70bb580). Compared with unitree_rl_lab's
G1 23dof USD this model has a different head mesh (it collides with the ground when lying),
nine collision spheres per sole instead of four, 34.1 kg instead of 32.1 kg, and the URDF
effort limits (hip 88 / knee 139 / ankle 50 / arms 25 Nm).

The actuators follow HoST's legged_gym setup: an explicit PD computed outside the physics
engine and clipped to the URDF effort limit (``IdealPDActuatorCfg``), HoST's gains, and
armature 0.01. The URDF's keyframe / auxiliary links (massless markers used by HoST's
rewards) are merged away on import; the get-up task computes the head point from the torso.
"""

import getpass
import os
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG, UnitreeArticulationCfg, UnitreeUrdfFileCfg

# HoST's URDF without its massless keyframe / auxiliary marker links (see the header of that file)
G1_23DOF_HOST_URDF = os.path.join(os.path.dirname(__file__), "g1_23dof_host", "g1_23dof_nomarkers.urdf")
# converted USD cache; per user, since Isaac Lab's default /tmp/IsaacLab may belong to another user
G1_23DOF_HOST_USD_DIR = os.path.join(tempfile.gettempdir(), f"IsaacLab-{getpass.getuser()}", "g1_23dof_host")

UNITREE_G1_23DOF_HOST_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=G1_23DOF_HOST_URDF,
        usd_dir=G1_23DOF_HOST_USD_DIR,
        fix_base=False,
        merge_fixed_joints=True,
        # legged_gym: density 0.001 for links without inertia, cylinders as capsules, self collisions on
        link_density=0.001,
        replace_cylinders_with_capsules=True,
        self_collision=True,
        collider_type="convex_hull",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # HoST: TGS, 8 position / 1 velocity iterations
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_joint": 0.8,
            ".*_hip_(roll|yaw)_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            ".*_shoulder_.*": 0.0,
            ".*_wrist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # HoST g1_config_ground.py: kp / kd = hip 150 / 4, knee 200 / 6, ankle 40 / 2, others 100 / 4
        "hips": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=88.0,
            velocity_limit=32.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.01,
        ),
        "knees": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=139.0,
            velocity_limit=20.0,
            stiffness=200.0,
            damping=6.0,
            armature=0.01,
        ),
        "ankles": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit=50.0,
            velocity_limit=37.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit=88.0,
            velocity_limit=32.0,
            stiffness=100.0,
            damping=4.0,
            armature=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit=25.0,
            velocity_limit=37.0,
            stiffness=100.0,
            damping=4.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=UNITREE_G1_23DOF_CFG.joint_sdk_names,
)
