"""Configuration for the ALTs Anaguma quadruped (wall-cross port, 2026-10-14).

Built from the NAS copy of the URDF that S.Eguchi published on 2026-10-13
(``192.168.100.4:/data/ort/store/models/Anaguma``). Unlike ``Anaguma_v1.zip`` used for the
long-jump port, that file is already in REP103 axes with base_link at the CoM and Go2-style
leg names, so ``/home/tanaka/isaacsim/anaguma/tools/normalize_nas_urdf.py`` only has to:

  * replace the 1000000 placeholder effort / velocity limits,
  * rename base_link -> base (the task configs match the trunk as "base"),
  * swap the visual-mesh colliders for a narrow trunk box, leg cylinders and foot spheres.
    With the raw meshes the trunk's convex hull swallows the leg roots and ``base_contact``
    fires on the first step of every episode (measured on the long-jump port).

Joint axes and angle signs are identical to the long-jump port's normalised URDF (checked:
same thigh axis after the old frame rotation), so its IK stance below carries over.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg

ANAGUMA_URDF = "/home/tanaka/isaacsim/anaguma/anaguma_nas/anaguma.urdf"

# Actuators. ALTs' own Isaac config (tsubame_isaac_env ANAGUMA_CFG, quoted in the NAS
# ``mjcf/anaguma_scene_position.xml``) trains with hip/thigh 80 / 4.0, calf 60 / 2.0 and
# armature 8e-3, peak torque 70 N.m hip/thigh and 60 N.m knee. The same file's <position>
# elements say kp=120 for every joint, contradicting its own comment -- the comment is the
# one that names its source, so it is what is used here. Unconfirmed, ask ALTs.
#
# Speed ceiling 13.4 / 15.6 rad/s is from the internal doc「Anagumaの性能限界」, which does
# not say which joint is which; hip/thigh = 13.4 and calf = 15.6 is an assumption. X1 is
# Go2's knee-point ratio (13.5 / 30 = 0.45) applied to those numbers.
ANAGUMA_HIP_SPEED = 13.4
ANAGUMA_CALF_SPEED = 15.6

ANAGUMA_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ANAGUMA_URDF,
        fix_base=False,
        # Keep the ``*_foot`` links as bodies so the ``.*_foot`` contact sensors resolve.
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # On: the collider swap above is what makes this safe. Folded legs passing
            # through the trunk would invalidate a hind-leg stand and a wall landing.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Trunk (CoM) at 0.38 m, 1 cm of drop to settle onto the feet.
        pos=(0.0, 0.0, 0.39),
        # Each foot directly under its own thigh pivot with the trunk at 0.38 m (68% of the
        # 0.558 m leg, Go2's ratio), solved on the NAS MJCF. In the NAS model all four legs
        # share one geometry -- knees point backward at zero, as in ALTs' all-zero "stand"
        # keyframe -- so one pair of angles serves every leg. The long-jump port's per-leg
        # "mirrored" angles (rear thigh +0.0517 / calf +0.446) put the rear knees 8 cm off the
        # floor here and the thigh cylinders dragged on the ground (measured 971 N in Isaac).
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -0.02,
            ".*_calf_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_thigh": unitree_actuators.UnitreeActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
            Fs=0.01,
            Y1=70.0,
            Y2=70.0,
            X1=6.0,
            X2=ANAGUMA_HIP_SPEED,
            armature=8e-3,
        ),
        "calf": unitree_actuators.UnitreeActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            stiffness=60.0,
            damping=2.0,
            friction=0.0,
            Fs=0.01,
            Y1=60.0,
            Y2=60.0,
            X1=7.0,
            X2=ANAGUMA_CALF_SPEED,
            armature=8e-3,
        ),
    },
    # fmt: off
    # Unitree SDK ordering. The URDF body order is FL, FR, RL, RR -- index joints by name.
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ],
    # fmt: on
)
