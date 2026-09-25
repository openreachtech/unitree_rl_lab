"""Configuration for the ALTs Anaguma quadruped.

Anaguma is not a Unitree machine, but it has the same 12-DoF layout, so the task
configs in this repository port across once the robot is described here. The URDF
shipped by ALTs (``Anaguma_v1.zip``, received 2026-09-13) is not usable as-is; the
normalised copy referenced below is produced by
``/home/tanaka/isaacsim/anaguma/tools/normalize_urdf.py``, which fixes three things:

  * the base_link frame is "forward = +z, left = -y, up = +x" in the shipped file,
  * every joint limit is the placeholder 1000000 for both effort and velocity,
  * link and joint names are CAD exports (``FL_link1``, ``base_link_rev-1``), so none
    of the ``.*_foot`` / ``.*_calf_joint`` / ``base`` patterns this repository uses
    would resolve.

See project memory プロジェクト_Anaguma移植の基礎.md for the measurements behind the
numbers here and for what is still unverified.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg

ANAGUMA_URDF = "/home/tanaka/isaacsim/anaguma/anaguma_description/anaguma.urdf"

# Actuator model.
#
# The internal doc「Anagumaの性能限界：早く走らせる」measured 70 N.m of torque and a
# joint-speed ceiling of 13.4 / 15.6 rad/s, with an observed average of 6.0 rad/s. It
# does NOT say which joint is which, so hip/thigh = 13.4 and calf = 15.6 is an
# assumption -- flagged because a long jump's take-off saturates the knee, so this is
# the one number worth confirming with ALTs before reading much into a jump result.
#
# X1 (knee point of the torque-speed curve) is not published at all. Go2's ratio is
# X1/X2 = 13.5/30 = 0.45, and the same ratio here puts X1 at 6.0 / 7.0 rad/s, which is
# where the doc's measured average sits. Randomised at reset by
# ``randomize_actuator_torque_speed_curve`` in the long-jump task, the same way Go2's is.
ANAGUMA_CALF_SPEED = 15.6
ANAGUMA_HIP_SPEED = 13.4

# PD gains. Go2 runs 25.0 / 0.5 at 15 kg; Anaguma is 24.3 kg on legs 1.33x as long, so
# the torque needed per radian of error is roughly double. 60 / 2.0 is what held the
# robot upright in the MuJoCo static test (3 s, 3.4 deg of tilt, 51.5 N.m peak).
ANAGUMA_STIFFNESS = 60.0
ANAGUMA_DAMPING = 2.0

ANAGUMA_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ANAGUMA_URDF,
        fix_base=False,
        # Keep the ``*_foot`` links as bodies. Every foot in this URDF hangs off a fixed
        # joint, and the default merge would fold them into the calf -- taking with them
        # the ``.*_foot`` contact sensors that feed feet_air_time, feet_slide and the
        # whole jump-landing reward family.
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
            # On, but only because the colliders were replaced first. With the raw visual
            # meshes as colliders the trunk's convex hull swallows the leg roots (the hips
            # mount at y = +-0.1175, inside the trunk's +-0.146 half-width), PhysX exempts
            # only parent/child pairs, and ``base_contact`` fired on the FIRST step of
            # every episode. normalize_urdf.py now gives the trunk a box narrower than the
            # leg roots and the thigh/calf capsules along their own axes, measured down
            # from 92% of random in-range postures self-colliding to 32%, and from 100% to
            # 0% at the stance pose. What is left is mostly large hip abduction, where a
            # real leg would foul the body too.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link sits at the FRONT of the machine, not at the CoM (the CoM is 0.275 m
        # behind it), so this is the height of the front edge rather than of the centre.
        # 0.35 m of stance plus the foot radius: the feet start just touching rather
        # than 7 cm in the air.
        pos=(0.0, 0.0, 0.38),
        # Solved by IK so that each foot sits directly under its own hip with the trunk
        # at 0.35 m. Front and rear legs are mirror images in this URDF -- feeding both
        # the same angles splays the robot front-to-back and it collapses.
        joint_pos={
            ".*_hip_joint": 0.0,
            "F[LR]_thigh_joint": -0.2501,
            "R[LR]_thigh_joint": 0.0517,
            "F[LR]_calf_joint": 0.4395,
            "R[LR]_calf_joint": 0.4460,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "ANAGUMA": unitree_actuators.UnitreeActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=ANAGUMA_STIFFNESS,
            damping=ANAGUMA_DAMPING,
            # PhysX's own joint friction is left at 0 because Fs/Fd below model the same
            # loss once, matching what the MJCF does (frictionloss 0.2 / damping 0.1).
            friction=0.0,
            Fs=0.2,
            Fd=0.1,
            Y1=70.0,
            Y2=70.0,
            X1={".*_hip_joint": 6.0, ".*_thigh_joint": 6.0, ".*_calf_joint": 7.0},
            X2={
                ".*_hip_joint": ANAGUMA_HIP_SPEED,
                ".*_thigh_joint": ANAGUMA_HIP_SPEED,
                ".*_calf_joint": ANAGUMA_CALF_SPEED,
            },
            armature=0.01,
        ),
    },
    # fmt: off
    # Unitree SDK ordering. The URDF/USD body order is FL, FR, RL, RR, which is NOT this
    # order -- indexing joints positionally instead of by name puts the PD controller on
    # the wrong joint (measured: the robot cannot stand).
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ],
    # fmt: on
)
