"""G1 29dof get-up from lying on the back: the 23dof task (``getup_env_cfg``) on the 29dof model.

The 29dof G1 adds waist roll / pitch and wrist pitch / yaw. They take HoST's arm & waist gains
(100 / 4) except the small wrist motors (W4010, 5 Nm: 40 / 1); once standing they are pulled to
0 with the rest of the upper body. The mirroring (``symmetry.py``) already handles them: roll /
yaw joints flip sign, pitch joints keep it.

Trained from scratch, so the head height reward uses a 1.0 m margin (HoST's) around the 1.22 m
target: the 0.3 m margin of the 23dof posture fine-tune gives ~0 reward on the ground and no
gradient to get up.
"""

import copy
import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

from .getup_env_cfg import DEFAULT_JOINT_POS, RobotEnvCfg, RobotPlayEnvCfg, _supine

DEFAULT_JOINT_POS_29 = {
    **DEFAULT_JOINT_POS,
    "waist_(roll|pitch)_joint": 0.0,
    ".*_wrist_(pitch|yaw)_joint": 0.0,
}
UPPER_JOINTS_29 = [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*", "waist_.*_joint"]
ARM_JOINTS_29 = [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"]


def _make_robot_cfg_29() -> ArticulationCfg:
    robot = copy.deepcopy(UNITREE_G1_29DOF_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.pos = (0.0, 0.0, 0.5)
    robot.init_state.rot = _supine(-math.pi / 2)
    robot.init_state.joint_pos = DEFAULT_JOINT_POS_29
    # HoST PD gains (kp / kd): hip 150/4, knee 200/6, ankle 40/2, arms & waist 100/4
    a = robot.actuators
    a["N7520-14.3"].stiffness = {".*_hip_.*": 150.0, "waist_yaw_joint": 100.0}
    a["N7520-14.3"].damping = {".*_hip_.*": 4.0, "waist_yaw_joint": 4.0}
    a["N7520-22.5"].stiffness = {".*_hip_roll_.*": 150.0, ".*_knee_.*": 200.0}
    a["N7520-22.5"].damping = {".*_hip_roll_.*": 4.0, ".*_knee_.*": 6.0}
    # the 29dof groups the ankles with the arms and the waist roll / pitch
    a["N5020-16"].stiffness = {".*_ankle_.*": 40.0, ".*_shoulder_.*|.*_elbow_.*|.*_wrist_roll.*|waist_(roll|pitch)_joint": 100.0}
    a["N5020-16"].damping = {".*_ankle_.*": 2.0, ".*_shoulder_.*|.*_elbow_.*|.*_wrist_roll.*|waist_(roll|pitch)_joint": 4.0}
    a["W4010-25"].stiffness = 40.0
    a["W4010-25"].damping = 1.0
    # unitree_rl_lab's 29dof cfg puts the ankles and the waist roll / pitch in the 25 N*m N5020-16
    # group; the G1 29dof URDF gives them 35 N*m (as the 23dof cfg does for the ankles). At 25 N*m the
    # waist could not hold the torso (it rested on the pitch limit) and the robot stood on its joint
    # limits (locked knees, elbows, wrists).
    a["N5020-16"].effort_limit_sim = {".*_ankle_.*": 35.0, "waist_(roll|pitch)_joint": 35.0,
                                      ".*_shoulder_.*|.*_elbow_.*|.*_wrist_roll.*": 25.0}
    a["N5020-16"].velocity_limit_sim = {".*_ankle_.*": 30.0, "waist_(roll|pitch)_joint": 30.0,
                                        ".*_shoulder_.*|.*_elbow_.*|.*_wrist_roll.*": 37.0}
    return robot


def _apply_29dof(cfg) -> None:
    cfg.scene.robot = _make_robot_cfg_29()
    r = cfg.rewards
    r.task_orientation_head_height.params["target_head_margin"] = 1.0
    r.target_upper_dof_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=UPPER_JOINTS_29)
    r.target_arm_pose_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINTS_29)


@configclass
class Robot29EnvCfg(RobotEnvCfg):
    """G1 29dof get-up from supine."""

    def __post_init__(self):
        super().__post_init__()
        _apply_29dof(self)


@configclass
class Robot29PlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_29dof(self)


# HoST's own reward set (the 23dof run ``g1_getup/*_host`` that first learned to stand from scratch;
# the posture terms of getup_env_cfg were added later by fine-tuning from it). From scratch, the
# extra terms gated at a 0.5 m pelvis height penalized the robot as it rose past 0.5 m, and the
# 29dof run settled sitting with the head below the 0.9 m curriculum threshold.
HOST_REMOVED_TERMS = (
    "style_kneeling", "style_feet_width", "style_hip_yaw_l1", "style_hip_roll_l1",
    "target_lower_body_pose", "target_leg_pose_l2", "target_arm_pose_l2",
    "target_stand_base_vel", "target_stand_feet_vel", "target_stand_joint_vel",
)
HOST_PHASE3_HEIGHT = 0.65


def _apply_host_rewards(cfg) -> None:
    r = cfg.rewards
    for name in HOST_REMOVED_TERMS:
        setattr(r, name, None)
    r.task_orientation_head_height.params["target_head_height"] = 1.0
    r.task_orientation_head_height.params["target_head_margin"] = 1.0
    for name in ("style_left_foot_displacement", "style_right_foot_displacement"):
        getattr(r, name).params["standup_height"] = HOST_PHASE3_HEIGHT
    for name in ("target_ang_vel_xy", "target_lin_vel_xy", "target_feet_height_var", "target_upper_dof_pos",
                 "target_orientation", "target_base_height"):
        getattr(r, name).params["min_base_height"] = HOST_PHASE3_HEIGHT
    r.target_base_height.params["sharpness"] = 20.0
    r.target_upper_dof_pos.params["target"] = {"left_shoulder_roll_joint": 0.3, "right_shoulder_roll_joint": -0.3}
    r.target_upper_dof_pos.params["sigma"] = -0.1


@configclass
class Robot29HoSTEnvCfg(Robot29EnvCfg):
    """G1 29dof get-up from supine with HoST's reward set (train from scratch with this first)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_host_rewards(self)


@configclass
class Robot29HoSTPlayEnvCfg(Robot29PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_host_rewards(self)


@configclass
class Robot29StagedEnvCfg(Robot29EnvCfg):
    """G1 29dof from scratch in one run: HoST's rewards, then the posture terms faded in.

    Slower than the 23dof: with the 1000-iteration ramp started as soon as the pulling force reached
    0, the 29dof (which got up later and less steadily) started kneeling and falling again; wait 1000
    iterations for it to stand reliably, then fade the posture terms in over 2000."""

    def __post_init__(self):
        super().__post_init__()
        from .getup_env_cfg import G1_HOST_PARAMS
        from .staged import make_staged

        make_staged(self, G1_HOST_PARAMS, ramp_iterations=2000, delay_iterations=1000)


@configclass
class Robot29StagedV2EnvCfg(Robot29EnvCfg):
    """29dof staged as the 23dof (ramp over 1000 iterations once the pulling force is 0, after a
    500-iteration delay), with the URDF effort limits (ankles, waist roll / pitch 35 N*m)."""

    def __post_init__(self):
        super().__post_init__()
        from .getup_env_cfg import G1_HOST_PARAMS
        from .staged import make_staged

        make_staged(self, G1_HOST_PARAMS, ramp_iterations=1000, delay_iterations=500)
