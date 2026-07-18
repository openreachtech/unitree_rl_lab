from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.go2.jump_env_cfg import RobotEnvCfg, RobotPlayEnvCfg


@configclass
class RobotEnvCfgPhase1(RobotEnvCfg):
    """Phase 1: learn quiet standing with the final jump observation layout."""


@configclass
class RobotPlayEnvCfgPhase1(RobotPlayEnvCfg):
    """Play configuration for Phase 1."""
