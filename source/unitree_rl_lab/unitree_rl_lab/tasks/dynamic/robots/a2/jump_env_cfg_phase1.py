from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.robots.a2.jump_env_cfg import RobotEnvCfg, RobotPlayEnvCfg


@configclass
class RobotEnvCfgPhase1(RobotEnvCfg):
    """Phase 1: learn quiet standing with the final jump observation layout.

    Nothing jumps here -- ``auto_trigger`` is off, so the jump command never fires and the
    reward set is the standing one. Its only job is to produce a checkpoint whose
    observation layout already matches the jump tasks, so ``A2-Jump-60`` can resume from it
    and spend its iterations on the jump rather than on learning to stand first.
    """


@configclass
class RobotPlayEnvCfgPhase1(RobotPlayEnvCfg):
    """Play configuration for Phase 1."""
