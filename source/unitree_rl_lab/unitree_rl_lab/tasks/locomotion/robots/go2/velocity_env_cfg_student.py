"""Student distillation env: Phase3-balance MDP + noisy policy / clean teacher obs."""

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoiseCfg

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    CriticCfgGo2,
    ObservationsCfgGo2,
    POLICY_HEIGHT_SCAN_CFG,
    PolicyCfgGo2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase3 import (
    RobotEnvCfgPhase3Balance,
)


@configclass
class StudentPolicyCfgGo2(PolicyCfgGo2):
    """Policy obs for the student: same layout as teacher training, noisy height-scan.

    Noise matches the paper ("Learning Robust Perceptive Locomotion"): N(0, 0.1).
    """

    height_scan = POLICY_HEIGHT_SCAN_CFG.replace(noise=GaussianNoiseCfg(mean=0.0, std=0.1))


@configclass
class TeacherObsCfgGo2(PolicyCfgGo2):
    """Clean proprio|extero for teacher actions and reconstruction targets.

    Same term order / dims as ``StudentPolicyCfgGo2`` so extero slices align, but
    ``enable_corruption=False`` and no height-scan noise.
    """

    height_scan = POLICY_HEIGHT_SCAN_CFG

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfgStudent(ObservationsCfgGo2):
    """Student distillation observations: noisy policy, clean teacher, critic for priv."""

    policy: StudentPolicyCfgGo2 = StudentPolicyCfgGo2()
    teacher: TeacherObsCfgGo2 = TeacherObsCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class RobotEnvCfgStudent(RobotEnvCfgPhase3Balance):
    """Phase3-balance scene/rewards with student distillation observation groups."""

    observations: ObservationsCfgStudent = ObservationsCfgStudent()


@configclass
class RobotPlayEnvCfgStudent(RobotEnvCfgStudent):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
