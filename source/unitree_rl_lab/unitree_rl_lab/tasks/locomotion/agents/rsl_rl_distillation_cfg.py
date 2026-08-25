# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TCN student distillation configs (behavior cloning + latent matching).

Lee et al. 2020, Eq. 1 / Table S8. Policy is the TCN-100 student plus a frozen
``PrivilegedMlp`` teacher loaded from a ``Go2w-v2-Teacher-*`` PPO checkpoint.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class TcnStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Isaac student-teacher policy cfg + TCN / privileged-encoder widths."""

    class_name: str = "TcnStudentTeacher"
    init_noise_std: float = 0.1
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False
    student_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    tcn_history_length: int = 100
    tcn_channels: int = 34
    tcn_kernel_size: int = 5
    tcn_latent_dim: int = 64
    tcn_concat_current_obs: bool = True
    privileged_encoder_hidden_dims: list[int] = [72]
    privileged_latent_dim: int = 64


@configclass
class TcnDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Isaac distillation algorithm + paper latent-matching term."""

    class_name: str = "TcnDistillation"
    num_learning_epochs: int = 10
    learning_rate: float = 1.0e-4
    gradient_length: int = 10
    max_grad_norm: float | None = 1.0
    latent_loss_coef: float = 1.0
    """Weight on ``||l̄_t - l_t||^2`` (paper Eq. 1 uses 1.0)."""
    weight_decay: float = 1.0e-4


@configclass
class StudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """DistillationRunner cfg; train/play construct UnitreeDistillationRunner."""

    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 100
    max_iterations: int = 1000
    save_interval: int = 100
    experiment_name: str = ""
    empirical_normalization: bool = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "teacher": ["policy", "critic"]}
    policy: TcnStudentTeacherCfg = TcnStudentTeacherCfg()
    algorithm: TcnDistillationAlgorithmCfg = TcnDistillationAlgorithmCfg()
