# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Belief-encoder distillation configs (BC + height-map reconstruction).

Extends Isaac Lab distillation cfgs. Policy adds ``extero_obs_dim`` /
``proprio_obs_dim`` / ``priv_obs_dim``; Isaac's MLP ``student_hidden_dims`` /
``teacher_hidden_dims`` / ``activation`` are filled with placeholders because
``StudentTeacher`` builds throwaway MLPs in ``super().__init__`` then replaces
them with ``StudentPolicy`` / ``TeacherPolicy`` from ``config.yaml``.

Algorithm adds ``reconstruction_loss_coef`` only.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class BeliefStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Isaac student-teacher policy cfg + belief-net observation widths."""

    init_noise_std: float = 0.1
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False
    # Placeholders for Isaac/rsl_rl MLP StudentTeacher.__init__; discarded after swap.
    student_hidden_dims: list[int] = [256, 256, 256]
    teacher_hidden_dims: list[int] = [256, 256, 256]
    activation: str = "lrelu"
    proprio_obs_dim: int = MISSING
    """Width of the proprioceptive block (commands + proprio) before extero."""
    extero_obs_dim: int = MISSING
    """Width of the height-scan block within proprio|extero observation vectors."""
    priv_obs_dim: int = MISSING
    """Privileged width for TeacherPolicy's encoder (must match teacher checkpoint)."""
    transfer_extero_encoder_from_teacher: bool = True
    """If True, copy teacher extero_encoder → student when loading a PPO teacher checkpoint."""


@configclass
class BeliefDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Isaac distillation algorithm + height-map reconstruction term."""
    # 収集した各サンプルに対して何回updateを行うか
    num_learning_epochs: int = 2
    # 論文と同じ値
    learning_rate: float = 5.0e-4
    # Paper uses 10 (TBPTT trunc length). Keep in sync with num_steps_per_env % this == 0.
    gradient_length: int = 10
    max_grad_norm: float | None = 1.0
    reconstruction_loss_coef: float = 0.5
    """Weight on height-map reconstruction loss (paper default 0.5)."""


@configclass
class BeliefDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """DistillationRunner cfg shell; train/play construct UnitreeDistillationRunner."""

    class_name: str = "DistillationRunner"
    # Paper uses 400; 200 is a middle ground for belief GRU context vs memory/wall-clock.
    num_steps_per_env: int = 200
    max_iterations: int = 20000
    save_interval: int = 100
    experiment_name: str = ""
    empirical_normalization: bool = False
    # Student sees noisy policy obs; teacher / recon target use clean teacher group.
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "teacher": ["teacher"]}
    policy: BeliefStudentTeacherCfg = BeliefStudentTeacherCfg()
    algorithm: BeliefDistillationAlgorithmCfg = BeliefDistillationAlgorithmCfg()
