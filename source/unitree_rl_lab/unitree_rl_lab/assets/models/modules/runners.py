# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-specific wrappers around rsl-rl runners."""

from __future__ import annotations

from rsl_rl.runners import DistillationRunner


def _register_distillation_classes(train_cfg: dict) -> None:
    """Register TCN student/alg classes into DistillationRunner eval scope.

    Uses names that do not collide with rsl-rl's default ``StudentTeacher`` /
    ``Distillation``, so a later stock DistillationRunner in the same process
    still resolves to the MLP implementations.
    """
    import rsl_rl.runners.distillation_runner as distillation_runner_module

    policy_cfg = train_cfg.get("policy", {})
    if policy_cfg.get("class_name") == "TcnStudentTeacher":
        from unitree_rl_lab.assets.models.modules.student_teacher import TcnStudentTeacher

        distillation_runner_module.TcnStudentTeacher = TcnStudentTeacher

    alg_cfg = train_cfg.get("algorithm", {})
    if alg_cfg.get("class_name") == "TcnDistillation":
        from unitree_rl_lab.assets.models.modules.distillation import TcnDistillation

        distillation_runner_module.TcnDistillation = TcnDistillation


class UnitreeDistillationRunner(DistillationRunner):
    """DistillationRunner with TCN StudentTeacher + latent-matching Distillation."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_distillation_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
