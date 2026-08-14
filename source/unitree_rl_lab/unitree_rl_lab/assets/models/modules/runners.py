"""Project-specific wrappers around rsl-rl runners."""

from __future__ import annotations

from rsl_rl.runners import DistillationRunner, OnPolicyRunner


def _register_on_policy_classes(train_cfg: dict) -> None:
    policy_cfg = train_cfg.get("policy", {})
    class_name = policy_cfg.get("class_name")
    if class_name == "ActorCriticTeacher":
        from unitree_rl_lab.assets.models.teacher_actor import ActorCriticTeacher
        import rsl_rl.runners.on_policy_runner as on_policy_runner_module

        on_policy_runner_module.ActorCriticTeacher = ActorCriticTeacher
    elif class_name == "ActorCriticTcn":
        from unitree_rl_lab.assets.models.student_actor import ActorCriticTcn
        import rsl_rl.runners.on_policy_runner as on_policy_runner_module

        on_policy_runner_module.ActorCriticTcn = ActorCriticTcn


def _register_distillation_classes(train_cfg: dict) -> None:
    """Register TCN StudentTeacher / Distillation into DistillationRunner eval scope."""
    import rsl_rl.runners.distillation_runner as distillation_runner_module

    policy_cfg = train_cfg.get("policy", {})
    if policy_cfg.get("class_name") == "StudentTeacher":
        from unitree_rl_lab.assets.models.modules.student_teacher import StudentTeacher

        distillation_runner_module.StudentTeacher = StudentTeacher

    alg_cfg = train_cfg.get("algorithm", {})
    if alg_cfg.get("class_name") == "Distillation":
        from unitree_rl_lab.assets.models.modules.distillation import Distillation

        distillation_runner_module.Distillation = Distillation


class UnitreeOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with custom actor-critic registration hooks."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_on_policy_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)


class UnitreeDistillationRunner(DistillationRunner):
    """DistillationRunner with TCN StudentTeacher + latent-matching Distillation."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_distillation_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
