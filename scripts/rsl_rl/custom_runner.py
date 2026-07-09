"""Project-specific wrapper around rsl-rl OnPolicyRunner."""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner


class UnitreeOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with custom actor-critic registration hooks."""

    @staticmethod
    def _register_custom_policy_classes(train_cfg: dict) -> None:
        policy_cfg = train_cfg.get("policy", {})
        class_name = policy_cfg.get("class_name")
        if class_name != "TeacherActorCritic":
            return

        from unitree_rl_lab.assets.models.modules.teacher_actor import TeacherActorCritic
        import rsl_rl.runners.on_policy_runner as on_policy_runner_module

        # rsl-rl resolves policy class_name via eval(...) in on_policy_runner module scope.
        on_policy_runner_module.TeacherActorCritic = TeacherActorCritic

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._register_custom_policy_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
