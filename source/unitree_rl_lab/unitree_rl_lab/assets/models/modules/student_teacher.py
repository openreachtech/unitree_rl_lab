import torch
from torch.distributions import Normal
from typing import Any, Optional
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml

from rsl_rl.modules import StudentTeacher as RslStudentTeacher
from rsl_rl.networks import HiddenState

from unitree_rl_lab.assets.models.student_actor import StudentPolicy
from unitree_rl_lab.assets.models.teacher_actor import TeacherPolicy


class StudentTeacher(RslStudentTeacher):
    """
    Belief-encoder student + frozen teacher actor for rsl-rl Distillation.

    Extends ``rsl_rl.modules.StudentTeacher``: keeps parent obs grouping, normalizers,
    action noise, ``act()``, ``train()``, ``update_normalization()``, and
    ``load_state_dict()``. Replaces the MLP student/teacher with:
        student → StudentPolicy  (recurrent belief encoder)
        teacher → TeacherPolicy  (non-recurrent, loaded from PPO TeacherActorCritic)

    Observation layout (same as TeacherActorCritic):
        policy / teacher obs = [proprio | extero]
    Widths ``proprio_obs_dim``, ``extero_obs_dim``, ``priv_obs_dim`` come from the
    policy cfg (same as ``extero_obs_dim`` for TeacherActorCritic). ``priv_obs_dim``
    must match the teacher checkpoint's privileged_encoder.
    If the teacher obs group is longer (e.g. includes privileged), only the leading
    proprio|extero slice is fed to TeacherPolicy.
    """

    is_recurrent = True  # StudentPolicy carries a GRU belief state

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        extero_obs_dim,
        proprio_obs_dim,
        priv_obs_dim,
        model_cfg_path=None,
        student_model_cfg_key="student_model",
        teacher_model_cfg_key="teacher_model",
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        transfer_extero_encoder_from_teacher: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        # Parent builds placeholder MLPs + normalizers + action noise; we swap the nets below.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            student_obs_normalization=student_obs_normalization,
            teacher_obs_normalization=teacher_obs_normalization,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self.proprio_dim = int(proprio_obs_dim)
        self.extero_dim = int(extero_obs_dim)
        self.priv_dim = int(priv_obs_dim)
        self.action_dim = int(num_actions)
        self.transfer_extero_encoder_from_teacher_enabled = bool(transfer_extero_encoder_from_teacher)
        assert self.proprio_dim > 0, f"proprio_obs_dim must be > 0, got {self.proprio_dim}."
        assert self.extero_dim > 0, f"extero_obs_dim must be > 0, got {self.extero_dim}."
        assert self.priv_dim >= 0, f"priv_obs_dim must be >= 0, got {self.priv_dim}."

        # Sanity-check cfg widths against live observation tensors.
        num_student_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_teacher_obs = sum(obs[group].shape[-1] for group in obs_groups["teacher"])
        expected_student = self.proprio_dim + self.extero_dim
        assert num_student_obs == expected_student, (
            f"Student obs width ({num_student_obs}) != proprio+extero "
            f"({self.proprio_dim}+{self.extero_dim}={expected_student}). "
            f"Check policy cfg dims vs env observation groups."
        )
        assert num_teacher_obs >= expected_student, (
            f"teacher obs ({num_teacher_obs}) must at least cover proprio|extero "
            f"({expected_student})."
        )

        if model_cfg_path is None:
            model_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        student_cfg = deepcopy(self._load_model_cfg(model_cfg_path, student_model_cfg_key))
        teacher_cfg = deepcopy(self._load_model_cfg(model_cfg_path, teacher_model_cfg_key))

        args = SimpleNamespace(
            proprio_obs_dim=self.proprio_dim,
            extero_obs_dim=self.extero_dim,
            priv_obs_dim=self.priv_dim,
            action_dim=self.action_dim,
        )

        # Replace parent MLP student / teacher with belief / teacher policies.
        self.student = StudentPolicy(args, student_cfg)
        self.teacher = TeacherPolicy(args, teacher_cfg)
        self.teacher.eval()
        print(f"StudentPolicy: {self.student}")
        print(f"TeacherPolicy: {self.teacher}")

        # GRU hidden: (num_layers, num_envs, hidden) — managed like rsl_rl Memory
        self.student_hidden_state: Optional[torch.Tensor] = None

    @staticmethod
    def _load_model_cfg(model_cfg_path, model_cfg_key):
        cfg_path = Path(model_cfg_path).expanduser().resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if model_cfg_key not in raw:
            raise KeyError(f"Config key '{model_cfg_key}' not found in {cfg_path}.")
        return raw[model_cfg_key]

    def transfer_extero_encoder_from_teacher(self) -> None:
        """Warm-start student extero encoder from the frozen teacher (paper / reference agent).

        ``extero_encoder`` is architecturally identical in teacher and student
        (``config.yaml``: shape [80, 60], output 24). ``base_net`` is not copied:
        teacher input is [proprio | extero_latent | priv_latent], student is
        [proprio | belief_state].
        """
        self.student.extero_encoder.load_state_dict(self.teacher.extero_encoder.state_dict())
        print("[INFO] Transferred teacher extero_encoder → student extero_encoder.")

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load teacher from PPO checkpoint (then warm-start student encoder), or resume distillation."""
        if any("actor" in key for key in state_dict):
            # PPO TeacherActorCritic → teacher only, then copy shared extero encoder to student.
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            if self.transfer_extero_encoder_from_teacher_enabled:
                self.transfer_extero_encoder_from_teacher()
            else:
                print("[INFO] Skipped teacher→student extero_encoder transfer (disabled in cfg).")
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Distillation does not resume
        elif any("student" in key for key in state_dict):
            # Resume previous distillation (student already has its own encoder weights).
            super(RslStudentTeacher, self).load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def _split_proprio_extero(self, obs_tensor: torch.Tensor):
        proprio = obs_tensor[..., : self.proprio_dim]
        extero = obs_tensor[..., self.proprio_dim : self.proprio_dim + self.extero_dim]
        return proprio, extero

    def _student_step(self, student_obs: torch.Tensor, use_decoder: bool = False):
        """One env-step through the belief student; updates ``student_hidden_state``.

        Returns:
            action if ``use_decoder`` is False, else ``(action, estimated_extero)``.
        """
        proprio, extero = self._split_proprio_extero(student_obs)
        out = self.student(
            proprio.unsqueeze(0),
            extero.unsqueeze(0),
            self.student_hidden_state,
            use_decoder=use_decoder,
        )
        self.student_hidden_state = out["recurrent_hidden"]
        action = out["action"].squeeze(0)
        if use_decoder:
            return action, out["estimated_extero_state"].squeeze(0)
        return action

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """Parent ``act()`` passes normalized student obs; run belief step then reuse noise logic."""
        mean = self._student_step(obs, use_decoder=False)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        self.distribution = Normal(mean, std)

    def act_inference(self, obs, use_decoder: bool = False):
        """Student forward for distillation update.

        Args:
            use_decoder: If True, also run the belief decoder and return
                ``(action, estimated_extero)`` for reconstruction loss.
        """
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self._student_step(obs, use_decoder=use_decoder)

    def get_clean_extero(self, obs) -> torch.Tensor:
        """Clean height-scan target for reconstruction (from teacher obs group).

        Builds a student-shaped vector with clean extero, runs the student normalizer
        (Identity when disabled), then returns the extero slice so the target lives in
        the same space as the decoder output.
        """
        teacher_obs = self.get_teacher_obs(obs)[..., : self.proprio_dim + self.extero_dim]
        student_obs = self.get_student_obs(obs).clone()
        student_obs[..., self.proprio_dim : self.proprio_dim + self.extero_dim] = teacher_obs[
            ..., self.proprio_dim : self.proprio_dim + self.extero_dim
        ]
        student_obs = self.student_obs_normalizer(student_obs)
        return student_obs[..., self.proprio_dim : self.proprio_dim + self.extero_dim]

    def evaluate(self, obs) -> torch.Tensor:
        """Teacher action mean (supervision target for behavior cloning)."""
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        # TeacherPolicy was trained on proprio|extero only (privileged is critic-side).
        obs = obs[..., : self.proprio_dim + self.extero_dim]
        with torch.no_grad():
            return self.teacher(obs)["action_mean"]

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: tuple[HiddenState, HiddenState] = (None, None),
    ) -> None:
        """Reset / restore student GRU hidden state (teacher is non-recurrent)."""
        if dones is None:
            self.student_hidden_state = hidden_states[0]
        elif self.student_hidden_state is not None:
            self.student_hidden_state[..., dones == 1, :] = 0.0

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.student_hidden_state, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        if self.student_hidden_state is None:
            return
        if dones is None:
            self.student_hidden_state = self.student_hidden_state.detach()
        else:
            self.student_hidden_state[..., dones == 1, :] = self.student_hidden_state[
                ..., dones == 1, :
            ].detach()
