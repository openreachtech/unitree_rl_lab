"""Belief-encoder distillation: behavior cloning + height-map reconstruction."""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.algorithms import Distillation as RslDistillation
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent


class Distillation(RslDistillation):
    """Distillation with height-map reconstruction loss.

    Extends ``rsl_rl.algorithms.Distillation``. Parent handles rollout collection,
    storage, optimizer, and multi-GPU sync. Only ``update()`` is overridden:

        L = L_bc + reconstruction_loss_coef * L_re

    where ``L_bc`` is behavior cloning (student vs teacher actions) and ``L_re`` is
    MSE between the belief decoder's estimated exteroception and the clean
    height-scan from the teacher observation group (see Miki / Zhuang student training).
    """

    policy: StudentTeacher | StudentTeacherRecurrent

    def __init__(
        self,
        policy: StudentTeacher | StudentTeacherRecurrent,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        reconstruction_loss_coef: float = 0.5,
    ) -> None:
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.reconstruction_loss_coef = reconstruction_loss_coef

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_reconstruction_loss = 0.0
        loss = 0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                # Student forward with belief decoder (noisy policy extero → estimate)
                actions, estimated_extero = self.policy.act_inference(obs, use_decoder=True)
                clean_extero = self.policy.get_clean_extero(obs)

                behavior_loss = self.loss_fn(actions, privileged_actions)
                reconstruction_loss = self.loss_fn(estimated_extero, clean_extero)
                step_loss = behavior_loss + self.reconstruction_loss_coef * reconstruction_loss

                loss = loss + step_loss
                mean_behavior_loss += behavior_loss.item()
                mean_reconstruction_loss += reconstruction_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        mean_reconstruction_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            "behavior": mean_behavior_loss,
            "reconstruction": mean_reconstruction_loss,
            "total": mean_behavior_loss + self.reconstruction_loss_coef * mean_reconstruction_loss,
        }
