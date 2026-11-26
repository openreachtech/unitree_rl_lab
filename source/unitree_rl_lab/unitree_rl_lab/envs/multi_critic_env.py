"""Manager-based RL env that also returns its reward split into groups, one per critic.

For multi-critic PPO (``assets/models/modules/multi_critic.py``). The env config sets
``reward_groups``: an ordered ``{group: [term-name regex, ...]}``. Every reward term goes to the
first group with a matching pattern, and must match one. After each step,
``extras["reward_groups"]`` (num_envs, num_groups) holds each group's share of the step reward;
the groups sum to the usual total reward, which ``step`` still returns.
"""

from __future__ import annotations

import re
import torch

from isaaclab.envs import ManagerBasedRLEnv


class MultiCriticRLEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        groups: dict[str, list[str]] = self.cfg.reward_groups
        self.reward_group_names = list(groups)
        term_names = self.reward_manager.active_terms
        mask = torch.zeros(len(term_names), len(groups), device=self.device)
        for i, term in enumerate(term_names):
            for k, patterns in enumerate(groups.values()):
                if any(re.fullmatch(p, term) for p in patterns):
                    mask[i, k] = 1.0
                    break
            else:
                raise ValueError(f"Reward term '{term}' matches no reward group of {groups}.")
        self._reward_group_mask = mask
        print("[INFO] Reward groups:")
        for k, name in enumerate(self.reward_group_names):
            print(f"  {name}: {[t for i, t in enumerate(term_names) if mask[i, k]]}")

    def step(self, action: torch.Tensor):
        out = super().step(action)
        # _step_reward holds each term's weighted value per second; times dt it is the term's share
        # of reward_buf (0 for terms skipped at weight 0)
        step_reward = self.reward_manager._step_reward * self.step_dt
        self.extras["reward_groups"] = step_reward @ self._reward_group_mask
        return out
