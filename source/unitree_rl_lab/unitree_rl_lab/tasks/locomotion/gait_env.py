from collections.abc import Sequence
from enum import IntEnum

import torch
from isaaclab.envs import ManagerBasedRLEnv


class GaitID(IntEnum):
    STAND = 0
    WALK = 1
    WALK_TO_STAND = 2
    RUN = 3
    RUN_TO_WALK = 4


NUM_GAITS = len(GaitID)


class GaitManager:
    def __init__(self, num_envs: int, device):
        self.gait_id = torch.full((num_envs,), GaitID.WALK, dtype=torch.long, device=device)

        self.w2s_timer = torch.zeros(num_envs, device=device)
        self.r2w_timer = torch.zeros(num_envs, device=device)

    def update(
        self,
        v_cmd: torch.Tensor,
        base_lin_vel: torch.Tensor,
        dt: float,
        leg_length: float,
        phase: int,
    ):
        cmd_speed = torch.norm(v_cmd[:, :2], dim=-1)
        actual_speed = torch.norm(base_lin_vel[:, :2], dim=-1)
        froude = actual_speed.square() / (9.81 * leg_length)

        prev = self.gait_id.clone()

        # -------------------------
        # Phase 1: WALK ONLY
        # -------------------------
        if phase == 1:
            self.gait_id[:] = GaitID.WALK
            self.w2s_timer[:] = 0.0
            self.r2w_timer[:] = 0.0
            return self.gait_id

        # =========================
        # WALK → W2S
        # =========================
        enter_w2s = (cmd_speed < 0.1) & (prev == GaitID.WALK)
        self.gait_id[enter_w2s] = GaitID.WALK_TO_STAND
        self.w2s_timer[enter_w2s] = 0.0

        # -------------------------
        # W2S state
        # -------------------------
        in_w2s = self.gait_id == GaitID.WALK_TO_STAND

        # W2S stability: command-only.
        stable_w2s = cmd_speed < 0.1

        self.w2s_timer[in_w2s & stable_w2s] += dt

        unstable_w2s = in_w2s & (~stable_w2s)
        self.w2s_timer[unstable_w2s] = 0.0

        # transition
        to_stand = in_w2s & (self.w2s_timer > 1.5)
        self.gait_id[to_stand] = GaitID.STAND

        # cancel transition
        cancel_w2s = in_w2s & (cmd_speed >= 0.1)
        self.gait_id[cancel_w2s] = GaitID.WALK

        # =========================
        # WALK → RUN (intent + actual)
        # =========================
        enter_run = (froude > 0.5) & (prev == GaitID.WALK)
        self.gait_id[enter_run] = GaitID.RUN

        # =========================
        # RUN → R2W
        # =========================
        enter_r2w = (cmd_speed < 0.5) & (prev == GaitID.RUN)
        self.gait_id[enter_r2w] = GaitID.RUN_TO_WALK
        self.r2w_timer[enter_r2w] = 0.0

        # -------------------------
        # R2W state
        # -------------------------
        in_r2w = self.gait_id == GaitID.RUN_TO_WALK

        # stable decay = slow + optional contact gate if re-enabled
        stable_r2w = actual_speed < 0.5

        self.r2w_timer[in_r2w & stable_r2w] += dt

        # reset if instability
        unstable_r2w = in_r2w & (~stable_r2w)
        self.r2w_timer[unstable_r2w] = 0.0

        # transition
        to_walk = in_r2w & (self.r2w_timer > 2.5)
        self.gait_id[to_walk] = GaitID.WALK

        # cancel if re-accelerate
        cancel_r2w = in_r2w & (froude > 0.5)
        self.gait_id[cancel_r2w] = GaitID.RUN

        return self.gait_id

    def reset(self, env_ids: Sequence[int] | torch.Tensor):
        self.gait_id[env_ids] = GaitID.WALK
        self.w2s_timer[env_ids] = 0.0
        self.r2w_timer[env_ids] = 0.0




class VelocityManagerBasedRLGaitEnv(ManagerBasedRLEnv):
    """Task-specific RL gait env with gait state management."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.gait_manager = GaitManager(self.num_envs, self.device)
        self.current_gait_id = torch.full((self.num_envs,), GaitID.WALK, dtype=torch.long, device=self.device)
        # Phase 1: Walking only
        # Phase 2: Standing and Walk to Stand (W2S)
        # Phase 3: Running and Run-to-Walk (R2W)
        self.curriculum_phase = getattr(cfg, "curriculum_phase", 1)
        self._last_curriculum_phase = -1

        self.leg_length = cfg.leg_length
        self._apply_command_curriculum(force=True)

    def _apply_command_curriculum(self, force: bool = False):
        if not force and self.curriculum_phase == self._last_curriculum_phase:
            return

        term = self.command_manager.get_term("base_velocity")
        cfg = term.cfg

        phase_cfgs = getattr(self.cfg, "phase_command_curriculum", None)
        if phase_cfgs is None:
            raise ValueError("Missing `phase_command_curriculum` in env cfg.")
        phase_key = str(self.curriculum_phase)
        if phase_key not in phase_cfgs:
            raise ValueError(f"Missing command curriculum for phase {self.curriculum_phase}.")

        phase_cfg = phase_cfgs[phase_key]
        cfg.resampling_time_range = phase_cfg["resampling_time_range"]
        cfg.rel_standing_envs = phase_cfg["rel_standing_envs"]
        cfg.ranges.lin_vel_x = phase_cfg["lin_vel_x"]
        cfg.ranges.lin_vel_y = phase_cfg["lin_vel_y"]
        cfg.ranges.ang_vel_z = phase_cfg["ang_vel_z"]
        cfg.ranges.heading = phase_cfg["heading"]

        term.time_left[:] = 0.0  # Force immediate command resampling so new phase settings take effect now.
        self._last_curriculum_phase = self.curriculum_phase

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.gait_manager.reset(env_ids)
        self.current_gait_id[env_ids] = GaitID.WALK

    def step(self, actions):
        obs, rew, terminated, truncated, info = super().step(actions)

        # if self.common_step_counter > 2_000_000:
        #     self.curriculum_phase = 3
        # elif self.common_step_counter > 1_000_000:
        #     self.curriculum_phase = 2
        # else:
        #     self.curriculum_phase = 1
        # self._apply_command_curriculum()

        # ---- velocity ----
        base_lin_vel = self.scene["robot"].data.root_lin_vel_b

        # ---- command ----
        v_cmd = self.command_manager.get_command("base_velocity")

        self.current_gait_id = self.gait_manager.update(
            v_cmd=v_cmd,
            base_lin_vel=base_lin_vel,
            dt=self.step_dt,
            leg_length=self.leg_length,
            phase=self.curriculum_phase,
        )

        return obs, rew, terminated, truncated, info