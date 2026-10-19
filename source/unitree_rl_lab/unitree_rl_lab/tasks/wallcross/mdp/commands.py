"""Commands for the hind-stance wall crossing task.

The task is written on the multi-task policy's *unified* observation, unchanged, so the crossing is
cued the way every other move is cued: through the columns that already exist. Concretely:

* the robot walks toward the wall in the hind-leg stance, with ``handstand_command`` on, exactly as
  in ``Go2-Multitask-Biped-Hind``;
* when it is close enough, the stance is released and ``jump_command`` fires in the same step -- the
  order the merged environment already uses, where a jump waits while a stance is held;
* ``jump_command``'s ``target_height`` carries how far the base has to rise to clear this wall, in
  the column's own meaning (apex above the 0.40 m nominal standing height).

Nothing here edits a shared module: :class:`WallHandstandCommand` subclasses the stance command to
add a release, and the jump is fired through ``JumpCommand.set_command``, its public entry point.
:class:`WallCrossCommand` is bookkeeping only -- it schedules the trigger, measures progress over the
wall, and decides success. It contributes no observation column.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped.mdp.handstand import HandstandCommand, HandstandCommandCfg
from unitree_rl_lab.tasks.wallcross import terrains

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WallHandstandCommand(HandstandCommand):
    """The pinned hind stance, with a release the wall command can call.

    Releasing sets ``hold_duration`` to the time already held, so every quantity the shared gates
    derive from it -- the 0.25 s reward crossfade, ``since_release``, ``descending`` -- reads exactly
    as it does when the merged environment ends a stance on its own schedule."""

    def release(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.hold_duration[env_ids] = self.elapsed_since_trigger[env_ids]
        self.enabled[env_ids] = False


@configclass
class WallHandstandCommandCfg(HandstandCommandCfg):
    class_type: type = WallHandstandCommand


class WallCrossCommand(CommandTerm):
    """Schedules the jump at the wall and keeps score of the crossing.

    Per episode: with probability ``jump_episode_probability`` the episode carries a crossing. Its
    jump fires once the base is within a drawn distance of the wall's inner face, or at
    ``trigger_max_time_s`` wherever the robot is, but never before ``trigger_min_time_s`` (the rise
    takes 0.3-0.4 s). The remaining episodes never jump: the robot holds the stance next to a wall it
    is not told to cross, which is what keeps the jump column meaningful to the policy.

    Geometry: every quantity is measured in the tile's own frame, with the Chebyshev distance
    ``max(|x|, |y|)`` from the tile centre -- the square ring's inner face is at ``half`` and its
    outer face at ``half + thickness`` whatever the robot's heading.
    """

    cfg: WallCrossCommandCfg

    def __init__(self, cfg: WallCrossCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if terrains.WALL_TABLE is None:
            raise RuntimeError("WallCrossCommand needs the wall terrain (terrains.WALL_TERRAIN_CFG).")
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.table = torch.as_tensor(terrains.WALL_TABLE, dtype=torch.float, device=self.device)
        self.part_ids, self.part_names = self.robot.find_bodies(cfg.part_body_names, preserve_order=True)
        self.foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.sensor_foot_ids, _ = sensor.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.sensor_trunk_ids, _ = sensor.find_bodies(cfg.trunk_body_names, preserve_order=True)

        n = self.num_envs
        z = lambda dtype=torch.float: torch.zeros(n, dtype=dtype, device=self.device)  # noqa: E731
        self.jump_episode = z(torch.bool)
        self.parked = z(torch.bool)
        self.trigger_gap = z()
        self.triggered = z(torch.bool)
        self.forced = z(torch.bool)
        self.trigger_time = z()
        self.gap_at_trigger = z()
        self.success_hold = z()
        self.success = z(torch.bool)
        self.success_time = z()
        self.success_paid = z(torch.bool)
        self.phi_prev = z()
        self.z_trigger = z()
        self.z_max = z()
        self._command = torch.zeros(n, 1, device=self.device)
        for name in (
            "jump_episode", "triggered", "forced_trigger", "gap_at_trigger", "success",
            "parts_past_max", "time_to_success", "max_foot_force", "max_trunk_force", "wall_height",
            "wall_thickness", "rise", "rise_fraction",
        ):
            self.metrics[name] = torch.zeros(n, device=self.device)

    # -- geometry ---------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def wall(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(half, height, thickness)`` of each environment's wall."""
        terrain = self._env.scene.terrain
        w = self.table[terrain.terrain_levels, terrain.terrain_types]
        return w[:, 0], w[:, 1], w[:, 2]

    def chebyshev(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Chebyshev distance from the tile centre, for positions shaped ``(N, ..., 3)``."""
        origin = self._env.scene.env_origins
        shape = (origin.shape[0],) + (1,) * (pos_w.dim() - 2) + (3,)
        rel = pos_w - origin.view(shape)
        return torch.maximum(rel[..., 0].abs(), rel[..., 1].abs())

    @property
    def base_gap(self) -> torch.Tensor:
        """Distance from the base to the wall's inner face; negative once the base is past it."""
        half, _, _ = self.wall()
        return half - self.chebyshev(self.robot.data.root_pos_w)

    @property
    def parts_past(self) -> torch.Tensor:
        """How many of the tracked parts (base and feet) are beyond the wall's outer face."""
        half, _, thick = self.wall()
        cheb = self.chebyshev(self.robot.data.body_pos_w[:, self.part_ids])
        return (cheb > (half + thick)[:, None]).sum(dim=1)

    @property
    def elapsed(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    def rise_fraction(self) -> torch.Tensor:
        """How much of the rise this wall needs the base has made since the jump fired, 0..1.

        The base has to peak ``apex_clearance`` above the wall's top -- the height ``target_height``
        already asks for. Measured from the base height at the trigger (about 0.51 m in the hind
        stance), on the highest point so far, so it only ever grows. A wall the stance is already
        tall enough for counts as done from the trigger on, so nothing is paid for it."""
        _, height, _ = self.wall()
        need = height + self.cfg.apex_clearance - self.z_trigger
        frac = ((self.z_max - self.z_trigger) / need.clamp(min=1.0e-3)).clamp(0.0, 1.0)
        return torch.where(need > self.cfg.min_rise_needed, frac, torch.ones_like(frac))

    def stand_fraction(self) -> torch.Tensor:
        """How close to the success pose the robot is once over the wall, 0..1.

        Base height from ``stand_height_floor`` up to ``min_success_height`` times level from 30 deg
        down to ``max_tilt_rad``; zero until base and feet are past the outer face. Gives the stricter
        success of run4 a slope: under it not one of run3's own crossings (model_4498, 2048 envs x
        30 s) counted, all of them crouched at 0.19 m."""
        half, _, thick = self.wall()
        past = (-self.base_gap > thick + self.cfg.base_past_margin) & (
            self.chebyshev(self.robot.data.body_pos_w[:, self.foot_ids]) > (half + thick)[:, None]
        ).all(dim=1)
        z = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        tall = ((z - self.cfg.stand_height_floor) / (self.cfg.min_success_height - self.cfg.stand_height_floor))
        level = (self.cfg.stand_tilt_ceiling_rad - tilt) / (self.cfg.stand_tilt_ceiling_rad - self.cfg.max_tilt_rad)
        return tall.clamp(0.0, 1.0) * level.clamp(0.0, 1.0) * past.float()

    def potential(self) -> torch.Tensor:
        """Progress toward and over the wall, in points; read by the shaping reward.

        Approach: from where the jump fired to the inner face, worth ``approach_points``. Rise: the
        base's highest point against the height this wall needs, worth ``rise_points``. Crossing:
        each tracked part beyond the outer face, worth ``cross_points`` in total. Standing: the pose
        past the wall against the success pose, worth ``stand_points``. Only the change of
        this is paid, so walking back and forth or hovering at the top earns nothing net."""
        gap0 = self.gap_at_trigger.clamp(min=1.0e-3)
        approach = ((gap0 - self.base_gap.clamp(min=0.0)) / gap0).clamp(0.0, 1.0)
        cross = self.parts_past.float() / len(self.part_ids)
        return (
            self.cfg.approach_points * approach
            + self.cfg.rise_points * self.rise_fraction()
            + self.cfg.cross_points * cross
            + self.cfg.stand_points * self.stand_fraction()
        )

    # -- schedule ---------------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device)
        k = len(ids)
        self.jump_episode[ids] = torch.rand(k, device=self.device) < self.cfg.jump_episode_probability
        self.trigger_gap[ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.trigger_gap_range)
        for buf in (self.triggered, self.forced, self.success, self.success_paid, self.parked):
            buf[ids] = False
        for buf in (self.trigger_time, self.gap_at_trigger, self.success_hold, self.success_time, self.phi_prev,
                    self.z_trigger, self.z_max):
            buf[ids] = 0.0

    def _update_command(self):
        t = self.elapsed
        half, height, thick = self.wall()
        z = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        rising = self.triggered & ~self.success
        self.z_max = torch.where(rising, torch.maximum(self.z_max, z), self.z_max)

        # Trigger: release the stance and fire the jump in the same step.
        waiting = self.jump_episode & ~self.triggered & (t >= self.cfg.trigger_min_time_s)
        near = self.base_gap <= self.trigger_gap
        late = t >= self.cfg.trigger_max_time_s
        fire = waiting & (near | late)
        if torch.any(fire):
            ids = fire.nonzero(as_tuple=False).squeeze(-1)
            self.triggered[ids] = True
            self.forced[ids] = ~near[ids]
            self.trigger_time[ids] = t[ids]
            self.gap_at_trigger[ids] = self.base_gap[ids].clamp(min=0.0)
            self.z_trigger[ids] = z[ids]
            self.z_max[ids] = z[ids]
            target = (height[ids] + self.cfg.target_height_offset).clamp(*self.cfg.target_height_clip)
            jump = self._env.command_manager.get_term(self.cfg.jump_command_name)
            jump.set_command(ids, True, target_height=None, target_pitch_turns=0.0, target_roll_turns=0.0)
            jump.target_height[ids] = target
            self._env.command_manager.get_term(self.cfg.handstand_command_name).release(ids)
            self.phi_prev[ids] = self.potential()[ids]

        # Episodes without a jump: walk up to the wall, then stand there. Pushing on into it would
        # charge the stance for obeying its velocity command (measured: -43 points of shin contact
        # per episode on the 0.2 m walls), and "walked to the wall, no jump given" is the case the
        # jump column has to be told apart from.
        self.parked |= ~self.jump_episode & (self.base_gap <= self.trigger_gap)
        if torch.any(self.parked):
            velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
            velocity.vel_command_b[self.parked] = 0.0

        # Success: over the wall, standing on its feet, level -- held for `success_hold_s`.
        past_base = -self.base_gap > thick + self.cfg.base_past_margin
        feet_past = (
            self.chebyshev(self.robot.data.body_pos_w[:, self.foot_ids]) > (half + thick)[:, None]
        ).all(dim=1)
        sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        foot_force = torch.linalg.norm(sensor.data.net_forces_w[:, self.sensor_foot_ids], dim=-1)
        feet_down = (foot_force > self.cfg.contact_threshold).sum(dim=1) >= self.cfg.min_feet_in_contact
        tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        standing = z > self.cfg.min_success_height
        landed = self.triggered & past_base & feet_past & feet_down & standing & (tilt < self.cfg.max_tilt_rad)
        self.success_hold = torch.where(landed, self.success_hold + self._env.step_dt, torch.zeros_like(t))
        newly = ~self.success & (self.success_hold >= self.cfg.success_hold_s)
        self.success_time[newly] = t[newly]
        self.success |= newly

    def _update_metrics(self):
        _, height, thick = self.wall()
        m = self.metrics
        m["jump_episode"][:] = self.jump_episode.float()
        m["triggered"][:] = self.triggered.float()
        m["forced_trigger"][:] = self.forced.float()
        m["gap_at_trigger"][:] = self.gap_at_trigger
        m["success"][:] = self.success.float()
        m["parts_past_max"][:] = torch.maximum(m["parts_past_max"], self.parts_past.float() * self.triggered)
        m["time_to_success"][:] = torch.where(self.success, self.success_time - self.trigger_time, 0.0)
        m["wall_height"][:] = height
        m["wall_thickness"][:] = thick
        m["rise"][:] = (self.z_max - self.z_trigger) * self.triggered
        m["rise_fraction"][:] = self.rise_fraction() * self.triggered
        # For choosing the landing-impact and wall-slam thresholds later: peak forces after the jump.
        sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        after = self.triggered.float()
        foot = torch.linalg.norm(sensor.data.net_forces_w[:, self.sensor_foot_ids], dim=-1).amax(dim=1)
        trunk = torch.linalg.norm(sensor.data.net_forces_w[:, self.sensor_trunk_ids], dim=-1).amax(dim=1)
        m["max_foot_force"][:] = torch.maximum(m["max_foot_force"], foot * after)
        m["max_trunk_force"][:] = torch.maximum(m["max_trunk_force"], trunk * after)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class WallCrossCommandCfg(CommandTermCfg):
    """Configuration for :class:`WallCrossCommand`."""

    class_type: type = WallCrossCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Scheduled per episode, not on a timer -- same convention as the jump and stance commands."""

    jump_command_name: str = "jump"
    velocity_command_name: str = "base_velocity"
    handstand_command_name: str = "handstand"
    contact_sensor_name: str = "contact_forces"
    part_body_names: list[str] = ["base", "FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    foot_body_names: list[str] = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    trunk_body_names: list[str] = ["base", ".*_hip"]

    jump_episode_probability: float = 0.85
    trigger_gap_range: tuple[float, float] = (0.3, 0.7)
    """Base-to-inner-face distance at which the jump fires. At deployment the operator presses the
    key about 0.5 m short of the wall; the range is what makes that press forgiving."""
    trigger_min_time_s: float = 1.0
    trigger_max_time_s: float = 8.0
    """Fire anyway at this time. Without it, staying away from the wall would keep the stance's
    +2.7/s coming indefinitely and the jump would never be asked for."""

    target_height_offset: float = -0.25
    """``target_height`` = wall height + this. The column means base apex above the 0.40 m nominal
    standing height, so -0.25 puts the apex 0.15 m above the wall's top."""
    target_height_clip: tuple[float, float] = (0.05, 1.0)

    approach_points: float = 1.0
    rise_points: float = 4.0
    """Paid for the base rising to ``apex_clearance`` above the wall. Added after run3, which crossed
    walls up to 0.5 m without jumping: the base rose only 0.05-0.12 m (p50) after the trigger at any
    wall height, and the robot tipped over the wall from the stance height (0.51 m). On 0.6 m and
    higher the base peaked 0.08 m under the top (p50) and not one of ~8000 jumps got over."""
    apex_clearance: float = 0.15
    """Base apex above the wall's top that the rise is measured against; the same apex that
    ``target_height`` = wall height + ``target_height_offset`` names in the jump column."""
    min_rise_needed: float = 0.02
    cross_points: float = 4.0

    base_past_margin: float = 0.2
    min_feet_in_contact: int = 3
    min_success_height: float = 0.24
    """Base above the ground for a success. Added after run3, whose successes sat at 0.19 m (p50) and
    19 deg of tilt -- crouched, not standing. The quadruped stands at 0.287 m (Isaac Lab, measured in
    ``jump_env_cfg_jump.py``)."""
    max_tilt_rad: float = 0.26
    """15 deg. Was 30 deg (0.52) in run3."""
    stand_points: float = 3.0
    stand_height_floor: float = 0.15
    stand_tilt_ceiling_rad: float = 0.52
    contact_threshold: float = 1.0
    success_hold_s: float = 0.3
    success_end_s: float = 2.0
    """Episode ends this long after success (terminations.wall_success_done)."""
    wall_timeout_s: float = 8.0
    """Episode ends if the crossing has not succeeded this long after the jump fired."""
