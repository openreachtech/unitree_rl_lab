"""MDP terms for jumping a circular wall from the hind stance (``Go2w-Biped-WallJump``).

After ``feat/wallcross``'s hind-stance wall crossing (Go2), on this package's circular walls: the
robot approaches in the hind stance; when its base is within a drawn distance of the next wall's
inner face, the stance is released and ``jump_command`` fires in the same step, its
``target_height`` saying how far the base has to rise for this wall; the crossing is scored by a
potential (approach, rise, base and wheels past the outer face, standing up) and judged a success
once the robot stands past the wall.

Differences from ``feat/wallcross``: the wall is a ring, so distances are radial rather than
Chebyshev; every episode jumps (its FromWalk runs: holding the stance beside a wall it is not told
to cross is a separate skill, and the source of most of its failure modes); and the stance command
here is pinned, so the release is written straight into it.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .walls import _geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "WallJumpCommand",
    "WallJumpCommandCfg",
    "before_jump",
    "after_jump",
    "after_success",
    "while_driving",
    "bad_orientation_driving",
    "wall_progress",
    "wall_alive",
    "wall_success",
    "terminated_by",
    "wall_success_done",
    "wall_timeout",
    "wall_jump_levels",
]


class WallJumpCommand(CommandTerm):
    """Fires the jump at the wall and keeps score of the crossing. Contributes no observation."""

    cfg: WallJumpCommandCfg

    def __init__(self, cfg: WallJumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.part_ids, _ = self.robot.find_bodies(cfg.part_body_names, preserve_order=True)
        self._sensor_foot_ids = None
        n = self.num_envs
        z = lambda dtype=torch.float: torch.zeros(n, dtype=dtype, device=self.device)  # noqa: E731
        self.trigger_gap = z()
        self.triggered = z(torch.bool)
        self.forced = z(torch.bool)
        self.trigger_time = z()
        self.gap_at_trigger = z()
        self.z_trigger = z()
        self.z_max = z()
        self.phi_prev = z()
        self.success_hold = z()
        self.success = z(torch.bool)
        self.success_time = z()
        self.success_paid = z(torch.bool)
        # Driving mode (``stance_on_gap_range``): the stance is switched on in front of the wall.
        self.stance_on = z(torch.bool)
        self.stance_time = z()
        self.on_gap = z()
        self._command = torch.zeros(n, 1, device=self.device)
        for name in ("triggered", "forced_trigger", "gap_at_trigger", "success", "rise", "rise_fraction", "parts_past_max", "wall_height"):
            self.metrics[name] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def elapsed(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    # -- geometry: the first ring ---------------------------------------------------------------

    def wall(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(inner face radius, height, thickness)`` of each environment's first ring."""
        geometry = _geometry(self._env)
        height, thickness = geometry.height_thickness
        return geometry.radii[0] - 0.5 * thickness, height, thickness

    def radial(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _geometry(self._env).radial(pos_w)

    @property
    def base_gap(self) -> torch.Tensor:
        """Base to the inner face; negative once the base is past it."""
        inner, _, _ = self.wall()
        return inner - self.radial(self.robot.data.root_pos_w)

    @property
    def parts_past(self) -> torch.Tensor:
        """How many of the tracked parts (base and wheels) are beyond the outer face."""
        inner, _, thick = self.wall()
        r = self.radial(self.robot.data.body_pos_w[:, self.part_ids])
        return (r > (inner + thick).unsqueeze(-1)).sum(dim=1)

    def _base_z(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

    def _tilt(self) -> torch.Tensor:
        return torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))

    def _past(self) -> torch.Tensor:
        inner, _, thick = self.wall()
        base = -self.base_gap > thick + self.cfg.base_past_margin
        feet = (self.radial(self.robot.data.body_pos_w[:, self.foot_ids]) > (inner + thick).unsqueeze(-1)).all(dim=1)
        return base & feet

    def rise_fraction(self) -> torch.Tensor:
        """Share of the rise this wall needs that the base has made since the jump, 0..1."""
        _, height, _ = self.wall()
        need = height + self.cfg.apex_clearance - self.z_trigger
        frac = ((self.z_max - self.z_trigger) / need.clamp(min=1.0e-3)).clamp(0.0, 1.0)
        return torch.where(need > self.cfg.min_rise_needed, frac, torch.ones_like(frac))

    def stand_fraction(self) -> torch.Tensor:
        """How close to the success pose the robot is once over the wall, 0..1."""
        tall = (self._base_z() - self.cfg.stand_height_floor) / (self.cfg.min_success_height - self.cfg.stand_height_floor)
        level = (self.cfg.stand_tilt_ceiling_rad - self._tilt()) / (self.cfg.stand_tilt_ceiling_rad - self.cfg.max_tilt_rad)
        return tall.clamp(0.0, 1.0) * level.clamp(0.0, 1.0) * self._past().float()

    def potential(self) -> torch.Tensor:
        """Progress toward and over the wall, in points; its change is the shaping reward."""
        gap0 = self.gap_at_trigger.clamp(min=1.0e-3)
        approach = ((gap0 - self.base_gap.clamp(min=0.0)) / gap0).clamp(0.0, 1.0)
        cross = self.parts_past.float() / len(self.part_ids)
        return (
            self.cfg.approach_points * approach
            + self.cfg.rise_points * self.rise_fraction()
            + self.cfg.cross_points * cross
            + self.cfg.stand_points * self.stand_fraction()
        )

    # -- schedule -------------------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device)
        self.trigger_gap[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.trigger_gap_range)
        self.stance_time[ids] = 0.0
        if self.cfg.stance_on_gap_range is None:
            self.stance_on[ids] = True
        else:
            # Every episode starts on four wheels (the pinned stance command has just switched
            # itself on for the new episode; it runs before this term).
            self.stance_on[ids] = False
            self.on_gap[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.stance_on_gap_range)
            stance = self._env.command_manager.get_term(self.cfg.handstand_command_name)
            stance.enabled[ids] = False
            stance.release_step[ids] = -1
        for buf in (self.triggered, self.forced, self.success, self.success_paid):
            buf[ids] = False
        for buf in (self.trigger_time, self.gap_at_trigger, self.z_trigger, self.z_max, self.phi_prev,
                    self.success_hold, self.success_time):
            buf[ids] = 0.0

    def _update_command(self):
        t = self.elapsed
        z = self._base_z()
        self.z_max = torch.where(self.triggered & ~self.success, torch.maximum(self.z_max, z), self.z_max)

        _, height, _ = self.wall()
        has_wall = ~torch.isnan(height)

        # Driving mode: the stance comes on in front of the wall.
        if self.cfg.stance_on_gap_range is not None:
            rise = has_wall & ~self.stance_on & ~self.triggered & (self.base_gap <= self.on_gap)
            if bool(rise.any()):
                ids = rise.nonzero(as_tuple=False).squeeze(-1)
                stance = self._env.command_manager.get_term(self.cfg.handstand_command_name)
                stance.enabled[ids] = True
                stance.trigger_step[ids] = self._env.episode_length_buf[ids]
                stance.release_step[ids] = -1
                self.stance_on[ids] = True
                self.stance_time[ids] = t[ids]

        # Trigger: release the stance and fire the jump in the same step -- not before the stance
        # has had trigger_min_time_s to rise, and at trigger_max_time_s into it wherever the robot is.
        in_stance = t - self.stance_time
        waiting = has_wall & self.stance_on & ~self.triggered & (in_stance >= self.cfg.trigger_min_time_s)
        near = self.base_gap <= self.trigger_gap
        fire = waiting & (near | (in_stance >= self.cfg.trigger_max_time_s))
        if bool(fire.any()):
            ids = fire.nonzero(as_tuple=False).squeeze(-1)
            _, height, _ = self.wall()
            self.triggered[ids] = True
            self.forced[ids] = ~near[ids]
            self.trigger_time[ids] = t[ids]
            self.gap_at_trigger[ids] = self.base_gap[ids].clamp(min=0.0)
            self.z_trigger[ids] = z[ids]
            self.z_max[ids] = z[ids]
            target = (height[ids] + self.cfg.target_height_offset).clamp(*self.cfg.target_height_clip)
            jump = self._env.command_manager.get_term(self.cfg.jump_command_name)
            jump.set_command(ids, True, target_pitch_turns=0.0, target_roll_turns=0.0)
            jump.target_height[ids] = target
            stance = self._env.command_manager.get_term(self.cfg.handstand_command_name)
            stance.enabled[ids] = False
            stance.release_step[ids] = self._env.episode_length_buf[ids]
            self.phi_prev[ids] = self.potential()[ids]

        # Success: past the wall, on its wheels, standing and level -- held for success_hold_s.
        sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        if self._sensor_foot_ids is None:
            self._sensor_foot_ids = sensor.find_bodies(list(self.cfg.foot_body_names), preserve_order=True)[0]
        force = torch.linalg.norm(sensor.data.net_forces_w[:, self._sensor_foot_ids], dim=-1)
        feet_down = (force > self.cfg.contact_threshold).sum(dim=1) >= self.cfg.min_feet_in_contact
        landed = (
            self.triggered & self._past() & feet_down & (z > self.cfg.min_success_height) & (self._tilt() < self.cfg.max_tilt_rad)
        )
        self.success_hold = torch.where(landed, self.success_hold + self._env.step_dt, torch.zeros_like(t))
        newly = ~self.success & (self.success_hold >= self.cfg.success_hold_s)
        self.success_time[newly] = t[newly]
        self.success |= newly

    def _update_metrics(self):
        _, height, _ = self.wall()
        m = self.metrics
        m["triggered"][:] = self.triggered.float()
        m["forced_trigger"][:] = self.forced.float()
        m["gap_at_trigger"][:] = self.gap_at_trigger
        m["success"][:] = self.success.float()
        m["rise"][:] = (self.z_max - self.z_trigger) * self.triggered
        # Tiles without a wall (driving columns) have NaN geometry; they never trigger.
        m["rise_fraction"][:] = torch.nan_to_num(self.rise_fraction()) * self.triggered
        m["parts_past_max"][:] = torch.maximum(m["parts_past_max"], self.parts_past.float() * self.triggered)
        m["wall_height"][:] = torch.nan_to_num(height)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class WallJumpCommandCfg(CommandTermCfg):
    class_type: type = WallJumpCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    jump_command_name: str = "jump"
    handstand_command_name: str = "handstand"
    contact_sensor_name: str = "contact_forces"
    part_body_names: tuple[str, ...] = ("base", "FR_foot", "FL_foot", "RR_foot", "RL_foot")
    foot_body_names: tuple[str, ...] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

    stance_on_gap_range: tuple[float, float] | None = None
    """Driving mode: start on four wheels and switch the stance on when the base is this far from
    the inner face. ``None``: the stance is on from the first step (the pinned command's own)."""

    trigger_gap_range: tuple[float, float] = (0.3, 0.7)
    """Base-to-inner-face distance at which the jump fires (feat/wallcross's)."""
    trigger_min_time_s: float = 1.0
    """The rise into the stance comes first (measured from the stance's switch-on)."""
    trigger_max_time_s: float = 8.0
    """Fire anyway this long into the stance: otherwise keeping away from the wall keeps the stance
    rewards coming."""

    target_height_offset: float = -0.25
    """``target_height`` = wall height + this: the column is base apex above the nominal 0.40 m stand,
    so the apex is 0.15 m over the wall's top."""
    target_height_clip: tuple[float, float] = (0.05, 1.0)

    approach_points: float = 1.0
    rise_points: float = 4.0
    apex_clearance: float = 0.15
    min_rise_needed: float = 0.02
    cross_points: float = 4.0
    stand_points: float = 3.0

    base_past_margin: float = 0.2
    min_feet_in_contact: int = 3
    min_success_height: float = 0.30
    """Go2w stands at 0.40 m; feat/wallcross's 0.24 m is Go2's 0.287 m stand less a crouch margin."""
    max_tilt_rad: float = 0.26
    stand_height_floor: float = 0.18
    stand_tilt_ceiling_rad: float = 0.52
    contact_threshold: float = 1.0
    success_hold_s: float = 0.3
    success_end_s: float = 2.0
    wall_timeout_s: float = 8.0


# -- reward gates and terms ---------------------------------------------------------------------


def _jump(env: ManagerBasedRLEnv, command_name: str) -> WallJumpCommand:
    return env.command_manager.get_term(command_name)


def before_jump(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "walljump"):
    """``term`` until the jump fires."""
    return term(env, **(term_params or {})) * (~_jump(env, command_name).triggered).float()


def after_jump(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "walljump"):
    """``term`` from the jump on."""
    return term(env, **(term_params or {})) * _jump(env, command_name).triggered.float()


def after_success(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "walljump"):
    """``term`` once the crossing has been judged a success."""
    return term(env, **(term_params or {})) * _jump(env, command_name).success.float()


def while_driving(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "walljump"):
    """``term`` while on four wheels: weighted by ``1 - stance_weight`` (so off in the stance and
    faded back in after a release) and off between the jump and the success, when the robot is
    in the air or getting itself over the wall rather than driving."""
    jump = _jump(env, command_name)
    stance = env.command_manager.get_term(jump.cfg.handstand_command_name)
    crossing = jump.triggered & ~jump.success
    return term(env, **(term_params or {})) * (1.0 - stance.stance_weight) * (~crossing).float()


def bad_orientation_driving(
    env: ManagerBasedRLEnv, limit_angle: float, command_name: str = "walljump"
) -> torch.Tensor:
    """The four-wheel orientation limit, off in the stance, over the grace after a release and
    between the jump and the success."""
    jump = _jump(env, command_name)
    stance = env.command_manager.get_term(jump.cfg.handstand_command_name)
    crossing = jump.triggered & ~jump.success
    return (jump._tilt() > limit_angle) & ~(stance.enabled | stance.descending | crossing)


def wall_progress(env: ManagerBasedRLEnv, command_name: str = "walljump") -> torch.Tensor:
    """Change of :meth:`WallJumpCommand.potential` since the last step, between jump and success."""
    jump = _jump(env, command_name)
    phi = jump.potential()
    active = jump.triggered & ~jump.success
    delta = torch.where(active, phi - jump.phi_prev, torch.zeros_like(phi))
    jump.phi_prev = torch.where(active, phi, jump.phi_prev)
    return delta / env.step_dt


def wall_alive(env: ManagerBasedRLEnv, command_name: str = "walljump", min_height: float = 0.2) -> torch.Tensor:
    """1 per step between the jump and success with the base off the floor; sized to cancel the
    always-on penalties, so that holding still after the jump nets about zero."""
    jump = _jump(env, command_name)
    up = jump._base_z() > min_height
    return (jump.triggered & ~jump.success & up).float()


def wall_success(env: ManagerBasedRLEnv, command_name: str = "walljump") -> torch.Tensor:
    """Paid once, on the step the crossing is first judged a success."""
    jump = _jump(env, command_name)
    pay = jump.success & ~jump.success_paid
    jump.success_paid |= pay
    return pay.float() / env.step_dt


def terminated_by(env: ManagerBasedRLEnv, term_keys: list[str]) -> torch.Tensor:
    """1 if any of the named termination terms ended the episode this step (``is_terminated`` would
    also charge the success and timeout terminations)."""
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for key in term_keys:
        done |= env.termination_manager.get_term(key).bool()
    return done.float()


# -- terminations and curriculum ----------------------------------------------------------------


def wall_success_done(env: ManagerBasedRLEnv, command_name: str = "walljump") -> torch.Tensor:
    """End the episode ``success_end_s`` after the crossing succeeded."""
    jump = _jump(env, command_name)
    return jump.success & (jump.elapsed - jump.success_time >= jump.cfg.success_end_s)


def wall_timeout(env: ManagerBasedRLEnv, command_name: str = "walljump") -> torch.Tensor:
    """End the episode if the crossing has not succeeded ``wall_timeout_s`` after the jump."""
    jump = _jump(env, command_name)
    return jump.triggered & ~jump.success & (jump.elapsed - jump.trigger_time >= jump.cfg.wall_timeout_s)


def wall_jump_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "walljump") -> torch.Tensor:
    """Up a row on a success, down on a failure (feat/wallcross's ``wall_terrain_levels``): with the
    episode ending soon after the landing, distance from spawn says little. Rows of tiles without a
    wall are left alone, and the logged level is the mean over the wall tiles."""
    terrain = env.scene.terrain
    jump = _jump(env, command_name)
    success = jump.success[env_ids]
    # Tiles without a wall (driving columns) keep their row.
    has_wall = ~torch.isnan(jump.wall()[1])
    walled = has_wall[env_ids]
    terrain.update_env_origins(env_ids, success & walled, ~success & walled)
    return torch.mean(terrain.terrain_levels[has_wall].float()) if bool(has_wall.any()) else torch.tensor(0.0)
