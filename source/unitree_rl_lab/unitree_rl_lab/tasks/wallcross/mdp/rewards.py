"""Rewards for the hind-stance wall crossing task.

Everything here reads the ``wallcross`` command term. Values are per step and are multiplied by
``weight * step_dt`` by the reward manager, so a term returning ``x / step_dt`` once pays ``weight * x``
points, and a term returning 1 every step pays ``weight`` points per second.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["before_jump", "during_crossing", "after_jump", "after_success", "floor_contact_off_wall", "floor_contact_once", "sensor_height_deficit", "landing_pose", "landing_pitch_rate", "approach_overspeed", "stance_height_floor", "wall_progress", "wall_alive", "wall_success", "terminated_by"]


def _wall(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def before_jump(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "wallcross"):
    """``term`` until the jump fires, zero after it.

    For ``undesired_contacts`` on the lifted legs: in the stance they must stay off everything, but a
    crossing puts the front legs on the wall by design."""
    return term(env, **(term_params or {})) * (~_wall(env, command_name).triggered).float()


def during_crossing(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "wallcross"):
    """``term`` from the jump until success, zero outside that window."""
    wall = _wall(env, command_name)
    return term(env, **(term_params or {})) * (wall.triggered & ~wall.success).float()


def after_jump(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "wallcross"):
    """``term`` from the jump to the end of the episode (the crossing, the landing and the time after
    success), zero before the jump."""
    return term(env, **(term_params or {})) * _wall(env, command_name).triggered.float()


def after_success(env: ManagerBasedRLEnv, term, term_params: dict | None = None, command_name: str = "wallcross"):
    """``term`` once the crossing has been judged a success, zero before it.

    For the fall after the landing: success is judged after 0.3 s standing past the wall and paid
    then, and the episode runs on for ``success_end_s``. A fall in that window cost only the -4 of
    the termination penalty, and 68% of run11's "crossed" episodes (model_11997, from the walk)
    ended that way -- the robot is handed back to the walking expert after the landing, so it has
    to stay up."""
    return term(env, **(term_params or {})) * _wall(env, command_name).success.float()


def landing_pitch_rate(env: ManagerBasedRLEnv, command_name: str = "wallcross", asset_name: str = "robot") -> torch.Tensor:
    """Squared pitch rate [rad^2/s^2] once the base is past the wall's outer face, until success.

    run12 (model_14996, thin walls, from the walk): 69% of the falls after the jump happened with the
    base already past the wall, 0.58 s (p50) after the trigger, tilted 44 deg and coming down at
    1.1 m/s -- a nose-first landing, not a failure to clear. The rotation from the stance to four legs
    is needed on the way over, so this only charges the pitch rate on the far side. The same lever
    that raised the long jump's landings (``jump_pitch_rate``)."""
    wall = _wall(env, command_name)
    _, _, thick = wall.wall()
    past = -wall.base_gap > thick
    rate = env.scene[asset_name].data.root_ang_vel_b[:, 1]
    return rate.square() * (wall.triggered & ~wall.success & past).float()


def approach_overspeed(
    env: ManagerBasedRLEnv,
    command_name: str = "wallcross",
    velocity_command_name: str = "base_velocity",
    margin: float = 0.2,
) -> torch.Tensor:
    """Squared horizontal base speed above the commanded speed (+ ``margin``), until the jump fires.

    Added after run6, whose approach ignored the command: 1.9 m/s (p50 of the per-episode maximum)
    against a 0.5-0.8 m/s command, so the jump fired at ``trigger_min_time_s`` already at the wall.
    Without a cue there is then no stopping: 94% of the no-jump episodes (1914 of 2040, model_5999,
    2048 envs x 30 s) ran into the wall and fell -- the 16% "fall before the jump" of run3-run6. The
    hind-stance expert (model_1499) approaches at 0.5 m/s (p50), under this margin. Speeds are norms,
    so the frame the command lives in does not matter; a parked episode's command is zero."""
    wall = _wall(env, command_name)
    speed = torch.linalg.norm(wall.robot.data.root_lin_vel_w[:, :2], dim=-1)
    cmd = torch.linalg.norm(env.command_manager.get_command(velocity_command_name)[:, :2], dim=-1)
    excess = (speed - cmd - margin).clamp(min=0.0)
    return excess.square() * (~wall.triggered).float()


def stance_height_floor(
    env: ManagerBasedRLEnv,
    command_name: str = "wallcross",
    min_height: float = 0.45,
    upright_gx: float = -0.8,
) -> torch.Tensor:
    """Metres of base height below ``min_height`` while standing on the hind legs, until the jump fires.

    Added after run9. The approach sank run by run -- base height at the jump cue (p50) 0.51 m in
    run3, 0.43 in run6, 0.38 in run7, 0.33 in run9, against 0.56 for the hind-stance expert -- until
    the rear hips were a few centimetres off the floor. Parked in that crouch, the policy folds the
    calves further and draws the rear hips in, and 0.2 s later a rear hip touches down: 99.9% of the
    no-jump falls of run7 and run9 ended on ``RR_hip``/``RL_hip`` (`probe_park_fall.py`). The jump's
    own wind-up comes after the cue (calves extend, base rises 0.33 -> 0.40 m), so the floor does not
    reach it. Gated on the upright stance (projected gravity x below ``upright_gx``, nose up) so the
    rise from four legs at the start is not charged. A parked episode never triggers, so it stays on."""
    wall = _wall(env, command_name)
    z = wall.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    upright = wall.robot.data.projected_gravity_b[:, 0] < upright_gx
    return (min_height - z).clamp(min=0.0) * (upright & ~wall.triggered).float()

def wall_progress(env: ManagerBasedRLEnv, command_name: str = "wallcross") -> torch.Tensor:
    """Change of :meth:`WallCrossCommand.potential` since the last step, after the jump.

    A potential difference, so the episode total is the final potential minus the one at the
    trigger: bounded, and not farmable by moving back and forth."""
    wall = _wall(env, command_name)
    phi = wall.potential()
    active = wall.triggered & ~wall.success
    delta = torch.where(active, phi - wall.phi_prev, torch.zeros_like(phi))
    wall.phi_prev = torch.where(active, phi, wall.phi_prev)
    return delta / env.step_dt


def wall_alive(env: ManagerBasedRLEnv, command_name: str = "wallcross", min_height: float = 0.2) -> torch.Tensor:
    """1 per step between the jump and success while the base is off the floor.

    Sized (by its weight) to cancel the always-on smoothness penalties, so that after the jump,
    holding still nets about zero: not worth dying to escape, not worth staying for."""
    wall = _wall(env, command_name)
    up = wall.robot.data.root_pos_w[:, 2] > min_height
    return (wall.triggered & ~wall.success & up).float()


def wall_success(env: ManagerBasedRLEnv, command_name: str = "wallcross") -> torch.Tensor:
    """Paid once, on the step the crossing is first judged a success."""
    wall = _wall(env, command_name)
    pay = wall.success & ~wall.success_paid
    wall.success_paid |= pay
    return pay.float() / env.step_dt


def terminated_by(env: ManagerBasedRLEnv, term_keys: list[str]) -> torch.Tensor:
    """1 if any of the named termination terms ended the episode this step.

    Replaces ``is_terminated`` for the fall penalty: that one also fires on the success and timeout
    terminations, which are not time-outs, and would charge a successful crossing the fall price."""
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for key in term_keys:
        done |= env.termination_manager.get_term(key).bool()
    return done.float()


class floor_contact_off_wall(ManagerTermBase):
    """Number of the given bodies touching something other than the wall, from the jump on.

    Mid360 model_20994 lands with the front-right leg thrown forward against its limits (thigh -1.571,
    calf -0.837 for ~0.76 s of the first second) and the shin on the floor: 151 of 159 landings in
    Isaac, 1100-1700 N in MuJoCo. Only the base and hips count as a fall, so nothing charged it. The
    wall exemption is the one of ``illegal_contact_off_wall`` (within ``margin`` of the wall and at
    least ``min_height`` up), so legs and shins on the wall stay free."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        sensor = env.scene.sensors[sensor_cfg.name]
        names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.robot_ids, _ = robot.find_bodies(names, preserve_order=True)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        margin: float = 0.15,
        min_height: float = 0.15,
        command_name: str = "wallcross",
        asset_name: str = "robot",
        past_wall_only: bool = False,
        body_weights: bool = False,
    ) -> torch.Tensor:
        return self.count(env, sensor_cfg, threshold, margin, min_height, command_name, past_wall_only, body_weights)

    def count(self, env, sensor_cfg, threshold, margin, min_height, command_name, past_wall_only, body_weights=False) -> torch.Tensor:
        """Bodies touching off the wall; with ``body_weights`` their contact force in body weights instead
        (a hard shin strike costs more than a graze)."""
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        fmax = torch.linalg.norm(forces, dim=-1).amax(dim=1)
        touching = fmax > threshold
        wall = _wall(env, command_name)
        pos = wall.robot.data.body_pos_w[:, self.robot_ids]
        half, _, thick = wall.wall()
        cheb = wall.chebyshev(pos)
        near_wall = (cheb > (half - margin)[:, None]) & (cheb < (half + thick + margin)[:, None])
        raised = (pos[..., 2] - env.scene.env_origins[:, None, 2]) >= min_height
        off_wall = touching & ~(near_wall & raised)
        if body_weights:
            mg = wall.robot.data.default_mass.sum(dim=1).to(env.device) * 9.81
            off_wall = off_wall.float() * fmax / mg[:, None]
        gate = wall.triggered
        if past_wall_only:
            # the landing only: base past the wall's outer face. Over the top the long thigh links can touch
            # the wall's edge with their origin (the hip joint) more than ``margin`` away from it.
            gate = gate & (-wall.base_gap > thick)
        return off_wall.float().sum(dim=1) * gate.float()


class floor_contact_once(floor_contact_off_wall):
    """Paid once per episode, on the first step any of the bodies touches something but the wall after the
    jump: the price of a fall without ending the episode. As a termination (SafeFall, first try) it ended
    97% of model_20994's episodes before the success was judged -- nothing left to learn from."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.paid[:] = False
        else:
            self.paid[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        margin: float = 0.15,
        min_height: float = 0.15,
        command_name: str = "wallcross",
        asset_name: str = "robot",
        past_wall_only: bool = True,
    ) -> torch.Tensor:
        hit = self.count(env, sensor_cfg, threshold, margin, min_height, command_name, past_wall_only) > 0
        pay = hit & ~self.paid
        self.paid |= pay
        return pay.float() / env.step_dt


def sensor_height_deficit(
    env: ManagerBasedRLEnv,
    offset: tuple[float, float, float] = (0.295, 0.0, -0.075),
    min_height: float = 0.15,
    command_name: str = "wallcross",
    asset_name: str = "robot",
) -> torch.Tensor:
    """How far [m] the chin-mounted MID-360 (``offset`` in the base frame, the robot-side mount) is
    below ``min_height`` above the floor, from the jump on. model_20994 brings it down to 5-9 cm
    after the landing in MuJoCo."""
    robot = env.scene[asset_name]
    off = torch.tensor(offset, device=env.device).expand(env.num_envs, 3)
    p = robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, off)
    z = p[:, 2] - env.scene.env_origins[:, 2]
    return (min_height - z).clamp(min=0.0) * _wall(env, command_name).triggered.float()


def landing_pose(env: ManagerBasedRLEnv, sigma: float = 1.0, command_name: str = "wallcross", asset_name: str = "robot") -> torch.Tensor:
    """exp(-|q - q_default|^2 / sigma) once the base is past the wall's outer face (from the jump on).

    A reward, not a penalty: the safety arms (2026-10-11) charged the bad landing (legs on the floor,
    joints at their limits) and both learned to stop crossing instead -- arm A within 200 iterations
    (timeouts 97%), arm B kept crossing in Isaac but fell to 1-13% in MuJoCo. A term paid only on the far
    side cannot be dodged by staying on the near side; it pulls the landing towards the standing pose
    (model_20994 lands with the front-right thigh at -1.571 against 0.8 default and the hips at +-1.047)."""
    wall = _wall(env, command_name)
    _, _, thick = wall.wall()
    past = -wall.base_gap > thick
    robot = env.scene[asset_name]
    err = (robot.data.joint_pos - robot.data.default_joint_pos).square().sum(dim=1)
    return torch.exp(-err / sigma) * (wall.triggered & past).float()
