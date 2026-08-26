"""The command-state gate shared by the multi-task rewards and terminations.

Both source tasks are kept exactly as they were trained -- same reward functions, same weights, same
terminations -- and are switched on and off by *when* they apply rather than being rewritten. This
module owns that switch so the rewards and the terminations cannot drift apart.

Why the window is not ``enabled``
---------------------------------
``JumpCommand`` holds ``enabled`` for ``command_duration_s`` (0.5 s), but the move is not over then:
``minimum_landing_time_s`` is 0.8 s, so at 0.5 s the robot is still airborne. Handing it back to the
locomotion rewards there would fire ``flat_orientation_l2``, ``track_lin_vel_xy``, ``feet_slide`` and
``undesired_contacts`` at a robot that is upside down *because it was told to be* -- a large penalty
for correct behaviour, at the moment the behaviour is hardest to learn. The window therefore runs
from the trigger for ``rearm_after_s`` (1.5 s), long enough to cover flight, landing and recovery.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GATE_LOCOMOTION = "locomotion"
"""Outside the acrobatics window: the ordinary running task."""

GATE_ACROBATICS = "acrobatics"
"""Inside the acrobatics window."""

GATE_ACROBATICS_STANDING = "acrobatics_standing"
"""Inside the window *and* barely moving.

For ``motion_progress_standing``, which rewards coming to a stable stand after landing. It needs the
window because ``motion_progress_reward`` keys off ``motion_code`` alone and so returns a small
non-zero value from the moment the episode's motion is drawn -- well before any trigger.
"""

GATE_STANDING = "standing"
"""Barely moving, regardless of the window.

For the terms that carry their *own* timing gate internally and only need "and the robot is not
supposed to be running" added: ``pre_jump_pose`` (multiplies by ``~enabled``) and
``pre_motion_standing`` (``~enabled | still_waiting_for_assist``). Both describe the *idle* period --
``pre_jump_pose`` exists to charge for an anticipatory crouch held before the command fires -- so
wrapping them in the acrobatics window would confine them to the post-landing tail and drop the part
they were written for. Letting their own gate decide the timing, and adding only the speed
condition, keeps them behaving exactly as they did in the acrobatics task wherever the robot is
stationary, and silences them where the commanded gait would otherwise be penalised for not standing
in a default pose.
"""


def gate_mask(
    env: ManagerBasedRLEnv,
    gate: str,
    command_name: str = "jump",
    window_s: float = 1.5,
    crossfade_s: float = 0.25,
    standing_speed: float = 0.1,
) -> torch.Tensor:
    """Per-environment weight in ``[0, 1]`` for the requested gate.

    Args:
        env: The environment.
        gate: One of :data:`GATE_LOCOMOTION`, :data:`GATE_ACROBATICS`,
            :data:`GATE_ACROBATICS_STANDING`.
        command_name: Name of the jump command term.
        window_s: Length of the acrobatics window, measured from the trigger.
        crossfade_s: Linear ramp at the end of the window. A hard 0/1 switch would step the
            reward discontinuously at a fixed time after the trigger; fading matches the soft
            routing the rest of the design uses, and costs nothing.
        standing_speed: Commanded speed under which :data:`GATE_ACROBATICS_STANDING` counts as
            standing.
    """
    command = env.command_manager.get_term(command_name)
    elapsed = command.elapsed_since_trigger
    triggered = command.trigger_step >= 0

    if crossfade_s > 0.0:
        # 1 until window_s - crossfade_s, then ramping down to 0 at window_s.
        ramp = ((window_s - elapsed) / crossfade_s).clamp(0.0, 1.0)
    else:
        ramp = (elapsed < window_s).float()
    acrobatics = torch.where(triggered, ramp, torch.zeros_like(ramp))

    if gate == GATE_ACROBATICS:
        return acrobatics
    if gate == GATE_LOCOMOTION:
        return 1.0 - acrobatics
    if gate in (GATE_ACROBATICS_STANDING, GATE_STANDING):
        speed = torch.linalg.norm(
            env.command_manager.get_command(command.cfg.velocity_command_name)[:, :2], dim=-1
        )
        standing = (speed < standing_speed).float()
        return standing if gate == GATE_STANDING else acrobatics * standing
    raise ValueError(
        f"Unknown gate {gate!r}; expected one of {GATE_LOCOMOTION!r}, {GATE_ACROBATICS!r},"
        f" {GATE_ACROBATICS_STANDING!r}, {GATE_STANDING!r}."
    )
