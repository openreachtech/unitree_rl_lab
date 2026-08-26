"""Jump command that fires repeatedly within one episode, gated on how fast the robot is going.

``JumpCommand`` fires once per episode: ``command_issued`` latches on the first rising edge and is
only cleared by an episode reset. That suits the 4 s acrobatics task, where one flip *is* the
episode. The multi-task environment runs 20 s episodes in which the robot is mostly running, and an
acrobatic move is an interruption of that -- so the command has to re-arm.

Everything else the parent already does per trigger: ``trigger_step``, ``max_height``,
``accumulated_pitch``/``roll`` and ``success`` are all reset on each rising edge. Only the latch and
the choice of the next motion need adding.

Take-off speed gate
-------------------
The acrobatics expert learned to flip from a standstill and has never taken off with forward speed;
the locomotion expert has never left the ground. Neither has seen the state the merged task is
actually about. Allowing flips at any speed from the first iteration asks the policy to learn that
transition with no foothold -- and a failed landing at 3 m/s terminates the episode, so the reward
signal there is thin to begin with.

So a trigger is only armed while the *commanded* speed is under ``takeoff_speed_limit``, which a
curriculum raises as flips keep succeeding (see :func:`..curriculums.takeoff_speed_levels`). At the
start that means flips happen only in near-stationary environments -- exactly the regime the expert
was trained in -- and the moving take-off is grown into rather than demanded up front.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.dynamic.mdp.commands import JumpCommand, JumpCommandCfg


class MultiTriggerJumpCommand(JumpCommand):
    """:class:`JumpCommand` that re-arms after each motion, subject to a speed gate."""

    cfg: MultiTriggerJumpCommandCfg

    def __init__(self, cfg: MultiTriggerJumpCommandCfg, env) -> None:
        super().__init__(cfg, env)
        # Raised by the curriculum. Stored on the command because the curriculum term, the
        # observation and the metrics all need to read the same value.
        self.takeoff_speed_limit = float(cfg.initial_takeoff_speed_limit)
        self.metrics["takeoff_speed_limit"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["triggers_per_episode"] = torch.zeros(self.num_envs, device=self.device)
        # Conditioned on the environments that actually attempted a move -- see _update_metrics.
        self.metrics["max_height_attempted"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_attempted"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["attempting_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["attempt_success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.trigger_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Attempts scored once each, when they end -- see _score_attempts.
        self._attempts = 0.0
        self._successes = 0.0
        # Monotonic totals, for consumers that keep their own window (the take-off curriculum).
        self.total_attempts = 0
        self.total_successes = 0

    @property
    def commanded_speed(self) -> torch.Tensor:
        """Magnitude of the commanded planar velocity, used by the take-off gate."""
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        return torch.linalg.norm(command[:, :2], dim=-1)

    @property
    def episode_time(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        # Reached on episode reset. An attempt still in progress here was cut short -- the robot
        # fell, or time ran out -- so it is scored on the way past, as a failure unless it had
        # already landed. Scoring only in _rearm would quietly drop exactly the failures that
        # matter most.
        self._score_attempts(env_ids)
        super()._resample_command(env_ids)
        if len(env_ids) > 0:
            self.trigger_count[env_ids] = 0

    def _score_attempts(self, env_ids: Sequence[int]) -> None:
        """Record the outcome of any attempt ending for ``env_ids``, once each.

        ``success`` is a *state*: false from the trigger until the landing check passes, which
        cannot happen before ``minimum_landing_time_s`` (0.8 s) of a 1.5 s window. Averaging it
        over timesteps therefore caps a flawless policy at roughly 0.47, and comparing that number
        against the acrobatics task's own success metric -- averaged over a 4 s episode where the
        flag stays true to the end -- compares two different quantities. This counts attempts, so
        the rate means what its name says and is comparable across both tasks.
        """
        if len(env_ids) == 0:
            return
        ending = self.trigger_step[env_ids] >= 0
        count = int(ending.sum().item())
        if count == 0:
            return
        succeeded = float(self.success[env_ids][ending].sum().item())
        self._attempts += count
        self._successes += succeeded
        self.total_attempts += count
        self.total_successes += int(succeeded)
        # Age out old attempts so the rate tracks current behaviour instead of the run's history.
        if self._attempts > 4096.0:
            self._attempts *= 0.5
            self._successes *= 0.5

    def _update_command(self):
        self._rearm()
        self._defer_while_too_fast()
        # Snapshot before the parent runs: its last statement is
        # ``previous_enabled.copy_(enabled)``, so comparing the two afterwards can never show a
        # rising edge -- the counter would sit at zero however many moves actually fired.
        was_enabled = self.enabled.clone()
        super()._update_command()
        self.trigger_count += (self.enabled & ~was_enabled).long()

    def _rearm(self) -> None:
        """Clear the one-shot latch once a motion has finished, and pick the next one."""
        finished = (
            self.command_issued
            & (self.trigger_step >= 0)
            & (self.elapsed_since_trigger >= self.cfg.rearm_after_s)
        )
        env_ids = finished.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self._score_attempts(env_ids)

        # Re-use the parent's own sampling so the next motion is drawn exactly the way a fresh
        # episode would draw it -- motion type, target height, rotation targets, and the state
        # reset that goes with them. It also clears trigger_step, which ends the acrobatics
        # reward window (see ..rewards.gated).
        counts = self.trigger_count[env_ids].clone()
        super()._resample_command(env_ids)
        self.trigger_count[env_ids] = counts

        # The parent schedules relative to the start of the episode, which is already in the
        # past; re-base onto the current time plus a cooldown so the robot gets an interval of
        # ordinary running between moves.
        interval = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.retrigger_interval_range[0], self.cfg.retrigger_interval_range[1]
        )
        self.scheduled_trigger_time[env_ids] = self.episode_time[env_ids] + interval

    def _defer_while_too_fast(self) -> None:
        """Push a due trigger back while the commanded speed is above the curriculum's limit.

        Deferring rather than cancelling keeps the environment eligible: as soon as the command
        resamples to something slow enough, or the curriculum raises the limit, the move happens.
        """
        due = (~self.command_issued) & (self.episode_time >= self.scheduled_trigger_time)
        defer = due & (self.commanded_speed > self.takeoff_speed_limit)
        if torch.any(defer):
            self.scheduled_trigger_time[defer] = (
                self.episode_time[defer] + self.cfg.takeoff_retry_interval_s
            )

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["takeoff_speed_limit"][:] = self.takeoff_speed_limit
        self.metrics["triggers_per_episode"][:] = self.trigger_count.float()

        # The manager averages every metric over all environments, and only the fraction of them
        # under the take-off speed limit ever attempts a move -- so the inherited max_height and
        # success read roughly `true value x eligible fraction`. That dilution is easy to mistake
        # for a policy that cannot jump: a real 0.12 m flip in 13% of environments logs as 0.016,
        # which looks like a robot glued to the floor. These report the same quantities over the
        # environments that actually triggered, by broadcasting the conditional mean so the
        # manager's own averaging returns it unchanged.
        attempted = self.trigger_step >= 0
        count = attempted.sum()
        if count > 0:
            self.metrics["max_height_attempted"][:] = self.max_height[attempted].mean()
            self.metrics["success_attempted"][:] = self.success[attempted].float().mean()
        else:
            self.metrics["max_height_attempted"][:] = 0.0
            self.metrics["success_attempted"][:] = 0.0
        self.metrics["attempting_fraction"][:] = attempted.float().mean()
        self.metrics["attempt_success_rate"][:] = (
            self._successes / self._attempts if self._attempts > 0 else 0.0
        )


@configclass
class MultiTriggerJumpCommandCfg(JumpCommandCfg):
    class_type: type = MultiTriggerJumpCommand

    velocity_command_name: str = "base_velocity"
    """Command term whose planar magnitude the take-off gate reads."""

    rearm_after_s: float = 1.5
    """Time after a trigger at which the motion counts as over and the next one may be scheduled.

    Also the length of the acrobatics reward window. ``command_duration_s`` (0.5 s) is far too
    short for this: ``minimum_landing_time_s`` is 0.8 s, so at 0.5 s the robot is still airborne,
    and handing it back to the locomotion rewards there would penalise being inverted -- which is
    precisely what it was told to do.
    """

    retrigger_interval_range: tuple[float, float] = (3.0, 6.0)
    """Cooldown sampled after each motion, so the robot spends time running in between. With a
    1.5 s window and a 20 s episode this yields roughly three to four moves per episode, about a
    quarter of the time acrobatic."""

    initial_takeoff_speed_limit: float = 0.3
    """Commanded speed below which a flip may be triggered, before the curriculum raises it."""

    takeoff_retry_interval_s: float = 1.0
    """How long to defer a trigger that was blocked by the speed gate before re-testing."""
