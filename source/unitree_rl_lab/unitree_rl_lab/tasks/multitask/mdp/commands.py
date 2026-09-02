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
        self.metrics["locomotion_error"] = torch.zeros(self.num_envs, device=self.device)
        # Velocity-tracking error accumulated only while NOT mid-move -- see locomotion_error.
        self._loco_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._loco_error_steps = torch.zeros(self.num_envs, device=self.device)

        # --- diagnostics: where the locomotion error actually comes from ---------------------
        # Splitting the same error by how long ago the last move ended separates two things the
        # aggregate cannot: a locomotion expert that has genuinely got worse, and a robot that is
        # still collecting itself after a landing. Both raise the average, and they call for
        # opposite fixes -- one is a reward/training problem, the other means the exclusion window
        # simply ends too early.
        self._loco_recent_sum = torch.zeros(self.num_envs, device=self.device)
        self._loco_recent_steps = torch.zeros(self.num_envs, device=self.device)
        self._loco_settled_sum = torch.zeros(self.num_envs, device=self.device)
        self._loco_settled_steps = torch.zeros(self.num_envs, device=self.device)
        for name in ("locomotion_error_recent", "locomotion_error_settled",
                     "success_slow_takeoff", "success_fast_takeoff"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # Commanded speed latched at each trigger, so an attempt can be scored against the speed it
        # was actually launched from rather than the limit that permitted it.
        self.trigger_speed = torch.zeros(self.num_envs, device=self.device)
        self._slow_attempts = 0.0
        self._slow_successes = 0.0
        self._fast_attempts = 0.0
        self._fast_successes = 0.0

        # Per-motion, split the same way. A backflip rotates the body backwards and a sideflip
        # rolls it to the left, so either one fights a forward run head-on -- but the aggregate
        # rate cannot show whether the failures sit there or are spread evenly, and those two
        # answers call for completely different fixes.
        # All five, taken from the base class's own motion table rather than listed here. Listing
        # three by hand is what left the handspring and the right sideflip without slow/fast
        # tallies once they were added -- the two motions a direction-selected task fires most.
        self._motion_names = {
            code: name.removeprefix("success_") for code, name in self._motion_metric_names.items()
        }
        self._motion_tally = {
            (name, speed): [0.0, 0.0]
            for name in self._motion_names.values() for speed in ("slow", "fast")
        }
        for (name, speed) in self._motion_tally:
            self.metrics[f"success_{speed}_{name}"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["motion_reselected"] = torch.zeros(self.num_envs, device=self.device)
        self._reselected = 0.0
        self._triggered = 0.0
        # When success was first registered, per motion. The reward window and the scoring both
        # close at rearm_after_s; a move that lands after that is counted as a failure however well
        # it went, which is worth being able to see rather than infer.
        self._success_time = torch.zeros(self.num_envs, device=self.device)
        # Keyed off `_motion_names`, like the slow/fast tally, so the two cannot disagree about
        # which motions exist. Hardcoding three here while `_motion_names` carried five raised a
        # KeyError on the first re-arm of a right sideflip -- two iterations in.
        self._time_tally = {name: [0.0, 0.0] for name in self._motion_names.values()}
        for name in self._time_tally:
            self.metrics[f"success_time_{name}"] = torch.zeros(self.num_envs, device=self.device)
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
    def in_motion(self) -> torch.Tensor:
        """Whether a commanded move is in progress, on the same window the rewards use."""
        return (self.trigger_step >= 0) & (self.elapsed_since_trigger < self.cfg.rearm_after_s)

    @property
    def locomotion_error(self) -> torch.Tensor:
        """Mean velocity-tracking error over the steps this episode spent *not* flipping.

        Isaac Lab's own ``error_vel_xy`` accumulates on every step without exception, so a flip --
        during which the robot cannot possibly follow a ground velocity command -- counts as
        tracking failure. Feeding that to the take-off curriculum made the gate punish flipping
        itself: raising the speed limit means more flips, more flips means more error, and the limit
        gets pulled straight back down. The measured run did exactly that, stepping 0.3 -> 0.5 and
        back to 0.3 while the error swung 0.42 -> 0.67.

        Restricting the average to the steps outside the acrobatics window -- the same window the
        gated locomotion rewards use -- measures what the gate is actually asking about: whether the
        robot still runs properly when it is supposed to be running. It also makes the number
        directly comparable to the locomotion expert's own 0.30-0.40, which the unrestricted one was
        not.
        """
        return self._loco_error_sum / self._loco_error_steps.clamp(min=1.0)

    @property
    def episode_time(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    def _accumulate_locomotion_error(self) -> None:
        """Add this step's tracking error, but only for environments that are not mid-move."""
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        actual = self.robot.data.root_lin_vel_b[:, :2]
        error = torch.linalg.norm(command[:, :2] - actual, dim=-1)
        outside = (~self.in_motion).float()
        self._loco_error_sum += error * outside
        self._loco_error_steps += outside

        # Same error, split by how long since the move ended.
        elapsed = self.elapsed_since_trigger
        settled = ((self.trigger_step < 0) | (elapsed >= self.cfg.settle_s)).float() * outside
        recent = outside - settled
        self._loco_recent_sum += error * recent
        self._loco_recent_steps += recent
        self._loco_settled_sum += error * settled
        self._loco_settled_steps += settled

    def _resample_command(self, env_ids: Sequence[int]):
        # Reached on episode reset. An attempt still in progress here was cut short -- the robot
        # fell, or time ran out -- so it is scored on the way past, as a failure unless it had
        # already landed. Scoring only in _rearm would quietly drop exactly the failures that
        # matter most.
        self._score_attempts(env_ids)
        super()._resample_command(env_ids)
        if len(env_ids) > 0:
            self.trigger_count[env_ids] = 0
            # Zeroed after the curriculum has read them: curriculum_manager.compute() runs before
            # command_manager.reset() in _reset_idx, so the episode's figures are still intact there.
            self._loco_error_sum[env_ids] = 0.0
            self._loco_error_steps[env_ids] = 0.0
            self._loco_recent_sum[env_ids] = 0.0
            self._loco_recent_steps[env_ids] = 0.0
            self._loco_settled_sum[env_ids] = 0.0
            self._loco_settled_steps[env_ids] = 0.0

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

        # Split by the speed the move was launched from. An aggregate rate cannot say whether a
        # policy fails *because* it is moving, which is the whole question the take-off curriculum
        # is trying to answer.
        speeds = self.trigger_speed[env_ids][ending]
        wins = self.success[env_ids][ending]
        codes = self.motion_code[env_ids][ending]
        fast = speeds >= self.cfg.fast_takeoff_speed
        for code, name in self._motion_names.items():
            is_motion = codes == code
            for speed, mask in (("fast", fast), ("slow", ~fast)):
                selected = is_motion & mask
                tally = self._motion_tally[(name, speed)]
                tally[0] += float(selected.sum().item())
                tally[1] += float((wins & selected).sum().item())
            landed = is_motion & wins & (self._success_time[env_ids][ending] > 0)
            if torch.any(landed):
                times = self._time_tally[name]
                times[0] += float(landed.sum().item())
                times[1] += float(self._success_time[env_ids][ending][landed].sum().item())
        self._fast_attempts += float(fast.sum().item())
        self._fast_successes += float((wins & fast).sum().item())
        self._slow_attempts += float((~fast).sum().item())
        self._slow_successes += float((wins & ~fast).sum().item())
        self._attempts += count
        self._successes += succeeded
        self.total_attempts += count
        self.total_successes += int(succeeded)
        # Age out old attempts so the rate tracks current behaviour instead of the run's history.
        if self._attempts > 4096.0:
            self._attempts *= 0.5
            self._successes *= 0.5

    def _update_command(self):
        # Before _rearm, which can end a window and would otherwise let the landing step count as
        # ordinary running.
        self._accumulate_locomotion_error()
        self._rearm()
        self._defer_while_too_fast()
        # Snapshot before the parent runs: its last statement is
        # ``previous_enabled.copy_(enabled)``, so comparing the two afterwards can never show a
        # rising edge -- the counter would sit at zero however many moves actually fired.
        was_enabled = self.enabled.clone()
        super()._update_command()
        fired = self.enabled & ~was_enabled
        self._select_motion_for_direction(fired)
        self.trigger_count += fired.long()
        self.trigger_speed[fired] = self.commanded_speed[fired]
        self._success_time[fired] = 0.0

        # First moment success turns true, before _rearm can clear it.
        just_won = self.success & (self._success_time == 0.0) & (self.trigger_step >= 0)
        self._success_time[just_won] = self.elapsed_since_trigger[just_won]

    def _select_motion_for_direction(self, fired: torch.Tensor) -> None:
        """Give the commanded heading the rotation that goes *with* it.

        Every acrobatic move carries a direction, and asking for one that fights the gait it
        interrupts teaches nothing except that flips fail. Measured at a 0.7 m/s take-off limit,
        with the moves assigned at random: the plain jump held 0.94 slow and 0.90 fast because it
        is vertical and has no direction to conflict with, while the backflip fell 0.90 -> 0.64 and
        the sideflip 0.30 -> 0.17. MuJoCo showed the same thing from outside -- the sideflip landed
        cleanly while running left and failed while running right.

        An earlier version answered that by substituting a plain jump whenever the sampled move
        conflicted. That kept the interruption but threw away the rotation, and left half the
        headings with nothing but a jump. Now that the expert has all four rotations -- a
        handspring mirroring the backflip, a right sideflip mirroring the left -- the heading can
        simply choose:

            forward     handspring       pitch +1
            backward    backflip         pitch -1
            left        sideflip         roll  -1
            right       sideflip right   roll  +1

        Below ``direction_conflict_speed`` the robot is near enough to standing that no move
        conflicts, so whatever was sampled is kept -- which is also what keeps a move available
        while the robot is stationary.

        The dominant axis decides: a command with more lateral than forward speed is a sideways
        command, whatever its sign on x. Ties go to fore-aft, which only matters on the diagonal.
        """
        if not torch.any(fired):
            return
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        limit = self.cfg.direction_conflict_speed
        # Body frame: +x forward, +y left.
        forward_speed = command[:, 0]
        lateral_speed = command[:, 1]
        moving = (forward_speed.abs() > limit) | (lateral_speed.abs() > limit)

        fore_aft = forward_speed.abs() >= lateral_speed.abs()
        directed = torch.where(
            fore_aft,
            torch.where(
                forward_speed > 0,
                torch.full_like(self.motion_code, self.MOTION_HANDSPRING),
                torch.full_like(self.motion_code, self.MOTION_BACKFLIP),
            ),
            torch.where(
                lateral_speed > 0,
                torch.full_like(self.motion_code, self.MOTION_SIDEFLIP),
                torch.full_like(self.motion_code, self.MOTION_SIDEFLIP_RIGHT),
            ),
        )

        self._triggered += float(fired.sum().item())
        selected = fired & moving
        self._reselected += float((selected & (directed != self.motion_code)).sum().item())
        if not torch.any(selected):
            return

        self.motion_code[selected] = directed[selected]

        # Targets have to move with the code, and they are taken from the same configured
        # magnitudes the acrobatics task samples, with the sign that distinguishes a motion from
        # its mirror. Reading them from the config rather than writing 1.0 here means a task that
        # trains a different number of turns stays consistent through this substitution.
        pitch_turns = abs(self.cfg.target_pitch_turns_range[0])
        roll_turns = abs(self.cfg.target_roll_turns_range[0])
        code = self.motion_code
        self.target_pitch_turns[selected] = torch.where(
            code[selected] == self.MOTION_HANDSPRING,
            torch.full_like(self.target_pitch_turns[selected], pitch_turns),
            torch.where(
                code[selected] == self.MOTION_BACKFLIP,
                torch.full_like(self.target_pitch_turns[selected], -pitch_turns),
                torch.zeros_like(self.target_pitch_turns[selected]),
            ),
        )
        self.target_roll_turns[selected] = torch.where(
            code[selected] == self.MOTION_SIDEFLIP_RIGHT,
            torch.full_like(self.target_roll_turns[selected], roll_turns),
            torch.where(
                code[selected] == self.MOTION_SIDEFLIP,
                torch.full_like(self.target_roll_turns[selected], -roll_turns),
                torch.zeros_like(self.target_roll_turns[selected]),
            ),
        )
        # Flight time, not a height target: a rotation needs air under it, and the acrobatics task
        # gives every flip `flip_target_height` for exactly that reason.
        self.target_height[selected] = self.cfg.flip_target_height
        self.launch_height[selected] = (
            self.cfg.flip_launch_height
            if self.cfg.flip_launch_height > 0.0
            else self.cfg.flip_target_height
        )

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
        self.metrics["locomotion_error"][:] = self.locomotion_error
        self.metrics["locomotion_error_recent"][:] = (
            self._loco_recent_sum / self._loco_recent_steps.clamp(min=1.0)
        )
        self.metrics["locomotion_error_settled"][:] = (
            self._loco_settled_sum / self._loco_settled_steps.clamp(min=1.0)
        )
        self.metrics["success_slow_takeoff"][:] = (
            self._slow_successes / self._slow_attempts if self._slow_attempts > 0 else 0.0
        )
        self.metrics["success_fast_takeoff"][:] = (
            self._fast_successes / self._fast_attempts if self._fast_attempts > 0 else 0.0
        )
        for (name, speed), (attempts, wins) in self._motion_tally.items():
            self.metrics[f"success_{speed}_{name}"][:] = wins / attempts if attempts > 0 else 0.0
        for name, (count, total) in self._time_tally.items():
            self.metrics[f"success_time_{name}"][:] = total / count if count > 0 else 0.0
        self.metrics["motion_reselected"][:] = (
            self._reselected / self._triggered if self._triggered > 0 else 0.0
        )
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

    settle_s: float = 3.0
    """Time from the trigger after which the robot counts as having settled back into running.

    Diagnostic only. ``rearm_after_s`` (1.5 s) ends the acrobatics *reward* window, but a robot that
    has just landed is not necessarily running yet, and every step in between is currently scored as
    ordinary locomotion. Splitting at 3 s separates "the gait got worse" from "it is still
    collecting itself", which the aggregate cannot distinguish and which need opposite fixes."""

    fast_takeoff_speed: float = 0.3
    """Commanded speed at or above which an attempt is scored as a moving take-off."""

    direction_conflict_speed: float = 0.3
    """Commanded speed above which the move is chosen by heading rather than sampled at random.

    Below it the robot is near enough to standing that no move conflicts, so the sampled motion is
    kept -- which is what leaves a move available while stationary. Zero disables the selection
    entirely, which is what every run before this one did. See ``_select_motion_for_direction``."""
