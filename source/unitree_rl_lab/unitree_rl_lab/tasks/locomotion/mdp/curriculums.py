from __future__ import annotations

import json
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    increase_threshold: float = 0.8,
    decrease_threshold: float | None = None,
    step: float = 0.1,
    state_file: str | None = None,
    assist_free_only: bool = False,
    tow_command_name: str = "tow_assist",
    min_episodes: int | None = None,
) -> torch.Tensor:
    """Widen the commanded velocity range as tracking improves.

    ``decrease_threshold`` and ``state_file`` both default to off, so passing neither reproduces
    the original one-way, non-persisted behaviour exactly -- this term is shared by go2, h1, and
    g1, and no existing task should change because these were added.

    ``decrease_threshold`` (a fraction of the reward term's weight, like ``increase_threshold``)
    makes the ratchet two-way: below it, the range steps back *down*, never past the range the
    task started with. Without it the ratchet only ever adds, which is a trap when paired with an
    exponential tracking reward. Sandbox Try 9 walked into exactly that: the range reached 4.6 m/s
    against a robot that topped out near 3.7, ``exp(-error^2/0.25)`` went numerically to zero over
    most of the command distribution, and with no tracking gradient left but every penalty still
    live, the reward-maximizing policy became "stand still". It then lost the speeds it *had*
    mastered -- 3.72 m/s down to 1.94 m/s -- and 2600 further iterations produced no recovery,
    because nothing could lower the range again. Set this on any task whose ``limit_ranges``
    ceiling is above what the robot can actually do.

    ``assist_free_only`` judges the range using ONLY the environments held out from the tow
    assist (``TowAssistCommandCfg.eval_env_fraction``), so the commanded speed rises only once
    the robot can hold it *unaided*. Without this the curriculum reads a reward the assist force
    is inflating, and cannot distinguish "runs this fast" from "is towed this fast": sandbox
    Try 11 ratcheted to a commanded 4.8 m/s on a policy whose real top speed was 1.78, because a
    frozen assist kept the reward high. Gating on the held-out set breaks that loop at the
    measurement, which also keeps the tracking error -- and therefore the error-proportional tow
    force -- small, since the command never runs far ahead of the robot.

    ``min_episodes`` is the number of completed episodes to average before each decision;
    ``None`` means "one full sweep of the judging population", i.e. all envs normally and the
    held-out set under ``assist_free_only``. That keeps the historical cadence of at most one
    +/-``step`` per episode cycle while making each decision from ~1000 episodes rather than from
    whichever handful of envs happened to reset on one particular simulation step -- see the long
    comment in the body for why the old version could stop firing altogether.

    ``state_file`` persists the range across ``--resume``. rsl_rl checkpoints store only network
    weights, and this function mutates ``command_term.cfg.ranges`` in memory, so without it every
    resume restarts the range from the task's initial value -- Try 9's resume spent ~1300
    iterations re-climbing ground it had already covered. Same rationale as
    ``TowAssistCommandCfg.state_file``.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Remember where the task started, so the down-step has a floor and cannot shrink the range
    # away to nothing. Captured on first call, before any mutation below has run.
    if not hasattr(command_term, "_initial_lin_vel_ranges"):
        command_term._initial_lin_vel_ranges = (tuple(ranges.lin_vel_x), tuple(ranges.lin_vel_y))
        command_term._lin_vel_reward_sum = 0.0
        command_term._lin_vel_episode_count = 0
        command_term._lin_vel_judge = float("nan")
        command_term._lin_vel_start_step = env.common_step_counter
        if state_file is not None and os.path.isfile(state_file):
            with open(state_file) as f:
                saved = json.load(f)
            ranges.lin_vel_x = list(saved["lin_vel_x"])
            ranges.lin_vel_y = list(saved["lin_vel_y"])

    # Ignore the first episode cycle. rsl_rl's ``init_at_random_ep_len`` starts every env at a
    # random point in its episode, so the first environments to "time out" have only accumulated
    # part of an episode's reward -- measured at 0.731 against the 1.466 the same policy scores
    # once episodes are whole. Judging on that would step the range DOWN at every restart.
    if env.common_step_counter - command_term._lin_vel_start_step < env.max_episode_length:
        return torch.tensor(ranges.lin_vel_x[1], device=env.device)

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    qualifying = env.num_envs
    if assist_free_only:
        eval_mask = env.command_manager.get_term(tow_command_name).eval_mask
        ids = ids[eval_mask[ids]]
        qualifying = int(eval_mask.sum().item())

    # Judge on completed episodes only. An env that fell at step 300 carries 30% of an episode's
    # reward, and counting it would read as bad tracking rather than as a fall.
    if len(ids) > 0:
        ids = ids[env.episode_length_buf[ids] >= env.max_episode_length]

    if len(ids) > 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        command_term._lin_vel_reward_sum += (episode_sums[ids].sum() / env.max_episode_length_s).item()
        command_term._lin_vel_episode_count += len(ids)

    # Decide once per full sweep of the judging population -- i.e. about once per episode cycle,
    # the same cadence as before, but on a sample of ~1000 episodes instead of whatever happened
    # to reset on one particular simulation step.
    #
    # THIS IS NOT COSMETIC. The previous version acted only when
    # ``common_step_counter % max_episode_length == 0``, which assumes environments reset together.
    # They do not: ``train.py`` calls ``runner.learn(..., init_at_random_ep_len=True)``, so rsl_rl
    # scatters every env's episode phase at the start of training and each env keeps that offset.
    # With 4096 envs spread over 1000 steps only ~4 reset on any given step, and after the
    # ``assist_free_only`` filter that is ~1 -- so the commanded range was being raised or held on
    # the evidence of a SINGLE episode. Worse, an env only changes phase by terminating early, so
    # once a policy stops falling the phase histogram freezes; if nothing sits on the exact step
    # the modulo selects, the ratchet stops for good. That is what happened to sandbox Try 16:
    # the range froze at 1.9 m/s at the iteration where falls ceased and never moved again over
    # 1500 further iterations, while tracking stayed at 1.43 -- far above the 1.2 raise threshold.
    if command_term._lin_vel_episode_count >= max(1, min_episodes or qualifying):
        reward = command_term._lin_vel_reward_sum / command_term._lin_vel_episode_count
        command_term._lin_vel_judge = reward
        command_term._lin_vel_reward_sum = 0.0
        command_term._lin_vel_episode_count = 0

        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        direction = 0.0
        if reward > reward_term.weight * increase_threshold:
            direction = 1.0
        elif decrease_threshold is not None and reward < reward_term.weight * decrease_threshold:
            direction = -1.0

        if direction != 0.0:
            delta_command = torch.tensor([-step, step], device=env.device) * direction
            initial_x, initial_y = command_term._initial_lin_vel_ranges
            # Clamp to limit_ranges going up, and to the task's initial range going down. The
            # per-axis min/max keeps a shrinking range from crossing over itself.
            new_x = torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command
            new_y = torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command
            ranges.lin_vel_x = [
                min(max(new_x[0].item(), limit_ranges.lin_vel_x[0]), initial_x[0]),
                max(min(new_x[1].item(), limit_ranges.lin_vel_x[1]), initial_x[1]),
            ]
            ranges.lin_vel_y = [
                min(max(new_y[0].item(), limit_ranges.lin_vel_y[0]), initial_y[0]),
                max(min(new_y[1].item(), limit_ranges.lin_vel_y[1]), initial_y[1]),
            ]
            if state_file is not None:
                os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
                with open(state_file, "w") as f:
                    json.dump({"lin_vel_x": ranges.lin_vel_x, "lin_vel_y": ranges.lin_vel_y}, f)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def custom_terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain-difficulty ratchet tuned for slow, hard terrain (stairs).
    Replacement for ``mdp.terrain_levels_vel``. The stock function:
      * move_up  when distance-from-spawn > terrain_size/2  (= 4.0 m)
      * move_down when distance < commanded_speed * episode_time * 0.5
    On stairs at a throttled command speed, a 4.0 m net displacement inside one
    episode is essentially unreachable, while the velocity-scaled move_down floor
    fires almost every reset -> levels collapse to 0 (observed: 0.057).
    This version:
      * move_up  at a reachable fraction of the tile (35 % = ~2.8 m), so a robot
        that genuinely climbs a few steps forward is promoted.
      * move_down only when the robot barely moved (< 0.5 m), i.e. it actually
        failed. A robot making partial progress stays on its level and keeps
        practising instead of being demoted. This turns the curriculum into a
        one-way ratchet that tracks real skill instead of net wandering.
    """
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None or terrain.cfg.terrain_generator is None:
        return torch.tensor(0.0, device=env.device)

    asset = env.scene[asset_cfg.name]
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    tile_size = terrain.cfg.terrain_generator.size[0]
    move_up = distance > tile_size * 0.35
    move_down = (distance < 0.5) & (~move_up)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _save_tow_assist_state(command) -> None:
    """Persist the tow-assist EFGCL curriculum state so it survives a training restart."""
    if command.cfg.state_file is None:
        return
    state = {
        "assist_scale": command.assist_scale,
        "curriculum_success_rate": command.curriculum_success_rate,
        "curriculum_episode_count": command.curriculum_episode_count,
        "curriculum_success_count": command.curriculum_success_count,
    }
    os.makedirs(os.path.dirname(command.cfg.state_file), exist_ok=True)
    tmp_path = command.cfg.state_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, command.cfg.state_file)


def tow_assist_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    tow_command_name: str = "tow_assist",
    velocity_command_name: str = "base_velocity",
    error_threshold: float = 0.4,
    success_threshold: float = 0.6,
    decay_step: float = 0.01,
    minimum_episodes: int = 1024,
    min_episode_duration_s: float = 2.0,
) -> torch.Tensor:
    """EFGCL-style success-rate-gated decay of :class:`TowAssistCommand`'s assist force,
    following the same ``assist_scale = max(0, assist_scale - decay_step)`` schedule used for
    the jump/backflip/sideflip assist (see ``feat/jump``'s ``assist_force_decay``), but
    triggered by continuous velocity-tracking success instead of a one-shot landing check.

    Reuses ``velocity_command_name``'s own per-env running tracking-error metric
    (``UniformVelocityCommand._update_metrics`` accumulates this every step and it is only
    reset -- by ``CommandManager.reset()`` -- *after* the curriculum manager's ``compute()``
    runs for the same ``env_ids``, so it is still valid here) rather than duplicating that
    bookkeeping inside the tow-assist command itself.

    That metric is a running sum pre-divided by a *fixed* window
    (``resampling_time_range[1] / step_dt``), meant to approximate a mean error over a full
    command window -- but an episode that ends after only a few steps (e.g. the robot falls
    almost immediately) has barely accumulated anything, so it reads as a tiny, "good" error
    regardless of how badly tracking actually went. Left uncorrected, early failures under a
    random initial policy get miscounted as tracking successes, decaying the assist force
    before it's earned (confirmed via smoke test: assist_scale dropped from 1.0 to ~0.95
    within 3 iterations). Episodes shorter than ``min_episode_duration_s`` are still counted
    toward ``curriculum_episode_count`` (a fall is a real failure, not a no-op) but are never
    counted as a success, regardless of what the under-accumulated metric reads.
    """
    tow_command = env.command_manager.get_term(tow_command_name)
    if len(env_ids) == 0:
        return torch.tensor(tow_command.assist_scale, device=env.device)

    tracking_error = env.command_manager.get_term(velocity_command_name).metrics["error_vel_xy"][env_ids]
    elapsed_s = env.episode_length_buf[env_ids].float() * env.step_dt
    ran_long_enough = elapsed_s >= min_episode_duration_s
    successes = int(((tracking_error < error_threshold) & ran_long_enough).sum().item())

    tow_command.curriculum_episode_count += len(env_ids)
    tow_command.curriculum_success_count += successes

    if tow_command.curriculum_episode_count >= minimum_episodes:
        success_rate = tow_command.curriculum_success_count / tow_command.curriculum_episode_count
        tow_command.curriculum_success_rate = success_rate
        if success_rate >= success_threshold:
            tow_command.assist_scale = max(0.0, tow_command.assist_scale - decay_step)
        tow_command.curriculum_episode_count = 0
        tow_command.curriculum_success_count = 0
        _save_tow_assist_state(tow_command)

    return torch.tensor(tow_command.assist_scale, device=env.device)
