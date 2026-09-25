from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    promote_fraction: float = 0.8,
    demote_fraction: float = 0.5,
) -> torch.Tensor:
    """Widen (or now also narrow) the velocity command range based on tracking reward.

    2026-09-02: added demotion. The original was promotion-only, so once the range
    ratcheted open it could never close again even as the policy fell behind. In the
    Stage B v2 run it reached 2.2 m/s by iteration 1400 and stayed pinned there while
    the policy degraded to a 100%-fall rate -- commanding speeds the policy could no
    longer track, which is what drove the reward budget negative and made early
    termination the optimum. Hysteresis (promote above 0.8*weight, demote below
    0.5*weight) lets the curriculum walk back to a level the policy can actually hold.
    See プロジェクト_StageB崩壊の原因分析.md.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        delta = 0.0
        if reward > reward_term.weight * promote_fraction:
            delta = 0.1
        elif reward < reward_term.weight * demote_fraction:
            delta = -0.1
        if delta != 0.0:
            delta_command = torch.tensor([-delta, delta], device=env.device)
            # Never shrink below the initial narrow range's width, so demotion cannot
            # collapse the command to a single point (which would remove the task).
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_x[1] = max(ranges.lin_vel_x[1], ranges.lin_vel_x[0] + 0.1)
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def jump_assist_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "jump_command",
    success_threshold: float = 0.60,
    decay_step: float = 0.02,
    minimum_episodes: int = 1024,
    force_zero_at_step: int = 0,
) -> torch.Tensor:
    """Decay the EFGCL assist force once the policy lands jumps on its own.

    Ported from tak's mdp.assist_force_decay (tag ``jump-demo``). "Success" here is a
    jump that actually left the ground and came back down upright, tracked per episode
    by JumpCommand. The assist is a teacher: it demonstrates the crouch-and-launch
    physically (which pure reward shaping was never able to elicit -- see
    プロジェクト_StageB崩壊の原因分析.md), then gets out of the way.

    2026-09-03 (v5) -- two changes, both because the v4 run finished with the assist
    still at 0.94 after 3000 iterations, i.e. the policy was never made to jump alone:

    1. The gate is now ``episode_real_jump`` (a jump that cleared the height and airtime
       thresholds) instead of ``success`` (which additionally demands all four feet down,
       upright, before max_jump_duration_s). Once airborne time grew past ~0.6 s that
       conjunction stopped fitting inside the window: real_jump_fraction ~90% against
       success_rate ~15% for 2900 iterations, so the 0.60 threshold was never met.
    2. ``force_zero_at_step`` is a hard ceiling that ramps the assist to 0 on a fixed
       schedule regardless of performance. A purely performance-gated teacher can, as v4
       showed, simply never leave -- and a policy that has never trained unaided cannot
       be deployed. The ceiling guarantees the last stretch of training is assist-free.

    See プロジェクト_StageB着地失敗の原因分析.md.
    """
    command = env.command_manager.get_term(command_name)

    # Hard schedule ceiling, applied on every call so the gate cannot outrun it.
    if force_zero_at_step > 0:
        ceiling = max(0.0, 1.0 - env.common_step_counter / float(force_zero_at_step))
        command.assist_scale = min(command.assist_scale, ceiling)

    if len(env_ids) == 0:
        return torch.tensor(command.assist_scale, device=env.device)

    command.curriculum_episode_count += len(env_ids)
    command.curriculum_success_count += int(command.success[env_ids].sum().item())
    command.curriculum_real_jump_count += int(command.episode_real_jump[env_ids].sum().item())

    # Throttled to at most one decay step per max_episode_length simulation steps, the
    # same gate lin_vel_cmd_levels uses. Without it this term decayed 1.0 -> 0.71 inside
    # a single PPO iteration during the 2026-09-02 smoke test: this function runs on
    # every environment reset, and with a flailing early policy (plus an assist strong
    # enough to launch the robot by itself, so "success" is easy at scale 1.0) the
    # episode counter crosses minimum_episodes many times per iteration. Withdrawing the
    # teacher that fast defeats the point of having one.
    interval_elapsed = env.common_step_counter % env.max_episode_length == 0
    if interval_elapsed and command.curriculum_episode_count >= minimum_episodes:
        episodes = max(command.curriculum_episode_count, 1)
        # Keep publishing the landing-quality rate under its established metric name --
        # it is still the number that says whether landings are clean -- but decay on
        # the "did a real jump happen at all" rate.
        command.curriculum_success_rate = command.curriculum_success_count / episodes
        rate = command.curriculum_real_jump_count / episodes
        if rate >= success_threshold:
            command.assist_scale = max(0.0, command.assist_scale - decay_step)
        command.curriculum_episode_count = 0
        command.curriculum_success_count = 0
        command.curriculum_real_jump_count = 0

    return torch.tensor(command.assist_scale, device=env.device)


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


def jump_vel_target_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "jump_sparse_reward",
    limit_v_target: float = 5.5,
    step: float = 0.1,
    min_liftoffs_to_judge: int = 1,
) -> torch.Tensor:
    """Raise mdp.jump_sparse_reward's ``v_target`` once it is reliably earned.

    Same reward-gated expansion pattern as lin_vel_cmd_levels/ang_vel_cmd_levels,
    applied to the jump reward's target speed instead of a command range.
    Deliberate deviation from [[reference_paper_impedance_matching_running_jump]]
    (which trains toward one fixed 2.5 m/s target): here the point is to
    push toward Go2's theoretical limit
    ([[project_long_jump_theoretical_calc]]: ~2.0-3.9 m/s depending on
    approach speed), not to hit one target, so the target itself ratchets
    up -- mirroring how 田村's team's vertical-jump task progressed target
    height in discrete steps (40cm -> 70cm -> ...) rather than a single
    fixed goal.

    2026-08-31 FIX: the original gate (episode_sum / max_episode_length_s >
    weight * 0.8) assumed a *dense* per-step reward, matching
    lin_vel_cmd_levels' use of the same pattern for track_lin_vel_xy.
    mdp.jump_sparse_reward instead fires on at most a handful of single
    steps per episode (one per liftoff), so dividing its episode sum by the
    full episode duration in seconds crushed the ratio by ~3 orders of
    magnitude below any reachable threshold -- confirmed empirically in the
    2026-08-30/31 Stage B run, where this stayed at its initial value for
    all 10000 iterations. Normalize by the number of liftoff *attempts*
    instead (tracked by mdp.jump_sparse_reward in
    ``env._ljb_liftoff_count``), so the gate reads as "how close to
    v_target were recent liftoffs, on average" rather than being diluted by
    how long the episode ran.
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if not hasattr(env, "_ljb_liftoff_count") or env._ljb_liftoff_count is None:
        # mdp.jump_sparse_reward hasn't run yet this env (e.g. very first
        # curriculum call before any reward computation) -- nothing to judge.
        return torch.tensor(reward_term.params["v_target"], device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        liftoff_count = env._ljb_liftoff_count[env_ids].float()
        # episode_sums accumulates weight * raw_reward per step; undo the
        # weight to recover the raw exp(...) value (bounded in [0, 1]) so
        # the "* 0.8" threshold means the same "80% of max" as elsewhere.
        raw_sum = env.reward_manager._episode_sums[reward_term_name][env_ids] / reward_term.weight
        enough_attempts = liftoff_count >= min_liftoffs_to_judge
        if enough_attempts.any():
            avg_per_liftoff = (raw_sum[enough_attempts] / liftoff_count[enough_attempts]).mean()
            if avg_per_liftoff > 0.8:
                reward_term.params["v_target"] = min(reward_term.params["v_target"] + step, limit_v_target)

    return torch.tensor(reward_term.params["v_target"], device=env.device)


def jump_dense_reward_decay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "jump_dense_reward",
    early_weight: float = 2.5,
    late_weight: float = 0.25,
    decay_at_step: int = 60000,
) -> torch.Tensor:
    """Lower mdp.jump_dense_reward's weight once jumping is established.

    Mirrors [[reference_paper_impedance_matching_running_jump]]'s Phase
    2a -> 2b schedule (Table II: Dense Jump scale -2.5 -> -0.25; the sign is
    flipped here because mdp.jump_dense_reward's own -std(F_foot) already
    carries the minus sign, so a positive RewTerm weight is what keeps the
    penalty direction correct in this codebase's convention). The paper
    describes this as easing off the heavy symmetric-push-off shaping once
    the robot can already jump, so it stops fighting fine-tuning of other
    behaviors. Gated on ``env.common_step_counter`` (simulation steps, not
    PPO iterations) rather than jump performance, to keep this simple and
    deterministic -- pick decay_at_step for the run's actual
    --max_iterations (decay_at_step = num_steps_per_env * iteration).
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term.weight = late_weight if env.common_step_counter >= decay_at_step else early_weight
    return torch.tensor(reward_term.weight, device=env.device)
