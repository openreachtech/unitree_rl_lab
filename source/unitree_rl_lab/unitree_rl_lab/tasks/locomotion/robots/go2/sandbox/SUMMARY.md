# Sandbox Try-1 .. Try-9: Phase3-balance-floating flat-idle fix

Summary of the sandbox experiments that shaped the current
`Unitree-Go2-Velocity-v2-Phase3-balance-floating` default in
`velocity_env_cfg_phase3.py`. All of Try-1 through Try-9 targeted that one
task; none touched Phase4. The `tryN.py` files themselves have been deleted
after this summary was written -- this is the historical record of what was
tried, why, and what happened.

## Goal

MuJoCo deploy testing of the Phase3-balance-floating checkpoint (which
already climbed stairs, including floating/open-riser ones, well) showed the
policy flattening/flapping its legs on flat ground at zero command. The
project across these tries was to fix that without regressing stair-climbing.

## Try-by-try

| Try | Change | Result |
|---|---|---|
| 1 | Anti-stall rewards: `base_height_climb`, `stall_penalty`, `stair_commit` + relaxed `bad_orientation` (`limit_angle` 0.8→1.0). Fixes the robot freezing at a stair edge instead of climbing. | `terrain_levels` 5.514 (up from 4.899 without the fix). **Promoted into the default.** Not about flat-idle yet -- this is the fix that made stair-climbing work in the first place. |
| 2 | `rel_standing_envs` 0.01 → 0.1 (`CommandsCfgGo2` drops this to 0.01 for all Go2 tasks -- only 1% of envs ever got a near-zero command). | MuJoCo: **worse**, not better -- more aggressive leg flapping. Root cause found afterward: `base_height_climb_reward` is unconditional on command, so 10x more standing episodes just gave a always-on "chase terrain height" term 10x more exposure to zero-command states, on terrain that isn't flat. |
| 3 | Gate `base_height_climb_reward` to 0 whenever `\|command\| <= 0.1`. | MuJoCo: much improved, but still some residual flapping. `wild_foot_clearance` was still unconditional on command. |
| 4 | Gate `wild_foot_clearance` (`adaptive_foot_clearance_reward`) the same way. Its swing-phase gate (`_cpg_leg_phases_rad`) is a pure open-loop clock (elapsed episode time only) -- independent of command or motion -- so it kept paying for a marching gait even at rest. | MuJoCo: **four-leg marching gone.** Only a small single-leg twitch remained, climbing unaffected. **Later confirmed (via an 8-way MuJoCo comparison across Try-1..Try-8) to be the best-tested config of the whole lineage**, though see the "confound" note below. |
| 5 | Added `quiet_standing_reward`: positive reward for all 4 feet planted + low joint velocity, gated only by command (weight 0.5). | Isaac metrics fine, but MuJoCo: flat-ground twitch fully gone, but **stair-climbing regressed** -- front legs would reach the 4th step, then the robot would fall, instead of driving the hind legs up. |
| 6 | Added a terrain-flatness gate to `quiet_standing_reward` (reused `wild_foot_clearance`'s roughness-gate pattern) so it can't fire on/near a stair; cut weight 0.5 → 0.15 as a second precaution. | MuJoCo: climbing recovered, but the flat-ground twitch **came back** -- 0.15 was calibrated for the term firing ~10x more often (any zero-command env), and the flatness gate made it fire far more rarely (only standing-env-on-a-flat-cell, ~1% of the population). |
| 7 | Raised `quiet_standing_reward`'s weight back to 0.5, keeping Try-6's flatness gate (the gate, not the weight, is what protects climbing). | `terrain_levels` 5.26, `stair_commit` 0.27 (best of the lineage). MuJoCo confirmed both the flat twitch and the climbing regression resolved together. **Promoted into the default** (twice -- see below). |
| 8 | Terrain-mix-only variant: dropped `pyramid_stairs_wide`, moved its 0.10 proportion onto `pyramid_stairs_inv` (0.20→0.30). Rewards unchanged from Try-7/default. | `terrain_levels` 5.355 -- comparable to, marginally better than, Try-7. Not conclusive enough to promote over Try-7's terrain mix. |
| 9 | Replaced `quiet_standing_reward` with `idle_joint_vel_penalty`: a plain continuous quadratic penalty on joint velocity (no contact dependency at all), gated only by command. Motivated by the diagnosis that `quiet_standing`'s hard "all 4 feet planted" AND-gate is fragile -- ordinary contact-sensor noise can flicker it to zero even while genuinely standing still, giving the policy something to actively probe/chase, which is a plausible explanation for why it made things *worse* in Try-5/6. | First attempt was **contaminated by a real bug**: because Try-3/4's classes inherit transitively from the *mutable* `RewardsCfgPhase3BalanceFloating` in the default file (not a frozen snapshot), Try-9 silently also inherited whatever `quiet_standing` currently was there (restored by that point) -- so it trained with both terms fighting over the same regime, not `idle_joint_vel_penalty` in isolation. Fixed by explicitly zeroing the inherited `quiet_standing`, retrained with a pinned seed (42) for a controlled comparison -- still landed notably degraded (`terrain_levels` 4.67 vs the usual ~5.3, `bad_orientation` 0.27 vs ~0.05-0.09). Never fully isolated whether that's `idle_joint_vel_penalty` itself or the seed/other confounds (see below) -- **abandoned, not promoted.** |

## Two important cross-cutting findings

**1. PPO training-run variance confounds single-run comparisons.** No run in
this lineage pinned a random seed (except Try-9's retry). Re-training the
*exact* Try-4 reward config fresh from Phase2 -- meant to reproduce a known-good
result -- landed "aggressively flattening," markedly worse than both the
original Try-4 checkpoint and Try-7. That means several of the verdicts above
("Try-4 beat Try-7," then later "Try-7 beat a fresh Try-4 retrain") may partly
reflect which random seed a given run happened to get, not just the reward
design. Take the specific numbers above as directional, not definitive.

**2. Sandbox tries built by inheriting from the shared default class are not
frozen snapshots.** `try3.py` imports `RewardsCfgPhase3BalanceFloating`
directly from `velocity_env_cfg_phase3.py`, and every later try (4 through 9)
chains off try3/try4. Every time that shared class was edited (promoting
Try-7, reverting to Try-4, restoring Try-7), every sandbox try built on top of
it silently picked up whatever the *current* state of that file was, not the
state it had when that try was originally created. This caused a real,
initially-undetected bug in Try-9's first attempt. If sandbox tries are used
again in the future, make them self-contained rather than inheriting from a
class under active edit.

## Related fix (not a numbered try, but part of the same debugging arc)

A separate bug was found and fixed directly in `mdp/rewards.py`
(commit `9c9b986`, "fix ray scan related terrain reward"): `base_height_climb_reward`,
`foot_clearance_terrain_adaptive`, `adaptive_foot_clearance_reward`, and
`stair_commit_reward` all do a "nearest valid height-scan ray" lookup, and
three of them subtract two independently-selected heights. RayCaster reports
`±inf` for a ray that misses the terrain mesh entirely (plausible on the
floating/open-riser stairs terrain, where the robot can fall through a gap).
If a *whole* scan missed, `inf - inf = NaN` could result, and a NaN reward
corrupts PPO's batch-wide return/advantage normalization, not just the one
env. Fixed by sanitizing ray heights to a finite out-of-range sentinel before
any arithmetic, and explicitly zeroing the reward/detection flag when a scan
has no valid ray at all. Reviewed and confirmed sound.

## Where things landed

The default `RewardsCfgPhase3BalanceFloating` currently matches **Try-7's**
recipe: `rel_standing_envs=0.1`, `base_height_climb` and `wild_foot_clearance`
both gated by command, `stall_penalty`, `stair_commit`, and `quiet_standing`
(weight 0.5, gated by command *and* terrain flatness). This is the most
recently MuJoCo-confirmed-good configuration, kept as the default despite the
training-variance caveat above -- further chasing of the residual flat-idle
behavior was deprioritized in favor of moving on to Phase4 (thin-wall)
training.
