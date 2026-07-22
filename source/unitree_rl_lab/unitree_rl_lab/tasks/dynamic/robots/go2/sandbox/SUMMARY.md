# Jump task sandbox — summary of experiments

Context: reproducing the Jump task from the EFGCL paper ("Learning Dynamic
Motion through Spotting-Inspired External Force Guided Curriculum Learning")
on Unitree Go2, across `Unitree-Go2-Jump-Phase1` (quiet standing),
`Unitree-Go2-Jump-Phase2` (vertical jump), and `Unitree-Go2-Jump-Phase3`
(backflip/sideflip). The `try*.py` files that produced these results have
been deleted after this summary was written; the working, current task
configs remain in `jump_env_cfg_phase{1,2,3}.py`.

## Fixes made to the base tasks (before any sandbox tries)

- **Physics-based jump assist force.** Replaced a hardcoded 400N constant
  with the paper's projectile-motion formula, `F = m*sqrt(2*g*h_target) /
  assist_duration_s`, with robot mass auto-detected from the simulated
  asset. Lets `target_height` vary per episode instead of being fixed.
- **Curriculum state not persisting across training restarts.** The EFGCL
  assist-force curriculum (`assist_scale` and per-motion episode/success
  counters) lived only in the `JumpCommand` Python object, which rsl_rl's
  checkpoint format doesn't save. Every `--resume` silently reset the
  curriculum back to `initial_assist_scale=1.0`. Fixed by persisting state to
  a JSON file (`state_file` on `JumpCommandCfg`) tied to the task, loaded on
  init and saved on every curriculum decay step.
- **Reward-metric exploit (the big one).** `max_height`/`success` were
  measured relative to `standing_height`, which was re-captured at the exact
  instant the jump command triggered rather than fixed at
  `nominal_standing_height`. The policy learned to crouch low right before
  triggering, then simply stand back up — registering as a near-perfect
  "jump" with zero actual airtime. Caught via a play-mode diagnostic (robot
  visibly crouching to the ground, not leaving it) and confirmed by data:
  `standing_height` at trigger (~0.15m) vs. `root_height_max` (exactly
  0.40m, never exceeding nominal standing height). Fixed by removing the
  dynamic re-capture, always using the fixed `nominal_standing_height`.
  Phase2 was retrained from scratch after this fix and confirmed to jump for
  real (root height 0.40m -> 0.59m peak, ~44-55% of envs briefly fully
  airborne).
- **`export_deploy_cfg` crash on non-velocity tasks.** The shared
  `train_and_aggregate.py` wrapper always passes `--deploy-keyboard-commands`,
  which tried to rename a `velocity_commands` observation that jump/flip
  tasks don't have. Fixed by making the rename a no-op when the key is
  absent instead of raising.

## Try 1 & 2 — stronger angular-velocity penalty (tilt fix, attempt 1)

Play-mode observation after the metric fix: the robot genuinely jumps, but
the body tilts heavily in flight (quantified: pitch 37.9 deg, roll 16.8 deg
at peak height, very consistent std <1 deg across 128 envs — a learned
pattern, not noise).

- **Try1**: `non_target_rotation` weight -0.05 -> -0.3 (6x). Caused a flat
  training plateau — `assist_force` stuck at 0.35 for 2000 iterations,
  `max_height` flat, `success` collapsed near 0. Too strong; fights the
  height objective into a standstill.
- **Try2**: -0.05 -> -0.15 (3x). Converged cleanly (assist decayed to 0,
  success ~100%), but did **not** reduce tilt — pitch 41.7 deg (~same),
  roll 38.2 deg (**worse** than baseline). Angular-velocity penalty only
  discourages *continuing* to rotate; if the tilt is set by a brief
  high-omega impulse at push-off and the body then coasts at that angle,
  the velocity term barely touches the actual problem.

## Try 3 & 4 — direct root-orientation penalty (tilt fix, attempt 2)

Added a new always-on reward term using the existing `mdp.upright_reward`
(previously only active pre-trigger), so it also applies during flight —
this deviates from the paper's Table I reward set but was suggested as the
most reliable fix once angular-velocity regularization alone proved
insufficient.

- **Try3**: weight 2.0 (matching Phase1's standing-task convention). Same
  plateau failure mode as Try1 — `assist_force` stuck at 0.27, `success`
  collapsed to 0 after peaking at 0.73 mid-training.
- **Try4**: weight 0.5. Converged cleanly, but again did not fix tilt —
  pitch 39.6 deg (~same), roll 36.8 deg (worse than baseline).

**Conclusion on tilt**: across 4 tries and 2 different levers (angular
velocity, direct orientation) at 4 different weights, pitch consistently
lands in the 37-42 deg band regardless of intervention, and roll gets worse
whenever either penalty is added. This looks structural rather than a
reward-weight tuning problem — plausibly related to Go2's CoM being offset
~2cm forward of the geometric center between the front/rear hip attachment
points (confirmed via URDF inspection: hips are exactly symmetric at
+/-0.1934m, but the base link's own inertial CoM sits at x=+0.0211m), which
would give equal-magnitude front/rear assist forces unequal moment arms and
a resulting net pitch torque. Not further investigated (deprioritized to
avoid over-spending time on diagnosis per user direction). Tilt fix
abandoned as a dead end for now; current Phase2 checkpoint still tilts
~38 deg in flight but jumps successfully.

## Try 5, 6, 7 — raising the jump height target

Motivation: 0.20m of vertical liftoff gives very little airtime, likely not
enough margin for a full rotation for future flip-style motions.

- **Try5**: `target_height_range` 0.20 -> 0.40 (2x). Flat plateau —
  `assist_force` stuck at 0.73, `max_height` capped ~0.28-0.30m even with
  73% assist still active.
- **Try6**: 0.20 -> 0.30 (1.5x, smaller step). `assist_force` decayed slowly
  but never plateaued fully flat (still ~0.51 after 2400 iterations);
  `max_height` settled at ~0.19-0.20m — essentially the *same* ceiling as
  the working 0.20m baseline, despite 51-100% assist still active throughout.
- **Try7**: kept Try6's 0.30m target, added `mdp.jump_takeoff_velocity_reward`
  (new function in `mdp/rewards.py`, rewards upward CoM velocity in a 0.3s
  window right after trigger — a denser signal than waiting for height gain
  to already show up) plus relaxed `joint_torques`/`action_rate` weights
  (halved), mirroring a pattern that worked for a similar plateau in the
  locomotion/stair-climbing sandbox. Still plateaued at the same ~0.19-0.20m
  ceiling (`assist_force` stuck at 0.57).

**Conclusion on height**: three different approaches (bigger ask, smaller
ask, denser reward + relaxed effort penalties) all converge to the same
~0.20m ceiling. Cross-checked against Go2's actual actuator spec
(`UnitreeActuatorCfg_Go2HV`: Y1=20.2 N*m / Y2=23.4 N*m peak torque per
joint) — a rough energy estimate puts unassisted jump capability in the
0.3-0.45m range depending on leg-kinematics assumptions, genuinely
borderline. The consistent ~0.20m ceiling across every reward-engineering
attempt is more consistent with a real physical/actuator limit (or the
fixed 0.1s assist/push-off window not giving the robot's own legs enough
time to contribute as assist decays) than with a reward-shaping problem.
`mdp.jump_takeoff_velocity_reward` remains in `rewards.py` (unused by any
current task) in case this is revisited later.

**0.20m accepted as the practical target height for Phase2.**

## Phase 3 (backflip) — works, no sandbox needed

Given the height ceiling, tested whether a backflip (a different motion —
asymmetric front-leg push generating rotation, not a pure vertical jump,
using the paper's own fixed 350N force design, independent of the height
formula) is achievable at all. Trained the existing, unmodified
`Unitree-Go2-Jump-Phase3` task fresh from Phase1 — converged cleanly in a
single run: `assist_force` fully decayed to 0.0 by iteration 1200,
`success` reached and held 98.7-100% from iteration 1800 onward, and
`base_contact` (failed/crashed landing) dropped to 0.000 by iteration 1700.
The height ceiling found in Phase2 does not appear to block the backflip
motion.
