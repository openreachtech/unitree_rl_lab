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

## Try 1-8 — natural pre-jump crouch pose (backflip aesthetics fix)

Context: the backflip trained cleanly (above), but play-mode review showed
an unnatural pre-jump pose — legs splayed outward (hip abduction) instead of
tucking vertically, and the crouch starting immediately after spawn and
being held for the entire pre-trigger window instead of only briefly before
liftoff, since a real robot gets no advance notice of when a command fires.
Fixed in two independent stages; both are now merged into the default
`Unitree-Go2-Jump-Phase3` task (`jump_env_cfg_phase3.py`).

**Try 1 — leg-splay fix (promoted).** `.*_hip` joints are abduction/adduction;
splay is a hip problem, not a thigh/calf one. Added `hip_deviation`
(`mdp.joint_deviation_l1` restricted to `.*_hip_joint`, weight -0.4) as a
continuous penalty. Converged cleanly first try: `assist_force` decayed to
0.0, `success` 0.99. Root cause: `FlipRewardsCfg` never had any joint-pose
term at all pre-trigger (`pre_jump_standing_reward` only checks
upright/stillness), so hip splay was free.

**Try 2 — anticipation-timing fix, first attempt (not viable as tried).**
Widened `trigger_time_range` (0.8,1.2)->(0.5,2.5) and added
`pre_jump_pose_reward` (new function, penalizes joint-pose deviation from
default whenever the jump command is idle) to make holding an early crouch
costly. Result: unstable, oscillating success (0.44 up to 0.6 down to 0.006
repeatedly), never converged. Diagnosis (confirmed later by Try 8): the
policy still needed to *self-discover* a crouch-load to launch well, so the
new pose penalty fought a real dependency instead of just removing a free
exploit.

**Try 3-6 — external-force windup delay (dead end, but informative).**
Hypothesis: give the assist-force launch a delay after trigger
(`assist_delay_s`, new field on `JumpCommandCfg`) so the policy has a legal,
reward-gate-exempt window to crouch-load before the shove lands. Every
variant collapsed totally (100% `base_contact`, 0% success from the very
first logged iteration) regardless of delay length (0.12s or 0.04s
identically) or a reward-gate fix for the resulting dead zone
(`pre_jump_standing_reward_windup`, new function -- kept, later reused by
Try 8). Root cause: the network resumed from Phase1 has never once seen
`enabled=1` (Phase1 never triggers it), and a **hard** delay creates an
abrupt idle -> dead-window -> sudden-full-force discontinuity that a
never-before-seen-enabled=1 network handles catastrophically, independent of
window length or reward shaping.

**Try 7 — smooth force ramp fixes the discontinuity (promoted).** Replaced
the hard delay with `assist_ramp_s` (new field): launch force ramps linearly
0->full over the window instead of a step, removing the discontinuity
entirely (no dead window: force starts building immediately at trigger).
Converged as cleanly as Try 1 (success 0.99-1.00 sustained iterations
1000-3200). Confirms the lesson: it was never delay *length* or reward
shaping, purely the hard discontinuity.

**Try 8 — crouch-assist force + wider timing + strong pose penalty
(promoted, current default).** Combines everything: (1) `crouch_assist_force`
+ `crouch_assist_duration_s` (new fields) -- a brief downward pulse on all 4
legs (added `RL_hip` to `assist_body_names`, previously only 3 legs were
resolved since Phase3's launch profiles only ever needed FR/FL/RR) right at
trigger, shaped as a linear 0->peak->0 envelope so it starts and ends at
zero force; `assist_delay_s` set equal to `crouch_assist_duration_s` so the
(already-ramped, from Try 7) launch force begins exactly as the crouch pulse
ends -- both sides of that handoff are at ~0 force too, so the whole
timeline is discontinuity-free. (2) `trigger_time_range` widened to
(0.5, 2.0). (3) `pre_jump_pose_reward` at weight 1.0 (vs Try 2's 0.5).
Physically supplying the crouch-load removed the tension that broke Try 2:
the pose penalty no longer fights a real dependency. Converged cleanly:
success climbs smoothly through iterations 600-1000 (0 -> 0.56 -> 0.83 ->
0.97) then holds 0.99+ through 3200 with no oscillation; `pre_jump_pose`
reward climbs steadily to ~0.80 and holds (vs Try 2's plateau at 0.35-0.37);
`base_contact` 0.000.

Backflip launch force remains 2 front legs only (`FR_hip`, `FL_hip`) as
before -- only the new crouch-assist pulse uses all 4.

## Try 1 (new investigation) — sim2sim gap: MuJoCo backflip transfer (promoted)

Context: Try 8's backflip worked great in Isaac Lab (99%+ success) but sim2sim
testing in MuJoCo (via `State_Flip`/`go2_ctrl`) mostly failed -- the robot
didn't complete a full 360 deg rotation and landed on its back. Root-caused
via two rounds of investigation, both now promoted into the default task.

**Friction randomization.** Raising MuJoCo's foot friction to match
IsaacLab's fixed training value (1.0) made things *worse*, not better --
ruled out a simple "match the friction number" fix. Closer observation
showed successful (low-friction, 0.4) MuJoCo runs involved the hind legs
briefly sliding forward during push-off, a behavior never seen in IsaacLab,
suggesting contact dynamics (not just the nominal friction coefficient)
differ enough between PhysX and MuJoCo that no single friction value
reliably transfers -- especially significant here since a crouch-then-launch
is a short, high-force, impulsive contact event. Added `physics_material`
randomization (`mdp.randomize_rigid_body_material`, static/dynamic friction
0.4-1.2, matching the pattern already used in the locomotion task) so the
policy doesn't overfit to one assumed grip level. Trained cleanly (success
0.99+ by iteration ~1100). Sim2sim result: "a bit improved" over the
non-randomized policy, but still mostly failed at both 0.4 and 1.0 --
friction was a real but minor contributor, not the dominant cause.

**Actuator armature (dominant cause, fixed).** Added telemetry logging to
both the deploy-side `State_Flip` (new `telemetry.csv`, logging
accumulated pitch/tilt/joint velocity+torque) and `play.py` (new
`--telemetry-csv`/`--telemetry-steps` flags) to get directly comparable
traces. Isaac Sim: `RL_thigh_tau`/`RR_thigh_tau` hit and *sustain* their
20.2 N*m torque ceiling for ~0.18-0.20s straight during push-off, reaching a
full -360 deg rotation. MuJoCo (same policy): the same joints reach a
comparable *peak* torque (~22.8 N*m) for essentially one sample before
dropping away, reaching only ~-125 deg before completely stalling (residual
angular velocity ~0). Sustained-vs-momentary torque saturation at the same
peak is the signature of an effective-inertia mismatch: `armature` was unset
(defaulting to 0) on `UnitreeActuatorCfg_Go2HV` in
`assets/robots/unitree_actuators.py`, unlike every sibling actuator class in
that file (M107_15/24, N7520_*, N5010_16, N5020_16, W4010_25), which all
have an explicit, physically-derived armature from real rotor+gearbox
specs. A real geared motor's reflected rotor inertia is never exactly zero
-- this was almost certainly an oversight specific to Go2, making the
simulated joint "lighter"/quicker to track a fast target than reality, which
MuJoCo (and presumably real hardware) can't reproduce. Fixed with
`armature = 0.0122` (empirical standard for the real GO-M8010-6 actuator,
6.33:1 reduction -- close to but more targeted than MuJoCo's own generic
`armature="0.01"` default). This is a **shared asset change affecting every
Go2 task**, not sandbox-scoped.

Retrained with both fixes together (friction randomization + armature):
converged even faster than the friction-only run (success 0.99+ by
iteration ~900 vs ~1100-1400). Confirmed working in MuJoCo. Promoted into
the default `Unitree-Go2-Jump-Phase3` task (`events.physics_material` in
`jump_env_cfg_phase3.py`; armature fix lives in the shared actuator config).

## Try 1 (unified jump + backflip + sideflip) -- in progress

Enabling `enable_jump=True, enable_backflip=True, enable_sideflip=True`
together (plus restoring `target_height_range` to (0.20, 0.20), the
Phase2-proven achievable jump height) hit two separate issues.

**Crash: PPO numeric divergence at seed 42 (resolved -- not a config bug).**
Training crashed deterministically (`RuntimeError: normal expects all
elements of std >= 0.0`) at the exact same point on repeated launches --
`action_rate` reward exploded to ~-2.4e20 before the policy's action
distribution produced a negative/NaN std. Traced to `rsl_rl`'s default
`agent_cfg.seed = 42`, meaning "retries" of an unseeded run are not
independent trials -- both launches were byte-for-byte identical. Added a
`--seed` passthrough to `train_and_aggregate.py` (previously had no way to
override it). A different seed (7) trained the full 3000 iterations without
crashing, confirming this was a seed-specific instability, not a structural
problem with the combined config.

**Structural stall: sideflip's assist force was miscalibrated (root-caused
in Try 2-4 below).** Even without crashing, `assist_force` never decayed
from 1.0 despite the aggregate success sitting at 65-70% -- the saved
curriculum state showed `curriculum_success_rate: 0.0` (the *minimum*
success rate across all three enabled motions), meaning one motion
essentially never succeeded, which blocks `assist_force_decay` for all
three since it requires every enabled motion to individually clear the
60% threshold.

## Try 2-4 -- isolate and fix sideflip_assist_force (resolved)

Sideflip had never been trained even once before this project
(`TRAIN_SIDEFLIP` was `False` throughout). Isolating it alone (Try 2,
default 600N) never succeeded once in 3200 iterations: `max_height`
started very high (0.94-1.13m, far more than backflip's usual 0.1-0.3m)
and `motion_progress` *declined* over training instead of improving --
signature of a chaotic, uncontrolled launch rather than a clean in-place
roll, with the policy learning to tone down the wildness rather than
complete the rotation.

Bisected the force: Try 3 (200N) never left the ground at all (`max_height`
~0.000, `motion_progress` flat at the never-attempted baseline the whole
run). Try 4 (350N, matching backflip's proven value exactly) converged
cleanly: success climbed to 0.89 by iteration 700, 0.98 by 800, held
0.99-1.00 through 3200, `assist_force` decayed fully by iteration 900 --
even faster than backflip's own convergence. Root cause: 600N (near double
backflip's force) was simply too strong for the roll axis, which likely
has a smaller moment of inertia than the pitch axis backflip uses, so
sideflip needed *less* force than backflip, not more.

**Promoted**: `sideflip_assist_force` fixed to 350.0 (was 600.0) in the
default `jump_env_cfg_phase3.py`. `enable_sideflip` stays `False` in the
default task -- this only corrects the dormant value for whenever sideflip
is actually enabled (i.e., the eventual unified-motion retrain in Try 1).

Also found and fixed independently: the deploy-side `config.yaml` commanded
`target_roll_turns: 1.0` for the `5` (sideflip) key, but training always
used `target_roll_turns_range=(-1.0, -1.0)` -- an exact sign mismatch that
made the policy silently not react to the sideflip command at all (an
out-of-distribution target it had never seen). Fixed to `-1.0` to match
training.

**Next**: retry Try 1 (unified 3-motion) with the corrected sideflip force
and an explicit `--seed` override.

## Try 1 retry -- unified jump + backflip + sideflip (promoted, current default)

Retried with sideflip's corrected 350N force and `--seed 7` (avoiding the
deterministic seed-42 crash from before). Converged cleanly: success climbed
through iterations 400-1000 (0.31 -> 0.70 -> 0.98), dipped to ~0.77-0.82
around iterations 1200-1700 as the assist-force curriculum aggressively
decayed (0.90 -> 0.0 over that window -- a normal, temporary difficulty
spike during the assist-weaning transition), then recovered to hold
0.99-1.00 from iteration 2200 through 3200. The saved curriculum state
confirmed `curriculum_success_rate: 0.998` -- the *minimum* success rate
across all three motions, meaning jump, backflip, and sideflip are each
individually near 99.8% success, not just a favorable aggregate. Confirmed
working in MuJoCo.

Promoted into the default `Unitree-Go2-Jump-Phase3` task:
`TRAIN_JUMP = TRAIN_BACKFLIP = TRAIN_SIDEFLIP = True`, `target_height_range`
restored to `(0.20, 0.20)` (previously a backflip-only leftover at
`(0.0, 0.0)`). One motion is sampled uniformly at random per environment
per episode reset (`JumpCommand._resample_command`). Checkpoint copied from
`unitree_go2_jump_phase3_try_1/2026-07-24_13-52-13` (same recipe, no
retrain needed).
