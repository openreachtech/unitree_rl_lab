# Sandbox Try-1 .. Try-7: getting Go2-Biped onto two legs

Summary of the sandbox experiments that shaped the current `Go2-Biped`
default in `biped_env_cfg.py`. The `tryN.py` files themselves are kept in
place (not deleted) as historical record, matching the convention already
established in `tasks/locomotion/robots/go2/sandbox/`.

## New round: multi-style (quad/hind-biped) walking -- abandoned mode-switching-in-one-step approach

A second round (also numbered from Try-1, files since cleared/reverted --
not the same tryN.py as Round 1 above) explored combining ordinary 4-leg
walking and the hind-biped stance into a single policy that switches gait
mode within an episode via a `gait_mode` command. Every variant tried --
probabilistic mode resampling (`GaitModeCommand`), a repeating scripted
quad/biped schedule with an external assist force (`ScheduledGaitModeCommand`
+ shoulder-lift force), a single-transition-per-episode schedule
(`SingleTransitionGaitModeCommand`), and a direct multimodal edit of
`biped_env_cfg.py` itself -- converged to a policy that **correctly received
the `gait_mode` observation** (confirmed via the printed observation-term
table: present at the right index, right dimension, no wiring bug) but
**never produced any visible mode-dependent behavior change**, regardless of
reward-weight configuration (unmodified multimode defaults, a boosted
`biped_weight`, or weights matched exactly to Phase1's own proven values) or
training length (up to 8000-9900 iterations). The robot simply kept walking
on 4 legs the entire time, in every variant, even as the `mode_orientation`
reward metric itself sometimes rose substantially.

Conclusion: asking one policy to learn "how to stand on two legs" and "when
to switch based on gait_mode" simultaneously, from scratch, in a single
training run, did not work in any tried configuration -- this looks like an
optimization/exploration problem, not a reward-tuning or wiring problem.

## Current round: decouple "learn to use the gait_mode input" from "learn when to switch"

New approach (`try1.py`): keep the *proven* single-stance `Go2-Biped-Phase1`
reward/termination set completely unchanged, and only add a `gait_mode`
one-hot observation pinned constant to hind-biped the entire time (never
resampled, never quad -- see `mdp.PinnedGaitModeCommand`). Goal: a checkpoint
that already walks well on two legs (Phase1's own proven recipe, untouched)
and already has its network's `gait_mode` input dimension "warmed up" around
a real signal, before ever asking it to condition behavior on an actually
*varying* mode. A later stage can swap in a real switching command (e.g.
`GaitModeCommand`) and resume from this checkpoint -- same observation shape,
no architecture change -- rather than repeating the previous round's
all-at-once approach.

Training/play-testing result: **success, promoted into `../biped_env_cfg.py`.**
Trained 7000 iterations from scratch. `tilt_reward` climbed smoothly to
~0.75-0.77 by iteration 4000 and held steady there for the rest of training
(matches original Phase1's own historically validated ~0.72-0.78 range);
`time_out` stayed 0.98-0.99 throughout; `base_contact`/`collapsed` both stayed
very low (~0.7%/~0.2%). Play-mode video confirmed genuine, sustained 2-leg
standing/walking across all test envs -- indistinguishable from the original
Phase1's own quality. Confirms the hypothesis: adding a constant `gait_mode`
observation input doesn't hurt Phase1's proven training at all.

**Promoted directly into `../biped_env_cfg.py`** (the `Go2-Biped-Phase1` task
itself now carries the pinned-hind-biped `gait_mode` observation;
`RewardsCfg`/`TerminationsCfg` unchanged from before). `biped_env_cfg_front.py`
was also updated to pin its own `gait_mode` to front-biped instead of
inheriting the hind-biped default (cosmetic-only -- that file's reward set
never reads `gait_mode`). `biped_env_cfg_phase2.py` needed no changes at all
(its `front_contact_force` reward-field override and `base_velocity` override
are both untouched by this promotion). `try1.py` has been cleared back to a
placeholder per the established post-promotion convention.

Next stage (not yet started): swap the pinned `gait_mode` command for an
actually-varying one (e.g. `GaitModeCommand`) and resume training from this
promoted Phase1 checkpoint, to finally attempt real quad<->hind-biped mode
switching on top of a network that already has its `gait_mode` input "warmed
up" -- rather than repeating the previous round's all-at-once approach.

## Third round, continued: real switching attempt (Try-2), then reward-shaping on the pinned stance (Try-3..7)

**Try-2**: attempted the "next stage" above -- swapped the pinned `gait_mode`
for `SingleTransitionGaitModeCommand` (one quad -> hind-biped -> quad cycle per
episode) and resumed from the promoted Phase1 checkpoint, with
`BipedOnlyVelocityCommand` keeping the velocity command non-zero specifically
during the hind-biped stretch (Phase1's own Try-6 lesson: a demanded pace slow
enough that a tripod satisfies it removes all pressure to commit to a genuine
2-leg gait). **Result: the mode-ignoring failure from the second round recurred**
-- the policy stayed quad-postured for almost the entire episode regardless of
the commanded `gait_mode`, confirmed both numerically (`mode_orientation`
reward near zero, far below the reference `0.482` from an earlier attempt) and
visually (population-wide video check: 7/8 sampled envs never left quad
posture across a full 20 s episode). Root-cause investigation with the
operator concluded the paper's own quad<->biped transition demo is not
evidence this is learnable in one policy either -- it trains two *separate*
specialist policies (quad-only, biped-only) sharing a reward *framework* but
not weights, and demonstrates the transition by swapping which one drives the
robot at deployment, matching this project's own earlier (also successful)
inference-time policy-swap experiment. Single-policy live switching remains
unsolved; not pursued further this round.

**Try-3..7** pivoted to reward-shaping questions on the *pinned* hind-biped
stance itself, prompted by the operator noticing Go2-Biped-Phase1 stood
unnaturally straight-legged with stiffly forward-pointing front legs compared
to other published Go2 biped demos:

| Try | Change (all fresh-trained unless noted) | Result |
|---|---|---|
| 3 | `base_height` target 0.55 -> 0.45 m (`BASE_HEIGHT_TARGET` was copied from an unrelated reference robot and is "well above" Go2's own quadruped stance height by design). | Confirmed via play-mode video: visibly bent knees instead of near-full leg extension. Stability (`time_out`/`collapsed`) only marginally worse than Phase1. |
| 4 | `front_hip/thigh/calf_motion` (joint-deviation pins to the quadruped-flat default pose) weights -> 0.0, to see what the front legs do unconstrained. | Operator's own play-mode check: no visible change from Phase1's stuck-out-looking pose. Root cause: the action space is parameterized as an *offset from* `default_joint_pos`, so outputting ~0 action reproduces the old pose for free, and the remaining smoothness/energy reward terms (`action_rate`, `energy`, `joint_torques`) still bias toward small actions regardless of this term's weight -- removing a penalty doesn't create an incentive to move. |
| 5 | Added `front_leg_hang_penalty`: an explicit world-frame reward for the front feet hanging directly below the front hips (unlike a body-frame joint target, stays meaningful regardless of the ~80-90 degree torso pitch). | Front legs did visibly hang lower, but `front_contact_force` rose ~30x over Phase1's -- the pitched-up torso's shoulder height means a straight hang reaches close to the ground. **Not adopted**; function later deleted. |
| 6 | Combined the confirmed-good pieces: free front legs (try4-style, dropping try5's hang-down reward), `base_height` lowered further to 0.40 (below try3's 0.45), plus a new `stand_still_penalty` (operator observed the robot drifting in play mode even at zero commanded velocity -- `track_lin_vel_xy_yaw_frame_exp`'s wide Gaussian tolerance doesn't punish a few tenths of m/s of residual drift). Weight -1.0. | Best result of the round: `tilt_reward` 0.767 (better than Phase1's own 0.759), stability at or above Phase1's level, visibly bent knees, relaxed-looking front legs. `stand_still` reward plateaued early (~-0.002 to -0.003 from iteration 100 through 9000) and the operator confirmed via MuJoCo play mode the drift was still present -- weight too weak to matter next to terms like `tilt_reward`. |
| 7 | Same as Try-6, `stand_still` weight -1.0 -> -5.0. Tried 3000 iterations fresh, then resumed +5000 (8000 total) to rule out undertraining. | Raw drift magnitude roughly halved at 3k, but `front_contact_force` roughly doubled (bracing trade-off) and `undesired_contacts` rose too. At 8000 total, `tilt_reward` improved modestly (0.733 -> 0.755) but a still-meaningful fraction of the population **degenerated into a 3-leg ("tripod") stationary hold** instead of genuine 2-leg standing -- trivially satisfies both `stand_still` and the height/tilt rewards without the harder dynamic balance. **Not adopted.** |

**Try-6's recipe was briefly promoted into `../biped_env_cfg.py`** (`BASE_HEIGHT_TARGET`
0.55 -> 0.40, `front_hip/thigh/calf_motion` removed, new `stand_still` reward
term added at weight -1.0), **then reverted** after a side-by-side comparison of
the two checkpoints' final training metrics (both 10k iterations): Try-6 won on
`tilt_reward` (0.767 vs 0.759) but Phase1's original recipe was equal-or-better
on every stability/robustness metric that mattered most -- `time_out` (0.996 vs
0.993), `base_contact` (0.003 vs 0.0044), and especially `front_contact_force`
(raw front-foot-contact roughly 13x lower: ~0.008 vs ~0.106, since freeing the
front legs entirely also let them touch the ground more). Operator's call:
keep Phase1's original straight-legged/pinned-front-leg/no-stand_still recipe
as the default, since it's the more robust checkpoint even though Try-6 looked
more natural. `biped_env_cfg.py` is back to its pre-Try-6 state; `stand_still_penalty`
was deleted from `mdp/rewards.py` (no longer used anywhere). The bent-knee
(try3) and free-front-leg (try4) *findings* are still valid and could be
revisited with different weights/thresholds in a future round -- see the
try-by-try table above for what was tried.

`SingleTransitionGaitModeCommand`, `BipedOnlyVelocityCommand`, and
`front_leg_hang_penalty` -- used only by the now-cleared Try-2/5 -- were
deleted from `mdp/commands.py` / `mdp/rewards.py` regardless (dead code
independent of the promotion/revert). All of Try-1..7's `tryN.py` files have
been deleted (not just cleared to placeholders).

## Goal

The default task (128 CPU envs, 9999 iterations, tilt_reward weight 0.8) had
training curves that looked healthy (tilt_reward 0.76, 85-90% survival) but
the checkpoint diverged catastrophically at play time -- actions escalated to
absurd magnitudes within ~30 steps of any reset and crashed the simulation,
reproducible at 100% rate across 128 test envs regardless of device, cold vs.
warm reset, or deterministic vs. stochastic action sampling. Root cause was
never conclusively found for that specific checkpoint; the project moved to
retraining with more environments (GPU, 4096 envs) instead of debugging it
further.

## Try-by-try

| Try | Change | Result |
|---|---|---|
| 1 | tilt_reward weight 0.8 -> 1.2, on top of a healthy 4096-env/2698-iteration baseline that had plateaued at tilt_reward ~0.5 (walking like a quadruped, no lift). | tilt_reward maxed out (~0.99) but play testing showed the robot lifting only one leg and freezing in place -- no walking at all. Confirmed the previous checkpoint's flat/quadruped-like gait wasn't a training-time issue, and that tilt alone isn't sufficient to produce genuine bipedal locomotion. |
| 2 | Compared against the cloned reference repo (`bipedal_locomotion_for_quadrupedal_robots`), specifically `outputs/random_dog/Imi/test_reward/train_cfg_robot.py` (paired with a real deployed rear-leg-biped checkpoint). Added `front_contact_force` (front-foot ground-reaction-force penalty) and `feet_air_time` (step-cadence reward, requires `track_air_time=True`); relaxed `pendulum_angle`/`pendulum_instability`/`handle_length` 3x/100x/5x to the reference's own validated weights. | **Collapsed catastrophically**: every episode fell within ~5-8 steps, sustained for the full 2000 iterations. Root cause: `front_contact_force` used raw continuous force magnitude (matching the reference's own formula), but Isaac Lab's `RewardManager` has no equivalent of the reference's `only_positive_rewards=True` (which clips each step's *total* reward to >= 0). Without that clamp, a raw-Newton penalty term dominates the whole reward sum over a long episode and makes ending it immediately cheaper than enduring it. Fixed by reworking `front_contact_force` to a bounded 0/1-per-foot threshold count (matching every other contact reward/termination already in this codebase), weight -0.3. |
| 3 | Isolation diagnostic (~100-150 iterations): added *only* the two new reward terms, left the relaxed stability weights unchanged, to determine whether Try-2's collapse came from the new terms or from relaxing pendulum/handle_length simultaneously. | Also collapsed instantly at the raw-force scale; trained normally once the same threshold-count fix was applied. Confirmed the bug was in `front_contact_force`'s reward scale, not the stability-weight relaxation. |
| 4 | `front_contact_force` weight -0.3 -> -0.6 (still boolean/threshold, not raw force), to push harder from a tripod gait toward lifting both front feet. | **Collapsed catastrophically again**, identical signature to Try-2 (100% `base_contact` for the full 2000 iterations). Confirmed the "cheaper to die" vulnerability isn't specific to raw-force scale -- it's a property of Isaac Lab's unclamped reward sum, and *any* sufficiently large penalty term can trigger it once its weight crosses some threshold (between -0.3 safe and -0.6 unsafe here). |
| 5 | Added `termination_penalty` (`mdp.is_terminated`, weight -200.0 -- matches `isaaclab_tasks`' own h1/g1/cassie `rough_env_cfg.py`, which share our exact `step_dt=0.02s`) on top of Try-4's config. | Fixed the collapse completely (100-iteration diagnostic: time_out 0.996, base_contact 0.004). Full 2000-iteration run: front-foot ground-contact time dropped 41% -> 14% -> 6% and was still trending down; tilt_reward ~0.72-0.74, time_out ~0.99. Resumed for another 2000 iterations (4000 total): front-foot contact kept falling to ~6%, velocity tracking *improved* to ~0.88. Resumed again (6000 total): plateaued -- front-foot contact stopped improving (~5%, no longer trending down), everything else held steady. Play testing confirmed real 2-leg balancing, but inconsistent (some robots on 2 legs, most still tripod-ish). |
| 6 | Compared our velocity command range (`lin_vel_x` +-0.4, `lin_vel_y` +-0.2, `ang_vel_z` +-0.4) against the reference's own validated config (same `test_reward` config as Try-2): `lin_vel_x/y` +-1.0, `ang_vel_yaw` +-1.0. Widened to `lin_vel_x` +-1.0, `lin_vel_y` +-0.5 (user's choice, narrower than the reference's y), `ang_vel_z` +-1.0, on top of Try-5's reward recipe. Trained fresh (10000 iterations, matching the reference's own `BipedalBaseCfgPPO.runner.max_iterations`). | **Best result across every attempt.** front-foot contact fell far below Try-5's plateau, down to ~0.5-0.8% (roughly 6-10x lower) and was still gently improving at 10000 iterations; tilt_reward ~0.78 (higher than Try-5); tracking held up on a much harder task. Play testing **confirmed genuine, sustained 2-leg walking for the first time**. Hypothesis: at the old slower demanded pace, a tripod shuffle could still track the command well enough that there was little pressure to commit to a full swinging 2-leg gait -- that pressure only appears once the required pace exceeds what dragging a foot can sustain. **Promoted into the default `Go2-Biped` task.** |
| 7 | Play testing on Try-6 showed a remaining issue: the robot tends to sidestep rather than walk forward/backward (plausibly because the hind legs are naturally spaced side-by-side, making lateral weight-shift the more mechanically stable balance direction -- closer to tandem/heel-to-toe walking than ordinary forward walking, for a human analogy). `track_lin_vel_xy_yaw_frame_exp` computes one *combined* `sum((cmd_xy-actual_xy)^2)` error, so it can't express "x accuracy matters more than y." Split into `track_lin_vel_x` (weight 1.0) and `track_lin_vel_y` (weight 0.5), trained fresh, 7000 iterations. | Did **not** resolve the sidestepping (still observed in play testing). Raw x-tracking accuracy (~0.89-0.91) was only modestly better than y (~0.84-0.88) -- the split addresses relative priority when both are commanded simultaneously, but likely doesn't touch a "sideways gait as default mechanism" habit. Also, front-foot-contact progress was measurably slower than Try-6 at the same iteration count (~0.044 vs ~0.005), plausibly because splitting the tracking term raised the total achievable tracking reward from 1.0 to 1.5, diluting `front_contact_force`'s relative weight in the mix. **Not promoted** -- Try-6 remains the default's reward/command basis; the sidestepping issue is still open.

## Fourth round, Try-1: the front stance's hardware failure was actuation delay

`Go2-Biped-Front` rose reliably in Isaac Lab and in MuJoCo but could not get into the
handstand on hardware -- it crept to ~55 deg of tilt and stalled. Placed in the pose by
hand it held it indefinitely, so the failure was specific to the dynamic rise.

Deploy-side telemetry was added for this (`State_BipedRL`: a 50 Hz per-policy-step CSV, a
1 kHz `tau_cmd`/`tau_app` capture over the rise, and an always-on summary line) and the
hardware trace was read against an Isaac Lab play trace of the same checkpoint:

| | hardware | Isaac Lab, same checkpoint |
|---|---|---|
| sagittal lean reached | 43 deg | 72 deg |
| lateral lean | 33 deg (leads the sagittal rise) | 15 deg |
| stance-shoulder `dq_des` RMS/step | 0.51-0.58 rad | 0.05-0.07 rad |
| stance-shoulder tracking error | 0.89-0.93 rad (2.5 peak) | 0.11-0.36 rad |
| stance-shoulder `|dq|` max | 24-31 rad/s | 3-8 rad/s |
| stance-shoulder net power | ~39 W each | 0.4-1.2 W each |
| `action_rate_l2` | 12.08 | 0.98 |

The policy's rise strategy is to hold the stance shoulders at the torque ceiling and let
the load set the speed -- in simulation they sit pinned at exactly Y1 = 20.2 N*m at only
3-7 rad/s, which has almost no phase margin. On hardware that became a 4-6 Hz saturated
limit cycle; because the two shoulders oscillated incoherently (torque cross-correlation
-0.05) the net effect was a roll moment rather than a pitch-up.

A play sweep isolated the cause. Adding actuation delay -- and nothing else -- reproduced
the hardware failure, monotonically:

| delay | max tilt | end sagittal | max lateral | FT `|dq|`max | `action_rate` | falls |
|---|---|---|---|---|---|---|
| 0 ms | 80.9 | 71.2 | 20.1 | 7.1 | 0.98 | 0/8 |
| 20 ms | 81.1 | 67.7 | 33.1 | 19.9 | 4.20 | 1/8 |
| 30 ms | 77.1 | 63.3 | 46.6 | 22.5 | 7.30 | 4/8 |
| 40 ms | 81.1 | 61.0 | 48.1 | 23.8 | 9.63 | 8/8 |
| hardware | 58.9 | 38.7 | 33.1 | 30.7 | 12.08 | n/a |

Cross-correlating the published position command against the motor's torque response in
the 1 kHz hardware trace puts the command-side latency alone at 8-10 ms; sensing lag and
the 50 Hz zero-order hold are on top. The 40 ms run also reproduced the hardware's exact
signature: the rise fails, but an attempt that *does* reach the pose then holds it with
`|dq|` ~ 0 -- which is what "holds it when placed by hand" looks like.

Two other candidates were tested in the same sweep and **ruled out** -- neither changed the
outcome at all (all 8 envs still reached ~80 deg tilt with ~15 deg lateral):

- raising the actuator envelope to what hardware actually delivers (Y1/Y2 = 30/37 N*m,
  torque-speed derating removed). Hardware's stance shoulders do spend 40-45% of the time
  outside the trained envelope, but that is a symptom of the oscillation, not its cause.
- starting from a settled quadruped stand instead of the training reset's ~9 cm drop, i.e.
  the FixStand pose the deploy FSM actually hands the policy.

Also ruled out from the trace itself: control-loop timing (the 50 Hz policy step measured
0.83 ms of its 20 ms budget, zero overruns), and -- from the operator's own earlier tests --
total mass and peak torque (a Jump task matched hardware height to simulation) and ground
friction (rubber floor).

**Change: `min_delay`/`max_delay` = 0..6 physics steps (0-30 ms) on the Go2HV actuator,
promoted into `../biped_env_cfg.py`** so every biped variant inherits it (Phase1, Phase2,
Front, Single, Multimode). Scoped there rather than on the shared `UNITREE_GO2_CFG`, which
every other Go2 task also uses.

Result, resumed from the deployed Front checkpoint for 1500 iterations:

- `tilt_reward` 0.7287 -> **0.7380** (better than the zero-delay original, while now
  handling 0-30 ms of lag); `time_out` 0.994 -> 0.987, `base_contact` 0.006 -> 0.011.
  `base_contact` was 0.201 at iteration 49 and fell to 0.011, so the policy adapted rather
  than sitting in the old local optimum.
- Delay-sweep acceptance test: **8/8 environments rise past 60 deg and settle at 69-79 deg
  sagittal at every delay from 0 to 40 ms**, with 0/8 falls at 40 ms where the old
  checkpoint fell 8/8. `action_rate` at 40 ms: 9.63 -> 4.17.
- MuJoCo, through the full deploy stack: rises in 1.2 s and holds for 8.3 s (sagittal
  69-74 deg, lateral 13-16 deg, base 23 cm up). The 4-6 Hz shoulder limit cycle is gone
  (dominant command frequency 6.02 Hz -> 0.12 Hz), stance-shoulder `|dq|` max 13.3 rad/s
  (inside the X1 = 13.5 derating knee), net power 3-5 W instead of 39-51 W, and 0.0% of
  the time outside the trained envelope.

Not yet re-validated on hardware. The like-for-like control that is still missing is the
*old* checkpoint measured in MuJoCo with the same telemetry -- MuJoCo passed with the old
checkpoint too, so a MuJoCo success alone does not prove the hardware fix.

## Fourth round, continued: the single-leg stance's blocker is the reset, not the delay

Applying the delay fix to `Go2-Biped-Single` (resumed from its deployed checkpoint for 1000
iterations) produced training curves that looked fully converged -- `mean_episode_length`
1000, `base_contact` 0.92 at iteration 50 falling to 0.0098, `tilt_reward` 0.7799, i.e.
*better* than the zero-delay original's 0.7693 while now handling 0-30 ms of lag.

It then failed in MuJoCo through the deploy stack: the base rose only 0.365 -> 0.451 m
(the 2-leg front stance reaches 0.597 m in the same setup), never pitched nose-down at all
(sagittal lean stayed between -21 and +8 deg where the 2-leg stance reaches +70), rolled
out to 87 deg of lateral lean, and was lying on the ground 1.9 s after entry. The stance
leg was not the problem -- FR's shoulder ran at |tau| p99 17.9 N*m with a 0.52 Hz command,
nothing like the 4-6 Hz limit cycle seen on hardware. The three *free* legs were pinned at
their ceilings instead (FL_calf 39.22 N*m at 30.8 rad/s, RL_thigh 23.40, FL_thigh 20.20),
flailing as counterweights whose net effect was the roll that tipped it over.

An Isaac Lab play ablation separated the two candidate causes -- and it is the reset, not
the delay:

| start condition | delay | max base_z | end sagittal | max lateral | falls (8 envs, 5 s) |
|---|---|---|---|---|---|
| drop reset (as trained) | 0 ms | 0.615 | 75.9 | 14.1 | 0 |
| drop reset | 40 ms | 0.621 | 75.9 | 42.2 | 8 |
| settled 4-foot stand | 0 ms | 0.441 | 15.6 | 68.2 | 37 |
| settled 4-foot stand | 10 ms | 0.442 | 14.9 | 65.6 | 39 |
| MuJoCo, deploy stack | ~8-10 ms | 0.451 | ~5 | 87 | fell |

The settled-start rows reproduce the MuJoCo failure almost exactly, and delay barely moves
them (0 ms vs 10 ms are within noise). The delay work did what it was for: from the trained
start the policy still rises at 40 ms.

The **old, pre-delay checkpoint fails from a settled start too** (max base_z 0.488, end
sagittal 48.2, 16 falls), so this predates the delay work -- the delay retrain deepened it
(48 -> 16 deg, 16 -> 37 falls) but did not create it. `biped_env_cfg_single.py`'s
"MuJoCo-confirmed balancing on one leg" was therefore a play-mode observation from the drop
reset; the deploy entry path had never actually been validated for this task.

Root cause: the reset dropped the base 7.65 cm every episode (`init_state.pos` z = 0.400
against a 0.3235 m foot-contact height at the default joint pose), so the stances learned a
rise that depends on that fall -- the downward momentum and leg compression it buys.
Deployment never provides it; the FSM hands over a robot standing still on four feet. The
2-leg stances happen not to care (a settled start costs Front nothing: 79.7/70.9/26.5 vs
80.9/71.2/20.1), but a point support has no margin for it.

**Change: `reset_base` pose_range gains `"z": (-0.0765, 0.0)`, promoted into
`../biped_env_cfg.py`** alongside the delay. The lower bound is exactly the contact height
-- verified at runtime, the lowest foot sits 1.4 mm above the ground at the bottom of the
range and never penetrates -- so a settled four-foot stand is now in distribution together
with the old 7.65 cm drop. Not yet retrained.

### Retraining both stances with the two fixes together

`Go2-Biped-Single`, resumed 1500 iterations from its *pre-delay* checkpoint
(`2026-08-07_18-55-14` -- the delay-only retrain had degraded the settled-start case, so it
was not used as the base): `base_contact` 0.9146 at iteration 50 -> 0.0039, `tilt_reward`
0.7799, `front_contact_force` -0.0492 (better than the delay-only -0.0669). The settled
start is now indistinguishable from the drop:

| start | delay | max base_z | end sagittal | max lateral | falls |
|---|---|---|---|---|---|
| settled | 0 ms | 0.616 | 77.1 | 14.3 | 0 |
| settled | 10 ms | 0.618 | 76.2 | 14.6 | 0 |
| settled | 40 ms | 0.623 | 77.0 | 85.7 | 8 |
| drop | 0 ms | 0.616 | 76.9 | 14.2 | 0 |
| drop | 40 ms | 0.623 | 75.9 | 59.5 | 8 |

(compare: pre-delay checkpoint from a settled start reached 48.2 deg and fell 16 times;
the delay-only one reached 15.6 deg and fell 37 times.) 40 ms is past the margin -- one fall
per environment, then recovery -- but the deploy stack's measured lag is 8-10 ms, where it is
clean. **MuJoCo through the deploy stack: confirmed.**

`Go2-Biped-Front`, resumed 1500 iterations from the delay-only checkpoint (which was already
validated, unlike the single-leg one): `tilt_reward` 0.7406, the best of the three
generations (0.7287 zero-delay -> 0.7380 delay-only -> 0.7406). The reset change cost only a
brief transient (`base_contact` 0.0295 at iteration 50, settling to 0.0131).

| start | delay | max base_z | end sagittal | max lateral | FT \|dq\|max | falls |
|---|---|---|---|---|---|---|
| settled | 0 ms | 0.564 | 75.8 | 23.8 | 9.4 | 0 |
| settled | 10 ms | 0.562 | 76.2 | 16.7 | 12.7 | 0 |
| settled | 40 ms | 0.564 | 75.5 | 12.7 | 21.2 | 0 |
| drop | 0 ms | 0.562 | 75.0 | 20.3 | 10.6 | 0 |
| drop | 40 ms | 0.552 | 73.8 | 20.0 | 18.8 | 0 |

Strictly better than the delay-only generation at 40 ms (max lateral 26.4 -> 20.0,
`action_rate` 4.17 -> 3.15) with zero falls in all ten conditions, at the cost of a slightly
less extended pose at 0 ms (0.576 -> 0.564 m, 77.9 -> 75.8 deg). **MuJoCo through the deploy
stack: confirmed.** Hardware not yet retried.
