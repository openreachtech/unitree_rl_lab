# Go2 bipedal stance -- summary of experiments

The front-leg handstand (`Go2-Multitask-Biped-Front`), trained as a fourth expert for the multi-task
policy. Reward design follows two sources: the balance terms come from TumblerNet by way of
`feat/biped`, which took both a hind-leg and a front-leg stance to hardware, and the task terms
follow "Bipedalism for Quadrupedal Robots" (Zhang et al.), Table I. See
`tasks/biped/mdp/stance_rewards.py` for which is which.

`feat/biped`'s own `sandbox/SUMMARY.md` is the prior record and is worth reading first -- four
rounds of it, including why mode-switching was abandoned, why the velocity command range decides
whether the gait is real, and the actuation-delay and reset-height findings that transfer directly.

The observation-layout change this expert forced is recorded on the merged policy's side, in
`tasks/multitask/robots/go2/sandbox/SUMMARY.md`.

## Handstand, run 1: a real stance, but a crouched one

2000 iterations from scratch, `2026-09-02_23-24-37`. The robot rises onto its front legs and walks
there -- confirmed visually in play mode -- with `base_contact` 0.011 and full-length episodes. The
newly added `stance_roll` term does its job: lateral lean peaks at a median 12 deg, against the 33
deg that accompanied `feat/biped`'s hardware failure.

Measured with `scripts/rsl_rl/measure_stance.py`, 32 environments, 20 s, delay drawn 0-30 ms --
the same conditions and the same quantities as `feat/biped`'s own stance tables:

| | this run (median) | `feat/biped` Front |
|---|---|---|
| sagittal pitch, peak | 64.5 deg | 75.8 |
| sagittal pitch, settled | 60.0 deg | 75.0 |
| base height, peak | 0.396 m | 0.564 |
| lateral lean, peak | 12.1 deg | 23.8 |
| stance hip height, settled | 0.187 m | -- |
| falls | 1/32 | 0/8 |

**The stance is 17 cm shorter than the reference, and the shortfall is a deep crouch.** The stance
hips settle at 0.187 m against the 0.30 m `front_hip_height` asks for, and the distribution is
tight (p10 0.181, p90 0.199) -- every environment agrees on the posture.

**It also runs the stance thigh joints past their continuous rating during the rise**: peak |tau|
median 21.2 N*m, max 24.5, against Y1 = 20.2. Holding the stance is cheap (9.5 N*m settled); the
cost is all in getting there. That is the exact signature of `feat/biped`'s hardware failure, where
the stance shoulders sat pinned at 20.2 N*m with no phase margin and 8-10 ms of command latency
turned it into a 4-6 Hz limit cycle. A deeper crouch loads those joints harder, so this is a
transfer risk even though the policy is stable in simulation.

### What is not the cause

`dof_pos_limits` carries -10.0 here (the merged set's value) against the bipedal recipe's -1.0, and
a joint-limit barrier is invisible in the reward log precisely when it is working -- the term
logged -0.0006. Measuring the travel directly rules it out: the stance calf settles at -2.04 rad
with 0.22 rad of margin **on its flexion side**, and the thigh sits 1.71 rad clear of anything. The
legs are not being stopped from extending; they are being folded further than the default pose, and
the only barrier they approach is the one limiting how deep the crouch can go. Ruled out.

### What the cause looks like

The two terms this task added on top of the validated recipe, `stance_held` (+1.0) and
`upright_balance` (+0.5, up to +1.0 realised), are both gated at `upright_alignment = 0.6` -- 37
degrees of pitch. Together they pay about +1.8 for reaching a shallow stance and holding it still,
and both are fully banked long before the posture is good. Against that, everything asking the
robot to stand taller totals roughly 0.03 per step (`base_height` -0.019, `front_hip_height`
-0.013). The measured trajectory is what that arithmetic predicts: pitch and height creep upward
for 2000 iterations and never arrive.

`feat/biped` had no completion term at all, which is why its own `base_height` -0.5 / 0.55 m -- the
same weight and target used here -- was able to drive the stance to 0.564 m.

**Next: separate the gate threshold from the success threshold.** Success moves to 0.93 (68 deg,
just under what the reference stance reaches) so `stance_held` has to be earned by a real handstand,
and the upright gate becomes a ramp from 0.6 to 0.93 rather than a step, so the rise keeps a
continuous gradient instead of meeting a cliff. `base_height` and `action_rate` stay where they are,
so the change can be attributed.

## Handstand, run 2: separating the gate threshold from the success threshold

One change from run 1, so the result attributes: `success_alignment` 0.6 -> 0.93 (37 -> 68 deg), and
the upright reward gate became a ramp from 0.6 to 0.93 instead of a step at 0.6. `base_height`,
`front_hip_height` and `action_rate` were left exactly as they were.

2000 iterations from scratch, `2026-09-03_11-06-40`, measured the same way as run 1.

| | run 1 | run 2 | `feat/biped` Front |
|---|---|---|---|
| sagittal pitch, settled | 60.0 deg | **74.6** | 75.0 |
| sagittal pitch, peak | 64.5 deg | **77.9** | 75.8 |
| lateral lean, peak | 12.1 deg | **7.6** | 23.8 |
| base height, peak | 0.396 m | 0.399 | 0.564 |
| base height, settled | 0.355 m | 0.368 | -- |
| stance hip height, settled | 0.187 m | 0.183 | ~0.38 (derived) |
| stance thigh \|tau\|, peak | 21.2 N*m | **20.0** | "pinned at 20.2" |
| falls | 1/32 | 1/32 | 0/8 |

The change did what it was aimed at and nothing else. Pitch moved 15 degrees to the reference's own
value; height did not move, because height is governed by the two terms that were deliberately left
alone. Mid-run the policy passed through a rough patch -- `base_contact` 0.683 at iteration 404,
episode length 621 -- and came out at 0.012 with full-length episodes, the same shape `feat/biped`
records for this task family.

**The remaining difference from the reference is entirely the crouch**, and the geometry closes
exactly: `base_z = hip_z + 0.1934 * sin(pitch)` gives 0.183 + 0.186 = 0.369 against 0.368 measured.
The reference's 0.564 m at 75 degrees implies stance hips at ~0.38 m -- front legs nearly straight,
0.426 m of leg reaching 0.378 m.

Worth knowing before treating that as a defect: a straight-legged stance is what `feat/biped`'s
operator complained about, and Try-3/Try-4 of that round were spent trying to obtain bent knees.
Try-6 got them and was reverted for unrelated robustness reasons. The posture this run produces is
the one that work wanted and could not reach.

**Two things to watch rather than celebrate.** The stance calf now presses into its flexion soft
limit (margin median 0.071 rad, p10 -0.030), so the crouch is limit-constrained -- `dof_pos_limits`
at -10.0 (the merged set's weight, ten times the bipedal recipe's) is actively holding it back from
folding further. And the stance thigh still peaks at 20.0 N*m against a 20.2 N*m continuous rating
during the rise; better than run 1's 21.2 but no margin, and the same figure the reference's
hardware failure sat at. Holding the stance remains cheap at 8.8 N*m.

## Runs 3 and 4: two more reward weights, and why neither moved anything

Run 3 raised `base_height` from -0.5 to -2.0; run 4 reverted that and raised `front_hip_height`
from -0.5 to -5.0. Measured the same way as runs 1 and 2:

| | run 2 | run 3 | run 4 |
|---|---|---|---|
| head clearance, min | 0.045 m | 0.045 | 0.045 |
| stance hip height | 0.183 | 0.188 | 0.195 |
| base height, settled | 0.368 | 0.372 | 0.378 |
| stance knee \|tau\|, peak | 22.2 N*m | 26.0 | 27.0 |
| falls | 1/32 | 2/32 | 0/32 |

Four times the weight bought 1.2 mm. Ten times the weight, on the other term, bought a centimetre.
The minimum head clearance did not move at all across three runs with different reward
compositions -- 0.045 m, p10 to p90 spanning a millimetre.

Run 3's failure was predictable from `feat/biped`'s own note, which had been read and not acted on:
Go2's thigh and calf are 0.213 m each, so `BASE_HEIGHT_TARGET` = 0.55 m "was never reachable to
begin with". Quadrupling the weight on a squared penalty against an unreachable target is a weak
uniform pull carrying no information about what to move; it raised the stance knee's peak torque
17% and the stance not at all.

## The actual cause: the head was resting on the floor for free

A quantity that does not move under a 10x weight change is not being set by a reward. Measuring
contact force rather than height settled it in one run:

| | before | after |
|---|---|---|
| head/trunk contact force, peak | **4050 N** (p10 3192, max 5160) | 0 |
| head/trunk touching | **3.1% of steps** (p10 1.2, p90 5.5) | 0 |

Around 27 times body weight, in every environment. The 45 mm "clearance" reported for four runs was
not clearance at all -- it was the head's contact height, which is why no reward weight could move
it.

Nothing charged for it. `base_contact` terminates on `["base", ".*_hip"]` at 1 N and fired in 0.6%
of episodes, so the load was landing on `Head_upper`/`Head_lower`, which no termination covers. The
only term reaching them was `undesired_contacts`, a bounded per-body count shared with six other
links at -1.0: a 5000 N head strike cost about 0.05 per step. Resting the head on the floor means
never having to stand up, and it was nearly free.

**Lesson for the next round of this:** the symptom was reported from play mode as "the head is
skimming the ground". The response was to measure the height and read 45 mm as a margin. Measuring
the *contact* is what answered it, and it should have been the first thing measured, not the fifth.

## Tries 1-3: three ways to price the head strike

All three were run from scratch for 2000 iterations and measured identically.

| | Try 1 (terminate, 1 N) | Try 2 (terminate, 20 N) | **Try 3 (reward, -20)** |
|---|---|---|---|
| head contact force / duty | 0 / 0% | 0 / 0% | 0 / 0% |
| base height, peak | 0.563 m | 0.565 | 0.563 |
| stance hip height | 0.356 | 0.357 | 0.358 |
| sagittal pitch, settled | 83.6 deg | 83.1 | 83.5 |
| stance knee \|tau\|, peak | 33.9 N*m (max 46.0) | 34.8 (max 46.2) | **31.9 (max 43.0)** |
| falls (64 env) | 2/64 | 1/64 | **0/64** |
| forward tracking, normalised | 0.438 | **0.371** | **0.373** |
| forward tracking, p90 | 0.749 | 0.895 | **0.585** |
| yaw-rate error | 1.148 rad/s | 0.988 | **0.791** |

All three removed the contact completely and lifted the stance to the reference's own figure --
`feat/biped`'s front stance peaks at 0.564 m, and these reach 0.563-0.565. Pitch went past it, to
83 degrees against 75. `base_height` stayed at its original -0.5 throughout: it was never the
problem.

**Try 3 promoted.** It is best or joint-best on every secondary axis, and it is the only one that
keeps the episode alive -- a termination removes the learning signal for everything that was going
right in the episode as well as the strike. The two terminating variants both push the stance knee
past its 45.4 N*m peak rating.

Two measurement notes worth keeping. Try 2 led on every training-log metric and came last on falls
once measured in play conditions -- population means and play-condition distributions rank these
differently, again. And forward-tracking error must be normalised by the commanded speed before it
is compared: at 32 environments and unnormalised, the crouched stance appeared to track *better*
than the fixed ones, which reversed at 64 environments once the commanded magnitude was divided
out. The crouch tracks worse (0.431 against 0.371), consistent with `feet_air_time` rising 17-fold
once the legs extended -- the crouch was not really walking.

**Open.** The stance now runs with both stance-leg joints pressed against their travel limits
(margin medians near zero), which is the near-locked posture `feat/biped`'s operator disliked;
lowering `BASE_HEIGHT_TARGET` is the lever if a bent-knee stance is wanted. Stance-knee torque
roughly doubled with the taller stance (settled 8.8 -> 16.8 N*m), so the hardware-transfer question
has moved from the hip to the knee. Yaw-rate tracking is poor in every variant (0.79-1.15 rad/s
against a +-1.0 rad/s command range) and has not been looked at.

## The hind stance: the same free lunch, one link down

Built by mirroring the front stance -- sides swapped, `front_hip_height` dropped (its 0.30 m target
describes the front hips when they are the shoulders and nothing when they are the raised end), and
`lin_vel_y` back to +-0.5, which is what `feat/biped` validated for this stance and which is the
direction it drifts.

**First run, 1500 iterations.** Everything looked good and nothing was: `success` 0.907, pitch
0.911, falls 2/64. Reading it as "the hind stance is learning faster than the front did" was wrong,
and the mistake is worth naming -- it was reaching a *cheap* solution faster, and the population
metrics could not tell the difference.

The measurement could:

| | first run | front stance, for comparison |
|---|---|---|
| stance foot, one contact | 1400-3069 ms | 214-276 ms |
| stance foot, contacts/s | 0.89-1.76 | 2.78-2.83 |
| `RL_calf` on the ground | **8.7% of steps, peak 3215 N, 1.98/s, 39 ms each** | none |
| `feet_air_time` | 0.0034 | 0.0334 |
| base height, settled | 0.437 m | 0.547 |
| stance hip height | 0.246 m | 0.356 |

Both stance feet planted for one to three seconds at a time is not walking, it is standing and
sliding -- which is exactly what the operator reported seeing in play mode ("the standing legs are
dragging"). And the left shin was striking the floor twice a second, locked to the gait: the leg
was swinging and catching the ground instead of clearing it.

**The same structure as the head, one link further down.** `undesired_contacts` covered `.*_thigh`
and `.*_calf` as a bounded count at -1.0, so a 3.2 kN shin strike cost -0.09 per step. Cheap enough
to be worth doing, so the robot knelt.

**Fix: `stance_leg_contact`, -20.0, on the stance legs' thigh and calf.** Pulled out of the shared
term rather than raising it, so the lifted pair keeps its -1.0 -- the same operation the head got.
Both stances now carry both terms, each inert in the other's configuration, so neither has an
unpoliced way to rest part of itself on the floor.

**Second run, same 1500 iterations, one change:**

| | first run | **promoted** | front stance |
|---|---|---|---|
| stance foot, one contact | 1400-3069 ms | **217-281 ms** | 214-276 ms |
| stance foot, contacts/s | 0.89-1.76 | **2.62-2.68** | 2.78-2.83 |
| shin contact | 8.7% / 3215 N | **none** | none |
| base height, settled | 0.437 m | **0.569** | 0.547 |
| stance hip height | 0.246 m | **0.377** | 0.356 |
| sagittal pitch, settled | -- | **85.0 deg** | 83.5 |
| falls | 2/64 | **1/64** | 0-2/64 |

Contact on the two stance feet and nowhere else. The trajectory shows the causal order plainly:
the shin contact collapses by iteration 400, falls spike to 0.27 as the crutch goes, and only then
does base height start climbing -- 0.233 at 200, 0.260 at 400, 0.393 at 600, 0.505 at 800, 0.534 at
993.

**This stance is the gentler one on hardware.** Stance-knee torque settles at 0.445 N*m against the
front stance's 16.9, and peaks at 30.1 against 34.1 -- and its joints stand clear of their travel
limits (thigh margin 1.435 rad) where the front stance presses against them. The costs are lateral:
peak lean 26.5 deg median against 12.4, and forward tracking 0.571 normalised against 0.373. The
sideways bias `feat/biped` documented for this stance is visible but not disabling.

Promoted by editing the default config directly -- there was no sandbox try for this stance, the
change was made and measured in place. `2026-09-04_14-58-43` is the promoted checkpoint;
`2026-09-04_12-07-03_superseded_no_stance_leg_contact` is the first run, kept as the record behind
the table above and renamed so it cannot be picked up by accident.

## Both stances in one network

`Go2-Multitask-Biped` draws the stance per episode and reads it from the observation's stance
column. The reward terms that name a side follow it through `stance_aware`; the ones that do not
name a side are inherited from the front config untouched.

`feat/biped` spent a full round failing to make one policy respond to a mode command and concluded
that the paper's own transition demo trains two specialists and swaps them at deployment. That
conclusion did not transfer, and it was worth not taking on faith: its modes were quadruped against
biped, where one is far easier and ignoring the command is close to optimal. Front against hind are
mirror siblings, and a policy that ignores the column lifts the wrong end of the robot half the
time.

### Two design mistakes, both mine, both found by measuring

**The centre of pressure was handed all four feet.** Reasoning from `feat/biped`'s own note that
the force-weighted CoP settles onto whichever feet are loaded, and therefore needs no routing. The
premise is right and the conclusion was backwards: a tripod -- both stance feet plus one lifted
foot still down -- is precisely a case where a foot outside the stance carries load, so the CoP
moved onto it, the CoM-CoP vector straightened, and `pendulum_angle` scored the tripod at -0.0048
against the hind specialist's -0.0065. The pose that avoids the task measured *better balanced*
than the task. With `lifted_contact` at -0.25 per step the only thing pricing it, against -200 for
falling, the trade was worth taking above about a 2.5% fall risk -- and the policy took it, in
every hind-commanded environment, holding a pose the operator recognised on sight as a temple-lion
statue. Routed to the stance pair, `success_hind` went from a run that never once left zero across
3000 iterations to 0.42 by iteration 982.

**The command's foot lists were set to all four feet**, on the reasoning that both pairs are
candidates in this task. `success` requires `lifted_contact == 0`, so that turned "nothing lifted
is touching" into "no foot is touching at all" -- unreachable for a robot standing on two legs.
Two things followed. The metric counted the moments all four feet happened to leave the ground, so
every `success_front`/`success_hind` figure reported during that period was mislabelled: the
unified policy's hind rate was 0.737, not the 0.527 the log showed. And `stance_held`, the +1.0
completion reward, reads the same flag -- so the run was training with its completion term almost
never paid. That, not the curriculum and not the mirror, is what the later jump from 0.527 to 0.977
was mostly measuring.

Both were caught by measuring rather than reasoning: per-link contact force and duty for the first,
and reading what the command actually computes for the second. Four training runs were spent on the
front stance's equivalent before that habit set in.

### Try 1 (stance-share curriculum) and Try 2 (left-right mirror)

Aimed at the hind stance's remaining gap. Both ran 3000 iterations from scratch.

| | baseline | Try 1 | **Try 2** | single-stance specialists |
|---|---|---|---|---|
| `success_front` | 0.969 | 0.968 | **0.984** | 0.969 |
| `success_hind` | 0.527 | 0.977 | 0.966 | 0.969 |
| falls, 128 play episodes | -- | 9 | **3** | -- |
| planar velocity error, hind | -- | 0.274 | **0.222** | -- |
| planar velocity error, front | -- | 0.276 | **0.238** | -- |
| stance-foot duty, left vs right | 0.941 / 0.374 | 0.399 / 0.339 | **0.436 / 0.434** | 0.575 / 0.728 |

**Try 1's curriculum never fired.** `share_front` sat at 0.499 for the whole run: the hind stance
was never behind by the margin that would have moved the draw. Its gain over the baseline is the
`stance_held` fix described above, which both tries carry. So Try 1 is not evidence about the
share, and Try 1 against Try 2 is a clean read on the mirror alone.

**Try 2 promoted.** The mirror leaves the two sides even to within half a percentage point where
every previous policy in this project has invented a side, and it carries that into a third of the
falls and better tracking on every axis but yaw. Applied through `agents/rsl_rl_ppo_cfg.py`, which
is the only difference from the shared base configuration -- the environment is untouched.

The share curriculum and the per-stance episode tallies it needed were deleted with the tries.

**Still open.** Yaw-rate tracking is poor in every biped policy trained here (0.74-1.15 rad/s
against a +-1.0 rad/s command range) and has never been looked at. The single-stance tasks do not
carry the mirror, so their checkpoints and configurations still match each other but neither has
the symmetry the unified policy now gets.
