# Go2 bipedal stance -- summary of experiments

The front-leg handstand (`Go2-Multitask-Handstand`), trained as a fourth expert for the multi-task
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
