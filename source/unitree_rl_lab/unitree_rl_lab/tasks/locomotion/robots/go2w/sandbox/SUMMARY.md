# Go2W Phase 5 sandbox — what was tried and what it established

Phase 5 is the extreme-obstacle-crossing task for the wheeled Go2W. This file records two
sandbox campaigns that produced the current `velocity_env_cfg_phase5.py`: Try 1 – Try 9
(2026-08-02 … 2026-08-11) and Try 10 – Try 14 (2026-08-12 … 2026-08-13). **Every try in both
campaigns has been folded in and deleted** (2026-08-13); this file is the reasoning that
would otherwise be lost with them.

The second campaign's headline result reaches beyond just this phase: **switching the
policy network from an MLP to a GRU (`GruPPORunnerCfg`) fixed a MuJoCo command-following
problem that reward-shaping alone (Try 11/12) had only partially fixed**, on identical
environment config, and did so more convincingly than any environment/reward change tried
here — see Lesson 7. Consequently `Go2w-v1-Phase1` through `Go2w-v1-Phase5` (not just this
phase) now all register with `GruPPORunnerCfg` (`go2w/__init__.py`), and the calf/wheel
actuator correction from Try 13 (see the Try-by-try table) went into the *shared*
`UNITREE_GO2W_CFG` (`assets/robots/unitree.py`), not a Phase5-only override.

Read the "Lessons" section first if you are about to change something.

---

## Where it ended up

Best measured result, from the 6000-iteration run that first reshaped the terrain
(`logs/rsl_rl/go2w_v1_phase5_try_8/2026-08-10_11-53-06`):

| metric | value |
| --- | --- |
| `terrain_levels` | 9.2 / 19 → **0.44 m steps** |
| `foot_impact` termination | 3.0 % |
| `base_contact` termination | 4.4 % |
| `time_out` (survived full episode) | 91 % |
| `lin_vel_cmd_levels` | 1.2 m/s (ceiling) |

A play-mode check of that policy cleared **0.60 m** with the better individuals and failed
at 0.70 m.

The current default was then restarted from Phase 4 to pick up the reward and terrain
changes made after that run; at 3000 iterations it sits at `terrain_levels` 8.11 → 0.40 m,
i.e. it has not yet caught back up to the 0.44 m figure above.

### Second campaign (Try 10 – 14): MuJoCo is the result that matters, not `terrain_levels`

Isaac Lab `terrain_levels` and actual MuJoCo climbing **diverged** across Try 13 vs. 14:

| | Try 13 (MLP) | Try 14 (GRU) |
| --- | --- | --- |
| Isaac Lab `terrain_levels` | 12.5/19 → **~0.43 m** (higher) | 11.3/19 → ~0.40 m |
| MuJoCo climb, hands-on | **fails under 0.20 m** | **succeeds at 0.40 m** |
| MuJoCo command-following | still drives forward with no command | follows command, stops when told, relatively smooth |

Try 13 scored *better* in the training metric and *far worse* in the environment that
actually matters. Whatever generalizes to MuJoCo is not fully captured by `terrain_levels`,
survival rate, or any of the per-step reward logs in this file — see Lesson 7.

---

## Lessons

**1. Terrain *shape* dominated everything else.**
Three separate runs plateaued at ~0.36 m no matter which termination threshold was
relaxed. The cause was geometric. In an inverted pyramid the robot spawns on the pit floor
at `-(num_steps + 1) * step_height`, and `num_steps` is set by **`platform_width`**, not by
`step_height` (isaaclab `mesh_terrains.py:179-183`). On the original 8.0 m tile with
`platform_width=2.0` that meant `num_steps=3` — a pit four steps deep on a 1.2 m square
floor. "0.80 m steps" actually meant escaping a **3.13 m well**, so raising the height
ceiling made the task harder along an axis nobody intended. Reshaping to `num_steps=1`
(5.5 m tile, 1.00 m tread) cut `foot_impact` from 28 % to 3 % and broke the plateau
immediately.

> Any change to `size` / `step_width` / `platform_width` must be checked against the
> resulting pit depth and floor size, not just the nominal step height.

**2. Relaxing terminations has sharply diminishing returns — and can backfire.**
`base_contact` went 30 → 80 → 150 → 400 N across four runs chasing a plateau that turned
out to be terrain shape; `foot_impact` went 1500 → 2500 N for the same reason. Try 4
removed `base_contact` *entirely* and the policy got **worse**: with no contact risk at all
it charged into walls more recklessly instead of climbing carefully.

> Treat a high termination rate as a symptom to diagnose, not a number to raise.

**3. Curriculum resolution matters.**
Widening `step_height_range` without adding rows makes each promotion a bigger jump, and
`custom_terrain_levels_climb` is a one-way ratchet (it demotes only below 0.5 m of
progress). A robot promoted past its ability parks on a row it cannot clear and keeps
crashing there. Going 10 → 20 rows over 0.10–0.80 m restored a 0.035 m step.

**4. `terrain_levels` is not comparable across configs, and is easy to misread.**
It is a row index, not a length. The same value means different step heights under
different `num_rows` / `step_height_range`, and the raw number rose three separate times
while the *physical* step height stayed flat. Always convert:
`step_height = lo + ((level + 0.5) / num_rows) * (hi - lo)`.
Also note the ratchet's equilibrium is every robot parked on the hardest row it can still
make *some* progress on, so the mean reads as "ability plus a bit" and a non-zero
termination rate is the normal steady state.

**5. A reward that works on a legged robot can be inert on a wheeled one.**
`feet_contact_without_cmd` — the only term rewarding "stand still when told to" — rewards
*feet in contact*. That is a sound proxy for standing on Go2 and a meaningless one on
Go2W, which has all four wheels on the ground at 2 m/s and collects it in full while
driving. The metric rose during training and looked like progress; it was not.

**6. Restricting the command distribution creates out-of-distribution behaviour at deploy
time.** Raising `lin_vel_x`'s floor to 0.4 m/s meant the policy essentially never saw a
zero command, and in MuJoCo it drove off on its own. Exposure (`rel_standing_envs`) and a
gradient to learn from (`motion_without_cmd_penalty`) are both required — neither alone is
enough.

**7. An MLP genuinely could not fully solve "stop when told to" here; a GRU did, on the
identical environment.** Try 11 (gated `climb_progress`) and Try 12 (`wheel_motion_
without_cmd_penalty`, penalizing wheel joint velocity directly instead of only the base's
resultant velocity) both targeted this exact failure and both helped in training metrics —
but Try 13 (MLP, both fixes plus more) still drove forward with no command in MuJoCo, and
additionally failed to climb even a 0.20 m wall there despite the *highest* `terrain_levels`
of this whole campaign. Try 14, changing nothing but the policy network (MLP →
`GruPPORunnerCfg`, a GRU) on that same Try 13 environment config, fixed the command-drift
and climbed 0.40 m. The recurrent hidden state plausibly gives the policy actual memory of
"what was I just told to do", which an MLP has to reconstruct from single-step
observations alone every step; every fix up to Try 12 was trying to patch that gap from the
reward side. Concretely, this also means: **`terrain_levels` (or any Isaac-Lab-only metric)
is not sufficient evidence a change helped** — Try 13 would have looked like the best run
in this file by that metric alone. Check MuJoCo before trusting a training-metric win.

> Checkpoints cannot be shared across this change: RSL-RL's `OnPolicyRunner.load()` is a
> strict `state_dict` load, and `ActorCriticRecurrent`'s parameters (extra `memory_a`/
> `memory_c` RNN submodules, different first-layer input size) don't match `ActorCritic`'s.
> A GRU run has to warm-start through Phase1 → Phase2 → Phase5 again from scratch (Try 14
> did: 500 + 3000 + 2500 iterations) — there is no way to inherit an MLP lineage's weights,
> only its env/reward/terrain/actuator *config*.

---

## Try-by-try

| Try | Change | Outcome |
| --- | --- | --- |
| 1 | Height ramp 0.10–0.40 m; added `forward_command_progress` + `stall_penalty`; assist off | Real climbing skill to `terrain_levels` ~7.2/9 before eroding |
| 2 | Ramp widened to 0.50 m; `stall_penalty` → `forward_stall_penalty` | Fixed "spin in place near the wall": the old penalty gated on direction-agnostic planar speed, so rotational CoM wobble softened it without progress |
| 3 | `base_contact` 30 → 80 N | Insufficient |
| 4 | `base_contact` removed entirely | **Backfired** — more reckless, not less. Led directly to Try 5 |
| 5 | `base_contact` restored as `illegal_contact_excluding_top` | Direction-aware: resting on a step's flat top is exempt, slamming a vertical riser still terminates |
| 6 | (assist re-enabled at 150 N) | Not carried forward |
| 7 | Terrain thin_wall → stairs; added `climb_progress_reward` | Direct vertical-progress reward instead of relying on XY displacement |
| 8 | Height ceiling → 0.80 m; `num_rows` 10 → 20; **terrain geometry reshaped**; `bad_orientation` → 2.0 rad; `flat_orientation_l2` → −0.5 | The geometry fix is what broke the plateau (see Lesson 1) |
| 9 | Reverse enabled (`lin_vel_x` → (−0.4, 1.2)) | Folded in; operational recovery motion |
| 10 | `lin_vel_y` un-pinned on "rough" columns only (terrain-gated), Phase3's old (−0.1,0.1)/(−0.7,0.7) band | Confirmed in play mode: strafe restored on rough, still zero on pyramid. MuJoCo check of this checkpoint found the "drives forward with no command" bug (see Try 11) |
| 11 | `climb_progress` gated on command (was unconditional — the largest-weight reward in the set firing even while "standing"); `motion_without_cmd` weight −1.0 → −2.0; step_height_range narrowed (0.10,0.80)→(0.10,0.60) (the Try-8-era "Open items" entry, finally tried) | Fixed the *unbounded* forward drift (robot used to never stop once moving) — confirmed in MuJoCo play mode. A residual 2–3 s coast-to-stop remained |
| 12 | Added `wheel_motion_without_cmd_penalty` (mdp/rewards.py) — penalizes wheel joint velocity directly while cmd≈0, not just its effect on base velocity (which `motion_without_cmd_penalty` already covered) | Targeted the exact gap: `motion_without_cmd_penalty` only sees the base's *result*, not the wheel actuator command itself |
| 13 | Reverse terrain-gated the same way Try 10 gates `lin_vel_y` (rough-only, widened −0.4→−1.2); calf `effort_limit` raised 23.5→45.43 N·m to match the real motor spec (hip/thigh stay 23.7) | Reached the campaign's *highest* `terrain_levels` (12.5/19 ≈ 0.43 m) — but **failed under 0.20 m in MuJoCo and still drove forward with no command**. See Lesson 7 |
| 14 | Policy network only: MLP → GRU (`GruPPORunnerCfg`), identical env/reward/terrain/actuator config to Try 13. Cold-started through Phase1(500)→Phase2(3000)→Phase5(2500) since checkpoints can't cross this change | **Climbed 0.40 m in MuJoCo and follows commands correctly** — lower `terrain_levels` (11.3/19) than Try 13 but far better in the environment that matters. See Lesson 7 |

---

## Open items

* **`step_height_range`'s 0.80 m ceiling is larger than the policy can use.** At the 0.44 m
  equilibrium, levels ~13–19 are never reached. Narrowing to (0.10, 0.60) would put the
  working point near level 13 and tighten resolution to 0.025 m/level. Left at 0.80 because
  that is the range the best run actually used — the narrowing is untested.
* **The 0.4 m/s forward floor was never isolated.** It landed in the same run as a
  `foot_impact` threshold change, and the two runs after it still sat at 0.36 m, with the
  actual breakthrough coming from the terrain fix. Its contribution may be small. If the
  reverse range causes a regression, note that (−0.4, 1.2) puts 25 % of draws in the
  |cmd| < 0.2 band against 16.7 % for the (0.0, 1.2) config that plateaued; the principled
  fix would be a dead-band sampler on `lin_vel_x`.
* **MuJoCo climbing (updated 2026-08-13 — see Lesson 7 and the second-campaign table
  above for the full picture):** the underlying cause of most MuJoCo underperformance in
  this campaign turned out to be the **policy network**, not terrain shape or torque —
  Try 14 (GRU) climbs 0.40 m with correct command-following on the exact same environment
  Try 13 (MLP) fails under 0.20 m on. That doesn't fully close the earlier three
  candidates, though:
  1. *Wheel torque mismatch* — resolved as a decision, not yet as a MuJoCo-side change:
     confirmed 2026-08-13 that MuJoCo's `ctrlrange="-15 15"` is the real spec, not a
     placeholder. Tried lowering Isaac's wheel `effort_limit` 23.5→15.0 to match and
     reverted — judged too low to climb with. Decision is to raise MuJoCo's side to 23.5
     instead; that change is outside this repo (`/home/tak/unitree/unitree_mujoco`) and
     was not yet made as of this writing.
  2. *Obstacle shape* (thin walls in MuJoCo vs. 1.00 m-tread stairs in training) — not
     re-examined this campaign. Still an open confound between "network architecture" and
     "obstacle shape" for the Try13 vs. Try14 gap specifically, since both are MuJoCo-side
     mismatches against training. The wall heights/widths *were* updated 2026-08-12 (see
     `terrain_tool/terrain_generator_go2w.py`: heights 0.20/0.40/0.60 m, 2.0 m wide lanes
     flush against each other, depths 0.30/0.50 m) but the thin-wall-vs-stair shape
     mismatch itself wasn't addressed.
  3. *Undertrained* — Try 14 specifically: its `terrain_levels` curve was still short of
     as flat a plateau as Try 13's when its 2500-iteration budget ran out. Worth more
     iterations before concluding 0.40 m is where it settles.
* **Should GRU become the default architecture for Phase5 (and earlier phases)? Decided
  2026-08-13: yes.** `Go2w-v1-Phase1` through `Go2w-v1-Phase5` all register with
  `GruPPORunnerCfg` now, not just Phase5 — given Lesson 7, an MLP appears structurally the
  wrong tool for this task's "stop when told to" requirement, not just under-trained on
  it, and there is no reason to expect that to be Phase5-specific. Not yet re-validated by
  retraining the *full* Phase1 → Phase2 → Phase3 → Phase4 → Phase5 chain end-to-end with
  GRU throughout — Try 14 only ever exercised Phase1 → Phase2 → Phase5, skipping Phase3/4
  entirely. Worth doing before trusting Phase3/4 results under the new default.
* **Removed on 2026-08-10: the EFGCL wall-bump assist** (`WallBumpAssistCommand` +
  `wall_bump_assist_decay`, ported from feat/jump). Disabled since Try 1 and never
  re-enabled, so it contributed to no result here. A 2026-08-04 ablation had suggested it
  genuinely accelerated climbing rather than inflating `terrain_levels`, so the idea is
  worth revisiting — but it was written for thin walls and would need re-implementing
  against the current stair terrain.

---

## Deploy-side fixes found along the way

Not part of the training campaign, but discovered while validating it in MuJoCo and
recorded here because they are easy to reintroduce:

* **Action-to-motor mapping was cross-wired for 14 of 16 joints.**
  `deploy/robots/go2w/src/State_RLBase.cpp` applied `joint_ids_map` directly to the action
  index. `joint_ids_map` is indexed by *IsaacLab* joint id, while action element *k* refers
  to its term's `joint_ids[k]`, so the correct motor is `joint_ids_map[joint_ids[k]]`. The
  identity holds on Go2 (a single `joint_names=[".*"]` term) but not on Go2W, whose legs
  and wheels need separate position/velocity terms with explicit SDK-ordered names. Symptom
  was violent thrashing the instant the policy engaged.
* **The keyboard command could not reach zero.** The ported clamp used
  `[sx(0), sx(1)]` from the policy's ranges; with Phase 5's one-sided `lin_vel_x` that
  interval excluded 0, pinning the command at 0.32 m/s forward. Bounds are now widened to
  include zero (a no-op for symmetric ranges).
* **2026-08-13, `OrtRunner` (`deploy/include/isaaclab/algorithms/algorithms.h`, shared by
  all 7 robots' deploy binaries) had no support for a recurrent policy.** Loading Try 14's
  GRU-exported ONNX (which declares extra `h_in`/`h_out` I/O to carry hidden state across
  calls — see `isaaclab_rl/rsl_rl/exporter.py`'s `_OnnxPolicyExporter`) crashed immediately:
  `Input name h_in not found in observations.` A second, quieter bug was present too —
  the old code hardcoded reading only ONNX output index 0, so even supplying `h_in` would
  have silently dropped `h_out` every call, never actually carrying the recurrence forward.
  Fixed by having `OrtRunner` detect `h_in`/`c_in` by name and own that state internally
  (zero-initialized, updated from `h_out`/`c_out` each call) — no changes needed to
  `State_RLBase.cpp` or any other robot's deploy code, since the fix is transparent to the
  caller. Verified against Try 14's real `policy.onnx`: 5 calls with an *identical* input
  produced smoothly drifting actions (proof the hidden state evolves rather than resetting)
  and `reset()` reproduced call 1 bit-for-bit. Also verified no regression on an existing
  MLP robot (`go2`) whose export happens to have 4 outputs (none recurrent) — `action` is
  still exactly output 0, deterministic across repeated calls.

---

## 2026-08-18: superseded by the thin_wall / goal-directed redesign

The pyramid_stairs terrain, `UniformTerrainGatedVelocityCommand`, and climb_progress/
motion_without_cmd-style reward set this file documents were never fully able to solve
"stop when told to" in MuJoCo, even with the GRU network (Lesson 7). A Student built on
the TCN/V2 equivalent of this same design was later found still driving off under a zero
command in MuJoCo, prompting a full redesign rather than another round of reward tuning:

* Terrain: pyramid_stairs/pyramid_stairs_inv → a single free-standing thin_wall ring on
  flat ground (no pit), matching the real MuJoCo test scene directly.
* Command: `UniformTerrainGatedVelocityCommand` → `MixedGoalVelocityCommand` -- "rough"
  columns keep a full omnidirectional command, wall columns get a goal placed just beyond
  the wall with the command dropping to exactly zero on arrival.
* Reward: `forward_command_progress`/`forward_stall_penalty`/`climb_progress`/
  `motion_without_cmd`/`wheel_motion_without_cmd` all removed, replaced with direct ports
  of ANYmal Parkour's (Hoeller/Rudin et al. 2023) Table S2 goal-tracking terms.

Developed as `Go2w-v2-Teacher-Phase5-Try1` (V2/Teacher line) then
`Go2w-v1-Phase5-Try15` (this v1/GRU line, sandbox/velocity_env_cfg_phase5_try15.py).
Try15 confirmed in MuJoCo: controls correctly, no runaway under a zero command, crosses
0.40 m (stalls at 0.50 m -- the front legs get up onto the wall, but the robot can't
drive the rear end up and over; still unsolved as of this entry, see
Try16/Try17). Folded into the permanent `velocity_env_cfg_phase5.py` on 2026-08-18 --
see that file's own module docstring for which Lessons above are now obsolete (1, 2 --
terrain-shape-specific to the pyramid) versus still load-bearing (3, 6 -- re-confirmed
independently against the new terrain too). `Go2w-v2-Phase5`/`Go2w-v2-Teacher-Phase5`/
`Go2w-v2-Student-Phase5` inherit this fold as well (deliberate, not an oversight -- their
old pipeline had the same unsolved problem).

Try15's own file was *not* deleted after folding, unlike every prior fold recorded above
-- Try16/Try17 (chasing the 0.50 m stall) import `CommandsCfgPhase5Try15` from it
directly. Delete it once that investigation concludes.

---

## 2026-08-19: Try16-19 wrap-up and sandbox cleared

The 0.50 m stall investigation (Try16/Try17) did not reach a fix, and a follow-up thin-
wall-thickness experiment (Try18/Try19) came back inconclusive-to-negative. Per direct
instruction, the whole sandbox is cleared here (all six `velocity_env_cfg_*_try*.py`
files deleted, including `velocity_env_cfg_v2_teacher_phase5_try1.py`) rather than left
half-finished — this section is what would otherwise be lost with them.

**Try16** (continued from Try15's own checkpoint): added `front_leg_push_reward`, a
height-based detector for "front feet on the wall, rear feet dangling" meant to reward
pushing the front feet down in that exact stuck posture. Hit a `KeyError: 'thin_wall'` in
Play (a hardcoded sub-terrain name that doesn't exist under Play's per-height column
naming) — fixed with a generic column → sub-terrain lookup. Even after the fix, the
reward logged exactly `0.000` across two full training runs; root cause never
conclusively identified (suspected `FR_foot`/`RR_foot` `body_pos_w` reference-point
mismatch against the approximated wall height, not confirmed). Also disabled
`base_contact` for training to let the policy survive longer at the stuck position —
`terrain_levels` *regressed* instead of improving (re-confirming Lesson 2/3's "removing
base_contact backfires" finding against the new thin_wall terrain too), reverted.
Terrain grid shrunk 20x20 → 4x10 and rough/thin_wall reproportioned 30/70 → 25/75 for
memory — this part *was* kept, folded into the default `velocity_env_cfg_phase5.py` on
2026-08-18/19 independently of the reward work.

**Try17** (restarted fresh from `Go2w-v1-Phase2`, not continuing Try16's checkpoint):
MuJoCo showed the Try15→Try16 checkpoint chain had picked up a real regression (more
prone to tipping/inverting, harder to control), so this Try branched from the last known-
good common ancestor instead of stacking further. Redesigned `front_leg_push_reward` as
a pure contact-sensor test (front feet in contact, rear feet not) instead of the
height-based approximation — also logged `0.000` across its full run; tentatively
attributed to insufficient training depth (terrain_levels never reached the difficulty
row where "stuck" actually occurs) rather than a code bug, but never confirmed by further
training. Also relaxed `base_contact`'s `illegal_contact_excluding_top` `vertical_margin`
20.0 → 60.0 N (Play showed the base getting killed while visibly resting on the wall's
top surface). Over 2300 iterations from the same Phase2 start Try15 used,
`terrain_levels` *declined* 4.6 → 3.4, versus Try15's climb 4.5 → 12 — suspected the
`vertical_margin` relaxation backfired the same way Try16's full `base_contact` removal
did, just at smaller scale, but this was never isolated as a standalone variable before
the sandbox was cleared. **Open item for whoever picks this up next**: re-try
`front_leg_push_reward` (either formulation) with `base_contact` and `vertical_margin`
left at Try15's own values, so the reward is the only variable, and train long enough to
confirm whether it fires at all before judging it.

**Try18** (wall thickness pinned to a constant 1 cm, "steel plate", vs. the default's
fixed 40 cm) and **Try19** (thickness graded 10 cm → 1 cm by the same per-row difficulty
that already drives `wall_height_range` -- `thin_wall_terrain`'s `_lerp` already
supported this, just never used) both trained 3300 iterations from `Go2w-v1-Phase2`,
everything else identical to the post-shrink default (4x10 grid, rough 25 % / thin_wall
75 %). Results came back essentially identical to each other and both concerning:

| metric | Try18 (1 cm fixed) | Try19 (10→1 cm graded) | default Phase5 (40 cm, for reference) |
| --- | --- | --- | --- |
| `terrain_levels` (start → end) | ~4.49 → 3.24 (declined) | ~4.49 → 3.17 (declined) | climbs |
| `base_contact` termination | 72.2 % | 71.6 % | far lower |
| `time_out` | 27.6 % | 28.2 % | — |
| `goal_position_tracking` reward | 0.0114 | 0.0158 | — |

Gradual thinning (Try19) showed **no measurable advantage** over jumping straight to
1 cm (Try18) -- the premise that a curriculum would make thinning easier was not borne
out at this training budget. Both show the same "regressing instead of climbing" +
"very high base_contact rate" signature as every prior over-relaxation in this file
(Lesson 2/3), which reads as "1 cm is simply too hard for 3300 iterations from Phase2,"
but an untested alternative explanation was raised and never ruled out: a wall this thin
may interact badly with `illegal_contact_excluding_top`'s existing thresholds/contact-
history-window logic in a way the 40 cm baseline never exercised (e.g. a knife-edge
contact behaving differently from a wide flat-top rest). **Neither Try was checked in
Play or MuJoCo before the sandbox was cleared** -- the Isaac-Lab-only numbers above are
suggestive, not conclusive (Lesson 6: check MuJoCo before trusting a training-metric
reading). Not folded into the default; revisit by re-implementing from this record if the
thin-wall deploy target is still a priority.

`velocity_env_cfg_v2_teacher_phase5_try1.py` (the V2/Teacher-line original of Try15,
whose `goal_*` reward functions were long since promoted to `mdp/rewards.py` and whose
own registered task `Go2w-v2-Teacher-Phase5-Try1`/`Go2w-v2-Student-Phase5-Try1` had
become redundant with the default `Go2w-v2-Teacher-Phase5` once the fold made their env
configs identical) was deleted in the same pass, along with its two now-pointless
registrations.

---

## 2026-08-19..24: curriculum rework (Try20-22) and a goal-reward rework grounded in the
## ANYmal Parkour paper (Try23-25) -- Try24 folded in, Try25 still under evaluation

A close re-read of `doc/papers/ANYmal_Parkour_Learning_Agile_Navigation_for_Quadrupedal_
Robots.md` (Hoeller/Rudin et al. 2023 -- the actual source of the `goal_*` reward
functions, previously cited but the file itself was missing from this repo until added
mid-investigation) found that this project's port only used half the paper's relevant
tables. The paper is a two-level hierarchy: a 50 Hz Locomotion module tracks a *local*
target (r*/psi*/t*) reissued every ~0.2 s by a 5 Hz Navigation module, trained with
Table S2; the Navigation module itself is trained against the *global* target (r_G*/
t_G*) with Table S3, whose "Position tracking" term only fires once, on the actual last
step of the episode ("this sparse formulation allows the policy to explore the terrain
to find safer paths and take its time where needed"). This codebase has no such
hierarchy -- one flat policy, one `goal_pos_w` per episode -- and had ported only Table
S2's `goal_position_tracking_reward`/`goal_heading_tracking_reward` (gated to a single
7-8 s window via `arrival_deadline_s=8.0`/`activation_window=1.0`), applying them
directly to the single global goal they were never designed for. Concretely: a wall
crossing taking longer than 8 s got zero credit from either term for the rest of the
episode, including while correctly holding position at the goal afterward, while
`goal_dont_wait_penalty` actively penalised the climb's necessarily-slow motion along
the way -- net effect, "climbed slowly but successfully" could score *worse* than
"never tried."

**Curriculum side (Try20-22, GRU line)**: replaced `custom_terrain_levels_climb` (a
one-way ratchet that promotes on any single episode's success, essentially never
demotes) with `mdp.traversability_terrain_levels_climb` (an EMA of per-env episode
success, column-aware -- wall envs succeed on goal arrival, rough envs on the existing
displacement threshold -- promoting/demoting only once the EMA crosses a threshold),
and `lin_vel_cmd_levels` with `mdp.lin_vel_cmd_levels_column_aware` (excludes "wall"
envs from the average used to decide whether to widen a velocity range they never
actually draw from). Try20 (all three changes, including episode_length_s 20→10 s)
saw terrain_levels crash from its random initial draw (~3.5) down toward ~1.2-1.5 and
stay there; a further 5000-iteration continuation *regressed outright* (bad_orientation
10%→54%, time_out 90%→44%). Try21 isolated out the episode-length change (reverted to
20 s) and got a similarly-shaped but healthier result (terrain_levels ~1.4, bad_orientation
2.4%, time_out 97.4%). Try22 isolated the EMA change alone (lin_vel_cmd_levels also
reverted to the default) and reproduced the same decline-to-~1.0 pattern, confirming the
EMA curriculum itself (not the other two changes) drives this: the *interpretation*
settled on is that the old ratchet's numbers (e.g. Try18/19's ~3.2, this campaign's
historical "0.40 m/0.44 m" milestones) were likely inflated by its "promote on any
lucky success, rarely demote" design, and the EMA's lower, stabler numbers are a more
honest read of sustained ability -- not confirmed by Play/MuJoCo, still an
interpretation. None of Try20/21/22 folded into the default; the curriculum question is
open, revisit this record before re-attempting.

**Reward side (Try23-25, GRU line)**: Try23 both removed
`goal_position_tracking`/`goal_heading_tracking` (Table S2, misapplied here) and added
`goal_arrival_reward` (Table S3's terminal term, new function in mdp/rewards.py) in one
Try -- terrain_levels fell even further (4.49→0.68 over 2300 iterations) than any
curriculum-only variant, *using the default's own historically-lenient ratchet, not
even Try20/21/22's EMA* -- confirming the regression came from removing the dense
(if narrowly-windowed) Table S2 shaping, not from adding the sparse Table S3 term.
Try24 isolated the addition alone (goal_position_tracking/goal_heading_tracking left
untouched, goal_arrival added on top) -- terrain_levels behaved like the default (peak
~5.3, base_contact ~74%, essentially the same profile as measuring the literal default
Phase5 itself over the same budget: peak 4.56, base_contact 72.2%, confirmed by actually
training the unmodified default for direct comparison), i.e. adding the term doesn't
hurt and gives genuine credit for slow-but-successful crossings once they start
happening. **Try24 folded into the default `RewardsCfgPhase5` 2026-08-24** (see that
class's own docstring) and its sandbox file deleted. Try25 went further -- replacing
`goal_position_tracking` itself with a new function, `mdp.goal_progress_reward`
(potential-based: previous-step distance minus current distance, telescopes over an
episode to net distance closed, immune to "leave and come back" double-dipping by
construction, unlike a raw per-step distance value) -- and measured a large
survivability shift (base_contact 74%→1.4%, time_out 26%→98.3%) at the cost of a lower
terrain_levels (1.78 vs the default's likely-inflated ~4+), read as "less reckless, more
honest" rather than "worse," consistent with the curriculum-side EMA finding above.
`goal_progress`'s own weight (5.0) may be too low to provide strong shaping yet (its
logged contribution stayed order 1e-3, far smaller than `goal_move_in_direction`'s
~0.1-0.4). **2026-08-24: abandoned per direct instruction** -- `goal_progress_reward`
deleted from mdp/rewards.py and Try25's sandbox file/registration removed, despite the
measured survivability improvement above. Revisit from this record (the telescoping
potential-based formulation, and the "weight was likely too low" open question) if a
continuous distance-based shaping term is worth trying again.

**2026-08-24: `lin_vel_cmd_levels_column_aware` folded directly into the default**
`CurriculumCfgPhase5` (not via a sandbox try -- judged low-risk, since it only changes
which envs count toward the "rough" velocity-range curriculum's own decision, unrelated
to terrain_levels), then verified live with a 500-iteration continuation of the
*actual* default `Go2w-v1-Phase5` task (now genuinely trained end-to-end on this
codebase, checkpoint present on disk as of this entry) -- no regression observed
(terrain_levels 5.26, base_contact 25.8%, time_out 69.8%, bad_orientation 4.3%, all
healthy).

**2026-08-24: `traversability_terrain_levels_climb` abandoned per direct instruction.**
Deleted from mdp/curriculums.py along with all four tries that used it: Try20, Try21,
Try22 (GRU line), and Go2w-v2-Teacher-Phase5-Try2 (privileged Teacher line, the
10000-iteration run). Rationale: across all three GRU variants the EMA curriculum
consistently produced *lower* terrain_levels (1.0-1.8) than the default's own ratchet
(peak ~4.5-5.3), and while the working theory was that the default's higher numbers
were "inflated" by its rarely-demoting design, that was never independently confirmed
(no Play/MuJoCo check), and the EMA's own hyperparameters (`alpha=0.2`,
`promote_threshold=0.7`, `demote_threshold=0.2`) were never tuned beyond their initial
guess. The Teacher-line run (Try2, 10000 iterations) additionally found no sign of the
privileged-information climbing advantage Lee et al. 2020's own ablation would predict
-- terrain_levels stalled at ~1.45, similar to the much-cheaper GRU variants. The
`custom_terrain_levels_climb` one-way ratchet (this campaign's long-standing default at
the time) remained the terrain_levels curriculum immediately after this -- since
superseded by Try26 below (2026-08-25), not by a return to the EMA. Revisit this record
(particularly the EMA hyperparameters and the never-checked "inflated vs. honest"
question) if terrain_levels curriculum design is worth returning to.

---

## 2026-08-25: `terrain_levels_climb_demote_on_fail` (Try26) folded in -- first confirmed
## 0.50 m wall crossing in MuJoCo

Diagnosis: the default's terrain_levels trajectory (whether continuing from an existing
checkpoint or restarting fresh) tended to rise to a peak and then either plateau or
decline, without a clean, reliable recovery. `custom_terrain_levels_climb`'s move_down
only fires below 0.5 m of net displacement; between that floor and the promotion
threshold (35 % of the tile), it neither promotes nor demotes, by design (partial
progress isn't punished). But this creates a dead zone once an env has been promoted
past its actual ability: it can crash into the wall (`base_contact`) or tip over
(`bad_orientation`) after already covering, say, 0.8 m, never clearing 0.5 m and never
reaching the promotion threshold either -- stuck at a level it is genuinely failing at,
for the rest of training, with nothing pulling it back down. Envs piling up in exactly
this dead zone was the suspected cause.

**Try26** (`mdp.terrain_levels_climb_demote_on_fail`, mdp/curriculums.py): identical to
`custom_terrain_levels_climb` except it additionally demotes on `base_contact`/
`bad_orientation` termination regardless of distance travelled. Tested two ways:

| | continuing from the default's own checkpoint (+1500 iter) | fresh from Go2w-v1-Phase2 (2500 iter) |
| --- | --- | --- |
| terrain_levels (start → end) | 4.49 → 6.05 (peak 6.16) | 4.49 → 6.16 (peak 6.25) |
| `base_contact` | 42.3 % (vs. the default's own ~74 %) | 43.1 % |
| `time_out` | 55.3 % | 53.1 % |

Both runs reproduced the same result: meaningfully higher terrain_levels *and* a much
lower `base_contact` rate than the default's own comparable continuation (~74 %) --
demoting on genuine failure, not just low displacement, both climbs higher and fails
less recklessly getting there.

**Checked in MuJoCo (the fresh-from-Phase2 checkpoint): this project's first confirmed
0.50 m wall crossing.** Command-following was reported as sluggish/hard to control at
that checkpoint -- not yet resolved, an open item for whoever picks this back up
(possibly related to this being a relatively short, fresh-from-Phase2 run rather than a
long, thoroughly-converged one; not yet isolated).

**Folded into the default `CurriculumCfgPhase5` 2026-08-25** (see that class's own
docstring) and Try26's sandbox file/registration deleted. `Try27` (goal_arrival scaled
by terrain_levels difficulty, tested alongside Try26 from the same starting point) was
*not* folded initially -- it showed only a marginal terrain_levels improvement
(5.36 → 5.54) with no improvement to `base_contact` (stayed ~74-75 %), unlike Try26's
clear effect on both axes.

**Try27 re-run 2026-08-25, fresh from Go2w-v1-Phase2 (1500 iterations), now
automatically inheriting Try26's demote-on-fail fix** (it never overrode
`curriculum`, so it picked up the new default's `CurriculumCfgPhase5` for free).
Isaac Lab metrics looked dramatically better than Try26 alone -- `base_contact` 0.96 %,
`time_out` 96.0 %, terrain_levels 5.37 (peak 5.90) -- but **a Play check found this was
misleading**: the trained policy had learned to sway left-right in place near the wall
rather than ever attempt to climb it. The "healthy" termination stats reflected "never
engaging the obstacle at all" (so nothing crashes into anything), not genuine
wall-crossing competence -- the difficulty-scaled arrival bonus, stacked with
`goal_dont_wait_penalty`'s "don't be too slow" pressure, apparently made "wobble just
fast enough to dodge the don't-wait penalty, near a goal you never actually reach"
a viable low-risk strategy once the arrival bonus's own upside (paid only via a genuine
arrival) wasn't reliably attainable. A clear instance of Lesson 6/7 (Isaac-Lab-only
metrics, including termination-rate ones, are not sufficient evidence a change
helped -- check MuJoCo/Play before trusting them).

**2026-08-25: abandoned per direct instruction** -- `mdp.goal_arrival_reward_
difficulty_scaled` deleted from mdp/rewards.py and Try27's sandbox file/registration
removed. If a difficulty-scaled success bonus is worth retrying, this record's own
failure mode (reward-hacked non-engagement) is the first thing to guard against --
e.g. by verifying in Play *before* declaring victory on termination-rate metrics alone,
and/or reconsidering whether `goal_dont_wait_penalty` and a success-only bonus can be
made to interact safely together.

### 2026-08-26/27: Try29 (difficulty-scaled progress reward) and Try30 (leg-wall
contact exemption) -- one folded, one still undetermined

Two new tries, each isolating one candidate fix motivated by the accumulated lessons
above, trained sequentially from `Go2w-v1-Phase2` (2000 iterations each):

**Try29** -- revived `goal_progress_reward` (the potential-based term abandoned after
Try25) as a new class-based `mdp.goal_progress_reward`, this time scaling the
*progress* itself by terrain difficulty (`(prev_distance - distance) * (1 + max_scale
* difficulty)`) rather than scaling `goal_arrival_reward`'s success bonus the way
Try27 did. The design reasoning: a progress reward pays ~0 for standing still or
swaying in place (distance doesn't change), so the specific "wobble near the goal,
never engage it" exploit that sank Try27 shouldn't be available here, regardless of
the difficulty scale.

Result: terrain_levels 4.78, but `base_contact` 0.58% / `time_out` 94.8% -- a
termination distribution that superficially resembles Try27's reward-hack pattern.
The `goal_progress` episode-reward log itself stayed near zero throughout training
(±0.0001-0.0002), i.e. net closed distance averaged across the population was close
to nil. This does not confirm the reward-hack theory doesn't apply here (the
telescoping-to-zero argument only rules out one specific exploit mechanism, not every
possible one), nor does it confirm genuine climbing success.

**2026-08-27: abandoned per direct instruction, without a Play/MuJoCo check** --
the difficulty-scaled `mdp.goal_progress_reward` class deleted from mdp/rewards.py
and Try29's sandbox file/registration removed. Unlike Try27 (whose deletion followed
a confirmed reward-hack finding), Try29's outcome was never actually determined --
its metrics only *resembled* Try27's pattern, on a mechanism (`goal_progress_reward`)
whose own design should rule out that specific exploit. Treat this as an open
question, not a confirmed negative result, if a difficulty-scaled progress reward is
revisited later.

**Try30** -- split the default `undesired_contacts` term (Head/hip/thigh/calf,
weight -0.3, threshold 1 N, uniform across every column) into two: Head/hip stays
penalised everywhere unchanged, thigh/calf contact is exempted specifically on "wall"
columns via new `mdp.undesired_contacts_column_aware`. Motivated by the theory that
penalising the exact load-bearing leg-wall contact a climb requires (even at only
-0.3 weight/1 N threshold, continuously for every step of contact) gives an
independent, structural incentive to avoid the wall, on top of whatever an
arrival-side reward is doing.

Result: terrain_levels 6.53 (this project's highest yet), with a *healthy* termination
distribution -- `base_contact` 37.8%, `time_out` 57.5% (genuinely mixed outcomes, not
concentrated near either extreme the way Try27's or Try29's numbers are). Checked in
MuJoCo: reaches a front-leg foothold at 0.60 m -- the highest confirmed climb attempt
to date. Still trembles and creeps forward while meant to be holding position at the
goal -- the same not-yet-resolved issue first flagged on Try26's checkpoint; this fold
doesn't address it, remains open (see below).

**2026-08-27: Try30 folded into the default `RewardsCfgPhase5`** (see that class's
own docstring for the exact split) and Try30's sandbox file/registration deleted.

### Resolved 2026-08-28: stop-time trembling / creep -- Try31 (wheel_vel_without_cmd_penalty)

Every checkpoint validated in MuJoCo since Try26 (Try26 itself, and Try30) shared one
issue: once holding position at (or near) the goal -- and, confirmed via direct
question, also on plain flat-ground zero-command standing, not just post-climb
arrival -- the robot trembles and drifts/creeps forward rather than staying still.
Go2W drives with wheels, and nothing in the reward set specifically penalised wheel
*velocity* under a zero command: `joint_position_penalty`'s stand-still branch only
watches leg joint *position* (wheels are excluded, being continuous
velocity-controlled joints), and `track_lin_vel_xy_exp`'s tolerance (std
`sqrt(0.25)`) is loose enough that a small residual wheel-driven creep still scores
fairly well.

**Try31** added `mdp.wheel_vel_without_cmd_penalty` (new function: penalizes squared
wheel joint velocity, summed over 4 wheels, whenever the commanded velocity is
exactly zero). Bootstrapped from Try30's own checkpoint via a manual symlink into its
own log root, since `--previous-task` requires the referenced task ID to still be
*registered* in gym (an `argparse` `choices` restriction, not just the string-based
log-directory resolution below it) -- Try30's registration had already been removed
once folded.

**First attempt, weight -0.01, collapsed** within ~20 iterations of resuming:
terrain_levels 6.53 -> ~1.5, `bad_orientation` ~0% -> 77-85%, neither recovering over
the remaining ~1300 iterations. Discarded (checkpoint directory deleted). Root cause:
the raw (unweighted) signal has no upper bound -- squared wheel velocity summed over
4 wheels spikes arbitrarily high if a wheel is still spinning fast at the exact
instant a command drops to zero, a frequent, recurring transition (every
standing-env draw, every wall arrival) -- so even a small weight multiplied into an
occasional, destructively large single-step reward exactly at those transitions.

**Retried at weight -0.001 (10x smaller)**: stable throughout -- terrain_levels
recovered to 4.3-4.7 within the first ~100 iterations and reached 5.97 by the end
(1500 iterations total), `bad_orientation` stayed low (3.7% at the final iteration),
`base_contact` 45.2%, `time_out` 51.1% -- a healthy, mixed-outcome distribution, no
sign of the earlier collapse or of a Try27/29-style "avoid everything" pattern.
Checked in MuJoCo: **confirmed a significant improvement** to the trembling/creeping
behaviour.

**2026-08-28: Try31 folded into the default `RewardsCfgPhase5`** (weight -0.001; see
that class's own docstring) and Try31's sandbox file/registration deleted.

### 2026-08-28/29: the fold above regressed under long, from-scratch training --
### ablation (Try32/33/34), and the resolution (a separate `-Adjust` task)

A fresh, continuous 3000-iteration run of the newly-folded default (Try26 + Try30 +
Try31 all present, from `Go2w-v1-Phase2`) told a different story than any of the
three individual folds: terrain_levels rose to a peak of ~5.0 around iteration 3628,
then declined steadily for the remaining ~2370 iterations to 2.8 by the end, while
`bad_orientation` climbed from ~1-2% to 20.6% over that same stretch. `base_contact`
stayed low throughout (never above ~5%) -- a slow, sustained degradation, not the
sudden collapse Try31's first (weight -0.01) attempt showed. `Mean action noise std`
also grew across the run (0.79 -> 1.47), unusual for PPO (normally shrinks as a
policy converges), suggestive of an unstable/never-fully-converging policy rather
than simple bad luck on one run.

This was surprising because each of Try26/Try30/Try31 individually had only ever
been validated as a *short* refinement (1500-2500 iterations) on top of an
*already-converged* checkpoint -- none had been tested together, from scratch, for
this long. Separately, re-reading the *old* default's own archived tensorboard logs
(pre-Try26/30/31, `logs/rsl_rl/_archive/go2w_v1_phase5_pre_try31fold_2026-08-28/`)
for the same iteration range showed terrain_levels climbing to ~5.3-5.4 and
*staying* there (no decline) -- but only by way of `base_contact` climbing to and
staying at ~74%, i.e. that apparently-stable number reflected a policy constantly
crashing into the wall, not one that had stopped needing to (retroactively
confirming a suspicion held since Try26's own record above).

**Ablation, ~2000 iterations each from `Go2w-v1-Phase2`:**

| Try | Change vs Try32 | terrain_levels (final) | base_contact | bad_orientation | Decline? |
|---|---|---|---|---|---|
| Try32 | none (old-default reproduction, baseline) | 5.01 | 74.3% | 0.7% | n/a |
| Try33 | + Try26 (curriculum demote-on-fail) | 6.10 | 40.7% | 3.9% | No -- monotonic rise throughout |
| Try34 | + Try30 (undesired_contacts split) | 6.53 | 37.8% | 4.7% | No -- monotonic rise throughout |

Try32's own numbers (base_contact 74.3%, terrain_levels 5.01) closely reproduce the
old default's archived record, confirming the baseline reproduction was faithful.
Neither Try33 nor Try34 showed any sign of decline even checked at the iteration
equivalent to the full fold's peak (iter 3628) -- both were still climbing strongly
at that point (Try34: ~6.3 and still rising, versus the full fold which was *already*
turning over at essentially the same point in training). Conclusion: Try26 and Try30
are both unambiguously beneficial, individually and combined, with no long-run
downside found; the decline is specific to `wheel_vel_without_cmd_penalty`, combined
with continuous long-duration training from scratch (as opposed to a short polish on
an already-strong checkpoint, which is exactly how Try31 itself was originally
validated).

**Resolution (2026-08-29):**
  * `wheel_vel_without_cmd` removed from the permanent `RewardsCfgPhase5` --
    the default's reward set is now exactly Try26 + Try30 (i.e. equivalent to
    Try34), with `wheel_vel_without_cmd_penalty` withheld.
  * A new, permanent (non-sandbox) task, **`Go2w-v1-Phase5-Adjust`**
    (`velocity_env_cfg_phase5_adjust.py`, registered in `go2w/__init__.py`, not
    this sandbox), adds `wheel_vel_without_cmd_penalty` (weight -0.001) on top of
    the default and is meant to be run for ~1000 iterations against
    `Go2w-v1-Phase5`'s own latest checkpoint whenever the trembling/creeping issue
    needs addressing -- a deliberate, standing "polish pass" step, not a curriculum
    phase and not resumed-from by anything else.
  * Try34's own checkpoint (2000 iterations from Phase2, terrain_levels 6.53,
    MuJoCo-confirmed 0.50 m climbing) was promoted to be `Go2w-v1-Phase5`'s own
    checkpoint (the failed 3000-iteration from-scratch run was archived instead,
    at `logs/rsl_rl/_archive/`).
  * Try32/33/34 sandbox files and registrations deleted -- their conclusions are
    fully captured above and in `RewardsCfgPhase5`'s own docstring.

**2026-08-30: `Go2w-v1-Phase5-Adjust` checked in MuJoCo (1000 iterations against the
promoted default checkpoint above)**: climbs 0.50 m reliably, matching Try30/Try34's
own validated level -- the polish pass did not cost any climbing ability. 0.60 m is
still difficult (consistent with the default's own record -- a front-leg foothold at
0.60 m was the best result seen so far, never a full crossing). The trembling/creeping
issue is **improved but not fully resolved**: still creeps forward slightly while
meant to be holding position, just less than before. `wheel_vel_without_cmd_penalty`
at -0.001 is a genuine partial fix, not a complete one -- worth revisiting if further
stillness is wanted (e.g. a slightly larger weight now that it's known not to
destabilise training at -0.001, or addressing whichever of the earlier hypotheses --
GRU hidden-state momentum, sparse "just arrived" training experience -- turns out to
still apply).

### 2026-09-03/04: the 0.60 m wall -- a body-height reward (Try35-39, one folded) and
### the termination that was blocking it (Try40-42)

Starting point: the default's best confirmed result was a 0.50 m crossing, with 0.60 m
unreliable and MuJoCo/Play describing the failure as "gets front feet up but doesn't
pull the body up and over".

**Try35 (sloped wall terrain)** -- a new terrain
(`mdp.sloped_thin_wall_terrain`/`MeshSlopedThinWallTerrainCfg`) whose crossing face
ramps 30 -> 90 degrees with difficulty instead of being sheer at every row, so the
policy meets a walkable slope before a vertical face. Needed the tile grown 5.5 -> 11.0 m
(a 30-degree ramp at 0.6 m height runs ~1.04 m per face, and the goal still needs room
beyond the ramp's outer edge). Trained 1500 iterations: terrain_levels 6.05,
base_contact 25.0%, bad_orientation 13.1%. MuJoCo-checked: **did not solve the
crossing.** Still registered; the terrain itself is reusable if a slope-based approach
is revisited.

**Try36 (wall_body_height_reward)** -- `mdp.wall_body_height_reward`, rewarding the
base for reaching wall-top-plus-clearance height while near the wall and before
arriving, reading the wall's *known* geometry (terrain_levels -> `wall_height_range`
lerp, distance-from-spawn gate) rather than a height-scan sensor built for continuous
terrain. Trained 1500 iterations: terrain_levels 6.74 (highest at the time),
base_contact 35.1%, bad_orientation 4.0%. MuJoCo-checked: **the first real behavioural
progress of the whole campaign** -- the robot began rearing up and propping its front
legs on the wall's top edge -- but it settled into holding that leaning pose
indefinitely instead of continuing over. Diagnosis: a continuous per-step reward for
height alone pays exactly as well for "hold this pose forever" as for "pass through it
on the way over", and once at target height the reward can only go *down* by leaving.

**Try37/38/39** each isolated one fix for that specific local optimum, all 1000
iterations from the same default checkpoint:

| Try | fix | terrain_levels | base_contact | bad_orientation | wall_body_height |
|---|---|---|---|---|---|
| Try37 | one-time bonus (`wall_body_height_boost_reward`) | 6.65 | 36.9% | 4.4% | 0.0043 |
| Try38 | gated on progress toward goal (`min_progress_speed=0.15`) | 6.66 | 36.3% | 4.5% | 0.0387 |
| Try39 | far-side gate extended (`gate_width_far=1.5`) | 6.76 | 36.6% | 3.9% | 0.0555 |

**Try39 folded into the default `RewardsCfgPhase5` 2026-09-04** (see that class's own
docstring) and deleted; its checkpoint promoted to be the default's own.

**Try40/41/42 -- the termination was ending the crossing attempt itself.** Try39's Play
check found individuals dying *at the moment the torso touched the wall*. Reading
`mdp.illegal_contact_excluding_top` explained it: its exemption only ever spares
*upward*-dominant reaction force (resting on a wall's flat top), so the
horizontal-dominant reaction from a torso pressed against a vertical face is
indistinguishable, to that function, from slamming into it at speed -- and with
threshold 400 N against ~191.5 N body weight, plus the magnitude test using the contact
history's *maximum*, the impact spike as the torso first meets the wall clears the bar
easily. **The exemption written to protect climbing technique was blocking the specific
technique a 0.60 m wall needs.** Three fixes, 800 iterations each:

| Try | fix | terrain_levels | base_contact | bad_orientation | time_out | wall_body_height |
|---|---|---|---|---|---|---|
| (Try39 baseline) | none | 6.76 | 36.6% | 3.9% | 59.4% | 0.0555 |
| Try42 | horizontal contact <= 200 N exempt (`illegal_contact_excluding_supported`) | 6.35 | 10.1% | 7.7% | 82.1% | 0.0704 |
| Try41 | threshold 400 -> 900 N | 6.37 | 11.9% | 7.5% | 80.6% | 0.0742 |
| Try40 | speed gate 0.8 m/s (`illegal_contact_speed_gated`) | 6.03 | 0.44% | 9.6% | 90.0% | 0.0811 |

Monotone in how much each relaxes base_contact: base_contact rate falls, the
wall_body_height reward earned rises, terrain_levels falls. **Try41 Play-checked at
pinned 0.60 m: roughly 30 % of individuals clear it** (vs. almost none for the
unrelaxed baseline) -- MuJoCo still could not. Try41 and Try42 are nearly
indistinguishable on every metric; Try42 is the better one to pursue on design grounds
rather than numbers (it keeps the 400 N ceiling so a genuine high-speed collision still
terminates, where Try41's blanket 900 N tolerates any impact in any direction -- exactly
the move Lesson 3 warns about, and Try4 historically saw a policy charge into walls
more recklessly when base_contact was removed).

**terrain_levels is ranking these backwards -- do not use it to choose between them.**
Try39 measures higher terrain_levels than Try41/42 (6.77 vs ~6.35 at *matched*
iterations -- iteration count was checked and is not the explanation) while being
clearly worse at crossing 0.60 m in Play. Mechanism: the curriculum promotes on net
displacement from spawn (> tile_size*0.35 = 1.925 m) and demotes on base_contact/
bad_orientation, so relaxing base_contact removes a demotion trigger but simultaneously
enables a "leaning on the wall at ~1.25 m until time_out" state that neither promotes
nor demotes -- the ratchet stalls. Supporting evidence: Try41 plateaus at ~6.2-6.3
while Try39 keeps climbing to ~6.8; Try41's mean goal_distance is *higher* (1.93 vs
1.60) despite surviving longer (904 vs 703 steps), i.e. the extra survival time is not
being spent getting to the goal. Note this mechanism is inferred from those aggregates,
not from a direct measurement of the distance-from-spawn distribution at reset -- that
would settle it. Either way the whole gap is ~2 cm of wall height (level 6.77 -> 0.476 m
vs 6.35 -> 0.451 m), and both train on ~45-48 cm walls, not 60 cm. Same "inflated
plateau" pattern already recorded for the pre-Try26 default (terrain_levels 5.01 at
74 % base_contact).

### 2026-09-04/06: Try43-45 -- the curriculum's promotion rule, then friction from both
### sides. All null; sandbox cleared.

Continuing from Try40-42 (the base_contact relaxation, above), three more attempts on
the 0.60 m wall. None improved on Try42, and all were deleted 2026-09-06 along with the
rest of the sandbox.

**Try43 -- promotion distance realigned to the wall** (`promote_distance=1.6`, vs the
inherited `tile_size * 0.35` = 1.925 m). The motivating arithmetic still stands and is
preserved in `mdp.terrain_levels_climb_demote_on_fail`'s own docstring: the wall ring
sits at 1.25 m and its far face at 1.45 m, so clearing the wall leaves the robot ~0.48 m
short of promotion, and with `arrival_radius` 0.5 m wider than that whole window, a
robot that genuinely crosses *and* arrives can still fail to promote.

Result: terrain_levels **4.12** (vs Try42's 6.22) -- and the trajectory shows it never
climbed at all, sitting at 4.1-4.6 (its initial random draw is ~4.49) for the entire
2000 iterations, while `wall_body_height` ran 0.39 -> 0.60 (6x Try42's 0.08) and
`goal_arrival` stayed at ~0 throughout. base_contact 0.14 %, time_out 97 %.

**Two things to take from this, and one mistake to not repeat.** The mistake: this was
not a clean test. It changed the promotion rule *and* the bootstrap point (Phase2
instead of Try42's competent checkpoint), so `promote_distance` itself remains untested
rather than disproved. What it did establish, more sharply than the thing it was aimed
at: **trained from scratch, `wall_body_height` plus a relaxed `base_contact` is a trap.**
The policy finds "walk to the wall, rear up to target height, stand there for 20 s" long
before it could find "cross", and then nothing dislodges it -- it does not terminate
(base_contact relaxed), it is not required to arrive (the arrival reward is sparse and
tiny), and it does not demote (parked at ~1.25 m is neither < 0.5 m nor > the promotion
distance). The earlier "leaning" diagnosis from Try36 was the same phenomenon seen
through a checkpoint that already knew how to climb.

**Try44 / Try45 -- friction, from both sides.** The hypothesis: climbing needs the front
wheels to plant on the wall's top face and hold while the lower legs are pulled up, and
they cannot because they slip. Two separable mechanisms, so two tries, both 1000
iterations from Try42's checkpoint so they are directly comparable to it and to each
other:

| | terrain_levels | base_contact | bad_orientation | time_out | goal_distance | wall_body_height |
|---|---|---|---|---|---|---|
| Try42 (baseline) | 6.22 | 10.3 % | 6.5 % | 83.2 % | 1.98 | 0.080 |
| Try44 (surface friction) | 5.88 | 2.6 % | 8.7 % | 88.5 % | 2.34 | 0.088 |
| Try45 (wheel-joint friction) | 6.15 | 8.4 % | 9.0 % | 82.7 % | 1.90 | 0.069 |

*Try44* raised surface grip: terrain material 1.0 -> 2.0 **and** the startup robot-side
randomization 0.3-1.2 -> 1.0-1.5. Both were needed because
`friction_combine_mode="multiply"` makes effective friction the *product* -- at the
default values a third of envs train at 0.3-0.6, genuinely slippery. Together: effective
2.0-3.0.

*Try45* raised wheel-joint static friction 0.01 -> 2.0 N*m. Note **`ActuatorBaseCfg.friction`
is an effort (N*m) on Isaac Sim 5.x, not the unitless coefficient it was on 4.5** -- so
the stock 0.01 is 0.1 N of tangential resistance at the 0.086 m wheel radius against a
~191.5 N body weight, i.e. nothing. 2.0 N*m gives 23 N (12 % of body weight) for 8.5 % of
the wheel's torque budget. This tested what Try44 structurally could not: surface
friction resists *sliding*, not *rolling*, and a wheel on the wall top can roll off no
matter how grippy the surface is.

Both null. Neither moved terrain_levels off the same ~6.2 plateau, and no other metric
pointed positive. Together they rule out grip as the blocker from both directions --
which also removes the reason to build the targeted "high friction on the wall's top
face only" version. Worth recording how that would have to be done if it ever comes
back: `TerrainImporter.import_mesh` takes the whole generated terrain as **one prim with
one physics material** (the generator has already merged every sub-terrain by then), so
per-face friction means spawning separate high-friction cap prims over each wall top --
~160 of them for the 4x10 grid.

**Where this leaves things.** The best result of the whole campaign is still Try41's
~30 % of individuals clearing 0.60 m in Play (never in MuJoCo), and Try42 is the
config that survived. Everything since has been null. The single most likely remaining
explanation is a reward-structure one rather than a physics one: **while parking at the
wall outscores crossing it, no change to the physics will produce a crossing.**
`wall_body_height` pays weight 1.0 every step; `goal_arrival` pays weight 0.15 once per
episode. Two guards against exactly this were built and measured -- Try37's one-time
bonus and Try38's progress gate -- but both ran while `base_contact` still killed
leaners, so neither could show its value; they should be retried on top of the relaxed
termination. A related calibration issue was found and never tested:
`goal_radius_range` (1.75-2.5 m) with `arrival_radius` 0.5 m means a goal drawn at the
low end can be "arrived at" from ~1.25 m -- the near side of the wall -- so
`goal_arrival_reward` can pay out without a crossing at all.

### 2026-09-06/07: Try46/47 -- perception, and the parking trap finally isolated

**Try46 -- give the *actor* the height scan.** Phase5's reward design is a port of ANYmal
Parkour, whose Locomotion module is perceptive; this codebase took the reward tables and
left the actor proprioception-only, so the policy learns a wall exists only by touching
it. Try46 adds the critic's own privileged `mdp.height_scan` (17x11=187 rays, clip
(-2.0, 5.0), `history_length=1`, no noise) to `PolicyCfg`. Actor obs 141 -> 328, so
checkpoints are incompatible and the whole Phase1 -> Phase2 -> Phase5 lineage was
retrained. Explicitly a **diagnostic upper bound**, not a deployable policy.

Result: Phase5 came out at terrain_levels **3.89** with `wall_body_height` at 0.52-0.67
(~7x what the same term earns from a competent checkpoint) and `goal_arrival` pinned at
~0 for the whole run. Same signature as Try43. **Try46 measured the parking trap, not
perception.**

**The trap, stated properly.** Two from-scratch Phase5 runs have now collapsed into it
(Try43 from Phase2, Try46 from Phase1) while all six competent-checkpoint runs
(Try39-42, 44, 45) avoided it. **The bootstrap point was hiding the trap, not preventing
it** -- a policy that already climbs never explores "stand at the wall forever", but one
learning from scratch finds it long before it finds crossing, and then nothing dislodges
it: it does not terminate, is not required to arrive, and does not demote (parked at
~1.25 m is neither < 0.5 m nor past the promotion distance).

**Try47 -- Try46 minus `wall_body_height`.** One-variable difference, so Try46 is the
baseline. Reused Try46's Phase1/Phase2 checkpoints unchanged (`wall_body_height` lives
only in `RewardsCfgPhase5`), Phase5 only, 2000 iterations.

| | Try46 (trap) | Try47 (term removed) |
|---|---|---|
| terrain_levels | 3.89 (frozen 3.9-4.4) | **5.54** (peak 6.43) |
| `goal_arrival` | -0.0002 | **0.0027** |
| base_contact | 1.3 % | **41.9 %** |
| time_out | 94.2 % | 55.4 % |
| mean episode length | (long, parked) | 666 |

All four flipped from "standing around" to "actually engaging the wall". **Cause
confirmed: `wall_body_height` pays weight 1.0 every step against `goal_arrival`'s 0.15
once per episode, so parking simply outscores crossing.** Play-checked: the robot now
lifts its body *in front of* the wall rather than colliding and then lifting -- a
qualitatively better motion than anything earlier in the campaign.

**But perception did not help.** The right blind baseline is **Try34** (from Phase2,
2000 iterations, blind actor, no `wall_body_height` -- it did not exist yet, same Try26
curriculum and Try30 contact split as the default). Try34 vs Try47 differ *only* in the
actor's height scan:

| | terrain_levels | base_contact | bad_orientation | time_out |
|---|---|---|---|---|
| Try34 (blind) | **6.53** | 37.8 % | 4.7 % | 57.5 % |
| Try47 (perceptive) | 5.54 (peak 6.43) | 41.9 % | 2.7 % | 55.4 % |

Roughly equal, with the blind run ahead on terrain_levels. **A perfect privileged height
map did not improve climbing** -- which, on the upper-bound logic this try was built
for, argues against funding the two-stage LiDAR elevation-map project *for this problem*
(it may still be worth it for others). Caveat: terrain_levels has ranked things
backwards here before, and Try47 also shows the familiar peak-then-decline (6.43 ->
5.54).

**Standing conclusion for anyone starting a from-scratch Phase5 run:** check
`wall_body_height` and `goal_arrival` in the first few hundred iterations. If the former
is running several times its usual value while the latter sits at zero, the run is in
the trap and is measuring nothing else.

### 2026-09-07/08: Try48 -- a jointly-trained state estimator; and the height-scan
### transpose that made the whole perceptive lineage unverifiable in MuJoCo

**Try48 -- estimator (base_lin_vel + CoM-CoP) on the GRU's hidden output.** Try47 had
established that the actor's *observation* was not the blocker; the remaining reading was
that the missing behaviour needs **timing** (accelerate, commit weight forward, launch),
and timing needs to know your own speed and whether your weight is ahead of your contact
patch -- neither of which is in the actor's observation. So: a 6-dim auxiliary regression
head (`base_lin_vel(3)` + `com_cop_vector(3)`) hanging off the GRU's hidden state, trained
jointly with PPO, with the actor conditioned on the **estimate** rather than the ground
truth (so deployment is unchanged). Origin: TumblerNet (Xiao et al. 2025) via
`origin/feat/biped`, adapted -- feat/biped's estimator eats a separate stacked-history
block because that policy is feedforward, whereas this lineage's GRU already *is* that
history encoder, so the estimator reads `out_mem` instead.

One-variable difference from Try47, so Try47 is the baseline. Both are 2000 Phase5
iterations on top of their own Phase2 (numbers below re-aggregated at interval 200, which
is why Try47's differ slightly from the interval-100 figures in the table above):

| | terrain_levels (final / peak) | base_contact | time_out | goal_arrival | Loss/estimator |
|---|---|---|---|---|---|
| Try47 (no estimator) | 5.48 / 6.45 | 37.7 % | 59.5 % | 0.003 | -- |
| Try48 (estimator) | **6.84 / 6.90** | 37.4 % | 59.5 % | 0.002 | 0.017 -> 0.013 |

Two things worth separating:

* **The estimator learned.** `Loss/estimator` fell 0.017 -> 0.013 and stayed flat-low,
  i.e. the auxiliary task is being solved, not ignored. That is the check the try was
  built to make; a flat loss would have meant "no signal", not "no effect".
* **The peak-then-decline is much milder.** Try47 peaked at 6.45 and gave back a full
  level (-> 5.48); Try48 peaked at 6.90 and held 6.84. Given how often this campaign has
  seen a run climb and then collapse, "held its peak" is the more interesting half of the
  result than the absolute number.

`goal_arrival` stayed at ~0.002 in both, so **neither run crosses in bulk** -- the
difference is in how much wall they get up, not in completions. And per the standing rule,
`terrain_levels` has ranked tries backwards before, so this needed MuJoCo.

**Which is where it fell over: the observation the whole perceptive lineage trained on is
not the one the deploy stack produces.** Trying to MuJoCo-check Try48, the deploy binary
aborted with `Observation term 'height_scan' is not registered.` Porting the height-scan
pipeline into `deploy/robots/go2w` surfaced the real problem: Try46's actor height scan
was chosen to *match the critic's*, and nobody had checked it against the height map this
project already publishes (`unitree_mujoco`'s `HeightMapSimulator` -> `rt/height_scan` ->
the deploy parser). Two of the three defining properties disagreed:

1. **The grid was transposed.** `GridPatternCfg.ordering` defaults to `"xy"`, which is
   torch `meshgrid(indexing="xy")`: **x** is the fastest-varying axis, so ray `k` is
   `k = iy*Nx + ix`. Both deploy-side implementations walk **y** fastest
   (`height_map_simulator.cpp`: `idx = ix*kGridNy + iy`). Same 187 numbers, transposed --
   a policy trained on `"xy"` reads a 1.6 m *forward* elevation profile as a 1.0 m
   *lateral* one. Any MuJoCo run of Try46/47/48 would have been measuring a scrambled
   observation, not the policy.
2. **The height offset was wrong.** `mdp.height_scan` returns `sensor_z - hit_z - offset`
   with `offset=0.5` by default; the deploy stack's shared `height_scan::kOffset` is
   `0.273175`. A constant 0.226825 m bias -- absorbable by the network, but only if it was
   *trained* with it.

Size, resolution and yaw-alignment already agreed (17x11 at 0.1 m over 1.6x1.0 m, rays
world-vertical, grid following heading), so the mismatch was exactly these two.

**Honest note on how this happened:** this is a design error in Try46, not a discovery
about the environment. "Match what the critic sees" was a convenient default, chosen
without asking whether the project already had a *deployable* height-scan spec. It did --
the publisher and the C++ parser had been deliberately aligned with each other all along.
Three tries (46, 47, 48) then ran on an observation that could not be checked outside
Isaac Lab. The cost is not the compute; it is that Try47's "perception did not help"
conclusion above rests on an Isaac-only comparison that was never externally validated.

**Fixed on the training side (Try49), deliberately.** The deploy spec has to hold across
the publisher, the C++ parser and the real robot's utlidar; the training config is the
free variable. Try49 = Try48 with `ordering="yx"`, `offset=0.273175`, and the actor's
`clip` set to the publisher's own `(-1.0, 5.0)` -- Try46 had widened that to
`(-2.0, 5.0)` precisely because `offset=0.5` made a crouched robot saturate the floor,
which was a symptom of the wrong offset rather than a real need. Paired with moving
`unitree_mujoco`'s Go2W `utlidar` site back to the **base_link origin** (it had been
placed at `pos="0 0 -0.226825"` to fake Isaac's 0.5 without touching the shared
`kOffset`), so both sides now compute `base_z - hit_z - 0.273175` from one constant.
**The two changes only work as a pair -- do not reintroduce the site fudge.**

The observation *shape* is unchanged at 328, so a Try48 checkpoint loads without complaint
and is silently wrong (every input permuted and shifted) -- hence a full
Phase1 -> Phase2 -> Phase5 retrain rather than a resume.

**Standing rule this adds:** an observation term destined for deploy must be specified
against the deploy side's definition, not against the critic's, and the grid *ordering* is
part of that definition.

**Try49 -- the realignment, measured.** Phase1 500 / Phase2 2000 / Phase5 **900** (Phase5
deliberately shorter than Try48's 2000: nothing here was meant to change behaviour, and
the deliverable was a *checkable* policy rather than a longer one).

| | terrain_levels (final / peak) | base_contact | time_out | bad_orientation | goal_arrival | Loss/estimator |
|---|---|---|---|---|---|---|
| Try48 (2000 iter, misaligned) | 6.84 / 6.90 | 37.4 % | 59.5 % | 3.1 % | 0.002 | 0.017 -> 0.013 |
| Try49 (900 iter, aligned) | 6.65 / 6.75 | 46.2 % | 49.3 % | 4.4 % | 0.002 | 0.026 -> 0.014 |

**The realignment cost nothing measurable** -- Try49 is within ~0.2 of a level of Try48
on less than half the Phase5 budget, and the estimator loss converges to the same 0.014.
That is the expected outcome and worth stating as such: a fixed permutation plus a
constant offset is exactly the kind of structure a network relearns from scratch without
difficulty, so the transpose was never going to show up as an Isaac-side *training*
problem. It only ever mattered at the boundary. Nothing about the wall itself improved
(`goal_arrival` still ~0.002); what Try49 buys is a policy whose observation the deploy
binary can reproduce.

`deploy/robots/go2w/config/config.yaml`'s `policy_dir` now points at
`go2w_v1_phase5_try49`. Exported graph verified: ONNX `obs [1, 328]` + `h_in [1, 1, 256]`,
and `deploy.yaml` carries both `keyboard_velocity_commands` (so MuJoCo keyboard input
works) and `height_scan`.

**Try49 Play-checked: still cannot climb 0.60 m. The perceptive/estimator line is closed
and was not promoted.** Isaac-side numbers looked respectable (terrain_levels 6.65 on 900
Phase5 iterations), which is exactly the trap this file has warned about twice already --
`terrain_levels` measures how far up the curriculum a run got, and level 6.65 corresponds
to ~0.47 m walls. Play showed no 0.60 m crossing. Sandbox cleared 2026-09-09.

### What the whole 0.60 m campaign (Try35-49) settled

Four distinct classes of hypothesis were tested and all came back null:

| class | tries | outcome |
|---|---|---|
| terrain shape (sloped approach, tile size) | 35, 36 | null; Try36 produced elbow-propping, no crossing |
| a reward that pays for lifting the body | 37, 38, 39 | folded `wall_body_height`, later shown to be a **trap** from scratch |
| the termination that killed leaners | 40, 41, 42 | Try41 was the campaign best (~30 % in Play) and never reproduced in MuJoCo |
| curriculum promotion rule | 43 | null, and confounded by a bootstrap change (my error) |
| friction, from both sides | 44, 45 | null (surface 0.3-1.2 -> 2.0-3.0; wheel joint 0.01 -> 2.0 N*m, ~200x) |
| perception (privileged height scan in the actor) | 46, 47, 49 | null against the blind Try34 baseline |
| state estimation (lin_vel + CoM-CoP, TumblerNet) | 48 | estimator learned (loss 0.017 -> 0.013), held its peak better, still no crossing |

**Where that leaves the problem.** 0.50 m is solved and reproduces in MuJoCo; 0.60 m
never did, under any of the above. The two things never actually tested are worth writing
down so nobody re-derives them:

  * **The goal geometry lets "arrival" happen on the near side of the wall.**
    `goal_radius_range=(1.75, 2.5)` with `arrival_radius=0.5` means a goal at 1.75 m is
    satisfied from 1.25 m -- which is exactly where the wall ring sits. Found by reading
    the config, never tested. If true, `goal_arrival` has been partly payable without
    crossing for the whole campaign, which would undercut every reward-shaping result
    here.
  * **The curriculum's promotion window is narrower than its arrival radius.** Promotion
    needs net displacement > `tile_size * 0.35` = 1.925 m; the wall's far face is at
    1.45 m. So a robot that crosses cleanly and stops is not promoted, and the 0.60 m
    wall only exists at the top curriculum row -- see
    `terrain_levels_climb_demote_on_fail`'s docstring, which records this as untested
    rather than disproved.

Both are geometry bugs in the *measurement*, not in the policy. A next campaign should
fix those before shaping anything else, because every reward result above was read through
them.

### What was deleted, and what was kept

Deleted (2026-09-09): `velocity_env_cfg_perceptive.py` (Try46), `velocity_env_cfg_phase5_try47.py`,
`velocity_env_cfg_estimator.py` (Try48), `velocity_env_cfg_try49.py`, all their gym
registrations, `assets/models/actor_critic_recurrent_estimator.py`,
`assets/models/modules/estimator_ppo.py`, `GruEstimatorPPORunnerCfg`,
`mdp.observations.com_cop_vector`, and ~2.9 GB of Try46-49 checkpoints.

Kept deliberately:

  * **The default Phase5 and the `-Adjust` polish task**, unchanged throughout -- Try49
    was never promoted. `deploy/robots/go2w/config/config.yaml`'s `policy_dir` is back on
    `go2w_v1_phase5_adjust`.
  * **`unitree_mujoco`'s Go2W support** on `feat/height-map`: foot friction 0.8 -> 1.0 and
    wheel `ctrlrange` 15 -> 23.7 N*m (matching the Isaac actuator), the `utlidar` site at
    the base_link origin, and the wall staircase in `scene_terrain.xml` (tops at 0.08 to
    0.92 m, the 0.62 m step being the target). Independently useful for any future wall
    work, and the friction/ctrlrange values are corrections rather than experiment
    scaffolding.

Also deleted (2026-09-09): the **deploy-side height-scan pipeline** --
`HeightScanUpdater.{h,cpp}`, `REGISTER_OBSERVATION(height_scan)`, `apply_height_scan_noise`,
`resolve_bool_option`, the `FSM.Velocity.height_scan` config block, and the
`main.cpp`/`CMakeLists.txt` hooks. `deploy/robots/go2w/` is now byte-identical to its
committed state. It was built and verified end to end against `unitree_mujoco`'s publisher,
so if a future perceptive policy needs it, the specification is worth recovering rather than
re-deriving:

  * one float32 `"z"` field, `point_step = 4`, `height = Nx = 17`, `width = Ny = 11`,
    `row_step = 44`, cell `(ix, iy)` at flat index **`ix * Ny + iy`** (y fastest);
  * values are `base_z - hit_z - kOffset` with `kOffset = 0.273175`, already clipped to
    `(-1.0, 5.0)` by the publisher, so the deploy side passes them through untouched;
  * the training side must therefore set `ordering="yx"` and `offset=0.273175` on
    `mdp.height_scan` (Isaac's `"xy"` default is the transpose);
  * the under-body exclusion (`mdp.height_scan_excluding_body`, 35 masked cells -> 152
    values at this resolution, **not** the Go2 implementation's literal 117, which was
    sized for a 29x21 grid at 0.05 m) belongs behind a config option defaulting to the
    Go2 behaviour, not hard-coded;
  * the missing-topic fallback must be the **trained flat-ground reading**, never the
    `kEmpty` sentinel -- see the defect note below.

One deploy-side defect found while debugging this, worth remembering even though the code
it lived in is gone: `HeightScanUpdater::get()` filled 187 cells with `kEmpty = -1.0` when
the topic was missing or stale. In this convention -1.0 means "terrain 0.73 m above the base, on every
side" -- an input no policy has ever seen. The robot thrashes and falls the instant it
enters the policy state, which reads as a bad policy and says nothing about the missing
topic. Measured on Try49's own weights: max|action| 7.13 vs 2.80 on a correct flat scan,
with the estimator hallucinating 0.70 m/s of forward velocity while standing still. **The
signal that an input is missing belongs in the log, not in the robot's behaviour** -- the
fallback is now the trained flat-ground reading (`nominal_base_z - kOffset` = 0.176825)
with a warning every 2 s. `config.yaml`'s `flat_value: 0.0` was wrong for the same reason
(0.0 means "ground 0.273 m below the base" -- a step down, not flat; the correct value is
`nominal_base_z - kOffset` = 0.176825). **Rule: a missing input must degrade to something
the policy was trained on, and announce itself in the log -- never encode "missing" as an
extreme value the network has never seen.**
