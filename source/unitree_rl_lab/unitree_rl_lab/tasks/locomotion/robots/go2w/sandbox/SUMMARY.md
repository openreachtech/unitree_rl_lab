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

### Open item, still unresolved as of 2026-08-27: stop-time trembling / creep

Every checkpoint validated in MuJoCo since Try26 (Try26 itself, and now Try30) shares
one un-investigated issue: once holding position at (or near) the goal, the robot
trembles and drifts/creeps forward rather than staying still, despite the command
dropping to zero on arrival and `goal_dont_wait_penalty` gating off once arrived.
Candidate hypotheses, none yet tested:
  * the GRU's recurrent hidden state may carry momentum/context from the climbing
    motion for a few steps past arrival, before settling;
  * arrival is a rare outcome (most episodes end via `time_out` or a failure
    termination before ever reaching it), so the policy may simply have little
    training experience of the "just arrived, now hold" sub-task specifically;
  * possible boundary-condition jitter right at the `goal_dont_wait_penalty`
    speed-threshold transition, though this is a guess, not yet checked against the
    actual gating logic in a running policy.
Not yet isolated to a specific cause; the next sandbox try in this line should target
this directly, one hypothesis at a time.
