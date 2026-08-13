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
here — see Lesson 7. Consequently `Go2W-v1-Phase1` through `Go2W-v1-Phase5` (not just this
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
  2026-08-13: yes.** `Go2W-v1-Phase1` through `Go2W-v1-Phase5` all register with
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
