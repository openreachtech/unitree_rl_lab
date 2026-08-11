# Go2W Phase 5 sandbox — what was tried and what it established

Phase 5 is the extreme-obstacle-crossing task for the wheeled Go2W. This file records the
sandbox campaign that produced the current `velocity_env_cfg_phase5.py` (Try 1 – Try 9,
2026-08-02 … 2026-08-11). Every try has been folded into the default and deleted; this is
the reasoning that would otherwise be lost with them.

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
* **MuJoCo cannot yet reproduce a 0.50 m climb.** Three candidate causes, none yet
  isolated:
  1. *Wheel torque mismatch.* Isaac declares one actuator group, `[".*"]` with
     `effort_limit=23.5`, so the wheels get the leg budget; the MuJoCo model shipped with
     `ctrlrange="-15 15"` on the wheels, 36 % less. MuJoCo was raised to 23.5 as a
     **diagnostic**. If 15 Nm is the real hardware limit, the correct end state is the
     opposite — lower Isaac's wheel limit and retrain — otherwise the policy depends on
     torque the robot does not have.
  2. *Obstacle shape.* The MuJoCo scene uses thin walls (0.10 / 0.30 m thick); training
     uses stairs with a 1.00 m tread. Landing space after the riser is completely
     different. Phase 4's `MeshThinWallTerrainCfg` still exists if walls should be mixed
     back into training.
  3. *Undertrained.* The current default is at 0.40 m mean, below the 0.44 m the earlier
     lineage reached and well below 0.50 m.
  A useful next diagnostic: put a staircase matching training geometry (2 steps, 1.0 m
  tread, 0.30 / 0.40 / 0.50 m rises) into the MuJoCo scene. If that climbs, the policy is
  fine and the wall shape is the problem.
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
