# Sandbox summaries

Historical records of experiments whose code has since been deleted. Each section says
what was tried, why, what happened, and where the surviving conclusion lives now.

- [Try-1 .. Try-9: Phase3-balance-floating flat-idle fix](#try-1--try-9-phase3-balance-floating-flat-idle-fix)
- [Height-map source comparison (Go2-HMBench)](#height-map-source-comparison-go2-hmbench)

---

## Try-1 .. Try-9: Phase3-balance-floating flat-idle fix

Summary of the sandbox experiments that shaped the current
`Unitree-Go2-Velocity-v2-Phase3-balance-floating` default in
`velocity_env_cfg_phase3.py`. All of Try-1 through Try-9 targeted that one
task; none touched Phase4. The `tryN.py` files themselves have been deleted
after this summary was written -- this is the historical record of what was
tried, why, and what happened.

### Goal

MuJoCo deploy testing of the Phase3-balance-floating checkpoint (which
already climbed stairs, including floating/open-riser ones, well) showed the
policy flattening/flapping its legs on flat ground at zero command. The
project across these tries was to fix that without regressing stair-climbing.

### Try-by-try

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

### Two important cross-cutting findings

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

### Related fix (not a numbered try, but part of the same debugging arc)

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

### Where things landed

The default `RewardsCfgPhase3BalanceFloating` currently matches **Try-7's**
recipe: `rel_standing_envs=0.1`, `base_height_climb` and `wild_foot_clearance`
both gated by command, `stall_penalty`, `stair_commit`, and `quiet_standing`
(weight 0.5, gated by command *and* terrain flatness). This is the most
recently MuJoCo-confirmed-good configuration, kept as the default despite the
training-variance caveat above -- further chasing of the residual flat-idle
behavior was deprioritized in favor of moving on to Phase4 (thin-wall)
training.

---

## Height-map source comparison (Go2-HMBench)

How much does a training iteration cost if the critic's height map comes off a LiDAR
instead of the top-down scanner, and how much does making the robot block its own rays
add on top? Four arms were built, timed, and then deleted; only the winner of the
question they answered survives, as `MID360_DYNAMIC_MESH` in
`velocity_env_cfg_mid360.py`.

Measured 2026-09-10 on an RTX 5080 (16 GB), Isaac Sim 5.1, 4,096 environments, 40
iterations with the first 10 discarded, median iteration.

### Setup

Four registrations, `Go2-HMBench-{TopDown,Fan,Mid360,Mid360Occ}-Phase1`, each
`Go2-Blind-GRU-Phase1` with a 609-cell height grid appended to the **critic** group and
nothing else changed. The actor stayed blind and identical in all four, so any difference
in iteration time is the cost of producing the grid.

| arm | height map source |
|---|---|
| TopDown | the existing `height_scanner`: 609 rays straight down from 20 m up |
| Fan | `velocity_env_cfg_lidar.py`'s 1,080-ray body-mounted fan, binned by `mdp.LidarElevationMap` |
| Mid360 | the MID-360 of `velocity_env_cfg_mid360.py`, 1,000 rays/step, terrain mesh only |
| Mid360Occ | the same MID-360 with the robot's collision geometry as a moving second mesh |

Held fixed: terrain, rewards, events, commands, actor observations, network, PPO
hyperparameters, and the grid itself -- same 29 x 21 cells, resolution, extent, centre,
offset and clip, so the critic's first layer was the same shape everywhere. The fan's
usual 0.80 x 0.60 m body exclusion was switched off to keep all four at 609.

Not held fixed, deliberately: each map kept the noise model it ships with (none for the
top-down scan, `GO2_LIDAR_NOISE_CFG` for the fan, `MID360_NOISE_CFG` for both MID-360
arms). Those are a few tensor ops on 609 cells, negligible against a raycast, and
stripping them would have timed a configuration none of the arms would be trained with.

`height_scanner` stayed in the scene in all four, because Phase 1's `wild_foot_clearance`
and `foot_clearance_terrain_adaptive` hold a `SceneEntityCfg("height_scanner")` and the
reward manager evaluates them even at weight 0.0. The three LiDAR arms therefore measure
the **marginal** cost of their sensor, not a substitution for the baseline.

### Results

| arm | iter median | collection | learning | steps/s | vs baseline |
|---|---:|---:|---:|---:|---:|
| TopDown | 1.306 s | 1.032 | 0.270 | 75,278 | -- |
| Fan | 1.474 s | 1.203 | 0.267 | 66,692 | x1.13 |
| Mid360 | 1.479 s | 1.211 | 0.267 | 66,478 | x1.13 |
| Mid360Occ | 1.852 s | 1.576 | 0.266 | 53,067 | x1.42 |

`learning` came out flat at 0.266-0.270 s across all four. That was the built-in validity
check: the critic input was identical by construction, so a difference there would have
meant the comparison was measuring something other than the sensor. It did not.

Marginal cost over the baseline, per iteration and per env step (24 steps/iteration):

| | +collection/iter | per env step |
|---|---:|---:|
| Fan (1,080 rays) | +0.171 s | 7.12 ms |
| Mid360 (1,000 rays) | +0.179 s | 7.44 ms |
| Mid360Occ | +0.544 s | 22.67 ms |

Projected wall clock at 7,300 iterations (the length of the Phase 4 run): 2.65 h, 2.99 h,
3.00 h, 3.76 h.

### What the numbers said

**Fan and Mid360 are indistinguishable** -- 1.474 vs 1.479 s, 0.3% apart. Ray count
(1,080 vs 1,000) is what sets the cost; the scan pattern does not. The MID-360's rolling
window costs one indexed read and one broadcast copy per step, which is nothing, so the
non-repetitive rosette is free relative to a fixed fan. Choose between them on what the
map looks like, not on what it costs.

**The dynamic mesh costs more than the raycast it protects** -- +0.365 s/iter over plain
Mid360, 2.05x the MID-360's own raycast. It is not the second `raycast_mesh`: the
dominant term is refitting a BVH over 4,898,816 triangles (18 bodies x 4,096 envs, 650
vertices per robot) 50 times a second. `occluder_body_names` is the knob -- `["base",
"F[LR]_.*"]` keeps the parts that shadow a forward-looking mount and roughly halves it.

**Self-occlusion removes 5.1 percentage points of returns** standing still on flat ground
(1,000 rays -> about 51 blocked per step), min 41.6% -> 34.1% and max 100.0% -> 97.0%
returned. Even at the best step something is in the way. A walking policy stepping over
Phase 4's walls swings the legs further and would lose more.

Caveat on the absolute numbers: the run logged `CPU performance profile is set to
powersave`, so everything is slower than the machine can go. The ratios are unaffected.
Phase 1 is also a flat plane, which is the cheap case for a raycast; rough terrain and
Phase 4's walls raise hit rates and would likely widen the gaps.

### Two bugs worth remembering

**Robot links are instanceable.** IsaacLab spawns them as instanceable references, so
`Usd.PrimRange` and `Prim.GetChildren()` stop at the instance boundary and report every
link as childless. That reads exactly like an asset with no collision geometry, and the
first version of the occluder failed with "no occluding bodies matched" because of it.
`Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)` is the fix; instance proxies are
read-only, which is all that is needed to read vertices and transforms.

**The sensor sits inside its own housing.** `radar_joint` puts the L1 at
`(0.28945, 0, -0.046825)` in the base frame and `Head_lower`'s collision sphere is 4.7 cm
in radius centred at `(0.293, 0, -0.06)` -- 1.4 cm away, so the mount is inside it. Warp's
`mesh_query_ray` does not cull back faces, so every ray would have exited through the
inside of that sphere within 6 cm and the whole map would have gone unobserved. Handled by
`occluder_drop_geometry_containing_sensor`, which tests the mount against each geometry at
init and drops the ones enclosing it, naming them in the log. That is also the physically
right answer: the shell a LiDAR looks out of is the one part of the robot it can never
range on.

Related, and the reason the occluder uses collision rather than visual geometry: the Go2's
**visual** trunk mesh alone is 233,784 points / 77,928 faces. Replicated across 4,096
environments that is a billion vertices. The collision geometry is 27 analytic prims
(boxes, cylinders, spheres, straight from `go2_description.urdf`) totalling 650 vertices
per robot.

### Where things landed

The comparison rig is gone: `velocity_env_cfg_hmbench.py`, the four `Go2-HMBench-*`
registrations, `HmBenchGruPPORunnerCfg`, and three throwaway tools
(`scripts/tools/benchmark_height_map_sources.py`, `dump_robot_collision_prims.py`,
`check_self_occlusion.py`). Re-running any of it means rebuilding from the setup described
above.

What survives is `sensors/robot_occluder.py` and its switch `MID360_DYNAMIC_MESH` in
`velocity_env_cfg_mid360.py`, **on by default** for `Go2-Blind-GRU-Mid360-Phase4`. The
+25% was judged worth paying: with the body transparent, the cells a leg should have
shadowed come back *measured* and correct, so the map's unobserved pattern is wrong in a
gait-correlated way -- exactly the structure a height-map encoder trained behind this
policy would otherwise learn to rely on.

One thing the comparison did **not** answer: whether any of these maps trains a better
policy or a better encoder. Every number above is cost. Phase 1 is a flat plane, where
all four maps are nearly the same picture.
