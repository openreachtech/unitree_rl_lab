# Merged multi-task policy — summary of experiments

A single policy that walks, runs and gallops in four directions and can be interrupted by an
acrobatic move at any moment. Built as a mixture of experts: expert 0 initialised from the
locomotion policy, expert 1 from the acrobatics policy, expert 2 and the gates random, all
fine-tuned together with PPO. `build_moe_checkpoint.py` does the initialisation; the sandbox
`try*.py` files that produced the results below have been deleted after this summary was written,
per the convention used by the jump task's own sandbox.

## What the sandbox runs established

**Try 1 / Try 2 / Try 3 — the take-off speed curriculum.** Acrobatic moves may only fire below
`takeoff_speed_limit`, which a curriculum raises as flips keep landing. Three things came out of
those runs and are now in `multitask_env_cfg_moe.py` and `mdp/curriculums.py`:

- *Judge on more than flip success.* Promoting on success alone let the limit climb 0.3 -> 3.5 m/s
  in 225 iterations while velocity tracking went 0.40 -> 1.49. The policy was buying flip success by
  giving up running. `max_velocity_error` now gates promotion.
- *Measure the error over the right window.* Using the velocity command's own `error_vel_xy`
  charges the gate for the steps spent mid-flip, where no ground velocity command can be followed
  by construction; the limit then oscillated 0.3 -> 0.5 -> 0.3 without settling.
  `MultiTriggerJumpCommand.locomotion_error` excludes those steps.
- *Size the window in attempts, not episodes.* `minimum_attempts` was copied from the assist
  curriculum, where one episode is one attempt. Here each 20 s episode holds three to five flips
  across thousands of environments, so 1024 attempts accumulated in about seven iterations -- 32
  promotions in 225, and the limit hit its ceiling long before anything consolidated.

**`Go2-Multitask-Measure` / `-Measure-Expert`** were per-attempt measurement tasks. Superseded by
`scripts/rsl_rl/measure_motion.py`, which does the same job for any task and prints a distribution
rather than a mean.

## The two settings that decided the outcome

**Command window (`ACRO_WINDOW_S`), 0.5 -> 1.5 -> 1.0 s.** This is not a deploy-side timeout: it
sets how long the `enabled` flag stays high in the observation, and the gate keys its expert-routing
prior off that flag. At 0.5 s the prior lapsed a third of the way through every flip and handed an
inverted robot back to the locomotion expert -- 59% of the action while upside down, flip success
capped at 0.55. At 1.5 s that became 0.7% and 0.79. But 1.5 s outlasts the motion, which is felt as
a move that will not hand back cleanly to the gait; measured trigger-to-landing under zero assist is
0.82 s for both sideflips, 0.84 s for the handspring and 0.88 s for the backflip. 1.0 s covers the
motion without overstaying it. The expert's `command_duration_s`, this constant, and the deploy
state's `command_duration_s` must all agree.

**Acrobatics speed ceiling (`ACRO_SPEED_CEILING`), 3.5 -> 1.0 m/s.** The take-off limit gates
*whether a move fires at all*. Raising it to the locomotion ceiling asks the policy to land a flip
while being commanded 3.5 m/s, and the only way to do that is to stop: over 3000 iterations the
limit reached 3.5 halfway through, velocity error went 0.03 -> 2.11, and the flips kept landing 74%
of attempts. A flip from 3.5 m/s is not a skill being withheld; it is one that does not exist.
Capping at 1.0 also evens out *which* move fires -- the heading picks the move by dominant axis, and
`lin_vel_x` spans [-1.0, 3.5] against `lin_vel_y`'s [-1.0, 1.0], so above 1.0 m/s lateral commands,
and therefore both sideflips, are a small minority.

## Direction-matched motions

The merged environment fires no plain jump (`enable_jump = False`). Every move carries a direction
and `_select_motion_for_direction` hands the commanded heading the one that goes with it:

    forward   handspring       pitch +1
    backward  backflip         pitch -1
    left      sideflip         roll  -1
    right     sideflip right   roll  +1

Below `direction_conflict_speed` the sampled motion is kept, which is what leaves a move available
while stationary. The earlier design substituted a vertical jump whenever the sampled move fought
the commanded direction -- it kept the interruption but discarded the rotation, and left half the
headings with nothing else to do. Flip failure had tracked direction almost perfectly: at a 0.7 m/s
limit the jump held 0.94 slow and 0.90 fast because it has no direction, while the backflip fell
0.90 -> 0.64 and the sideflip 0.30 -> 0.17.

## Result (3000 iterations, capped ceiling, verified in MuJoCo)

| motion | success per attempt | share |
|---|---|---|
| backflip | 0.543 | 0.251 |
| handspring | 0.405 | 0.247 |
| sideflip left | 0.272 | 0.251 |
| sideflip right | 0.851 | 0.251 |

Gate routing: 0.997 acrobatics inside the command window, 0.991 locomotion outside it. Locomotion
held -- `error_vel_xy` 0.533 against 2.106 for the uncapped run, `track_lin_vel_xy` 0.907.

**Open.** The left sideflip scores 0.272 against the right's 0.851, from an expert where both were
1.000 at zero assist. The sign of the gap has flipped since the previous run, which is the signature
of the gradient interference the jump task's own SUMMARY records -- one network serving both
directions with nothing tying its response to a left roll to its response to a right roll. The tool
for that is a mirrored objective (`RslRlSymmetryCfg` / rsl_rl's `symmetry_cfg`), not more training.

The take-off limit also settles at 0.3-0.8 rather than reaching 1.0, held down by
`locomotion_error` crossing the 0.6 gate. Whether that is the right trade or the gate is too tight
has not been measured.

**Also open.** The locomotion expert is still trained under the stock actuator model while the
merged environment now runs the corrected one (see `multitask_env_cfg.py`). Go2-Gallop reaches its
robot through `velocity_env_cfg`'s stock `ROBOT_CFG`; retraining it under the corrected model is
the remaining piece.

## Which expert actually drives (`scripts/rsl_rl/measure_gate.py`, model_3000)

Training logs the gate split by the jump command's `enabled` flag, which averages over whatever
velocity command happened to be sampled -- a gallop and a backward walk land in the same bucket.
Holding one command fixed for the whole run separates them. 64 environments, checkpoint
`2026-09-02_11-42-39/model_3000.pt`.

| condition | locomotion | acrobatics | transition |
|---|---|---|---|
| gallop, vx +3.0 m/s (400 steps, no move ever fires) | 0.989 | 0.011 | 0.000 |
| backward walk, vx -0.8 m/s, between moves | 0.989 | 0.010 | 0.000 |
| backward walk, inside the backflip's command window | 0.004 | 0.995 | 0.000 |
| backward walk, 1.0-2.0 s after the trigger | 0.989 | 0.010 | 0.001 |

**The transition expert is unused.** It never exceeds 0.001 in any condition, including the
0.9-1.0 s bin where the hand-back happens and it would have the most to contribute. It was given
random weights and no prior on purpose, to earn its weight from the gate; after 3000 iterations it
has not. The two pre-trained experts are carrying the policy alone, so the third expert's
parameters and its share of every forward pass are currently pure cost.

**The hand-back is a step function on the flag, not on the robot's state.** Binned against time
since the trigger, acrobatics holds 0.99+ through 0.9 s, then drops to 0.008-0.013 in the very next
bin -- exactly where `command_duration_s` (1.0 s) ends. The gate is following the command flag it
was given as a prior rather than anything physical, so a move still recovering at 1.0 s is handed
to the locomotion expert mid-recovery. That is a plausible mechanism for flips that rotate fully
and still fail to hold the landing, and it is measurable: lengthening `ACRO_WINDOW_S` moves the
step, and if landing success moves with it the flag is the binding constraint.

Note for anyone extending the tool: this environment sets `rearm_after_s` equal to
`command_duration_s`, so the command's own `in_motion` flag expires at the same instant `enabled`
does. Bucketing on `in_motion` therefore leaves the recovery in no bucket at all -- the script
follows a fixed `--follow` window from the rising edge instead.

## Next stage: a bipedal expert, and the observation change it forced

`Go2-Multitask-Biped-Front` trains a front-leg stance -- the robot rises onto its front legs, hind
legs tucked, and walks there tracking a velocity command. It is meant to become a fourth expert.
The task config carries its own rationale; what belongs here is what it did to the shared layout.

**The unified observation grew, and every existing checkpoint has to be re-widened.**

| | before | after |
|---|---|---|
| actor | 122 | 124 |
| critic | 330 | 335 |

The actor gains `handstand_command` = `(enabled, stance)` -- a flag and its sign, +1 front / -1
hind, so the mirror stance can be trained later without moving any column again. The critic gains
the same two plus `com_cop` (3), the vector from the centre of pressure to the centre of mass,
which is the state variable the bipedal balance rewards are written in.

`jump_command` did not move: the new block sits after `jump_time`, so the gate's routing prior
still points at column 9 and needs no change.

This cost a re-widen, not a retrain, and the re-widen has been done. `widen_checkpoint.py` carried
`Go2-Multitask-Jump-Phase2` (model_2298, five motions at success 1.000) and
`Go2-Multitask-Gallop-Phase2` (model_2498) onto the new widths, placing the old weights and zeroing
the new columns; verified column by column, with every copied block bit-identical and columns
14-15 / 17-18 / 25-27 all zero. A widened network computes exactly the function it did before, so
both experts keep everything they knew.

The snapshot of the old layout (`POLICY_UNIFIED_V1` / `CRITIC_UNIFIED_V1`) was deleted once that
was done, along with `widen_checkpoint.py`'s table entries for it -- a compatibility shim with
nothing left to be compatible with is just a second layout to confuse the next reader. Git history
has it if a 122-column checkpoint ever turns up.

The merged policy's own checkpoint was deliberately *not* widened: it is rebuilt from the experts by
`build_moe_checkpoint.py`, and it needs retraining anyway now that `resolve_gated_term_params`
changed what several of its reward terms compute.

Every non-handstand multi-task task now carries `IDLE_HANDSTAND_COMMAND`, scheduled past the end of
any episode so the two columns read `(0, 0)`. Filling the slot truthfully rather than omitting the
term is what keeps the column indices the expert weights are placed by from shifting.

**A latent bug came out with it.** `resolve_gated_term_params` and `assert_observation_layout` were
both written to run as startup events, documented as running as startup events, and registered
nowhere. The first one matters: Isaac Lab resolves a `SceneEntityCfg` only at the top level of a
term's `params`, and an unresolved one does not raise -- `body_ids` defaults to `slice(None)`,
meaning *every body*. So in the merged environment `feet_slide` has been evaluating over all 17
links instead of the four feet, and `paired_gait`'s `preserve_order=True` selection has been
getting the default order. Terms whose selector was `.*` are unaffected.

Both are now registered on `MultitaskEventCfg`, which every multi-task environment inherits. Note
what this means for comparisons: the merged policy's reward function is not the one `model_3000`
was trained against. Given the observation change already forces a retrain, fixing it now costs
nothing extra -- but a metric from before this point cannot be read against one from after.

The handstand expert's own experiments are recorded with the skill, in
`tasks/biped/robots/go2/sandbox/SUMMARY.md`, following the same split the acrobatics and locomotion
experts use -- what the merged observation had to become belongs here, what the stance had to learn
belongs there.
