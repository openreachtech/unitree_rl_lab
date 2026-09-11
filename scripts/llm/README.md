# Language-driven control of `Go2-Multitask-v2`

A language model turns an instruction into a short **skill program**; a deterministic **compiler**
turns the program into the 50 Hz command stream the policy was trained on. The model never emits
velocities or timings at the control rate, and it never has to know the policy's constraints -- the
compiler inserts the stops, settle gaps, spacing and clamps.

```
"5mくらい直進してから、バク転して"
        │  language model
        ▼
[{"skill":"move","dir":"forward","distance_m":5}, {"skill":"flip","kind":"backflip"}]
        │  compile_program()            (unitree_rl_lab.program)
        ▼
 0.0- 5.9 s  move forward at 1.0 m/s        ← duration from the capability table's calibration
 5.9- 6.4 s  settle                         ← inserted: flips start from standing
 6.4- 7.4 s  backflip                       ← 1.0 s window = rearm_after_s
 7.4- 7.9 s  settle
        │  to_array() / events()
        ▼
 velocity command, jump command (motion code + targets), handstand command  → policy
```

## The grammar

`unitree_rl_lab.program.grammar` -- `describe_grammar()` prints the same thing for a system prompt.

| skill    | fields                                                     | notes                                        |
|----------|------------------------------------------------------------|----------------------------------------------|
| `move`   | `dir` (8 compass points, robot frame), `speed`, `duration_s` **or** `distance_m` | diagonals are true 45 degrees |
| `turn`   | `dir` left/right, `speed`, `duration_s` **or** `angle_deg`  | left = positive yaw                          |
| `stop`   | `duration_s`                                                |                                              |
| `flip`   | `kind` backflip / frontflip / sideflip_left / sideflip_right / jump, `count` | maps to `JumpCommand.MOTION_*` |
| `stance` | `kind` handstand (front legs) / hindstand (hind legs), `duration_s`, optional `dir`+`speed` | `HandstandCommand` sign +1 / -1 |

Speed words are nominal command magnitudes in `CompilerConfig` (forward 0.5 / 1.0 / 2.0 m/s,
backward 0.4 / 0.6 / 0.9, lateral 0.4 / 0.5 / 0.8, yaw 0.6 / 0.8 / 1.0 rad/s -- the slow ones were
raised after the first calibration found them under the policy's dead zone). Change them there, not
in the model.

## The pipeline

```bash
# 1. Measure what the policy can do: one program per skill and setting, two lengths each.
#    Writes the capability table (success rates + distance/duration fits). Pushes off.
python scripts/llm/validate_programs.py --calibrate --capability data/llm/capability.json \
    [--checkpoint logs/rsl_rl/go2_multitask_v2/<run>/model_<n>.pt]

# 2. Draw human-shaped random programs (1-4 steps, round numbers, forward-heavy).
python scripts/llm/sample_programs.py --count 2000 --seed 0 \
    --capability data/llm/capability.json --out data/llm/programs.jsonl

# 3. Run each on 8 domain-randomised replicas; keep the ones that pass >= 80 %.
python scripts/llm/validate_programs.py --programs data/llm/programs.jsonl \
    --capability data/llm/capability.json --out data/llm/validated.jsonl
```

`validated.jsonl` carries, per program: the program, the compiled timeline (events and any
adjustments the compiler made), per-replica survival, per-flip success, per-stance hold fraction,
and per-move measured displacement. That is everything the next stage needs to write an
instruction *and* a truthful reply ("about 5 m forward, then one backflip, 8 s in all").

The capability table also says which skills to teach the model to **decline** -- see
`CapabilityTable.unreliable()`. A skill that fails in simulation is not left out of the dataset;
it becomes a "still can't do that one" reply.

## What the simulator run does

`validate_programs.py` builds the play environment with every self-scheduling command path off
(velocity resampling, the flip's auto-trigger, the stance's trigger time) and drives the three
command terms from the compiled array each step: `vel_command_b` directly, `JumpCommand.set_command`
with the motion code and rotation targets on the flip's step, and the new
`HandstandCommand.set_command` on stance edges. Scoring reads the terms' own `success` flags -- the
same ones training logged -- so a "landed" here means what it meant during training.

## Building the dataset

The procedure is written up in `DATASET.md` (Japanese); this is the short version.

```bash
# 4. Instructions + replies for 5000 entries, from the validated programs and the capability table.
python scripts/llm/build_dataset.py --out data/llm/dataset.jsonl --total 5000

# 5. Round-trip: read each instruction back into a program, using words only, and compare.
python scripts/llm/parse_instruction.py data/llm/dataset.jsonl -o data/llm/reparse.jsonl
python scripts/llm/roundtrip_check.py data/llm/dataset.jsonl data/llm/reparse.jsonl

# 6. Seven automated quality checks (grammar, declines, warnings, numerals, wording, dupes, balance).
python scripts/llm/check_dataset.py data/llm/dataset.jsonl
```

The wording lives in two hand-written modules and nothing else: `phrasebank.py` (Japanese surface
forms for every step, in three registers) and `negatives.py` (chit-chat, impossible requests,
over-asks, ambiguous requests). `build_dataset.py` holds the *decisions* -- which skill to decline,
which to warn about, how long the program takes -- and takes all of them from `capability.json`
and the sim measurements, never from the text. Keeping the two apart is what lets the checks in
step 6 mean something.

`data/llm/system_prompt.txt` holds the system prompt (the grammar spec plus the behavioural rules);
it is stored once rather than repeated on all 5000 rows.

## Next stages (not built yet)

- **Fine-tune** a small instruction model to emit `{"reply": ..., "program": [...]}`.
- **End-to-end evaluation**: run the fine-tuned model's programs back through
  `validate_programs.py` and check they still stand up at the end.
- **Robot executor**: consume `Timeline.events()` in `State_Multitask` (velocity set, flip request,
  stance request/release), later closing the loop on `distance_m` with odometry.
