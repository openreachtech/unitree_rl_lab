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

| skill    | fields                                                      | notes                              |
|----------|-------------------------------------------------------------|------------------------------------|
| `move`   | `dir` forward/backward/left/right (robot frame), `speed`, `duration_s` **or** `distance_m` **or** neither | neither = until something replaces it |
| `turn`   | `dir` left/right, `angle_deg`                               | left = positive yaw; always at full yaw rate |
| `flip`   | `kind` backflip / frontflip / sideflip_left / sideflip_right | one rotation; two of them is two steps |
| `stance` | `kind` handstand (front legs) / hindstand (hind legs), `duration_s` **or** nothing | `HandstandCommand` sign +1 / -1 |

A `move` or a `stance` written without a length runs until something replaces it, so only the last
step of a program can be written that way. A flip is done from standing unless the step before it is
a `move`, in which case it is fired out of that move without stopping -- that is not a field, it
follows from the order, and the compiler and the phrase bank both read it the same way
(`running_flags`). The policy pairs each heading with one rotation
(`JumpCommand._select_motion_for_direction`: forward -> frontflip, backward -> backflip, left/right
-> the matching sideflip), so a flip behind a move has to be the matching kind.

There is no `stop` step (an empty program is how the robot is stopped), no `jump`, no diagonals,
and no `count` -- each was a field the model had to reason about and the order already carries.

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
python scripts/llm/sample_programs.py --count 4000 --seed 0 \
    --capability data/llm/capability.json --out data/llm/programs.jsonl
```

Half of those programs get their Japanese from the phrase bank, half are written by hand as whole
conversations. Which half is which comes from a hash of the program id, so a program never lands on
both sides or on neither. The hand-written half is the slow part and has its own spec and running
order in [HANDWRITTEN.md](HANDWRITTEN.md); it ends in `data/llm/handwritten.jsonl`, and a person
looks at it before it is merged.

```bash
# 3. Look at what was written, before it goes anywhere.
python scripts/llm/check_handwritten.py data/llm/handwritten.jsonl     # structure only, no Japanese
python scripts/llm/show_handwritten.py data/llm/handwritten.jsonl --stats
python scripts/llm/show_handwritten.py data/llm/handwritten.jsonl --sample 30

# 4. Merge: the phrase bank writes its half (single turns plus dialogues.py's multi-turn rows),
#    the hand-written half goes in as it was written, ids and the train/eval split are stamped on.
python scripts/llm/build_dataset.py --out data/llm/dataset.jsonl

# 5. Round-trip the rendered instructions: read each back into a program, using words only, with a
#    lexicon that never imports the phrase bank. Hand-written rows are deliberately not in this.
python scripts/llm/parse_instruction.py data/llm/dataset.jsonl -o data/llm/reparse.jsonl
python scripts/llm/roundtrip_check.py data/llm/dataset.jsonl data/llm/reparse.jsonl

# 6. The wire format, rendered and re-read over every turn.
python scripts/llm/chat_format.py --check data/llm/dataset.jsonl

# 7. Fine-tune (env_llm).
python scripts/llm/train_sft.py --out logs/llm/qwen3-1.7b-dora-v2

# 8. Fold the adapter in, convert to GGUF, quantize, and run the grammar through the real sampler.
#    llama.cpp lives in /home/tak/isaacsim/llama.cpp (CPU build); the Jetson gets its own CUDA build.
python scripts/llm/export_gguf.py --adapter logs/llm/qwen3-1.7b-dora-v3/adapter --out logs/llm/gguf/v3 --test
```

`<out>/` then holds `model-Q8_0.gguf` (~1.75 GB; Q8 keeps the trained weights nearly intact and the Orin has the memory), `output.gbnf` and the `system_prompt.txt` the
adapter was trained under -- everything the serving side needs. Measured on the workstation CPU
(16 threads, Q4_K_M; Q8 is somewhat slower): prompt eval ~880 tok/s, generation ~78 tok/s; a 1.6k-token prompt costs ~1.8 s,
so the server must keep the KV cache between turns (`llama-server` with `cache_prompt`) and the
client must append to the conversation rather than re-render it -- which is what
`chat_format.conversation_text` is for.

The synthesized wording lives in two hand-written modules and nothing else: `phrasebank.py`
(Japanese surface forms for every step, in three registers, plus the idiom tables) and
`negatives.py` (chit-chat, impossible requests, ambiguous requests). `build_dataset.py` and
`dialogues.py` hold the *decisions* -- which program, which scenario, what is left in the queue --
and never invent Japanese. That split is what makes the round trip in step 5 mean anything: the
reparser is a third lexicon that has never seen either side. The hand-written half is outside all
of this by design, which is why it is checked for structure and read by a person instead.

`data/llm/system_prompt.txt` holds the system prompt (the grammar spec plus the behavioural rules);
it is stored once rather than repeated on every row.

## The wire format

`chat_format.py` is the single definition of what the model reads and writes; the trainer, the
eval, the robot client and the GBNF grammar all import it.

```
やっぱり後ろに2m下がって                                        ← user turn: what the person said

queued programs: [{"skill": "move", "dir": "forward", ..., "distance_m": 1.8}]   ← then the queue

了解、後ろに2m下がるで。                                        ← assistant turn: one-line reply

program: [{"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 2.0}]
```

**What comes back replaces what was queued**, and that one rule absorbs the five actions the format
used to carry: copy the list back to change nothing, `[]` to stop, put a step in front to do it now
and let the rest carry on, put one behind to add it at the end, or write a different list entirely.
The queue is rendered as *what is left* of each step at the moment the reply lands, so copying
「あと1.8m」 back is the same as not being interrupted and the model never does arithmetic. It sits
after the person's words, at the very end of the prompt, so the server re-reads as little as
possible between turns and the text being copied is next to where it is written.

History is the conversation only: past turns keep the words and drop both the queue and the
program, because a queue from two turns ago describes a robot that has moved on. There is no
`<think>` block anywhere, and the grammar forbids a reply starting with `<` so the base model's
habit cannot leak through.

## Running it

`conductor.py` is the serving side: console in, llama-server in the middle, UDP out to the
controller. The controller gained one port and nothing else (`ProgramLink`, ~230 lines); the
compiler, the state block and the conversation stay here, where the training data was made. The
whole shape is in [SYSTEM.md](SYSTEM.md).

```bash
# 1. simulator                  (or the real robot: --network <eth iface> below instead of lo)
cd ~/unitree/unitree_mujoco/simulate/build && ./unitree_mujoco     # interface: lo

# 2. the model
~/isaacsim/llama.cpp/build/bin/llama-server -m logs/llm/gguf/v3/model-Q8_0.gguf \
    --port 8080 -t 16 -c 4096 --keep -1

# 3. the controller             [1] FixStand, then [6] Multitask
cd deploy/robots/go2/build && ./go2_ctrl --network lo

# 4. the conductor
python scripts/llm/conductor.py --llm http://localhost:8080
> 前に3mくらい走って
> そのままハンドスプリング!
> あ、ストップ!
```

Without `--llm` the same queue is driven by hand, which is how the executor is brought up before
the model is in the loop: `/fwd 3`, `/turn left 90`, `/flip backflip running`, `/stance front 5`,
`/append <json>`, `/cancel`, `/state`.

Two things the conductor does that are not in the model:

- **Stopping is not special-cased.** There was a hot path that cancelled on 「ストップ」「止まって」
  before the model answered; it was removed (2026-09-21) because it guessed -- 「止まれというまで
  走って」 reads as a stop to a regex and not to a person. Stopping is a request like any other, and
  the backstop is the keyboard: [Space] zeroes the command inside the controller and latches the
  link off, which no generation can undo.
- **The state block is predicted, not observed.** It is built at `elapsed + <measured generation
  time>`, because the answer lands one to three seconds after the person spoke and a state block
  describing the moment of asking would have the model planning against a step that is over.

[Space] on the controller's keyboard still wins over all of it: it zeroes the command and latches
the link off until the conductor sends `RESUME`, which it does when the operator asks for something
new.

## Next stages

- **End-to-end evaluation**: run the fine-tuned model's programs back through
  `validate_programs.py` and check they still stand up at the end.
- **Closing the loop on distance**: `distance_m` is open-loop against the capability table's fit;
  odometry would make it a measurement.
