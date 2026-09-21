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
| `flip`   | `kind` backflip / frontflip / sideflip_left / sideflip_right / jump, `count`, optional `running` | maps to `JumpCommand.MOTION_*` |
| `stance` | `kind` handstand (front legs) / hindstand (hind legs), `duration_s`, optional `dir`+`speed` | `HandstandCommand` sign +1 / -1 |

A flip is done from standing unless `"running": true`, which fires it at the end of the `move` just
before it without stopping; the flip ends the move, and the stop (or the next step) follows as
soon as the window closes. Training mostly ran on after landing, so whether an immediate stop
costs landings is something `--calibrate` measures (`--running-recover-s` compares). The policy pairs each heading with one rotation
(`JumpCommand._select_motion_for_direction`: forward → frontflip, backward → backflip, left/right →
the matching sideflip, diagonals by their fore-aft component) and only offers a move under 1.0 m/s,
so `validate_program` rejects a running flip whose kind does not match the move's direction or
whose move is `fast`. The model is taught to propose the matching kind instead
(`RUNNING_FLIP_FOR` in `grammar.py`).

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
# 4. Instructions + replies: 5000 single-turn rows plus 2500 two/three-turn dialogues (dialogues.py),
#    from the validated programs and the capability table. Every user turn carries a state block.
python scripts/llm/build_dataset.py --out data/llm/dataset.jsonl --total 5000 --dialogues 2500

# 5. Round-trip: read each instruction back into a program, using words only, and compare.
python scripts/llm/parse_instruction.py data/llm/dataset.jsonl -o data/llm/reparse.jsonl
python scripts/llm/roundtrip_check.py data/llm/dataset.jsonl data/llm/reparse.jsonl

# 6. Ten automated checks per assistant turn (format, grammar, state, declines, warnings, numerals,
#    wording, dialogue logic, dupes, balance), then the wire format and the GBNF over every target.
python scripts/llm/check_dataset.py data/llm/dataset.jsonl
python scripts/llm/chat_format.py --check data/llm/dataset.jsonl
python scripts/llm/gbnf_match.py --dataset data/llm/dataset.jsonl

# 7. Fine-tune (env_llm), score every turn of the held-out split, and the hand-written set.
python scripts/llm/train_sft.py --out logs/llm/qwen3-1.7b-dora-v2
python scripts/llm/eval_model.py --adapter logs/llm/qwen3-1.7b-dora-v2/adapter
python scripts/llm/handwritten_eval.py && python scripts/llm/eval_model.py --adapter ... --dataset data/llm/handwritten_eval.jsonl

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

The wording lives in two hand-written modules and nothing else: `phrasebank.py` (Japanese surface
forms for every step, in three registers) and `negatives.py` (chit-chat, impossible requests,
over-asks, ambiguous requests). `build_dataset.py` holds the *decisions* -- which skill to decline,
which to warn about, how long the program takes -- and takes all of them from `capability.json`
and the sim measurements, never from the text. Keeping the two apart is what lets the checks in
step 6 mean something.

`data/llm/system_prompt.txt` holds the system prompt (the grammar spec plus the behavioural rules);
it is stored once rather than repeated on all 5000 rows.

## The wire format

`chat_format.py` is the single definition of what the model reads and writes; the trainer, the
eval, the robot client and the GBNF grammar all import it.

```
[状態] 実行中 2.1s/7.4s: 前へ5m(いま 残り3.1s) → 前方回転×1(走りながら) / 中断可     ← user turn: state block
ええ天気やなあ                                                                       ← then what the person said

ほんまやな、走ってて気持ちいいわ。                                                   ← assistant turn: one-line reply

action: none                                                                         ← what to do with the queue
program: []
```

`action` is `none` (leave the robot alone), `cancel` (stop, drop the queue), `replace` (stop and run
this), `insert` (do this now, then go back to the interrupted step -- a bare 「ハンドスプリングして！」
mid-run flips out of the run and the run continues) or `append` (run this after). `none`/`cancel`
carry `[]`, the other three a non-empty program; an inserted program is compiled against the step
under way (`compile_program(..., context=, resume=True)`), which is also what checks a running
flip's kind against the heading the robot actually has; `render_output` refuses other pairings and the GBNF cannot express them. The state block
heads every user turn so the history stays append-only. `gbnf_match.py` is a small reference
matcher that checks every training target against the generated grammar without llama.cpp.

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

- **Emergency words never wait for a generation.** 「ストップ」「止まって」「やめて」 cancel the queue
  on the console thread, in the time it takes to send a datagram; the line still goes to the model
  afterwards so the reply and the conversation match what the robot did.
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
