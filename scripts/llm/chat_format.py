"""The wire format between the model and the robot -- written, read and constrained in one place.

The model does not emit JSON. It emits a spoken reply, a blank line, then what to do about the
robot's queue and the program itself:

    了解、5m進んでからバク転するね。

    action: replace
    program: [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 5.0}, ...]

Keeping the reply outside the JSON is what lets a 1.7B model write natural Japanese: quotes,
brackets and long vowels in the reply can no longer break an escape sequence, and only the part
that is genuinely machine-shaped has to be valid JSON.

The ``action`` line exists because the conversation continues while the robot moves. An empty
program used to mean "nothing to do"; with a program running it would have to mean either "leave
it alone" (small talk) or "stop it" (「あ、ストップ」), and those are opposite instructions. So the
intent is spelled out:

    none     leave the robot alone -- small talk, a question, a proposal awaiting a yes    program: []
    cancel   stop now and drop whatever was queued                                          program: []
    replace  stop what is running and run this instead (also: a new instruction while idle)  non-empty
    insert   do this now, then go back to what was interrupted                               non-empty
    append   let the current program finish, then run this                                   non-empty

``insert`` is what a bare skill asked for mid-run means: 「ハンドスプリングして！」 while running
forward flips out of the run and the run continues; nobody said stop, so nothing stops. The
executor compiles the inserted program against the step under way (``compile_program(...,
context=, resume=True)``), which is also how a running flip's kind gets checked against the
heading the robot actually has.

The pairing of action and program is part of the format: :func:`render_output` refuses the other
combinations and the GBNF makes them unwritable.

The model also has to know what the robot is doing. That comes in as a one-line *state block* at
the head of every user turn (:func:`render_state`), describing the queue at the moment the person
spoke. Putting it on the user turn rather than in the system prompt keeps the history append-only
-- nothing already in the KV cache changes -- and keeps each instruction next to the state it was
given in.

Five things have to agree on all of this -- the dataset builder that writes the state blocks and
targets, the training script, the evaluation script, the robot client, and the GBNF grammar
llama.cpp constrains sampling with. They agree because they all call this module. A format
mismatch between any two of them would show up as "the parse occasionally fails", which is the
kind of bug that is noticed late and debugged slowly.

The step vocabulary itself is not restated here: ``dir``/``speed``/``kind`` and the field order come
from ``unitree_rl_lab.program.grammar``, so adding a skill updates the prompt, the grammar and the
parser together.

    python scripts/llm/chat_format.py                       # show rendered examples and the GBNF
    python scripts/llm/chat_format.py --check data/llm/dataset.jsonl   # render/parse every row
    python scripts/llm/chat_format.py --gbnf data/llm/output.gbnf      # write the grammar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

try:
    from unitree_rl_lab.program.grammar import (
        DIRECTIONS, FLIP_KINDS, JA, RUNNING_FLIP_FOR, RUNNING_FLIP_SPEEDS, SPEEDS, STANCE_KINDS, TURN_DIRECTIONS,
        program_from_json, program_to_json,
    )
except ModuleNotFoundError:  # the fine-tuning venv has no Isaac Lab install; the package is pure Python
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))
    from unitree_rl_lab.program.grammar import (
        DIRECTIONS, FLIP_KINDS, JA, RUNNING_FLIP_FOR, RUNNING_FLIP_SPEEDS, SPEEDS, STANCE_KINDS, TURN_DIRECTIONS,
        program_from_json, program_to_json,
    )

ACTIONS: tuple[str, ...] = ("none", "cancel", "replace", "insert", "append")
IDLE_ACTIONS: tuple[str, ...] = ("none", "cancel")
"""Actions that carry no program."""
QUEUE_ACTIONS: tuple[str, ...] = ("replace", "insert", "append")
"""Actions that must carry one."""

# What the model is trained to write. The reply is one line, so the separator is unambiguous.
SEPARATOR = "\n\naction: "
PROGRAM_LINE = "\nprogram: "

# What the model is allowed to have written. Slightly looser than the training text -- a generation
# that lands on "action:" without the space, or with a stray blank line, is a formatting slip and
# not a failure to understand the instruction, so it is read rather than thrown away.
_TAIL_RE = re.compile(r"\n\s*action\s*:\s*(\w+)\s*\n\s*program\s*:\s*", re.DOTALL)


class FormatError(ValueError):
    """A generation that does not carry a readable reply, action and program."""


class Output(NamedTuple):
    reply: str
    action: str
    program: list[dict]


def check_pairing(action: str, program: list) -> None:
    """The action/program rule, in one place. Raises :class:`FormatError`."""
    if action not in ACTIONS:
        raise FormatError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    if action in IDLE_ACTIONS and program:
        raise FormatError(f"action {action!r} takes an empty program, got {len(program)} steps")
    if action in QUEUE_ACTIONS and not program:
        raise FormatError(f"action {action!r} needs a program, got []")


def render_output(reply: str, action: str, program: list[dict]) -> str:
    """The assistant text for one turn: the training target, and what the robot will read.

    The program is passed through the grammar so the text is exactly what ``step_to_dict`` writes
    (field order, no ``"running": false``) and so an invalid program can never become a target.
    """
    if "\n" in reply:
        raise FormatError(f"a reply must be one line, got {reply!r}")
    check_pairing(action, program)
    return reply + SEPARATOR + action + PROGRAM_LINE + program_to_json(program_from_json(program))


def parse_output(text: str) -> Output:
    """Read a generation back. Raises :class:`FormatError` with the reason, for the eval to count."""
    matches = list(_TAIL_RE.finditer(text))
    if not matches:
        raise FormatError("no 'action:' / 'program:' lines")
    last = matches[-1]  # the reply is free text; only the final action line can be the real one
    reply = text[: last.start()].strip()
    action = last.group(1)
    body = text[last.end() :].strip()
    if not body.startswith("["):
        raise FormatError(f"program is not a list: {body[:40]!r}")
    try:
        program = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FormatError(f"program is not valid JSON: {exc}") from exc
    if not isinstance(program, list) or not all(isinstance(step, dict) for step in program):
        raise FormatError("program must be a list of objects")
    check_pairing(action, program)
    return Output(reply, action, program)


# --- The robot's state, as the model sees it -------------------------------------------------------


@dataclass
class RobotState:
    """What the executor knows at the moment the person speaks.

    ``program`` is the whole queue -- steps already done, the one under way, the ones still to
    come (an ``append`` extends it). ``last`` is the previous queue, kept so 「もう一回」 has a
    referent even after a cancel changed what actually ran.
    """

    program: list[dict] = field(default_factory=list)
    step_index: int = 0
    """Index of the step under way while running."""
    elapsed_s: float = 0.0
    duration_s: float = 0.0
    step_remaining_s: float = 0.0
    running: bool = False
    interruptible: bool = True
    """False during a flip or a stance: a cancel then takes effect when the move is over."""
    last: list[dict] = field(default_factory=list)
    last_outcome: str | None = None
    """``"completed"`` or ``"cancelled"``; with ``cancelled``, ``last_step_index`` says where."""
    last_step_index: int = 0

    @classmethod
    def idle(cls, last: list[dict] | None = None, outcome: str | None = None, at_step: int = 0) -> RobotState:
        return cls(last=list(last or []), last_outcome=outcome if last else None, last_step_index=at_step)


def state_from_timeline(program: list[dict], timeline, elapsed_s: float, last: list[dict] | None = None,
                        last_outcome: str | None = None, last_step_index: int = 0) -> RobotState:
    """The state ``elapsed_s`` seconds into a compiled program -- what the executor would report.

    The step under way is the one whose segments contain ``elapsed_s``; inserted settles and the
    final stop belong to the step before them. A flip or a stance cannot be interrupted, and the
    settle that follows one is counted as still part of it. Used by the dataset builder to write
    truthful state blocks and by the robot client for the real ones.
    """
    if elapsed_s >= timeline.duration or not timeline.segments:
        return RobotState.idle(program, "completed")
    current = None
    for seg in timeline.segments:
        if seg.step_index is not None:
            current = seg.step_index
        if seg.t0 <= elapsed_s < seg.t1:
            break
    step_index = current if current is not None else 0
    step_end = max(s.t1 for s in timeline.segments if s.step_index == step_index)
    if elapsed_s >= step_end:  # in the compiler's own final stop: every step is done, nothing is "now"
        step_index = len(program)
    interruptible = seg.kind not in ("flip", "stance", "gap") and not (
        seg.kind == "settle" and step_index < len(program) and program[step_index]["skill"] in ("flip", "stance"))
    return RobotState(program=list(program), step_index=step_index, elapsed_s=elapsed_s, duration_s=timeline.duration,
                      step_remaining_s=max(step_end - elapsed_s, 0.0), running=True, interruptible=interruptible,
                      last=list(last or []), last_outcome=last_outcome, last_step_index=last_step_index)


STATE_TAG = "[状態]"
DONE, NOW, PARTIAL = "(済)", "(いま", "(途中)"


def _amount(value: float, unit: str) -> str:
    return f"{value:g}{unit}"


def describe_step(step: dict) -> str:
    """A step in a few Japanese characters, for the state block. Not the reply's wording."""
    skill = step["skill"]
    if skill == "move":
        speed = "" if step.get("speed", "normal") == "normal" else JA[step["speed"]]
        amount = _amount(step["distance_m"], "m") if "distance_m" in step else _amount(step["duration_s"], "秒")
        return f"{speed}{JA[step['dir']]}へ{amount}"
    if skill == "turn":
        speed = "" if step.get("speed", "normal") == "normal" else JA[step["speed"]]
        amount = _amount(step["angle_deg"], "度") if "angle_deg" in step else _amount(step["duration_s"], "秒")
        return f"{speed}{JA[step['dir']]}回り{amount}"
    if skill == "stop":
        return f"停止{_amount(step.get('duration_s', 1.0), '秒')}"
    if skill == "flip":
        running = "(走りながら)" if step.get("running") else ""
        return f"{JA[step['kind']]}×{step.get('count', 1)}{running}"
    if skill == "stance":
        walking = f"({JA[step['dir']]}へ歩きながら)" if step.get("dir") else ""
        return f"{JA[step['kind']]}{_amount(step.get('duration_s', 5.0), '秒')}{walking}"
    raise ValueError(f"unknown skill {skill!r}")


def describe_program(program: list[dict], done_before: int | None = None, now: int | None = None,
                     now_remaining_s: float | None = None) -> str:
    """Steps joined with arrows, with progress marks when given."""
    parts = []
    for index, step in enumerate(program):
        text = describe_step(step)
        if done_before is not None and index < done_before:
            text += DONE
        elif now is not None and index == now:
            text += f"{NOW} 残り{now_remaining_s:.1f}s)" if now_remaining_s is not None else PARTIAL
        parts.append(text)
    return " → ".join(parts)


def running_flip_now(state: RobotState) -> str | None:
    """While a move is under way: the one flip it can be done out of, or なし when it is too fast.

    Written into the state block so the model compares names instead of looking a heading up in
    a table -- the first fine-tune got that lookup wrong about one time in eight.
    """
    if not state.running or state.step_index >= len(state.program):
        return None
    step = state.program[state.step_index]
    if step["skill"] != "move":
        return None
    if step.get("speed", "normal") not in RUNNING_FLIP_SPEEDS:
        return "なし(速すぎる)"
    return JA[RUNNING_FLIP_FOR[step["dir"]]]


def render_state(state: RobotState) -> str:
    """The one-line state block that heads a user turn."""
    if state.running:
        steps = describe_program(state.program, done_before=state.step_index, now=state.step_index,
                                 now_remaining_s=state.step_remaining_s)
        interrupt = "中断可" if state.interruptible else "中断不可(技の途中)"
        flip = running_flip_now(state)
        hint = f" / 走りながらの技: {flip}" if flip else ""
        return f"{STATE_TAG} 実行中 {state.elapsed_s:.1f}s/{state.duration_s:.1f}s: {steps} / {interrupt}{hint}"
    if not state.last:
        return f"{STATE_TAG} 待機中"
    if state.last_outcome == "cancelled":
        steps = describe_program(state.last, done_before=state.last_step_index, now=state.last_step_index)
        return f"{STATE_TAG} 待機中 / 直前(中断): {steps}"
    return f"{STATE_TAG} 待機中 / 直前(完了): {describe_program(state.last)}"


def render_user_turn(state: RobotState, text: str) -> str:
    """What goes into the ``user`` message: the state block, then what the person said."""
    if "\n" in text:
        raise FormatError(f"an instruction must be one line, got {text!r}")
    return render_state(state) + "\n" + text


_STATE_RE = re.compile(r"^\[状態\][^\n]*\n")


def strip_state(user_text: str) -> str:
    """The person's words without the state block, for checks and printing."""
    return _STATE_RE.sub("", user_text, count=1)


# --- GBNF -------------------------------------------------------------------------------------
#
# llama.cpp rejects any token that cannot continue a string this grammar accepts, so a constrained
# generation cannot produce an unknown direction, a missing field, a truncated list, or a program
# behind `none`/`cancel` -- whatever the model's own probabilities say. The field order below is
# the order `step_to_dict` writes, and `--check` at the bottom of this file is what keeps the two
# in step.


def _alt(values) -> str:
    return " | ".join(f'"{value}"' for value in values)


def gbnf_grammar() -> str:
    """A GBNF grammar for the whole output, built from the same vocabulary the prompt describes."""
    return f'''# Generated by scripts/llm/chat_format.py -- do not edit by hand.
root       ::= reply "\\n\\naction: " tail
reply      ::= [^\\n]+
tail       ::= idle | queue
idle       ::= ({_alt(IDLE_ACTIONS)}) "\\nprogram: []"
queue      ::= ({_alt(QUEUE_ACTIONS)}) "\\nprogram: [" step (", " step)* "]"
step       ::= move | turn | stop | flip | stance

move       ::= "{{\\"skill\\": \\"move\\", \\"dir\\": \\"" dir "\\", \\"speed\\": \\"" speed "\\", " move-len "}}"
move-len   ::= "\\"duration_s\\": " num | "\\"distance_m\\": " num
turn       ::= "{{\\"skill\\": \\"turn\\", \\"dir\\": \\"" turn-dir "\\", \\"speed\\": \\"" speed "\\", " turn-len "}}"
turn-len   ::= "\\"duration_s\\": " num | "\\"angle_deg\\": " num
stop       ::= "{{\\"skill\\": \\"stop\\", \\"duration_s\\": " num "}}"
flip       ::= "{{\\"skill\\": \\"flip\\", \\"kind\\": \\"" flip-kind "\\", \\"count\\": " int flip-run "}}"
flip-run   ::= (", \\"running\\": true")?
stance     ::= "{{\\"skill\\": \\"stance\\", \\"kind\\": \\"" stance-kind "\\", \\"duration_s\\": " num stance-dir ", \\"speed\\": \\"" speed "\\"}}"
stance-dir ::= (", \\"dir\\": \\"" dir "\\"")?

dir        ::= {_alt(DIRECTIONS)}
turn-dir   ::= {_alt(TURN_DIRECTIONS)}
speed      ::= {_alt(SPEEDS)}
flip-kind  ::= {_alt(FLIP_KINDS)}
stance-kind ::= {_alt(STANCE_KINDS)}
num        ::= [0-9]+ ("." [0-9]+)?
int        ::= [0-9]+
'''


# --- The token stream ---------------------------------------------------------------------------
#
# Qwen3's chat template renders a *past* assistant turn without the empty <think></think> block it
# put in front of the generation prompt, so re-rendering the history every turn would change tokens
# already in the KV cache. Instead the conversation is kept as the literal stream: the first turn
# from the template, then each generated answer as it was written, closed with <|im_end|>, then
# the next user turn and the same assistant opening. Trainer, eval and robot client all build the
# prompt with `conversation_text`, so the boundary the loss was taken on is the one inference has.

USER_WRAP = "<|im_start|>user\n{}<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TURN_END = "<|im_end|>\n"


def first_prompt(system_prompt: str, user_text: str) -> str:
    """Turn 1 as the Qwen3 template renders it with thinking off; train_sft asserts this."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n" + USER_WRAP.format(user_text) + ASSISTANT_OPEN


def conversation_text(system_prompt: str, user_texts: list[str], answers: list[str]) -> str:
    """The prompt for the assistant turn after ``user_texts[-1]``, given the earlier answers.

    ``answers`` are the assistant texts already spoken (``len(user_texts) - 1`` of them), exactly as
    generated -- the reply, the action line and the program line -- without the closing token.
    """
    if len(answers) != len(user_texts) - 1:
        raise ValueError(f"{len(user_texts)} user turns need {len(user_texts) - 1} earlier answers, got {len(answers)}")
    text = first_prompt(system_prompt, user_texts[0])
    for answer, user_text in zip(answers, user_texts[1:]):
        text += answer + TURN_END + USER_WRAP.format(user_text) + ASSISTANT_OPEN
    return text


# --- Dataset rows ------------------------------------------------------------------------------


def row_turns(row: dict) -> list[tuple[str, str, str, list[dict]]]:
    """``(user_text, reply, action, program)`` per exchange of a dataset row.

    Accepts the multi-turn shape (``turns``: alternating user/assistant messages, the user text
    already carrying its state block) and the earlier single-turn shape (``input`` / ``output``),
    where a missing ``action`` is read as ``replace`` for a program and ``none`` without one --
    the only meanings an empty or non-empty program could have had when the robot was always idle.
    """
    if "turns" in row:
        turns = row["turns"]
        out = []
        for user, assistant in zip(turns[0::2], turns[1::2]):
            out.append((user["content"], assistant["reply"], assistant["action"], assistant["program"]))
        return out
    output = row["output"]
    action = output.get("action", "replace" if output["program"] else "none")
    return [(row["input"], output["reply"], action, output["program"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="DATASET", help="render and re-read every turn of a dataset")
    parser.add_argument("--gbnf", metavar="PATH", help="write the grammar to a file")
    args = parser.parse_args()

    if args.gbnf:
        Path(args.gbnf).write_text(gbnf_grammar())
        print(f"wrote {args.gbnf}")
        return

    if not args.check:
        program = [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 5.0},
                   {"skill": "flip", "kind": "frontflip", "count": 1, "running": True}]
        print(render_user_turn(RobotState.idle(), "5mくらい走ってから、そのまま前転して"))
        print("---")
        print(render_output("了解、5m走ってそのまま前転するね。", "replace", program))
        print("---")
        running = RobotState(program=program, step_index=0, elapsed_s=2.1, duration_s=7.4, step_remaining_s=3.1,
                             running=True)
        print(render_user_turn(running, "ええ天気やなあ"))
        print("---")
        print(render_output("ほんまやな、走ってて気持ちいいわ。", "none", []))
        print("---")
        print(render_user_turn(RobotState.idle(program, "cancelled", at_step=1), "もう一回やって"))
        print("---")
        long_run = [{"skill": "move", "dir": "forward", "speed": "normal", "duration_s": 10.0}]
        print(render_user_turn(RobotState(program=long_run, elapsed_s=3.0, duration_s=11.0, step_remaining_s=7.0,
                                          running=True), "ハンドスプリングして！"))
        print("---")
        print(render_output("走りながらいくで！", "insert", [{"skill": "flip", "kind": "frontflip", "count": 1, "running": True}]))
        print("\n--- GBNF ---")
        print(gbnf_grammar())
        return

    total = failed = 0
    for line in open(args.check):
        row = json.loads(line)
        for user_text, reply, action, program in row_turns(row):
            total += 1
            try:
                text = render_output(reply, action, program)
                back = parse_output(text)
            except (FormatError, ValueError) as exc:
                print(f"  FAIL {row['id']}: {exc}")
                failed += 1
                continue
            if back.reply != reply or back.action != action or back.program != program:
                print(f"  FAIL {row['id']}: round-trip changed the content")
                failed += 1
    print(f"{total - failed}/{total} turns render and read back unchanged")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
