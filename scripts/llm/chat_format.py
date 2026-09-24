"""The wire format between the model and the robot -- written, read and constrained in one place.

The model is shown the robot's queue and hands back the queue it wants:

    user       やっぱり後ろに2m下がって

               queued programs: [{"skill": "move", "dir": "forward", ..., "distance_m": 1.8}]

    assistant  了解、後ろに2m下がるで。

               program: [{"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 2.0}]

**What comes back replaces what was queued.** There is no action to choose -- the five the old
format carried all fall out of the list itself:

    何も変えない   渡された列をそのまま書き写す
    止まる        []
    走りながら技   [{flip running}, {渡された move}]        -- 技を先に置けば「いま」
    後ろに足す     [{渡された列}, {追加}]
    差し替え      まったく違う列

Two properties make that work. The queue is rendered as **what is left** of each step at the moment
the reply will land, not what was originally asked for -- so copying 「あと1.8m」 back is the same
as not being interrupted, and the model never does arithmetic. And the model reads and writes the
same type: a program in, a program out, in the same JSON. The old format asked it to read a prose
state block and write an enum plus a fragment, and the enum was where most of the errors were.

The queue sits *after* the person's words, at the very end of the prompt. Both halves of that
matter: it changes every turn, so anything before it would be re-read by the server on every turn
(measured: 63 tokens re-read per turn with it last, growing to 110 with it after the system
prompt), and the text the model copies from is then adjacent to where it writes.

History is conversation only. Past turns keep the words and drop the queue and the program: a queue
from two turns ago describes a robot that has moved on, and nothing in the text says so. There is
no ``<think></think>`` block anywhere either -- Qwen3's template puts one in front of the
generation prompt, but we fine-tune, so the format is ours to define and an empty block costs four
tokens a turn. The grammar forbids a reply starting with ``<`` so the base model's habit of opening
one cannot leak through.

Five things have to agree on all of this -- the dataset builder, the training script, the
evaluation script, the robot client, and the GBNF grammar llama.cpp constrains sampling with. They
agree because they all call this module.

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
from pathlib import Path
from typing import NamedTuple

try:
    from unitree_rl_lab.program.grammar import (
        DIRECTIONS, FLIP_KINDS, SPEEDS, STANCE_KINDS, TURN_DIRECTIONS,
        program_from_json, program_to_json,
    )
except ModuleNotFoundError:  # the fine-tuning venv has no Isaac Lab install; the package is pure Python
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))
    from unitree_rl_lab.program.grammar import (
        DIRECTIONS, FLIP_KINDS, SPEEDS, STANCE_KINDS, TURN_DIRECTIONS,
        program_from_json, program_to_json,
    )

QUEUE_LINE = "\n\nqueued programs: "
"""Label for the queue shown to the model, at the end of the user turn."""

PROGRAM_LINE = "\n\nprogram: "
"""Label for the queue the model hands back. Deliberately the same shape as the one it read."""

_QUEUE_RE = re.compile(r"\n\s*queued programs\s*:\s*\[.*\]\s*$", re.DOTALL)
_PROGRAM_RE = re.compile(r"\n\s*program\s*:\s*", re.DOTALL)


class FormatError(ValueError):
    """A generation that does not carry a readable reply and program."""


class Output(NamedTuple):
    reply: str
    program: list[dict]


# --- What the model writes ----------------------------------------------------------------------


def render_output(reply: str, program: list[dict]) -> str:
    """The assistant text for one turn: the training target, and what the robot will read.

    The program is passed through the grammar so the text is exactly what ``step_to_dict`` writes
    (field order, no ``"running": false``) and so an invalid program can never become a target.
    """
    if "\n" in reply:
        raise FormatError(f"a reply must be one line, got {reply!r}")
    if not reply.strip():
        raise FormatError("a reply cannot be empty")
    if reply.lstrip().startswith("<"):
        raise FormatError(f"a reply cannot start with '<' -- that is how a stray <think> looks: {reply!r}")
    return reply + PROGRAM_LINE + program_to_json(program_from_json(program))


def parse_output(text: str) -> Output:
    """Read a generation back. Raises :class:`FormatError` with the reason, for the eval to count."""
    matches = list(_PROGRAM_RE.finditer(text))
    if not matches:
        raise FormatError("no 'program:' line")
    last = matches[-1]  # the reply is free text; only the final program line can be the real one
    reply = text[: last.start()].strip()
    if not reply:
        raise FormatError("no reply before the program line")
    body = text[last.end():].strip()
    if not body.startswith("["):
        raise FormatError(f"program is not a list: {body[:40]!r}")
    try:
        program = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FormatError(f"program is not valid JSON: {exc}") from exc
    if not isinstance(program, list) or not all(isinstance(step, dict) for step in program):
        raise FormatError("program must be a list of objects")
    return Output(reply, program)


# --- What the model reads -------------------------------------------------------------------------


def render_user_turn(text: str, queue: list[dict]) -> str:
    """One user turn: what the person said, then the queue as it will stand when the reply lands."""
    return text + QUEUE_LINE + program_to_json(program_from_json(queue))


def strip_queue(user_text: str) -> str:
    """The person's words alone -- how the turn is rendered once it is history."""
    return _QUEUE_RE.sub("", user_text).rstrip()


def _alt(values) -> str:
    return " | ".join(f'"{value}"' for value in values)


def gbnf_grammar() -> str:
    """A GBNF grammar for the whole output, built from the same vocabulary the prompt describes."""
    return f'''# Generated by scripts/llm/chat_format.py -- do not edit by hand.
root       ::= reply "\\n\\nprogram: " program
# A reply cannot open with '<': that is the shape of the <think> block Qwen3 wants to write, and
# without one in the prompt the base model's habit would otherwise come through as the reply.
reply      ::= [^<\\n] [^\\n]*
program    ::= "[]" | "[" step (", " step)* "]"
step       ::= move | turn | flip | stance

move       ::= "{{\\"skill\\": \\"move\\", \\"dir\\": \\"" dir "\\", \\"speed\\": \\"" speed "\\"" move-len "}}"
move-len   ::= (", \\"duration_s\\": " num | ", \\"distance_m\\": " num)?
turn       ::= "{{\\"skill\\": \\"turn\\", \\"dir\\": \\"" turn-dir "\\", \\"angle_deg\\": " num "}}"
flip       ::= "{{\\"skill\\": \\"flip\\", \\"kind\\": \\"" flip-kind "\\"}}"
stance     ::= "{{\\"skill\\": \\"stance\\", \\"kind\\": \\"" stance-kind "\\"" hold "}}"
hold       ::= (", \\"duration_s\\": " num)?

dir        ::= {_alt(DIRECTIONS)}
turn-dir   ::= {_alt(TURN_DIRECTIONS)}
speed      ::= {_alt(SPEEDS)}
flip-kind  ::= {_alt(FLIP_KINDS)}
stance-kind ::= {_alt(STANCE_KINDS)}
num        ::= [0-9]+ ("." [0-9]+)?
'''


# --- The token stream ---------------------------------------------------------------------------
#
# Built by appending, never by re-rendering: everything already written stays byte-identical, so the
# server's KV cache keeps it and each turn costs only the tokens it adds (measured: ~30 against a
# ~1400-token prompt). The one thing that is dropped on the way into history is the queue line,
# which is why `strip_queue` exists -- a queue is only true for the turn it was written on, and the
# stripping is consistent, so history still never changes once written.

USER_WRAP = "<|im_start|>user\n{}<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>\n"


def first_prompt(system_prompt: str, user_text: str) -> str:
    """Turn 1, with no thinking block: see the note above on why we leave Qwen3's template here."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n" + USER_WRAP.format(user_text) + ASSISTANT_OPEN


def conversation_text(system_prompt: str, user_texts: list[str], answers: list[str]) -> str:
    """The prompt for the assistant turn after ``user_texts[-1]``, given the earlier answers.

    ``answers`` are the assistant replies already spoken (``len(user_texts) - 1`` of them), without
    their program line: history is the conversation, and the programs in it were superseded by the
    queue the last turn carries.
    """
    if len(answers) != len(user_texts) - 1:
        raise ValueError(f"{len(user_texts)} user turns need {len(user_texts) - 1} earlier answers, got {len(answers)}")
    past = [strip_queue(t) for t in user_texts[:-1]]
    text = first_prompt(system_prompt, past[0] if past else user_texts[0])
    for answer, user_text in zip(answers, (past + [user_texts[-1]])[1:]):
        text += answer + TURN_END + USER_WRAP.format(user_text) + ASSISTANT_OPEN
    return text


# --- Dataset rows ------------------------------------------------------------------------------


def row_turns(row: dict) -> list[tuple[str, list[dict], str, list[dict]]]:
    """``(user_text, queue, reply, program)`` per exchange of a dataset row."""
    turns = row["turns"]
    return [(user["content"], user.get("queue", []), assistant["reply"], assistant["program"])
            for user, assistant in zip(turns[0::2], turns[1::2])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="DATASET", help="render and re-read every turn of a dataset")
    parser.add_argument("--gbnf", metavar="PATH", help="write the grammar to a file")
    args = parser.parse_args()

    if args.gbnf:
        Path(args.gbnf).write_text(gbnf_grammar())
        print(f"wrote {args.gbnf}")
        return

    if args.check:
        rows = [json.loads(line) for line in open(args.check)]
        ok = bad = 0
        for row in rows:
            for user_text, queue, reply, program in row_turns(row):
                rendered = render_output(reply, program)
                try:
                    back = parse_output(rendered)
                except FormatError as exc:
                    print(f"  {row['id']}: {exc}")
                    bad += 1
                    continue
                turn = render_user_turn(user_text, queue)
                if back.reply == reply and back.program == program and strip_queue(turn) == user_text:
                    ok += 1
                else:
                    print(f"  {row['id']}: round trip changed the turn")
                    bad += 1
        print(f"{ok}/{ok + bad} turns render and read back unchanged")
        sys.exit(1 if bad else 0)

    queue = [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 1.8},
             {"skill": "flip", "kind": "backflip"}]
    program = [{"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 2.0}]
    print("--- user turn ---")
    print(render_user_turn("やっぱり後ろに2m下がって", queue))
    print("\n--- once it is history ---")
    print(strip_queue(render_user_turn("やっぱり後ろに2m下がって", queue)))
    print("\n--- assistant turn ---")
    print(render_output("了解、後ろに2m下がるで。", program))
    print("\n--- grammar ---")
    print(gbnf_grammar())


if __name__ == "__main__":
    main()
