"""Structural check for the hand-written conversations in ``data/llm/handwritten.jsonl``.

The hand-written half is written as whole conversations in the dataset's own row shape (see
HANDWRITTEN.md), so what has to be checked is *structure*, not wording:

    the row shape        keys, alternating roles, at least two turns -- no single-turn rows
    the programs         every program and every queue is in the grammar and compiles
    the queue            turn 1 starts from a standstill, and each later queue is what was left of
                         the program the robot was last given: a tail of it, with at most the step
                         it is in the middle of reduced

Nothing here reads Japanese. An earlier version of this file ran the round trip's lexicon over
hand-written text and was used as a gate, which taught the writers to phrase everything the way
the phrase bank does -- 19% of the lines ended in 「〜ください」 and half were the same
「〜て、〜て」 chain. Wording is judged by reading it, not by a regex.

    python scripts/llm/check_handwritten.py data/llm/handwritten.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

import phrasebank as pb  # noqa: E402
from unitree_rl_lab.program import (  # noqa: E402
    CapabilityTable, CompilerConfig, ProgramError, compile_program, program_from_json,
)

AMOUNTS = ("duration_s", "distance_m", "angle_deg")


def steps_equal(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def is_remainder(queue: list[dict], program: list[dict]) -> bool:
    """Whether ``queue`` is what is left of ``program`` at some moment after it was given.

    Steps run in order, so the remainder is a tail of the program; the step the robot is in the
    middle of keeps its skill and its direction but a smaller amount. An empty queue means it all
    finished (or was stopped), which is always allowed.
    """
    if not queue:
        return True
    for start in range(len(program)):
        tail = program[start:]
        if len(tail) != len(queue):
            continue
        if not all(steps_equal(t, q) for t, q in zip(tail[1:], queue[1:])):
            continue
        head, first = tail[0], queue[0]
        if steps_equal(head, first):
            return True
        if head.get("skill") != first.get("skill") or head.get("dir") != first.get("dir"):
            continue
        if head.get("kind") != first.get("kind") or head.get("speed") != first.get("speed"):
            continue
        shrunk = [f for f in AMOUNTS
                  if head.get(f) is not None and first.get(f) is not None and first[f] <= head[f]]
        unchanged = [f for f in AMOUNTS if head.get(f) is None and first.get(f) is None]
        if len(shrunk) + len(unchanged) == len(AMOUNTS) and shrunk:
            return True
    return False


def check_row(row: dict, compiler: CompilerConfig) -> list[str]:
    bad = []
    for key in ("id", "category", "style", "turns"):
        if key not in row:
            bad.append(f"missing key {key!r}")
    if bad:
        return bad
    if row["style"] not in pb.STYLES:
        bad.append(f"style {row['style']!r} is not one of {pb.STYLES}")

    turns = row["turns"]
    if len(turns) % 2 or len(turns) < 4:
        bad.append(f"{len(turns)} turn entries -- a conversation is an even number, at least 4 "
                   f"(hand-written rows are never single-turn)")
        return bad

    pairs = list(zip(turns[0::2], turns[1::2]))
    previous: list[dict] | None = None
    for index, (user, assistant) in enumerate(pairs):
        where = f"turn {index + 1}"
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            bad.append(f"{where}: roles are not user then assistant")
            continue
        if not isinstance(user.get("content"), str) or not user["content"].strip():
            bad.append(f"{where}: empty user text")
        reply = assistant.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            bad.append(f"{where}: empty reply")
        elif "\n" in reply or "program" in reply or "{" in reply:
            bad.append(f"{where}: the reply is one line and never shows the program -- {reply!r}")

        queue, program = user.get("queue"), assistant.get("program")
        for name, steps in (("queue", queue), ("program", program)):
            if not isinstance(steps, list):
                bad.append(f"{where}: {name} is not a list")
                continue
            try:
                compile_program(program_from_json(steps), compiler)
            except (ProgramError, KeyError, TypeError, ValueError) as error:
                bad.append(f"{where}: {name} is not a program the compiler takes -- {error}")

        if isinstance(queue, list):
            if index == 0 and queue:
                bad.append(f"{where}: a conversation opens with the robot standing still, queue=[]")
            elif previous is not None and not is_remainder(queue, previous):
                bad.append(f"{where}: queue is not what is left of the program of turn {index} "
                           f"({json.dumps(queue, ensure_ascii=False)} after "
                           f"{json.dumps(previous, ensure_ascii=False)})")
        previous = program if isinstance(program, list) else previous
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", help="jsonl of dataset rows")
    parser.add_argument("--capability", default="data/llm/capability.json")
    parser.add_argument("-q", "--quiet", action="store_true", help="only the summary")
    args = parser.parse_args()

    compiler = CompilerConfig(calibration=CapabilityTable.load(args.capability).calibration())
    total = flagged = 0
    for number, line in enumerate(open(args.rows), 1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            flagged += 1
            print(f"line {number}: not JSON -- {error}")
            continue
        problems = check_row(row, compiler)
        if problems:
            flagged += 1
            if not args.quiet:
                print(f"{row.get('id', f'line {number}')}:")
                for problem in problems:
                    print(f"    {problem}")

    print(f"\n{total - flagged}/{total} rows structurally sound, {flagged} flagged")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
