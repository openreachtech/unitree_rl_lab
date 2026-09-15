"""Quality gate for the generated dataset -- everything that can be checked without a human.

    python scripts/llm/check_dataset.py data/llm/dataset.jsonl

Every row is a conversation; every assistant turn is checked on its own, against the state block
the person spoke into:

1. format      action/program pairing (none/cancel carry [], the rest do not)
2. grammar     the program parses and validates -- an inserted one against the step under way
3. state       cancel / insert / append only while the robot is running; small talk while running
               leaves it alone (none); an idle 「ストップ」 is none
4. declined    no program contains a skill whose population success rate is below 10%
5. cautions    a program containing a below-80% skill carries a warning sentence, and one that
               does not, does not (an unearned warning is as wrong as a missing one)
6. numerals    every number in a reply is one the program, the state, the timeline or a compiler
               limit justifies -- the model must not invent "5m" out of a 3m step
7. forbidden   no promise the measurements do not support (必ず / ぴったり / 絶対 / 100%)
8. dialogue    a yes after a proposal runs exactly what was proposed; 「もう一回」 re-runs the
               program that ran; 「続けて」 runs a suffix of it
9. duplicates  no identical conversation
10. balance    category / style / split / turns distribution, printed for the eye

Exits non-zero if any check fails. The round-trip (instruction -> program) is a separate script,
``parse_instruction.py`` + ``roundtrip_check.py``, because it needs its own parse of the text.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(Path(__file__).parent))

import phrasebank as pb
from chat_format import FormatError, check_pairing, strip_state
from unitree_rl_lab.program import CapabilityTable, CompilerConfig, ProgramError, compile_program, program_from_json

import negatives as ng

FORBIDDEN = ["必ず", "ぴったり", "絶対", "100%", "確実に"]
DECLINE_BELOW = 0.10
WARN_BELOW = 0.80

CAUTION_SENTENCES = {label: set(sum(by_style.values(), [])) for label, by_style in pb.CAUTIONS.items()}
CAUTION_FOR_KIND = {"frontflip": "frontflip_landing", "handstand": "handstand_descent", "hindstand": "hindstand_descent"}

NUMBER = re.compile(r"\d+(?:\.\d+)?")

NO_PROGRAM = {"chitchat", "impossible", "ambiguous"}
RUNNING_ONLY = {"cancel", "insert", "append"}


def load_rates(path: str) -> dict[str, float]:
    """Same rates ``build_dataset.py`` used, read back from the dataset's own provenance file."""
    return json.load(open(path))


def load_durations(path: str, capability: str) -> dict[str, float]:
    """Program id -> the timeline length the compiler produces today, for checking a stated total."""
    compiler = CompilerConfig(calibration=CapabilityTable.load(capability).calibration())
    out = {}
    for line in open(path):
        record = json.loads(line)
        out[record["id"]] = compile_program(program_from_json(record["program"]), compiler).duration
    return out


def rate_key(step: dict) -> str | None:
    if step["skill"] not in ("flip", "stance"):
        return None
    return f"{step['kind']}:running" if step.get("running") else step["kind"]


def allowed_numbers(program: list[dict], user_text: str, state: dict) -> set[float]:
    """Numbers a reply is allowed to contain."""
    allowed: set[float] = {float(ng.FLIP_LIMIT), ng.STANCE_MAX_S, ng.PROGRAM_MAX_S, 1.0}  # "1回だけ"
    for step in program:
        for field in ("duration_s", "distance_m", "angle_deg", "count"):
            value = step.get(field)
            if value is None:
                continue
            allowed |= {float(value), round(float(value)), float(value) * 100}  # 0.5 m may be said as 50cm
            if field == "angle_deg" and float(value) % 360 == 0:
                allowed.add(float(value) / 360)  # "1回転"
            if field == "angle_deg" and float(value) == 180:
                allowed.add(2)
    # a clamp reply repeats what was asked; a status reply reads the state block
    for number in NUMBER.findall(user_text):
        allowed.add(float(number))
    if state.get("running"):
        allowed |= {round(state["step_remaining_s"]), round(state["duration_s"] - state["elapsed_s"]),
                    float(len(state["program"]) - state["step_index"] - 1)}
    return allowed


def is_suffix(shorter: list[dict], longer: list[dict]) -> bool:
    return len(shorter) <= len(longer) and longer[len(longer) - len(shorter):] == shorter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--rates", default="data/llm/rates.json", help="written by build_dataset.py")
    parser.add_argument("--validated", default="data/llm/validated.jsonl")
    parser.add_argument("--capability", default="data/llm/capability.json")
    parser.add_argument("--total-tolerance", type=float, default=2.0, help="seconds a stated total may be off")
    args = parser.parse_args()

    rates = load_rates(args.rates)
    durations = load_durations(args.validated, args.capability)
    rows = [json.loads(line) for line in open(args.dataset)]
    failures: list[str] = []
    seen: set[tuple] = set()

    for row in rows:
        turns = row["turns"]
        pairs = list(zip(turns[0::2], turns[1::2]))
        first_program = pairs[0][1]["program"]
        proposed = None
        for k, (user, assistant) in enumerate(pairs):
            tag = f"{row['id']} t{k}"
            program, reply, action = assistant["program"], assistant["reply"], assistant["action"]
            state = user.get("state", {})
            running = bool(state.get("running"))
            words = strip_state(user["content"])

            # 1. format
            try:
                check_pairing(action, program)
            except FormatError as exc:
                failures.append(f"[format] {tag}: {exc}")

            # 2. grammar, against the step under way for an insert
            try:
                context = None
                if action == "insert" and running:
                    context = program_from_json([state["program"][state["step_index"]]])[0]
                compile_program(program_from_json(program), context=context, resume=context is not None)
            except (ProgramError, IndexError) as exc:
                failures.append(f"[grammar] {tag}: {exc}")

            # 3. state
            if action in RUNNING_ONLY and not running:
                failures.append(f"[state] {tag}: {action} while idle")
            if row["category"] == "dlg_chitchat" and k > 0 and action != "none":
                failures.append(f"[state] {tag}: small talk while running must be none, got {action}")
            if row["category"] == "dlg_stopped" and k > 0 and action != "none":
                failures.append(f"[state] {tag}: idle stop must be none, got {action}")

            # 4. declined
            keys = [rate_key(s) for s in program if rate_key(s)]
            for key in keys:
                if rates.get(key, 1.0) < DECLINE_BELOW:
                    failures.append(f"[declined] {tag}: emits {key} (rate {rates[key]:.1%})")

            # 5. cautions
            wanted = {CAUTION_FOR_KIND[key.split(":")[0]] for key in keys
                      if rates.get(key, 1.0) < WARN_BELOW and key.split(":")[0] in CAUTION_FOR_KIND}
            if any(s["skill"] == "flip" and s.get("running") and s["kind"] == "frontflip" and s.get("count", 1) > 1 for s in program):
                wanted.add("running_frontflip_repeat")
            present = {label for label, sentences in CAUTION_SENTENCES.items() if any(s in reply for s in sentences)}
            if row["category"] != "over_ask":
                for label in wanted - present:
                    failures.append(f"[cautions] {tag}: no warning for {label}")
                for label in present - wanted:
                    failures.append(f"[cautions] {tag}: unearned warning {label}")

            # 6. numerals
            if row["category"] not in NO_PROGRAM and not (row["category"] == "dlg_chitchat" and k > 0):
                allowed = allowed_numbers(program, user["content"], state)
                stated_total = durations.get(row["source"].get("program_id")) if k == 0 else None
                for token in NUMBER.findall(reply):
                    value = float(token)
                    if value in allowed:
                        continue
                    if stated_total is not None and abs(value - stated_total) <= args.total_tolerance:
                        continue
                    failures.append(f"[numerals] {tag}: reply says {token} which nothing justifies -- {reply}")

            # 7. forbidden
            for word in FORBIDDEN:
                if word in reply:
                    failures.append(f"[forbidden] {tag}: reply contains {word!r}")

            # 8. dialogue logic
            if assistant.get("confirms"):
                if proposed is None or program != proposed:
                    failures.append(f"[dialogue] {tag}: confirmation runs something other than what was proposed")
            if "proposed" in assistant:
                proposed = assistant["proposed"]
                if action != "none" or program:
                    failures.append(f"[dialogue] {tag}: a proposal must be none / []")
            if row["category"] == "dlg_repeat" and k == 1:
                same_shape = len(program) == len(first_program) and all(
                    a["skill"] == b["skill"] and a.get("kind") == b.get("kind") and a.get("dir") == b.get("dir")
                    for a, b in zip(program, first_program))
                if not same_shape:
                    failures.append(f"[dialogue] {tag}: repeat does not re-run the first program")
            if row["category"] == "dlg_resume" and k == 2 and not is_suffix(program, first_program):
                failures.append(f"[dialogue] {tag}: resume is not a suffix of the first program")
            if assistant.get("status") and not running:
                failures.append(f"[dialogue] {tag}: status answer about an idle robot")

        key = tuple((t.get("content"), t.get("reply"), json.dumps(t.get("program"), ensure_ascii=False)) for t in turns)
        if key in seen:
            failures.append(f"[duplicates] {row['id']}: identical to an earlier row")
        seen.add(key)

    print(f"{len(rows)} rows, {sum(len(r['turns']) // 2 for r in rows)} assistant turns")
    for field in ("category", "style", "split"):
        counts = collections.Counter(row[field] for row in rows)
        print(f"  {field:<9}", "  ".join(f"{k}={v}" for k, v in counts.most_common()))
    print("  turns    ", "  ".join(f"{k}={v}" for k, v in sorted(collections.Counter(len(r["turns"]) // 2 for r in rows).items())))
    actions = collections.Counter(a["action"] for r in rows for a in r["turns"][1::2])
    print("  actions  ", "  ".join(f"{k}={v}" for k, v in actions.most_common()))
    firsts = collections.Counter(strip_state(row["turns"][0]["content"]) for row in rows)
    print(f"  unique first instructions {len(firsts)} ({len(firsts) / len(rows):.0%});  most repeated: {firsts.most_common(1)}")

    if failures:
        print(f"\n{len(failures)} problems:")
        for line in failures[:400]:
            print("  " + line)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
