"""Quality gate for the generated dataset -- everything that can be checked without a human.

    python scripts/llm/check_dataset.py data/llm/dataset.jsonl

Seven checks, each of which has caught a real bug at some point:

1. grammar      every output program parses under ``unitree_rl_lab.program`` and validates
2. declined     no output program contains a skill whose population success rate is below 10%
3. cautions     a program containing a below-80% skill carries a warning sentence, and one that
                does not, does not (an unearned warning is as wrong as a missing one)
4. numerals     every number in the reply is one the program, the timeline or a compiler limit
                justifies -- the model must not invent "5m" out of a 3m step
5. forbidden    no promise the measurements do not support (必ず / ぴったり / 絶対 / 100%)
6. duplicates   no identical (input, reply, program) row
7. balance      category / style / split distribution, printed for the eye

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
from unitree_rl_lab.program import ProgramError, program_from_json

import negatives as ng

FORBIDDEN = ["必ず", "ぴったり", "絶対", "100%", "確実に"]
DECLINE_BELOW = 0.10
WARN_BELOW = 0.80

CAUTION_SENTENCES = {label: set(sum(by_style.values(), [])) for label, by_style in pb.CAUTIONS.items()}
CAUTION_FOR_KIND = {"frontflip": "frontflip_landing", "handstand": "handstand_descent", "hindstand": "hindstand_descent"}

NUMBER = re.compile(r"\d+(?:\.\d+)?")


NO_PROGRAM = {"chitchat", "impossible", "ambiguous"}


def load_rates(path: str) -> dict[str, float]:
    """Same rates ``build_dataset.py`` used, read back from the dataset's own provenance file."""
    return json.load(open(path))


def load_durations(path: str) -> dict[str, float]:
    """Program id -> the timeline length the sim actually ran, for checking a stated total."""
    return {json.loads(line)["id"]: json.loads(line)["timeline"]["duration_s"] for line in open(path)}


def allowed_numbers(row: dict) -> set[float]:
    """Numbers a reply is allowed to contain."""
    allowed: set[float] = {float(ng.FLIP_LIMIT), ng.STANCE_MAX_S, ng.PROGRAM_MAX_S}
    for step in row["output"]["program"]:
        for field in ("duration_s", "distance_m", "angle_deg", "count"):
            value = step.get(field)
            if value is None:
                continue
            allowed |= {float(value), round(float(value)), float(value) * 100}  # 0.5 m may be said as 50cm
            if field == "angle_deg" and float(value) % 360 == 0:
                allowed.add(float(value) / 360)  # "1回転"
            if field == "angle_deg" and float(value) == 180:
                allowed.add(2)  # "半回転" never prints a number, but "2分の1" would
    # a clamp reply repeats what was asked before saying what it will cut to
    for number in NUMBER.findall(row["input"]):
        allowed.add(float(number))
    return allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--rates", default="data/llm/rates.json", help="written by build_dataset.py")
    parser.add_argument("--validated", default="data/llm/validated.jsonl")
    parser.add_argument("--total-tolerance", type=float, default=2.0, help="seconds a stated total may be off")
    args = parser.parse_args()

    rates = load_rates(args.rates)
    durations = load_durations(args.validated)
    rows = [json.loads(line) for line in open(args.dataset)]
    failures: list[str] = []
    seen: set[tuple] = set()

    for row in rows:
        program = row["output"]["program"]
        reply = row["output"]["reply"]

        try:
            program_from_json(program)
        except ProgramError as exc:
            failures.append(f"[grammar] {row['id']}: {exc}")

        kinds = [s.get("kind") for s in program if s.get("kind")]
        for kind in kinds:
            if rates.get(kind, 1.0) < DECLINE_BELOW:
                failures.append(f"[declined] {row['id']}: emits {kind} (rate {rates[kind]:.1%})")

        wanted = {CAUTION_FOR_KIND[k] for k in kinds if rates.get(k, 1.0) < WARN_BELOW and k in CAUTION_FOR_KIND}
        present = {label for label, sentences in CAUTION_SENTENCES.items() if any(s in reply for s in sentences)}
        if row["category"] != "over_ask":
            for label in wanted - present:
                failures.append(f"[cautions] {row['id']}: no warning for {label}")
            for label in present - wanted:
                failures.append(f"[cautions] {row['id']}: unearned warning {label}")

        for word in FORBIDDEN:
            if word in reply:
                failures.append(f"[forbidden] {row['id']}: reply contains {word!r}")

        if row["category"] not in NO_PROGRAM:
            allowed = allowed_numbers(row)
            stated_total = durations.get(row["source"].get("program_id"))
            for token in NUMBER.findall(reply):
                value = float(token)
                if value in allowed:
                    continue
                if stated_total is not None and abs(value - stated_total) <= args.total_tolerance:
                    continue
                failures.append(f"[numerals] {row['id']}: reply says {token} which nothing justifies -- {reply}")

        key = (row["input"], reply, json.dumps(program, ensure_ascii=False))
        if key in seen:
            failures.append(f"[duplicates] {row['id']}: identical to an earlier row")
        seen.add(key)

    print(f"{len(rows)} rows")
    for field in ("category", "style", "split"):
        counts = collections.Counter(row[field] for row in rows)
        print(f"  {field:<9}", "  ".join(f"{k}={v}" for k, v in counts.most_common()))
    inputs = collections.Counter(row["input"] for row in rows)
    print(f"  unique inputs {len(inputs)} ({len(inputs) / len(rows):.0%});  most repeated: {inputs.most_common(1)}")

    if failures:
        print(f"\n{len(failures)} problems:")
        for line in failures[:40]:
            print("  " + line)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
