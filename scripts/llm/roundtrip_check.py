"""Round-trip check: does the instruction a paraphrase went through still imply its program?

Compares each dataset entry's ``output.program`` against an independently produced *reparse* of
its ``input`` text (what a reader would construct from the instruction alone, ignoring the
original program). Implements the tolerance rules from ``DATASET.md`` section 4:

- the skill sequence, and each step's ``kind``/``dir``, must match exactly
- ``speed`` must match, except a step with no speed word in the text may default to "normal"
  (a bare "動いて" example) -- there is no automated way to check "no speed word was used" from a
  program alone, so this is left to the reparse to encode via ``speed`` being absent upstream
- ``duration_s`` / ``distance_m`` / ``angle_deg`` / ``count`` match within 20% or round to the
  same number
- a ``stop``'s length is not checked at all

Two categories are compared differently, because their output program is not a paraphrase of the
instruction but a *decision* about it:

- ``partial_decline``: the reparse is the literal (impossible) full request; the output must equal
  the reparse with every declined skill's steps removed, and at least one step must actually have
  been declined.
- ``adjusted`` / ``over_ask``: the model is expected to emit the request as asked and let the
  compiler clamp it downstream, so output and reparse must match exactly like a normal entry.

Categories with an empty program on both sides (chitchat, impossible, ambiguous) trivially pass.

    python scripts/llm/roundtrip_check.py data/llm/pilot_dataset.jsonl data/llm/pilot_reparse.jsonl
"""

import argparse
import json
import sys

TOLERANCE = 0.20

# Categories whose correct output is an empty program whatever the sentence seems to ask for.
NO_PROGRAM = {"chitchat", "impossible", "ambiguous"}


def close(a: float, b: float) -> bool:
    if round(a) == round(b):
        return True
    return abs(a - b) <= TOLERANCE * max(abs(a), abs(b), 1e-9)


def step_matches(want: dict, got: dict) -> str | None:
    """None if ``got`` matches ``want`` under the tolerance rules; otherwise a mismatch reason."""
    if want["skill"] != got["skill"]:
        return f"skill {want['skill']} != {got['skill']}"
    if want.get("kind") != got.get("kind"):
        return f"kind {want.get('kind')} != {got.get('kind')}"
    if want.get("dir") != got.get("dir"):
        return f"dir {want.get('dir')} != {got.get('dir')}"
    if want["skill"] != "stop":
        want_speed = want.get("speed", "normal")
        got_speed = got.get("speed", "normal")
        if want_speed != got_speed:
            return f"speed {want_speed} != {got_speed}"
    for field in ("duration_s", "distance_m", "angle_deg", "count"):
        if want["skill"] == "stop" and field == "duration_s":
            continue  # a stop's length is never checked
        w, g = want.get(field), got.get(field)
        if (w is None) != (g is None):
            return f"{field}: one side has it, the other doesn't ({w} vs {g})"
        if w is not None and not close(w, g):
            return f"{field} {w} vs {g} (>{TOLERANCE:.0%} apart)"
    return None


def programs_match(want: list[dict], got: list[dict]) -> str | None:
    if len(want) != len(got):
        return f"{len(want)} steps vs {len(got)}"
    for i, (w, g) in enumerate(zip(want, got)):
        reason = step_matches(w, g)
        if reason:
            return f"step {i}: {reason}"
    return None


def check_partial_decline(output: list[dict], reparsed: list[dict]) -> str | None:
    """``output`` must equal ``reparsed`` with some steps dropped, and at least one must be dropped."""
    remaining = list(reparsed)
    for step in output:
        found = next((i for i, r in enumerate(remaining) if step_matches(step, r) is None), None)
        if found is None:
            return f"output step not found in the reparsed request: {step}"
        remaining.pop(found)
    if len(remaining) == 0:
        return "nothing was declined -- output equals the full reparsed request"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("reparse")
    args = parser.parse_args()

    reparsed_by_id = {json.loads(line)["id"]: json.loads(line)["reparsed_program"] for line in open(args.reparse)}

    total = passed = 0
    for line in open(args.dataset):
        row = json.loads(line)
        total += 1
        output = row["output"]["program"]
        reparsed = reparsed_by_id.get(row["id"])

        if reparsed is None:
            if output == []:
                print(f"  OK    {row['id']:<24} (no program on either side)")
                passed += 1
            else:
                print(f"  FAIL  {row['id']:<24} no reparse recorded but output is non-empty")
            continue

        if row["category"] in NO_PROGRAM:
            if output == []:
                print(f"  OK    {row['id']:<24} [{row['category']}] (program is empty, as it must be)")
                passed += 1
            else:
                print(f"  FAIL  {row['id']:<24} [{row['category']}] must not emit a program")
            continue

        if row["category"] in ("partial_decline", "declined"):
            reason = check_partial_decline(output, reparsed)
        else:
            reason = programs_match(reparsed, output)

        if reason is None:
            print(f"  OK    {row['id']:<24} [{row['category']}]")
            passed += 1
        else:
            print(f"  FAIL  {row['id']:<24} [{row['category']}]  {reason}")

    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
