"""Draw human-shaped random skill programs and write them as JSONL for validation.

    python scripts/llm/sample_programs.py --count 2000 --seed 0 --out data/llm/programs.jsonl
    python scripts/llm/sample_programs.py --count 5 --describe        # look at a few

Each line is ``{"id": ..., "program": [...], "duration_s": ...}``. Feed the file to
``validate_programs.py`` to find out which of them the policy can actually perform. Pass
``--capability`` so distances are converted with the measured table rather than nominal speeds.
"""

import argparse
import json
import random
from pathlib import Path

from unitree_rl_lab.program import CapabilityTable, CompilerConfig, SamplerConfig, program_to_json, sample_program
from unitree_rl_lab.program.grammar import step_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-total-s", type=float, default=20.0)
    parser.add_argument("--capability", type=str, default=None, help="Capability table for the compiler's calibration.")
    parser.add_argument("--describe", action="store_true", help="Print each program's timeline as well.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sampler = SamplerConfig(max_total_s=args.max_total_s)
    compiler = CompilerConfig(calibration=CapabilityTable.load_or_empty(args.capability).calibration())

    handle = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        handle = open(args.out, "w")
    for index in range(args.count):
        program, timeline = sample_program(rng, sampler, compiler)
        record = {
            "id": f"s{args.seed}-{index:06d}",
            "program": [step_to_dict(step) for step in program],
            "duration_s": round(timeline.duration, 2),
        }
        if handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"{record['id']}  {timeline.duration:5.1f} s  {program_to_json(program)}")
        if args.describe:
            print(timeline.describe())
            if timeline.adjustments:
                print("  adjustments:", "; ".join(timeline.adjustments))
            print()
    if handle:
        handle.close()
        print(f"wrote {args.count} programs to {args.out}")


if __name__ == "__main__":
    main()
