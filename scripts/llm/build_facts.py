"""Turn ``validate_programs.py`` output into fact sheets: the text a generation model reads.

A generation model must never invent a number -- it reads exactly what sim measured, phrased in
words a person would recognise ("about 5 m", "8 out of 8"), and writes the instruction and reply
from that. This script is the deterministic step between the two: it does not call any model.

    python scripts/llm/build_facts.py data/llm/validated.jsonl --capability data/llm/capability.json \
        --out data/llm/facts.jsonl

Each output line is ``{"id": ..., "program": [...], "passed": ..., "facts": "<text>"}``. See
``DATASET.md`` section 2 for the format the text follows.
"""

import argparse
import json
import math

from unitree_rl_lab.program import CapabilityTable
from unitree_rl_lab.program.grammar import JA


def ja_step(step: dict) -> str:
    skill = step["skill"]
    if skill in ("flip", "stance"):
        return JA.get(step.get("kind"), step.get("kind", ""))
    if skill in ("move", "turn"):
        return JA.get(step.get("dir"), step.get("dir", ""))
    return JA.get(skill, skill)


def fmt_m(x: float) -> str:
    return f"約{x:.1f}m" if abs(x) >= 0.05 else "ほぼその場"


def fmt_deg(rad: float) -> str:
    return f"約{abs(math.degrees(rad)):.0f}度"


def fmt_s(x: float) -> str:
    return f"約{x:.1f}秒"


def motion_facts(motion: dict) -> str:
    kind = "前進" if motion["kind"] == "move" else "旋回"
    label = f"{JA.get(motion['dir'], motion['dir'])}"
    n = len(motion["along"])
    mean_along = sum(motion["along"]) / n
    if motion["kind"] == "move":
        measured = fmt_m(mean_along)
    else:
        measured = fmt_deg(mean_along)
    falls = f"、転倒 {motion['fall_rate']:.0%}" if motion["fall_rate"] > 0 else ""
    return f"- {label}を{fmt_s(motion['duration_s'])}指令 -> 実測 {measured} (n={n}){falls}"


def flip_facts(flip: dict) -> str:
    n = len(flip["ok"])
    ok = sum(flip["ok"])
    return f"- {JA.get(flip['kind'], flip['kind'])} 1回 -> 成功 {ok}/{n} ({flip['success_rate']:.0%})"


def stance_facts(stance: dict, table: CapabilityTable) -> str:
    n = len(stance["ok"])
    ok = sum(stance["ok"])
    recovered = [r for r in stance["recover_s"] if r is not None]
    recover_txt = (
        f"、解除後に四足へ戻れたのは {len(recovered)}/{n}"
        + (f" (平均{sum(recovered)/len(recovered):.1f}秒で復帰)" if recovered else "")
    )
    return (
        f"- {JA.get(stance['kind'], stance['kind'])}を{fmt_s(stance['duration_s'])}保持 -> "
        f"起立+保持成功 {ok}/{n} ({stance['success_rate']:.0%}){recover_txt}"
    )


def build_facts(record: dict, table: CapabilityTable) -> str:
    """One line (or a few) per program step, in step order.

    The sim runner scores segments in the same order the program's steps produced them, so each of
    ``motions`` / ``flips`` / ``stances`` is consumed sequentially rather than matched by kind --
    matching by kind alone merges two steps that happen to share a direction (e.g. "turn left" then
    later "turn left" again) into one, duplicated, entry.
    """
    lines = []
    motions = iter(record["motions"])
    flips = iter(record["flips"])
    stances = iter(record["stances"])
    for step in record["program"]:
        skill = step["skill"]
        if skill == "stop":
            lines.append(f"- 停止 {fmt_s(step['duration_s'])}")
        elif skill == "flip":
            count = step.get("count", 1)
            taken = [next(flips) for _ in range(count)]
            if count > 1:
                lines.append(f"- {JA.get(step['kind'], step['kind'])} {count}回連続 (下記は各回)")
                lines.extend("  " + flip_facts(flip) for flip in taken)
            else:
                lines.append(flip_facts(taken[0]))
        elif skill == "stance":
            lines.append(stance_facts(next(stances), table))
        else:  # move / turn
            lines.append(motion_facts(next(motions)))

    lines.append(f"- 合計所要時間: {fmt_s(record['timeline']['duration_s'])}")
    if record["timeline"]["adjustments"]:
        lines.append("- コンパイラの調整: " + "; ".join(record["timeline"]["adjustments"]))
    verdict = f"合格 ({sum(record['replica_passed'])}/{record['replicas']})" if record["passed"] else \
              f"不合格 ({sum(record['replica_passed'])}/{record['replicas']})"
    lines.append(f"- 判定: {verdict}")
    if not record["passed"]:
        lines.append(f"- 主な原因: {dominant_cause(record)}")
    return "\n".join(lines)


def dominant_cause(record: dict) -> str:
    """The step most responsible for the failing replicas, as a phrase.

    Three kinds of number can be "the reason": a flip's landing rate, a stance's *descent*
    recovery rate (its rise/hold `success_rate` is usually fine -- it is coming back down that
    fails), and a motion's fall rate (a replica that never recovered from a stance and then fell
    while walking). Comparing raw `success_rate` across these would have picked a flip that
    succeeded 100% of the time over the stance whose descent is the actual problem, because a
    stance's `success_rate` does not cover its descent at all.
    """
    candidates = []
    for flip in record["flips"]:
        candidates.append((flip["success_rate"], f"{flip['kind']}の着地"))
    for stance in record["stances"]:
        n = len(stance["recover_s"])
        recover_rate = sum(1 for r in stance["recover_s"] if r is not None) / n
        candidates.append((recover_rate, f"{stance['kind']}の降り(復帰)"))
    for motion in record["motions"]:
        candidates.append((1.0 - motion["fall_rate"], f"{JA.get(motion['dir'], motion['dir'])}移動中の転倒"))
    rate, label = min(candidates, default=(1.0, "不明"))
    return f"{label} (成功率 {rate:.0%})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("validated", help="validate_programs.py output (JSONL)")
    parser.add_argument("--capability", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    table = CapabilityTable.load_or_empty(args.capability)
    with open(args.validated) as handle, open(args.out, "w") as out:
        for i, line in enumerate(handle):
            if args.limit and i >= args.limit:
                break
            record = json.loads(line)
            out.write(json.dumps({
                "id": record["id"],
                "program": record["program"],
                "passed": record["passed"],
                "success_rate": record["success_rate"],
                "facts": build_facts(record, table),
            }, ensure_ascii=False) + "\n")
    print(f"wrote facts for {i + 1} programs to {args.out}")


if __name__ == "__main__":
    main()
