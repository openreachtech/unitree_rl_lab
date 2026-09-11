"""Summarise a ``validate_programs.py`` output file.

    python scripts/llm/summarize_validation.py data/llm/validated.jsonl

Answers the questions the dataset builder asks: how many programs passed, whether length or a
particular skill is what fails them, and how a skill fares *in context* (a flip after running
against one from standing) as opposed to the standalone figure in the capability table.
"""

import argparse
import collections
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--min-success", type=float, default=None, help="Re-threshold instead of using the stored verdict.")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.path) if line.strip()]
    if args.min_success is not None:
        for r in records:
            r["passed"] = r["success_rate"] >= args.min_success
    total = len(records)
    passed = sum(r["passed"] for r in records)
    print(f"{passed}/{total} programs passed ({passed / total:.1%})")

    print("\nby number of steps:")
    by_len = collections.defaultdict(list)
    for r in records:
        by_len[len(r["program"])].append(r["passed"])
    for n in sorted(by_len):
        v = by_len[n]
        print(f"  {n} steps  {sum(v):4d}/{len(v):<4d}  {sum(v) / len(v):.1%}")

    print("\nby skills present (a program counts under every skill it contains):")
    by_skill = collections.defaultdict(list)
    for r in records:
        kinds = {step["skill"] + ":" + str(step.get("kind", step.get("dir", ""))) for step in r["program"]}
        for k in kinds:
            by_skill[k].append(r["passed"])
    for k in sorted(by_skill, key=lambda k: sum(by_skill[k]) / len(by_skill[k])):
        v = by_skill[k]
        print(f"  {k:<28} {sum(v):4d}/{len(v):<4d}  {sum(v) / len(v):.1%}")

    print("\nwhy replicas failed (first cause per failed replica):")
    causes = collections.Counter()
    for r in records:
        n = r["replicas"]
        for i in range(n):
            if r["replica_passed"][i]:
                continue
            fell = r["fell_at_s"][i] is not None
            flip_fail = next((f["kind"] for f in r["flips"] if not f["ok"][i]), None)
            stance_fail = next((s["kind"] for s in r["stances"] if not s["ok"][i]), None)
            if fell:
                where = "unknown"
                t = r["fell_at_s"][i]
                for f in r["flips"]:
                    if f["t"] <= t <= f["t"] + 1.5:
                        where = f"during {f['kind']}"
                for s in r["stances"]:
                    if s["t"] <= t <= s["t"] + s["duration_s"]:
                        where = f"during {s['kind']}"
                    elif s["t"] + s["duration_s"] < t <= s["t"] + s["duration_s"] + 2.0:
                        where = f"coming down from {s['kind']}"
                if where == "unknown":
                    moving = [m for m in r["motions"] if m["t"] <= t <= m["t"] + m["duration_s"]]
                    where = f"during {moving[0]['kind']} {moving[0]['dir']} {moving[0]['speed']}" if moving else "while standing/settling"
                causes[f"fell {where}"] += 1
            elif flip_fail:
                causes[f"{flip_fail} not landed"] += 1
            elif stance_fail:
                causes[f"{stance_fail} not held"] += 1
            elif not r["ended_upright"][i]:
                causes["not upright at the end"] += 1
            else:
                causes["other"] += 1
    for cause, count in causes.most_common(15):
        print(f"  {count:5d}  {cause}")

    print("\nflip success in context (preceded by movement vs from standing):")
    ctx = collections.defaultdict(lambda: [0, 0])
    for r in records:
        for f in r["flips"]:
            moved_before = any(m["t"] + m["duration_s"] <= f["t"] and f["t"] - (m["t"] + m["duration_s"]) < 1.0 for m in r["motions"])
            key = (f["kind"], "after moving" if moved_before else "from standing")
            ctx[key][0] += sum(f["ok"])
            ctx[key][1] += len(f["ok"])
    for (kind, how), (ok, n) in sorted(ctx.items()):
        print(f"  {kind:<16} {how:<14} {ok:4d}/{n:<4d}  {ok / n:.1%}")


if __name__ == "__main__":
    main()
