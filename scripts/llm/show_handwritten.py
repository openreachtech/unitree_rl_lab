"""Read hand-written conversations the way a person would, before they go into the dataset.

The review gate between writing and merging. Two things are worth looking at and they need
different views:

    会話    the rows themselves -- does the Japanese sound like someone talking to a robot, and
            does each turn's program say what the person just asked for
    分布    what 2000 of them look like in aggregate -- the first attempt failed here, not in any
            single row: 19% of its lines ended 「〜ください」 and it read fine one line at a time

    python scripts/llm/show_handwritten.py data/llm/handwritten.jsonl --sample 30
    python scripts/llm/show_handwritten.py data/llm/handwritten.jsonl --stats
    python scripts/llm/show_handwritten.py data/llm/handwritten.jsonl --id s0-000123
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

AXES = {
    "前置き":   r"^(ねえ|なあ|あのさ|あの|ちょっと|悪いけど|すまん|ごめん|そしたら|とりあえず|おい|えっと|ほな)",
    "後置き":   r"(お願い|頼む|頼み|いける\?|いけますか|できる\?|よろしく|よろしゅう|たのむ)",
    "言い直し": r"(いや|じゃなくて|やのうて|ちゃう|やっぱ)",
    "表記ゆれ": r"(ｍ|メーター|メートル|cm|センチ|[０-９])",
    "助詞落とし": r"(前|後ろ|左|右|倒立|逆立ち)[0-9０-９]",
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def show(row: dict) -> None:
    turns = row["turns"]
    print(f"=== {row['id']}  [{row.get('style', '?')}]  {len(turns) // 2}ターン")
    for user, assistant in zip(turns[0::2], turns[1::2]):
        queue = json.dumps(user.get("queue", []), ensure_ascii=False)
        program = json.dumps(assistant.get("program", []), ensure_ascii=False)
        print(f"  人   {user['content']}")
        if user.get("queue"):
            print(f"       ↑ いま動いてるもの {queue}")
        print(f"  Go2  {assistant['reply']}")
        print(f"       → {program}")
    print()


def stats(rows: list[dict]) -> None:
    utterances = [u["content"] for row in rows for u in row["turns"][0::2]]
    first = [row["turns"][0]["content"] for row in rows]
    later = [u["content"] for row in rows for u in row["turns"][2::2]]
    total = len(utterances)
    print(f"会話 {len(rows)}  発話 {total}（1ターン目 {len(first)} / 2ターン目以降 {len(later)}）")

    print(f"\nターン数        " + "  ".join(f"{n}:{c}" for n, c in
                                          sorted(Counter(len(r["turns"]) // 2 for r in rows).items())))
    print(f"style          " + "  ".join(f"{s}:{c}" for s, c in Counter(
        r.get("style") for r in rows).most_common()))

    tails = Counter(u.rstrip("。!?！？ ")[-4:] for u in utterances)
    print(f"\n文末4字 上位12（{len(tails)}種、上位10で{sum(c for _, c in tails.most_common(10)) / total:.0%}）")
    for tail, count in tails.most_common(12):
        print(f"   {count:5} ({count / total:5.1%})  …{tail}")

    print("\n軸のカバー率（全発話中）")
    for name, pattern in AXES.items():
        count = sum(1 for u in utterances if re.search(pattern, u))
        print(f"   {name:10} {count:5} ({count / total:5.1%})")
    short = sum(1 for u in utterances if len(u) <= 5)
    long_ = sum(1 for u in utterances if len(u) >= 60)
    print(f"   {'5字以下':10} {short:5} ({short / total:5.1%})")
    print(f"   {'60字以上':10} {long_:5} ({long_ / total:5.1%})")

    repeated = [(u, c) for u, c in Counter(utterances).most_common(8) if c > 1]
    print(f"\n同じ発話の重複 {sum(c - 1 for _, c in Counter(utterances).items() if c > 1)}件"
          + ("" if repeated else " （なし）"))
    for text, count in repeated:
        print(f"   {count}回  {text}")

    grams = Counter(u[i:i + 4] for u in utterances for i in range(len(u) - 3))
    print("\n頻出4-gram  " + "  ".join(f"{g}:{c}" for g, c in grams.most_common(15)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows")
    parser.add_argument("--sample", type=int, help="this many at random")
    parser.add_argument("--id", help="one row by program id")
    parser.add_argument("--stats", action="store_true", help="the distribution report instead of the rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load(args.rows)
    if args.stats:
        stats(rows)
        return
    if args.id:
        rows = [r for r in rows if r["id"].split("#")[0] == args.id]
    elif args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    for row in rows:
        show(row)


if __name__ == "__main__":
    main()
