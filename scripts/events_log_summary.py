#!/usr/bin/env python
"""tfevents から checkpoint 周辺 (±window iter) の平均を Markdown 要約にする。

外部に出す数値はこの出力に統一する（docs 内のループ節は 25 iter 移動平均で 1〜2% ずれるため）。
使い方: events_log_summary.py --run <run_dir> --iter 12700 [--window 100] [--out out.md]
"""
import argparse
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SECTIONS = [
    ("跳躍の実測値（外力なし・ポリシー単体）", lambda t: "unaided" in t),
    ("跳躍の実測値（全環境・補助込み）", lambda t: t.startswith("Metrics/jump_command/")),
    ("走行", lambda t: t.startswith("Metrics/")),
    ("終了条件", lambda t: t.startswith("Episode_Termination/")),
    ("学習の健全性", lambda t: t.startswith("Loss/")),
    ("報酬の内訳（エピソード積算）", lambda t: t.startswith("Episode_Reward/")),
    ("カリキュラム", lambda t: t.startswith("Curriculum/")),
    ("その他", lambda t: t.startswith(("Policy/", "Train/")) and not t.endswith("/time")),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ea = EventAccumulator(a.run, size_guidance={"scalars": 0})
    ea.Reload()
    lo, hi = a.iter - a.window, a.iter + a.window

    means, steps = {}, []
    for tag in ea.Tags()["scalars"]:
        vals = [e.value for e in ea.Scalars(tag) if lo <= e.step <= hi]
        if vals:
            means[tag] = sum(vals) / len(vals)
            steps.append(len(vals))

    lines = [f"# {a.title or a.run.rstrip('/').split('/')[-1]} — iter {a.iter} 学習ログ要約", ""]
    lines += [
        f"- run: `{a.run}`",
        f"- checkpoint: **iter {a.iter}**",
        f"- 集計範囲: iter {lo} 〜 {hi}（{min(steps) if steps else 0}〜{max(steps) if steps else 0} 点の平均）",
        "",
        f"すべて iter {a.iter} を中心に ±{a.window} iter で平均した値（学習ログのノイズを均すため）。",
        "`unaided_*` は EFGCL の外力を与えないホールドアウト群（全環境の25%）の実測値で、",
        "ポリシー単体の性能を表す。それ以外は補助を含む全環境の平均。",
        "",
    ]

    used, buckets = set(), defaultdict(list)
    for title, pred in SECTIONS:
        for tag in sorted(means):
            if tag not in used and pred(tag):
                used.add(tag)
                buckets[title].append(tag)

    for title, _ in SECTIONS:
        if not buckets[title]:
            continue
        lines += [f"## {title}", "", "| 指標 | 値 |", "|---|---|"]
        for tag in buckets[title]:
            label = tag.split("/", 1)[1] if tag.startswith(("Metrics/", "Episode_Reward/",
                     "Episode_Termination/", "Loss/", "Curriculum/")) else tag
            lines.append(f"| `{label}` | {means[tag]:.4f} |")
        lines.append("")

    text = "\n".join(lines)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out} ({len(used)} tags)")
    else:
        print(text)


if __name__ == "__main__":
    main()
