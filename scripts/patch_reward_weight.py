#!/usr/bin/env python3
"""Replace the `weight=` value of a single RewTerm block in velocity_env_cfg_go2.py.

Usage: patch_reward_weight.py <cfg_file> <func_name> <new_weight>

Anchors on the RewTerm's `func=mdp.<func_name>` line and replaces the first
`weight=<number>` found after it, within that same RewTerm(...) call. Aborts
(exit 1) instead of silently no-op'ing if the anchor isn't found exactly once,
so a bad automated patch never runs training on an unintended config.
"""
import re
import sys

def main() -> int:
    cfg_file, func_name, new_weight = sys.argv[1], sys.argv[2], sys.argv[3]
    text = open(cfg_file).read()

    anchor = f"func=mdp.{func_name},"
    count = text.count(anchor)
    if count != 1:
        print(f"FAIL: anchor '{anchor}' found {count} times (expected exactly 1)")
        return 1

    anchor_pos = text.index(anchor)
    # Search for the next `weight=<number>` after the anchor.
    m = re.compile(r"weight=(-?[0-9.eE+-]+)").search(text, anchor_pos)
    if not m:
        print(f"FAIL: no weight= found after anchor '{anchor}'")
        return 1

    old_weight = m.group(1)
    new_text = text[: m.start(1)] + new_weight + text[m.end(1) :]
    open(cfg_file, "w").write(new_text)
    print(f"OK: {func_name} weight {old_weight} -> {new_weight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
