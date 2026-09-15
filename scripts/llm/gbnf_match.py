"""A small GBNF matcher, so the grammar in ``chat_format.py`` can be tested without llama.cpp.

It is a reference implementation, not a sampler: it answers "does this whole string belong to the
language" by backtracking, which is enough to check that every training target is accepted and
that the malformed strings we care about are refused. It covers the GBNF subset the generated
grammar uses -- literals, character classes, rule references, alternation, grouping, ``?``, ``*``,
``+`` -- and nothing more.

    python scripts/llm/gbnf_match.py                              # self-test on rendered examples
    python scripts/llm/gbnf_match.py --dataset data/llm/dataset.jsonl   # every training target
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chat_format import RobotState, gbnf_grammar, render_output, row_turns  # noqa: E402

# --- parsing the grammar text -------------------------------------------------------------------

_TOKEN = re.compile(r'\s*(?:(::=)|("(?:[^"\\]|\\.)*")|(\[(?:[^\]\\]|\\.)*\])|([A-Za-z][\w-]*)|([()|?*+]))')


def _unescape(text: str) -> str:
    return text.encode().decode("unicode_escape").encode("latin-1").decode("utf-8")


def parse_grammar(text: str) -> dict[str, list]:
    """``rule name -> alternatives``; an alternative is a list of items ``(kind, payload, repeat)``."""
    rules: dict[str, list] = {}
    body = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    # Each rule starts at "name ::="; split on that boundary rather than on newlines so a rule may wrap.
    for match in re.finditer(r"([A-Za-z][\w-]*)\s*::=(.*?)(?=\n[A-Za-z][\w-]*\s*::=|\Z)", body, re.DOTALL):
        name, rhs = match.group(1), match.group(2)
        rules[name] = _parse_alternatives(_tokens(rhs))
    return rules


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            if text[pos:].strip():
                raise ValueError(f"cannot tokenise {text[pos:pos + 20]!r}")
            break
        pos = match.end()
        token = next(group for group in match.groups() if group is not None)
        out.append(token)
    return out


def _parse_alternatives(tokens: list[str]):
    alternatives, current = [], []
    while tokens:
        token = tokens[0]
        if token == "|":
            tokens.pop(0)
            alternatives.append(current)
            current = []
        elif token == ")":
            break
        else:
            current.append(_parse_item(tokens))
    alternatives.append(current)
    return alternatives


def _parse_item(tokens: list[str]):
    token = tokens.pop(0)
    if token == "(":
        inner = _parse_alternatives(tokens)
        assert tokens.pop(0) == ")", "unbalanced parentheses"
        item = ("group", inner)
    elif token.startswith('"'):
        item = ("lit", _unescape(token[1:-1]))
    elif token.startswith("["):
        item = ("class", re.compile(token))
    else:
        item = ("ref", token)
    repeat = ""
    if tokens and tokens[0] in "?*+":
        repeat = tokens.pop(0)
    return (*item, repeat)


# --- matching -------------------------------------------------------------------------------------


def _match_seq(rules, items, text, pos):
    if not items:
        yield pos
        return
    head, rest = items[0], items[1:]
    for after in _match_item(rules, head, text, pos):
        yield from _match_seq(rules, rest, text, after)


def _match_once(rules, kind, payload, text, pos):
    if kind == "lit":
        if text.startswith(payload, pos):
            yield pos + len(payload)
    elif kind == "class":
        if pos < len(text) and payload.match(text[pos]):
            yield pos + 1
    elif kind == "ref":
        for alternative in rules[payload]:
            yield from _match_seq(rules, alternative, text, pos)
    elif kind == "group":
        for alternative in payload:
            yield from _match_seq(rules, alternative, text, pos)


def _match_item(rules, item, text, pos):
    kind, payload, repeat = item
    if repeat == "":
        yield from _match_once(rules, kind, payload, text, pos)
        return
    if repeat in "?*":
        yield pos
    seen = {pos}
    frontier = [pos]
    while frontier:
        nxt = []
        for p in frontier:
            for after in _match_once(rules, kind, payload, text, p):
                if after not in seen:
                    seen.add(after)
                    nxt.append(after)
                    if repeat in "*+":
                        yield after
                    elif repeat == "?":
                        yield after
        frontier = nxt if repeat in "*+" else []


def matches(rules: dict, text: str, root: str = "root") -> bool:
    return any(end == len(text) for end in _match_seq(rules, [("ref", root, "")], text, 0))


# --- self-test ------------------------------------------------------------------------------------


def self_test(rules: dict) -> int:
    program = [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 5.0},
               {"skill": "flip", "kind": "frontflip", "count": 2, "running": True},
               {"skill": "turn", "dir": "left", "speed": "slow", "angle_deg": 90},
               {"skill": "stop", "duration_s": 1.0},
               {"skill": "stance", "kind": "handstand", "duration_s": 5.0, "dir": "forward", "speed": "slow"},
               {"skill": "flip", "kind": "backflip", "count": 1}]
    accept = [
        render_output("了解、走って前転するね。", "replace", program),
        render_output("終わったらやるで。", "append", program[:1]),
        render_output("走りながらいくで！", "insert", [program[1]]),
        render_output("止めるで。", "cancel", []),
        render_output("ほんまやな。", "none", []),
    ]
    refuse = [
        "止めるで。\n\naction: cancel\nprogram: " + json.dumps(program[:1]),          # cancel with a program
        "やるで。\n\naction: replace\nprogram: []",                                   # replace without one
        "やるで。\n\naction: go\nprogram: []",                                        # unknown action
        "やるで。\n\nprogram: []",                                                    # no action line
        "やるで。\n\naction: none\nprogram: [",                                       # truncated
        render_output("x", "replace", program).replace('"count": 2, "running": true', '"count": 2, "running": false'),
        render_output("x", "replace", program).replace('"dir": "forward"', '"dir": "north"'),
        render_output("x", "replace", program).replace('"speed": "normal", ', ''),
        "二行の\n返事\n\naction: none\nprogram: []",
    ]
    failed = 0
    for text in accept:
        if not matches(rules, text):
            print("  should ACCEPT:", text.replace("\n", "\\n")[:120]); failed += 1
    for text in refuse:
        if matches(rules, text):
            print("  should REFUSE:", text.replace("\n", "\\n")[:120]); failed += 1
    print(f"self-test: {len(accept) + len(refuse) - failed}/{len(accept) + len(refuse)} as expected")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="check every training target of a dataset against the grammar")
    args = parser.parse_args()
    rules = parse_grammar(gbnf_grammar())
    failed = self_test(rules)
    if args.dataset:
        total = bad = 0
        for line in open(args.dataset):
            row = json.loads(line)
            for _user, reply, action, program in row_turns(row):
                total += 1
                if not matches(rules, render_output(reply, action, program)):
                    bad += 1
                    print(f"  REFUSED {row['id']}")
        print(f"dataset: {total - bad}/{total} targets accepted by the grammar")
        failed += bad
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
