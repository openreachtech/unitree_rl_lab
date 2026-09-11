"""Read a Japanese instruction back into a program, using words only.

This is the round-trip half of the check in ``DATASET.md`` section 4. It is deliberately written
against a *lexicon* -- the words a person might use -- and never imports ``phrasebank`` or looks at
the program an entry was generated from: it only ever sees ``input``. What it catches is a
generated sentence that does not say what the attached program does (a dropped or duplicated step,
a number that did not survive rendering, a left/right swap, a missing speed word). It does not
judge whether the sentence sounds natural; that is what the sampled human read is for.

    python scripts/llm/parse_instruction.py data/llm/dataset.jsonl -o data/llm/reparse.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

# ------------------------------------------------------------------------------------- lexicon

FLIP_WORDS = [
    ("backflip", r"バク転|バックフリップ|後方宙返り|後ろ回転"),
    ("frontflip", r"前方回転|前転|フロントフリップ|前方宙返り"),
    ("sideflip_left", r"左側転|左へのサイドフリップ|左回りの側転|左にサイドフリップ"),
    ("sideflip_right", r"右側転|右へのサイドフリップ|右回りの側転|右にサイドフリップ"),
    ("jump", r"ジャンプ|垂直跳び|真上に跳"),
]

STANCE_WORDS = [
    ("handstand", r"倒立|逆立ち"),
    ("hindstand", r"後ろ足立ち|二足立ち|後ろ足だけで立つ|後ろ足で立"),
]

# Compound headings first: "左斜め後ろ" must not be read as "後ろ" or "左".
DIRECTION_WORDS = [
    ("forward_left", r"左斜め前|斜め左前|左前方"),
    ("forward_right", r"右斜め前|斜め右前|右前方"),
    ("backward_left", r"左斜め後ろ|斜め左後ろ|左後方"),
    ("backward_right", r"右斜め後ろ|斜め右後ろ|右後方"),
    ("forward", r"前方|まっすぐ前|正面|前"),
    ("backward", r"後ろ|後方|バック"),
    ("left", r"左|反時計回り"),
    ("right", r"右|(?<!反)時計回り"),
]

SPEED_WORDS = [
    ("slow", r"ゆっくり|のんびり|そろりと|ゆるく"),
    ("fast", r"速く|はよ|急いで|素早く|パッと|ダッシュ|全力で"),
]

TURN_MARKERS = r"回っ|回り|回る|回れ|旋回|向きを変え|回転|周|振り向"
STOP_MARKERS = r"止ま|停止|待っ|待ち|待つ|静止|じっとし"

CONNECTIVE_SPLIT = r"[、。!?！？]|そのあと|その後|そんで|ほんで|そっから|それから|続けて|次に|最後に"
LEAD_TRIM = r"^(?:に|、|。|\s)+"

_UNIT_SPLIT = re.compile(CONNECTIVE_SPLIT)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _search(pairs, text: str):
    for key, pattern in pairs:
        if re.search(pattern, text):
            return key
    return None


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _distance(text: str) -> float | None:
    metres = _number(r"(\d+(?:\.\d+)?)\s*(?:m(?![a-zA-Z])|ｍ|メートル)", text)
    if metres is not None:
        return metres
    cm = _number(r"(\d+(?:\.\d+)?)\s*(?:cm|センチ)", text)
    return cm / 100.0 if cm is not None else None


def _seconds(text: str) -> float | None:
    return _number(r"(\d+(?:\.\d+)?)\s*秒", text)


def _angle(text: str) -> float | None:
    degrees = _number(r"(\d+(?:\.\d+)?)\s*度", text)
    if degrees is not None:
        return degrees
    if re.search(r"半回転|半周", text):
        return 180.0
    if re.search(r"1回転|一回転|1周|一周|ぐるっと", text):
        return 360.0
    if re.search(r"直角", text):
        return 90.0
    if re.search(r"(\d+)回転", text):
        return float(re.search(r"(\d+)回転", text).group(1)) * 360.0
    return None


def _count(text: str) -> int:
    match = re.search(r"(\d+)\s*(?:回(?!転)|連続|発|本)", text)
    return int(match.group(1)) if match else 1


def _speed(text: str) -> str:
    return _search(SPEED_WORDS, text) or "normal"


def parse_clause(text: str) -> dict | None:
    """One clause -> one step, or None if it carries no skill."""
    text = normalize(text).strip()
    if not text:
        return None

    stance = _search(STANCE_WORDS, text)
    flip = _search(FLIP_WORDS, text)

    # "倒立のまま前に5秒進んで" is one stance step that walks, not a stance followed by a move.
    if stance is not None:
        step = {"skill": "stance", "kind": stance, "duration_s": _seconds(text) or 5.0, "speed": "slow"}
        if re.search(r"のまま|したまま|しながら", text):
            direction = _search(DIRECTION_WORDS, re.split(r"のまま|したまま|しながら", text, maxsplit=1)[1])
            if direction is not None:
                step["dir"] = direction
        return step

    if flip is not None:
        return {"skill": "flip", "kind": flip, "count": _count(text)}

    direction = _search(DIRECTION_WORDS, text)
    angle = _angle(text)
    turning = re.search(TURN_MARKERS, text) is not None or angle is not None

    if turning and direction in ("left", "right"):
        step = {"skill": "turn", "dir": direction, "speed": _speed(text)}
        if angle is not None:
            step["angle_deg"] = angle
        else:
            step["duration_s"] = _seconds(text) or 3.0
        return step

    if re.search(STOP_MARKERS, text) and direction is None:
        return {"skill": "stop", "duration_s": _seconds(text) or 1.0}

    if direction is not None:
        step = {"skill": "move", "dir": direction, "speed": _speed(text)}
        metres = _distance(text)
        if metres is not None:
            step["distance_m"] = metres
        else:
            step["duration_s"] = _seconds(text) or 3.0
        return step

    if re.search(STOP_MARKERS, text):
        return {"skill": "stop", "duration_s": _seconds(text) or 1.0}
    return None


def parse_instruction(text: str) -> list[dict]:
    steps: list[dict] = []
    for chunk in _UNIT_SPLIT.split(normalize(text)):
        chunk = re.sub(LEAD_TRIM, "", chunk)
        step = parse_clause(chunk)
        if step is not None:
            steps.append(step)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset JSONL; only `id` and `input` are read")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    written = 0
    with open(args.out, "w") as out:
        for line in open(args.dataset):
            row = json.loads(line)
            reparsed = parse_instruction(row["input"])
            out.write(json.dumps({"id": row["id"], "reparsed_program": reparsed}, ensure_ascii=False) + "\n")
            written += 1
    print(f"reparsed {written} instructions -> {args.out}")


if __name__ == "__main__":
    main()
