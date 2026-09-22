"""Read a Japanese instruction back into a program, using words only.

This is the round-trip half of the check in ``DATASET.md`` section 4. It is deliberately written
against a *lexicon* -- the words a person might use -- and never imports ``phrasebank`` or looks at
the program an entry was generated from: it only ever sees the person's words. What it catches is a
generated sentence that does not say what the attached program does: a dropped or duplicated step,
a number that did not survive rendering, a left/right swap, a missing speed word, an idiom whose
table says something other than the shape it was matched against.

    python scripts/llm/parse_instruction.py data/llm/dataset.jsonl -o data/llm/reparse.jsonl

Two things it reads the way a person would rather than the way a machine would:

* 「後ろ向いて」「一周して」 name no side. They come back as ``dir: "any"``, and the comparison in
  ``roundtrip_check.py`` accepts either -- the sentence really is satisfied by both.
* 「走りながらバク転して」 does not say how long the run-up is. A rotation out of a run with nothing
  in front of it gets the 3 s run-up a reader would assume, which is the only length the phrase
  bank writes it for.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

# ------------------------------------------------------------------------------------- lexicon

FLIP_WORDS = [
    ("backflip", r"バク転|バックフリップ|後方宙返り|後ろ回転"),
    ("frontflip", r"前方回転|前転|フロントフリップ|前方宙返り|ハンドスプリング"),
    ("sideflip_left", r"左側転|左へのサイドフリップ|左回りの側転|左にサイドフリップ"),
    ("sideflip_right", r"右側転|右へのサイドフリップ|右回りの側転|右にサイドフリップ"),
]

STANCE_WORDS = [
    ("handstand", r"倒立|逆立ち"),
    ("hindstand", r"後ろ足立ち|二足立ち|後ろ足だけで立つ|後ろ足で立"),
]

# 走 reads as "forwards": every template that uses it is a forward one, because a sideways shuffle
# is not called running in either language.
DIRECTION_WORDS = [
    ("forward", r"前方|まっすぐ前|正面|前|走"),
    ("backward", r"後ろ|後方|バック"),
    ("left", r"左|反時計回り"),
    ("right", r"右|(?<!反)時計回り"),
]

SPEED_WORDS = [
    ("slow", r"ゆっくり|のんびり|そろりと|ゆるく"),
    ("fast", r"速く|はよ|急いで|素早く|パッと|ダッシュ|全力で"),
]

TURN_MARKERS = r"回っ|回り|回る|回れ|旋回|向きを変え|回転|周|振り向|向い|向く|向け|曲が|左折|右折"

# A move or a stance with no end: 「止めるまで歩いてて」「ずっと倒立」「倒立のまま」. A move with no
# amount and no marker at all is open-ended too -- that is the grammar's rule, so it is the default
# and these words only have to be kept out of the *amount* search.
OPEN_MARKERS = r"ずっと|しばらく|止めるまで|止まるまで|ストップ.{0,3}まで|続け|そのまま|のまま|したまま|キープ"

# Fired out of the preceding run rather than from standing.
RUNNING_MARKERS = r"そのまま|進んだまま|止まら[ずんな]|走りながら|走って|勢い|助走"

# 「そのまま歩いてて」「あとはずっと走ってて」 -- a verb of going, a word that says it does not
# end, and no heading, because the person said which way a clause ago. Read as "the same move,
# carrying on": same direction, same speed. A clause that *does* name a heading is read on its own
# terms, which is why the phrase bank's tails name the speed whenever they name the heading.
MOVE_VERBS = r"進ん|進み|歩い|歩き|行っ|行き|下がっ|下がり|戻っ|戻り|移動|寄っ|寄り|走っ|走り"
CARRY_MARKERS = r"そのまま|あとも|あとは|ずっと|止めるまで|止まるまで|続け|それからも|てて|でて|といて|どいて"

# A clause that is nothing but "again".
REPEAT_MARKERS = r"^(?:もういっちょ|もう一丁|もっかい|もっぺん|もう一回|もう一度|ダブル|×\s*2|アンコール)"

# Turns that name their angle in words rather than degrees, and sometimes name no side at all.
# Order matters: 「斜め右向いて」 is 45°, not the 90° that 「右向いて」 would be, so the narrower
# patterns come first.
TURN_IDIOMS: list[tuple[str, str | None, float]] = [
    (r"回れ右|後ろ向け後ろ", "right", 180.0),
    (r"左向け左", "left", 90.0),
    (r"右向け右", "right", 90.0),
    (r"後ろ(?:を)?向|うしろ向|真後ろ|振り向|反転|Uターン|半回転|半周", None, 180.0),
    (r"4分の3周|3/4回転|四分の三", None, 270.0),
    (r"一周|1周|一回転|1回転|ぐるっと|くるっと一", None, 360.0),
    (r"斜め|半分だけ", None, 45.0),
    (r"少し|ちょっとだけ|ちょい|少しだけ", None, 30.0),
    (r"左折", "left", 90.0),
    (r"右折", "right", 90.0),
    (r"直角", None, 90.0),
    (r"向い|向く|向け|曲が", None, 90.0),
]

# 「回れ右」 and 「一周」 are turns on their own; 「斜め右」 and 「少し左」 only are when something in
# the clause says the robot is turning. Without this, 「ちょっと左に3m」 would come back as a 30°
# turn instead of a short walk.
SELF_TURNS = r"回れ右|後ろ向け後ろ|向け|Uターン|反転|半回転|半周|一周|1周|一回転|1回転|4分の3周|3/4回転|左折|右折"

CONNECTIVE_SPLIT = (r"[、。!?！？→]|そのあと|その後|のあと|そんで|ほんで|そっから|それから|次に|最後に"
                    r"|から")
LEAD_TRIM = r"^(?:に|、|。|\s)+"

_UNIT_SPLIT = re.compile(CONNECTIVE_SPLIT)
_RUN_UP = {"skill": "move", "dir": "forward", "speed": "normal", "duration_s": 3.0}
"""What 「走りながら」 leaves unsaid. See the module docstring."""


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
    if re.search(r"4分の3周|3/4回転|四分の三", text):
        return 270.0
    if re.search(r"1回転|一回転|1周|一周|ぐるっと", text):
        return 360.0
    if re.search(r"直角", text):
        return 90.0
    if re.search(r"斜め", text):
        return 45.0
    if re.search(r"少し|ちょっとだけ|ちょい|少しだけ", text):
        return 30.0
    repeats = re.search(r"(\d+)回転", text)
    return float(repeats.group(1)) * 360.0 if repeats else None


def _count(text: str) -> int:
    """How many rotations one clause asks for. One step is one rotation, so this is a step count."""
    if re.search(r"もういっちょ|もう一丁|もっかい|もっぺん|もう一回|もう一度|ダブル|×\s*2", text):
        return 2
    match = re.search(r"(\d+)\s*(?:回(?!転)|連続|連発|連|発|本)", text)
    return int(match.group(1)) if match else 1


def _speed(text: str) -> str:
    return _search(SPEED_WORDS, text) or "normal"


def _turn_from_idiom(text: str) -> dict | None:
    if not re.search(TURN_MARKERS, text) and not re.search(SELF_TURNS, text):
        return None
    for pattern, direction, angle in TURN_IDIOMS:
        if re.search(pattern, text):
            step = {"skill": "turn", "angle_deg": _number(r"(\d+(?:\.\d+)?)\s*度", text) or angle}
            # A named side wins over one that happens to be in the sentence: 「回れ右して左に3m」
            # is a right turn, and 「後ろ向いて」 is satisfied either way.
            step["dir"] = direction or _search(DIRECTION_WORDS[2:], text) or "any"
            if direction is None and re.search(r"一周|一回転|半回転|反転|Uターン|振り向|後ろ|真後ろ", text):
                step["dir"] = "any"
            return step
    return None


def parse_clause(text: str) -> tuple[list[dict], bool]:
    """One clause -> the steps it asks for, and whether they come out of a run."""
    text = normalize(text).strip()
    if not text:
        return [], False

    stance = _search(STANCE_WORDS, text)
    flip = _search(FLIP_WORDS, text)

    if stance is not None:
        step = {"skill": "stance", "kind": stance}
        seconds = _seconds(text)
        if seconds is not None:
            step["duration_s"] = seconds
        return [step], False

    if flip is not None:
        running = re.search(RUNNING_MARKERS, text) is not None
        return [{"skill": "flip", "kind": flip} for _ in range(_count(text))], running

    turn = _turn_from_idiom(text)
    if turn is not None:
        return [turn], False

    direction = _search(DIRECTION_WORDS, text)
    angle = _angle(text)
    turning = re.search(TURN_MARKERS, text) is not None or angle is not None

    if turning and direction in ("left", "right") and angle is not None:
        return [{"skill": "turn", "dir": direction, "angle_deg": angle}], False

    if direction is not None:
        step = {"skill": "move", "dir": direction, "speed": _speed(text)}
        metres = _distance(text)
        seconds = _seconds(text)
        if metres is not None:
            step["distance_m"] = metres
        elif seconds is not None:
            step["duration_s"] = seconds
        # Neither: the move runs until something replaces it, whether or not the sentence said
        # 「ずっと」 -- 「前に進んで」 means the same thing.
        return [step], False
    return [], False


BARE_CARRY = r"^(?:そのまま|ずっと|継続)[。!]*$"
"""The tail of 「助走3秒→前転→そのまま。」: no verb at all, and it still means keep going."""


def _carries_on(chunk: str) -> bool:
    """Whether this clause is 「〜のまま続けて」 about the move before it rather than a new step."""
    if re.match(BARE_CARRY, chunk):
        return True
    if not re.search(CARRY_MARKERS, chunk) or not re.search(MOVE_VERBS, chunk):
        return False
    if _search(FLIP_WORDS, chunk) or _search(STANCE_WORDS, chunk):
        return False
    if _distance(chunk) is not None or _seconds(chunk) is not None:
        return False
    # 走 is not a heading, it is the same verb again; anything else names a direction of its own.
    return _search(DIRECTION_WORDS, re.sub(r"走", "", chunk)) is None


def parse_instruction(text: str) -> list[dict]:
    steps: list[dict] = []
    for index, chunk in enumerate(_UNIT_SPLIT.split(normalize(text))):
        chunk = re.sub(LEAD_TRIM, "", chunk).strip()
        if not chunk:
            continue
        if _carries_on(chunk):
            carried = next((s for s in reversed(steps) if s["skill"] == "move"), None)
            if carried is not None:
                steps.append({"skill": "move", "dir": carried["dir"], "speed": carried["speed"]})
                continue
        found, running = parse_clause(chunk)
        if not found and re.match(REPEAT_MARKERS, chunk) and steps:
            steps.append(dict(steps[-1]))          # 「バク転、もういっちょ」「右側転して、もう一回して」
            continue
        if running and found and found[0]["skill"] == "flip" and not steps:
            steps.append(dict(_RUN_UP))            # 「走りながらバク転して」: the run-up is implied
        steps.extend(found)
    return _fill_run_ups(steps)


def _fill_run_ups(steps: list[dict]) -> list[dict]:
    """「ちょっと走ってから前転して」 -- a move with no length, and a rotation out of it.

    A move with no length runs until something replaces it, so one cannot be followed by anything;
    a move in front of a flip is a run-up, and a run-up nobody measured is the 3 s one the phrase
    bank writes those words for.
    """
    for index, step in enumerate(steps[:-1]):
        if (step["skill"] == "move" and steps[index + 1]["skill"] == "flip"
                and step.get("duration_s") is None and step.get("distance_m") is None):
            step["duration_s"] = _RUN_UP["duration_s"]
    return steps


def instruction_turns(row: dict) -> list[tuple[int, str]]:
    """``(assistant turn index, the person's words)`` for every turn marked ``roundtrip``.

    Those are the turns whose user text is a whole rendered instruction given from a standstill.
    A turn that quotes an instruction inside a frame (「終わったら〜」) or answers one word
    (「もう一回」) says less than its program does on purpose, and is not the round trip's business.
    """
    if "turns" not in row:  # the earlier single-turn shape
        return [(0, row["input"])]
    out = []
    turns = row["turns"]
    for k, (user, assistant) in enumerate(zip(turns[0::2], turns[1::2])):
        if assistant.get("roundtrip"):
            out.append((k, user["content"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset JSONL; only `id` and the instruction turns are read")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    written = 0
    with open(args.out, "w") as out:
        for line in open(args.dataset):
            row = json.loads(line)
            for turn, text in instruction_turns(row):
                reparsed = parse_instruction(text)
                out.write(json.dumps({"id": row["id"], "turn": turn, "reparsed_program": reparsed},
                                     ensure_ascii=False) + "\n")
                written += 1
    print(f"reparsed {written} instructions -> {args.out}")


if __name__ == "__main__":
    main()
