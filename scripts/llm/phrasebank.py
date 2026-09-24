"""Japanese surface forms for instructions and replies -- the hand-written half of the dataset.

``build_dataset.py`` walks a validated program step by step and asks this module for the words.
Everything a human would actually say lives here; nothing here knows about sim results, capability
rates or thresholds, which keeps the wording separable from the decisions made about it.

Three registers, chosen per entry:

    polite   標準語・丁寧   「5メートルほど前に進んでください。」
    kansai   関西弁・くだけた 「5mくらい前いってや。」
    short    体言止め・省略  「前5m。」

A step renders to a ``Clause``: everything up to the verb, plus the verb in three conjugations, so
the same clause can end an instruction ("進んでください"), sit in the middle of one ("進んでから、")
or appear in a reply ("進みます" / "進むで")。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

STYLES: tuple[str, ...] = ("polite", "kansai", "short")


@dataclass(frozen=True)
class Verb:
    te: str
    masu: str
    plain: str

    @property
    def stem(self) -> str:
        """The ます-stem -- 進み, 歩き, 移動し. What 「〜続ける」 attaches to."""
        return self.masu[:-2]

    @property
    def teru(self) -> str:
        """Standard continuous -- 進んでて, 歩いてて. 「止めるまで歩いてて」."""
        return self.te + "て"

    @property
    def toku(self) -> str:
        """Kansai continuous -- 進んどいて, 歩といて, 下がっといて.

        て→と and で→ど, then いて. Written out because 「進んでといて」 is not a word and the
        templates need the real form.
        """
        return self.te[:-1] + ("ど" if self.te[-1] == "で" else "と") + "いて"


V_SUSUMU = Verb("進んで", "進みます", "進む")
V_ARUKU = Verb("歩いて", "歩きます", "歩く")
V_IKU = Verb("行って", "行きます", "行く")
V_SAGARU = Verb("下がって", "下がります", "下がる")
V_MODORU = Verb("戻って", "戻ります", "戻る")
V_IDOU = Verb("移動して", "移動します", "移動する")
V_YORU = Verb("寄って", "寄ります", "寄る")
V_MAWARU = Verb("回って", "回ります", "回る")
V_SENKAI = Verb("旋回して", "旋回します", "旋回する")
V_MUKI = Verb("向きを変えて", "向きを変えます", "向きを変える")
V_TOMARU = Verb("止まって", "止まります", "止まる")
V_TEISHI = Verb("停止して", "停止します", "停止する")
V_MATSU = Verb("待って", "待ちます", "待つ")
V_SURU = Verb("して", "します", "する")
V_YARU = Verb("やって", "やります", "やる")
V_KIMERU = Verb("決めて", "決めます", "決める")
V_KEEP = Verb("キープして", "キープします", "キープする")


@dataclass
class Clause:
    """One step as words: ``body`` + a verb that can be conjugated three ways."""

    body: str
    verb: Verb

    def te(self) -> str:
        return self.body + self.verb.te

    def masu(self) -> str:
        return self.body + self.verb.masu

    def plain(self) -> str:
        return self.body + self.verb.plain


# --------------------------------------------------------------------------------------- numbers


def _n(value: float) -> str:
    """3.0 -> "3", 0.5 -> "0.5"."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def distance_word(metres: float, style: str, rng: random.Random) -> str:
    n = _n(metres)
    if style == "short":
        return rng.choice([f"{n}m", f"{n}m"])
    if metres == 0.5:
        return rng.choice(["50cmくらい", "0.5mほど", "50cmほど", "0.5mくらい", "0.5m"])
    forms = [f"{n}m", f"{n}mくらい", f"{n}mほど", f"約{n}m", f"だいたい{n}m", f"{n}メートル", f"{n}メートルほど"]
    if style == "kansai":
        forms += [f"{n}mぐらい", f"{n}mばかし"]
    return rng.choice(forms)


def seconds_word(seconds: float, style: str, rng: random.Random) -> str:
    n = _n(seconds)
    if style == "short":
        return f"{n}秒"
    forms = [f"{n}秒", f"{n}秒くらい", f"{n}秒ほど", f"{n}秒間", f"だいたい{n}秒"]
    if style == "kansai":
        forms += [f"{n}秒ぐらい"]
    return rng.choice(forms)


def angle_word(degrees: float, style: str, rng: random.Random) -> str:
    n = _n(degrees)
    if style == "short":
        return f"{n}度"
    forms = [f"{n}度"]
    if degrees == 90:
        forms += ["90度", "直角に"]
    if degrees == 180:
        forms += ["半回転", "180度くらい", "半周"]
    if degrees == 360:
        forms += ["1回転", "ぐるっと1周", "360度ぐるっと"]
    if degrees == 45:
        forms += ["45度ほど", "ちょっとだけ45度"]
    return rng.choice(forms)


def count_word(count: int, style: str, rng: random.Random) -> str:
    if count == 1:
        return rng.choice(["", "", "1回"]) if style != "short" else rng.choice(["", "1回"])
    n = str(count)
    if style == "short":
        return f"{n}回"
    forms = [f"{n}回", f"{n}回続けて", f"{n}連続で", f"{n}回連続で"]
    return rng.choice(forms)


# ------------------------------------------------------------------------------------ directions

DIR_WORDS: dict[str, list[str]] = {
    "forward": ["前に", "前へ", "まっすぐ前に", "前方へ", "正面に"],
    "backward": ["後ろに", "後ろへ", "後方へ", "バックで"],
    "left": ["左に", "左へ", "左方向に", "真横の左へ"],
    "right": ["右に", "右へ", "右方向に", "真横の右へ"],
    "forward_left": ["左斜め前に", "斜め左前へ", "左前方に", "左斜め前へ"],
    "forward_right": ["右斜め前に", "斜め右前へ", "右前方に", "右斜め前へ"],
    "backward_left": ["左斜め後ろに", "斜め左後ろへ", "左後方に", "左斜め後ろへ"],
    "backward_right": ["右斜め後ろに", "斜め右後ろへ", "右後方に", "右斜め後ろへ"],
}

DIR_WORDS_SHORT: dict[str, str] = {
    "forward": "前",
    "backward": "後ろ",
    "left": "左",
    "right": "右",
    "forward_left": "左斜め前",
    "forward_right": "右斜め前",
    "backward_left": "左斜め後ろ",
    "backward_right": "右斜め後ろ",
}

MOVE_VERBS: dict[str, list[Verb]] = {
    "forward": [V_SUSUMU, V_ARUKU, V_IKU, V_SUSUMU],
    "backward": [V_SAGARU, V_MODORU, V_SAGARU],
    "left": [V_SUSUMU, V_IDOU, V_YORU, V_ARUKU],
    "right": [V_SUSUMU, V_IDOU, V_YORU, V_ARUKU],
    "forward_left": [V_SUSUMU, V_IDOU, V_ARUKU],
    "forward_right": [V_SUSUMU, V_IDOU, V_ARUKU],
    "backward_left": [V_SAGARU, V_IDOU],
    "backward_right": [V_SAGARU, V_IDOU],
}

SPEED_WORDS: dict[str, dict[str, list[str]]] = {
    "polite": {
        "slow": ["ゆっくり", "ゆっくりと", "ゆっくりめに"],
        "normal": ["", "", "", "", "", "普通の速さで"],
        "fast": ["速く", "急いで", "素早く"],
    },
    "kansai": {
        "slow": ["ゆっくり", "ゆっくりめに", "のんびり"],
        "normal": ["", "", "", "", "", "普通の速さで"],
        "fast": ["はよ", "速く", "パッと", "急いで"],
    },
    "short": {"slow": ["ゆっくり"], "normal": [""], "fast": ["速く"]},
}

# ----------------------------------------------------------------------------------------- flips

FLIP_WORDS: dict[str, list[str]] = {
    "backflip": ["バク転", "バックフリップ", "後方宙返り", "バク転"],
    "frontflip": ["前方回転", "前転", "フロントフリップ", "前方宙返り", "ハンドスプリング"],
    "sideflip_left": ["左側転", "左へのサイドフリップ", "左回りの側転"],
    "sideflip_right": ["右側転", "右へのサイドフリップ", "右回りの側転"],
}

FLIP_VERBS: dict[str, list[Verb]] = {
    "backflip": [V_SURU, V_YARU, V_KIMERU],
    "frontflip": [V_SURU, V_YARU, V_KIMERU],
    "sideflip_left": [V_SURU, V_YARU],
    "sideflip_right": [V_SURU, V_YARU],
}

# A flip fired out of the preceding move, without stopping. The instruction has to say so, or the
# reader (and the model) would take it for the ordinary stop-then-flip.
RUNNING_WORDS: dict[str, list[str]] = {
    "polite": ["そのまま", "止まらずに", "走りながら", "その勢いで", "走った勢いで", "進んだまま", "止まらないで"],
    "kansai": ["そのまま", "止まらんで", "走りながら", "その勢いで", "勢いのまま", "走った勢いで", "進んだまま", "止まらんと"],
    "short": ["そのまま", "走りながら", "止まらんと"],
}

STANCE_WORDS: dict[str, list[str]] = {
    "handstand": ["倒立", "逆立ち", "倒立"],
    "hindstand": ["後ろ足立ち", "二足立ち", "後ろ足だけで立つやつ"],
}

TURN_DIR_WORDS: dict[str, list[str]] = {
    "left": ["左に", "左へ", "左回りに", "反時計回りに"],
    "right": ["右に", "右へ", "右回りに", "時計回りに"],
}

TURN_VERBS = [V_MAWARU, V_SENKAI, V_MUKI, V_MAWARU]
STOP_VERBS = [V_TOMARU, V_TEISHI, V_MATSU]

# Deliberately without 「そのまま」, which belongs to RUNNING_WORDS above. It is the most natural
# way to say "keep going and then do it", and having it also mean "without moving" made it the one
# word that could not be read: in the v3 dataset 358 instructions containing そのまま carried a
# running flip and 141 a standing one, and the model resolved the ambiguity by dropping `running`.
# A move or a stance with no length. The grammar gained these when 「前に進んで」 with no amount
# stopped meaning "pick a number" and started meaning "until something replaces it"; without the
# words for them, 23% of the sampled programs could not be put into Japanese at all.
OPEN_MOVE_WORDS: dict[str, list[str]] = {
    "polite": ["ずっと", "そのまま", "止まるまで", "しばらく", ""],
    "kansai": ["ずっと", "そのまま", "止まるまで", "しばらく", ""],
    "short": ["ずっと", ""],
}
OPEN_STANCE_WORDS: dict[str, list[str]] = {
    "polite": ["ずっと", "そのまま", "しばらく"],
    "kansai": ["ずっと", "そのまま", "しばらく"],
    "short": ["ずっと"],
}

IN_PLACE = ["その場で", "そこで", ""]

# -------------------------------------------------------------------------------------- endings

# The bare 「、」 is listed twice because it is how people actually join two actions -- 「3m前に
# 進んで、そのまま前転して」 -- and it was the one join the bank could not write: every clause was
# followed by a connective word. A model trained only on the wordy joins reads the plain one as
# unfamiliar, and v3/v4 both answered unfamiliar sentences with a question instead of a program.
CONNECTIVES: dict[str, list[str]] = {
    "polite": ["、", "、", "から、", "。そのあと、", "。次に、", "。それから、", "。続けて、", "、そのあとに"],
    "kansai": ["、", "、", "から、", "、そんで", "、ほんで", "。そのあと、", "、そっから", "。ほんで次に、"],
    "short": ["、", "、", "。", "、そのあと"],
}

# The bare ending -- 「…前方回転して。」 with no 「ください」 and no 「や」 -- is the plainest way to
# end a typed instruction and was the last shape the bank could not write: only 551 of ~7250 rows
# ended that way, nearly all of them Kansai, so a sentence with neutral vocabulary and a plain
# ending was outside the data. Measured on v5, that ending alone decided the answer: every step of
# 「前方へ約3m行って。それから、走りながら前方回転を1回してください。」 ->
# 「3メートル前に進んで、走りながら前方回転してください。」 was answered with the program, and only
# swapping the last 「ください」 for 「。」 turned it into a question.
ENDINGS: dict[str, list[str]] = {
    "polite": ["ください。", "ください。", "もらえますか。", "ほしいです。", "みましょう。", "くれますか。", "ください!", "。", "。", "!"],
    "kansai": ["や。", "な。", "。", "。", "。", "くれる?", "みて。", "や!", "ほしいねん。", "んか。"],
    "short": ["。", "。", "!"],
}

OPENERS: dict[str, list[str]] = {
    "polite": ["承知しました。", "かしこまりました。", "了解しました。", "はい、", "わかりました。"],
    "kansai": ["ええで、", "オッケー、", "ほな、", "よっしゃ、", "あいよ、", "了解、", "任しとき、"],
    "short": ["了解。", "おっけー。", "はいよ。", "うい、"],
}

REPLY_TAIL: dict[str, list[str]] = {
    "polite": ["", "", "", "以上です。"],
    "kansai": ["", "", "", "いくで!", "ほな行くわ。"],
    "short": ["", "", "いくで。"],
}


def pick(options: list[str], rng: random.Random) -> str:
    return rng.choice(options)


# ------------------------------------------------------------------------------ step -> clause

_WORD_STYLE = {"polite": "polite", "kansai": "kansai", "short": "kansai"}


def _speed_word(speed: str, style: str, rng: random.Random) -> str:
    table = style if style in SPEED_WORDS else _WORD_STYLE.get(style, "polite")
    return rng.choice(SPEED_WORDS[table][speed])


def _amount_move(step: dict, style: str, rng: random.Random) -> str:
    if step.get("distance_m") is not None:
        return distance_word(step["distance_m"], style, rng)
    if step.get("duration_s") is not None:
        return seconds_word(step["duration_s"], style, rng)
    return rng.choice(OPEN_MOVE_WORDS[style])


def move_clause(step: dict, style: str, rng: random.Random, running: bool = False,
                words: dict | None = None) -> Clause:
    """The three orders a person uses for the same thing.

    「ゆっくり前に3m」 / 「ゆっくり3m前に」 / 「3mゆっくり前に」 all say it, and the bank used to write
    the first almost always. Word order is not decoration here: it is the main thing that made a
    typed sentence look unlike the training data.
    """
    dirw = rng.choice(DIR_WORDS[step["dir"]])
    speed = _speed_word(step.get("speed", "normal"), style, rng)
    amount = _amount_move(step, style, rng)
    verb = rng.choice(MOVE_VERBS[step["dir"]])
    order = rng.random()
    if order < 0.22:
        body = f"{speed}{amount}{dirw}"
    elif order < 0.40:
        body = f"{amount}{speed}{dirw}"
    else:
        body = f"{speed}{dirw}{amount}"
    return Clause(body, verb)


def turn_clause(step: dict, style: str, rng: random.Random, running: bool = False,
                words: dict | None = None) -> Clause:
    dirw = rng.choice(TURN_DIR_WORDS[step["dir"]])
    speed = _speed_word(step.get("speed", "normal"), style, rng)
    place = rng.choice(IN_PLACE) if rng.random() < 0.5 else ""
    if step.get("angle_deg") is not None:
        amount = angle_word(step["angle_deg"], style, rng)
        verb = V_SURU if ("回転" in amount or "周" in amount) else rng.choice(TURN_VERBS)
    else:
        amount = seconds_word(step["duration_s"], style, rng)
        verb = rng.choice([V_MAWARU, V_SENKAI, V_MAWARU])
    return Clause(f"{place}{speed}{dirw}{amount}", verb)


def stop_clause(step: dict, style: str, rng: random.Random, running: bool = False,
                words: dict | None = None) -> Clause:
    amount = seconds_word(step.get("duration_s", 1.0), style, rng)
    head = rng.choice(["", "", "そこで", "その場で"])  # not そのまま -- see IN_PLACE
    return Clause(f"{head}{amount}", rng.choice(STOP_VERBS))


def flip_clause(step: dict, style: str, rng: random.Random, running: bool = False,
                words: dict | None = None) -> Clause:
    """One rotation. Whether it is done out of a run is decided by what precedes it.

    The step carries no ``running`` flag any more -- the order does -- so the wording has to look
    back too, through :func:`running_flags`. Saying 「その場で」 about a rotation the robot does at
    speed is a sentence the dataset would be teaching as a lie, and it was in 20% of them.
    """
    word = skill_word(step, rng, words, style)
    count = count_word(1, style, rng)
    if running:
        place = rng.choice(RUNNING_WORDS[style])
    else:
        place = "その場で" if rng.random() < 0.2 else ""
    particle = "を" if count and rng.random() < 0.6 else ""
    verb = rng.choice(FLIP_VERBS[step["kind"]])
    return Clause(f"{place}{word}{particle}{count}", verb)


def _remember(words: dict | None, key: str, choose):
    """Name a skill once per conversation and keep calling it that.

    The instruction and the reply are rendered by separate calls, so without this the person asks
    for 「前方宙返り」 and the robot answers about 「ハンドスプリング」 -- the same rotation under two
    names, one turn apart. ``words`` is one dict per row, shared by every call that writes a line of
    it; the key is the skill and its kind, so a conversation may still contain a backflip and a
    frontflip under their own names, and 「もう一回」 refers back to a word that was actually said.
    """
    if words is None:
        return choose()
    if key not in words:
        words[key] = choose()
    return words[key]


def skill_word(step: dict, rng: random.Random, words: dict | None = None, style: str = "kansai") -> str:
    """The name of one skill on its own -- 「バク転」, 「倒立」, 「左回り90度」.

    Same memo as the clause builders, so a skill named here reads the same as one named in a full
    sentence: the person says 「前方宙返りして!」 and the answer is about 前方宙返り, not 前転.
    """
    skill = step["skill"]
    if skill == "flip":
        choices = FLIP_WORDS[step["kind"]][:1] if style == "short" else FLIP_WORDS[step["kind"]]
        return _remember(words, f"flip:{step['kind']}", lambda: rng.choice(choices))
    if skill == "stance":
        choices = STANCE_WORDS[step["kind"]]
        if style in ("polite", "short"):
            choices = choices[:1] if style == "short" else choices[:2]
        return _remember(words, f"stance:{step['kind']}", lambda: rng.choice(choices))
    if skill == "turn":
        return f"{rng.choice(TURN_DIR_WORDS[step['dir']])}{angle_word(step['angle_deg'], style, rng)}"
    return rng.choice(DIR_WORDS[step["dir"]])


def _amount_stance(step: dict, style: str, rng: random.Random) -> str:
    if step.get("duration_s") is not None:
        return seconds_word(step["duration_s"], style, rng)
    return rng.choice(OPEN_STANCE_WORDS[style])


def stance_clause(step: dict, style: str, rng: random.Random, running: bool = False,
                  words: dict | None = None) -> Clause:
    amount = _amount_stance(step, style, rng)
    word = skill_word(step, rng, words, style)
    place = "その場で" if rng.random() < 0.25 else ""
    particle = rng.choice(["を", "を", ""])
    return Clause(f"{place}{word}{particle}{amount}", rng.choice([V_SURU, V_YARU, V_KEEP, V_SURU]))


_CLAUSE_BUILDERS = {
    "move": move_clause,
    "turn": turn_clause,
    "stop": stop_clause,
    "flip": flip_clause,
    "stance": stance_clause,
}


def running_flags(steps: list[dict], context: dict | None = None) -> list[bool]:
    """Which steps the robot begins while still moving. The compiler's rule, mirrored.

    A flip out of a move is a running one, and a flip straight after a running flip is one too --
    the gait is kept through the gap rather than stopped and restarted. Deciding it on "is the step
    before it a move" alone got [前進, 前転, 前転] wrong: the second rotation is done at speed, and
    the wording called it 「その場で」.

    ``context`` is the step these ones will follow when they are not the whole program -- what is
    under way, or the tail of the queue they are being appended to. The compiler takes the same
    argument for the same reason: a flip added after a queued move is fired out of that move.
    """
    flags: list[bool] = []
    running = False
    for index, step in enumerate(steps):
        before = steps[index - 1] if index else context
        if step["skill"] == "flip":
            running = before is not None and (before["skill"] == "move"
                                              or (before["skill"] == "flip" and running))
        else:
            running = False
        flags.append(running)
    return flags


def clause_for(step: dict, style: str, rng: random.Random, running: bool = False,
               words: dict | None = None) -> Clause:
    return _CLAUSE_BUILDERS[step["skill"]](step, style, rng, running, words)


def clauses_for(steps: list[dict], style: str, rng: random.Random, words: dict | None = None,
                context: dict | None = None) -> list[Clause]:
    """Every step of a program, each told whether the robot is already moving when it starts."""
    flags = running_flags(steps, context)
    return [clause_for(step, style, rng, flags[i], words) for i, step in enumerate(steps)]


# ------------------------------------------------------------------------------- short register


def short_phrase(step: dict, rng: random.Random, words: dict | None = None) -> str:
    """The clipped register: 「前3m」, 「バク転1回」. One name per skill, the shortest one.

    It goes through the row's memo like the full clauses do, so a row written in this register
    cannot answer a 「バク転」 with a 「後方宙返り」 either.
    """
    skill = step["skill"]
    speed = _speed_word(step.get("speed", "normal"), "short", rng)
    if skill == "move":
        amount = _amount_move(step, "short", rng)
        return f"{speed}{DIR_WORDS_SHORT[step['dir']]}{amount}"
    if skill == "turn":
        # "右5秒" would read as a sideways walk, so a timed turn always keeps a rotation word.
        if step.get("angle_deg") is not None:
            return f"{speed}{DIR_WORDS_SHORT[step['dir']]}{angle_word(step['angle_deg'], 'short', rng)}"
        amount = seconds_word(step["duration_s"], "short", rng)
        turn_word = rng.choice([f"{DIR_WORDS_SHORT[step['dir']]}回り", f"{DIR_WORDS_SHORT[step['dir']]}旋回", f"{DIR_WORDS_SHORT[step['dir']]}に旋回"])
        return f"{speed}{turn_word}{amount}"
    if skill == "stop":
        return rng.choice([f"停止{seconds_word(step.get('duration_s', 1.0), 'short', rng)}", f"{seconds_word(step.get('duration_s', 1.0), 'short', rng)}停止"])
    if skill == "flip":
        return f"{skill_word(step, rng, words, 'short')}{count_word(1, 'short', rng)}"
    word = skill_word(step, rng, words, "short")
    amount = _amount_stance(step, "short", rng)
    if step.get("dir"):
        return f"{word}{amount}のまま{DIR_WORDS_SHORT[step['dir']]}へ"
    return f"{word}{amount}"


# ------------------------------------------------------------------------------------ assembling


def _nounable(step: dict) -> bool:
    """Whether this step still says what it is once the verb is dropped.

    A flip or a stance is named by its noun and a move carries a direction, so 「バク転1回。」 and
    「前へ3m。」 are complete. A stop is not -- 「だいたい1秒。」 is one second of nothing -- and
    neither is a timed turn, which without 「回って」 reads as a sideways walk. That is the same
    reason turn_clause keeps a rotation word on a timed turn.
    """
    if step["skill"] in ("flip", "stance", "move"):
        return True
    return step["skill"] == "turn" and step.get("angle_deg") is not None


def instruction(steps: list[dict], style: str, rng: random.Random, words: dict | None = None,
                context: dict | None = None) -> str:
    if not steps:
        return ""
    if rng.random() < IDIOM_RATE:
        # Half the instructions are said the way a person says them rather than assembled out of
        # parts; see the idiom section at the bottom of this file. When no table fits the shape,
        # this falls through to the composition below.
        said = idiom_for(steps, style, rng, words, context)
        if said is not None:
            IDIOM_USE["idiom"] += 1
            return said
    IDIOM_USE["composed"] += 1
    if style == "short":
        body = rng.choice(["、", "、", "。"]).join(short_phrase(s, rng, words) for s in steps)
        return body + rng.choice(ENDINGS["short"])
    clauses = clauses_for(steps, style, rng, words, context)
    out = ""
    for clause in clauses[:-1]:
        out += clause.te() + rng.choice(CONNECTIVES[style])
    if rng.random() < 0.15 and _nounable(steps[-1]):
        # Noun-ended: 「後ろに3秒下がって、バク転1回。」 The verb is dropped and nothing is lost --
        # people write this constantly and the bank never did.
        return out + clauses[-1].body + rng.choice(["。", "。", "!"])
    return out + clauses[-1].te() + rng.choice(ENDINGS[style])


def restate(steps: list[dict], style: str, rng: random.Random, words: dict | None = None,
            context: dict | None = None) -> str:
    """The reply's "what I will do" sentence, in the same register.

    ``words`` is the memo the matching instruction was written with: pass it and the answer names
    the skills the way the person just did.
    """
    if not steps:
        return ""
    if style == "short":
        body = "、".join(short_phrase(s, rng, words) for s in steps)
        return body + rng.choice(["、いくで。", "、やるで。", "。いくで。", "な、いくで。"])
    clauses = clauses_for(steps, style, rng, words, context)
    head = "".join(clause.te() + rng.choice(["、", "、", "から、"]) for clause in clauses[:-1])
    last = clauses[-1]
    if style == "polite":
        return head + last.masu() + "。"
    return head + last.plain() + rng.choice(["で。", "で。", "わ。", "な。"])


# ======================================================================================= idioms
#
# Everything above builds a sentence out of a step: direction word + speed word + amount + verb.
# That is how the bulk of the data is written, and it is also why the bulk of it reads the same:
# the composition can only ever say 「左に180度回って」, never 「回れ右!」.
#
# The tables below are the other half. Each one matches a *shape* of program -- a lone 180° turn,
# a run-up followed by a rotation followed by an open-ended run -- and writes it the way a person
# says it, in words the composition cannot reach. They are the forms the operator will actually
# use, and the ones a model has no way to guess: 回れ右, 左向け左, 一周, 2連発, もういっちょ,
# 止めるまで歩いてて.
#
# Half of the instructions go through here (IDIOM_RATE). The other half stays compositional, which
# is what keeps the surface wide: the idioms are a fixed set, and a dataset made only of them would
# teach the set rather than the language.
#
# Two of these say less than the program does, on purpose, because people do:
#
#   「ちょっと走って前転」   -- the run-up length is not given
#   「少し左向いて」         -- the angle is not given
#
# Both are only drawn when the value is the one a reader would assume (3 s, 30°), so the round-trip
# parser reading it back with its own default lands on the same number rather than on a mismatch.

IDIOM_RATE = 0.75
"""How often an instruction is said idiomatically *when a table fits its shape*.

Roughly two thirds of the programs have a shape some table covers, so this lands about half of all
instructions on an idiom -- the split asked for. ``build_dataset.py`` prints what it actually came
out at; turn this knob, not the tables, to move it."""

IDIOM_USE: dict[str, int] = {"idiom": 0, "composed": 0}
"""Tally of how the instructions in one run were written. Printed by ``build_dataset.py``: the
share is what IDIOM_RATE buys in practice, which is lower than the rate itself because only a
program whose shape has a table can be said idiomatically."""

# ------------------------------------------------------------------------------------ turns

TURN_180_NEUTRAL: dict[str, list[str]] = {
    "polite": ["後ろを向いてください。", "振り向いてください。", "反転してください。", "Uターンしてください。",
               "真後ろを向いてください。", "半回転してください。", "後ろを向いて。", "くるっと反転して。"],
    "kansai": ["後ろ向いて。", "振り向いて!", "反転して。", "くるっと反転して。", "Uターンして。",
               "真後ろ向いて。", "半回転して。", "後ろ向いてや。", "うしろ向いて!"],
    "short": ["後ろ向いて。", "反転。", "Uターン。", "半回転。", "真後ろ向いて。"],
}
"""180° without a side. 「後ろを向いて」 is satisfied by turning either way, which is why these are
used whatever the program's direction says -- and why the round-trip has to accept either."""

TURN_180_RIGHT: dict[str, list[str]] = {
    "polite": ["回れ右してください。", "回れ右!", "回れ右お願いします。", "後ろ向け後ろ!"],
    "kansai": ["回れ右!", "回れ右や!", "回れ右して。", "後ろ向け後ろ!"],
    "short": ["回れ右。", "回れ右!"],
}
"""The drill command. It names a side, so it is only written for a right 180°."""

TURN_90: dict[str, list[str]] = {
    "polite": ["{dj}を向いてください。", "{dj}に曲がってください。", "{dj}折してください。",
               "直角に{dj}を向いてください。", "{dj}を向いて。", "{dj}に曲がって。"],
    "kansai": ["{dj}向いて。", "{dj}に曲がって。", "{dj}折して。", "{dj}向け{dj}!",
               "直角に{dj}や。", "{dj}向いてや。", "{dj}に曲がってや。"],
    "short": ["{dj}向いて。", "{dj}に曲がる。", "{dj}折。", "直角に{dj}。"],
}

TURN_360: dict[str, list[str]] = {
    "polite": ["一周してください。", "その場で一回転してください。", "ぐるっと一周お願いします。",
               "一回転してください。", "ぐるっと一回転して。"],
    "kansai": ["一周して。", "くるっと一周!", "ぐるっと一回転して。", "一回転しといて。", "一周な。"],
    "short": ["一周。", "一回転。", "ぐるっと一周。", "くるっと一回転。"],
}

TURN_270: dict[str, list[str]] = {
    "polite": ["4分の3周してください。", "3/4回転してください。", "4分の3周お願いします。"],
    "kansai": ["4分の3周して。", "3/4回転な。", "4分の3周や。"],
    "short": ["4分の3周。", "3/4回転。"],
}

TURN_45: dict[str, list[str]] = {
    "polite": ["斜め{dj}を向いてください。", "半分だけ{dj}に向いてください。", "斜め{dj}を向いて。"],
    "kansai": ["斜め{dj}向いて。", "半分だけ{dj}に向いて。", "ちょい斜め{dj}や。"],
    "short": ["斜め{dj}。", "斜め{dj}向いて。"],
}

TURN_30: dict[str, list[str]] = {
    "polite": ["少し{dj}を向いてください。", "ちょっとだけ{dj}に向いてください。", "少しだけ{dj}に回ってください。"],
    "kansai": ["ちょっとだけ{dj}向いて。", "少しだけ{dj}に回って。", "ちょい{dj}向いて。"],
    "short": ["少し{dj}。", "ちょい{dj}向く。"],
}
"""No angle in the words. Drawn only for 30°, the smallest one written, so 「ちょっとだけ」 read
back with a reader's default means what it said."""

# ------------------------------------------------------------------------- moves with no end

OPEN_MOVE_IDIOMS: dict[str, list[str]] = {
    "polite": ["{s}{d}{v}ください。", "ずっと{s}{d}{vt}ください。", "止めるまで{s}{d}{vt}ください。",
               "{s}{d}{vs}続けてください。", "そのまま{s}{d}{vt}ください。",
               "ストップと言うまで{s}{d}{vt}ください。", "しばらく{s}{d}{vt}ください。",
               "{s}{d}{vs}続けて。", "止めるまで{s}{d}{vt}。"],
    "kansai": ["{s}{d}{vk}。", "ずっと{s}{d}{vt}。", "止めるまで{s}{d}{vk}。", "{s}{d}{vs}続けて。",
               "そのまま{s}{d}{vk}。", "ストップ言うまで{s}{d}{vt}。", "{s}{d}{vt}や。",
               "{s}{d}{vk}や。", "ずっと{s}{d}{vs}続けてや。"],
    "short": ["{s}{ds}にずっと。", "{s}{ds}に{vs}続けて。", "ずっと{s}{ds}。", "{s}{ds}、止めるまで。",
              "{s}{ds}にそのまま。"],
}

# ------------------------------------------------------- a run-up, a rotation, and keep going

RUN_FLIP_RUN: dict[str, list[str]] = {
    "polite": ["{s}{d}{a}{v}から{f}して、そのまま{vt}ください。",
               "{a}{s}{d}{v}、止まらずに{f}して、そのあとも{vt}ください。",
               "{s}{d}{a}{v}、{f}、そのまま{vs}続けてください。",
               "{a}{s}{d}{v}から{f}を決めて、止めるまで{vt}ください。"],
    "kansai": ["{s}{d}{a}{v}から{f}して、そのまま{vk}。",
               "{a}{s}{d}{v}、止まらんと{f}して、あとはずっと{vt}。",
               "{s}{d}{a}{v}、{f}、そのまま{vs}続けてや。",
               "{a}{s}{d}{v}から{f}決めて、止めるまで{vt}。"],
    "short": ["{s}{ds}{a}、{f}、そのまま{s}{ds}ずっと。",
              "{s}{ds}{a}から{f}、あと{s}{ds}ずっと。"],
}
"""The tail either names the heading again, in which case it names the speed again too, or names
neither and is read as 「the same move, carrying on」. What it may not do is name the direction and
leave the speed out: 「ゆっくり右2秒、右側転、そのまま右ずっと」 reads as a slow run and an
ordinary-speed one, and the program says both are slow."""

RUN_FLIP_RUN_FORWARD: dict[str, list[str]] = {
    "polite": ["{s}{a}走ってから{f}して、そのまま走っててください。",
               "{s}{a}走って、{f}、それから走り続けてください。",
               "{s}助走{a}→{f}→そのまま走っててください。"],
    "kansai": ["{s}{a}走ってから{f}して、あとはずっと走ってて。",
               "{s}{a}走って、{f}、ほんで走り続けて。",
               "{s}{a}走ってから{f}して、そのまま走っといて。"],
    "short": ["{s}{a}走って、{f}、そのまま走り続け。", "{s}走り{a}→{f}→そのまま。",
              "{s}助走{a}→{f}→そのまま。"],
}
"""走る only reads right going forwards; sideways and backwards keep the ordinary verb."""

RUN_FLIP_RUN_VAGUE: dict[str, list[str]] = {
    "polite": ["走りながら{f}してください。そのまま走っててください。",
               "ちょっと走ってから{f}して、そのまま走っててください。",
               "少し助走して{f}、そのまま走り続けてください。",
               "助走をつけて{f}して、それからも走っててください。"],
    "kansai": ["走りながら{f}して、そのまま走っといて。",
               "ちょっと走って{f}、そのまま走ってて。",
               "ちょい助走して{f}、そのまま走り続けて。",
               "助走つけて{f}して、それからも走っといて。"],
    "short": ["走りながら{f}、そのまま走り続け。", "ちょっと走って{f}、そのまま走り続け。"],
}
"""The run-up length is not in the words. Drawn only when it is 3 s -- what a reader assumes, and
what the round-trip parser fills in -- so the sentence still says what the program does."""

RUN_FLIP: dict[str, list[str]] = {
    "polite": ["{s}{d}{a}{v}から、止まらずに{f}してください。", "{a}{s}{d}{v}、そのまま{f}してください。"],
    "kansai": ["{s}{d}{a}{v}から、止まらんと{f}して。", "{a}{s}{d}{v}、そのまま{f}や!"],
    "short": ["{s}{ds}{a}、そのまま{f}。", "{s}{ds}{a}→{f}。"],
}
"""A run-up and a rotation, stopping after it -- ``move → flip`` with nothing following."""

RUN_FLIP_VAGUE: dict[str, list[str]] = {
    "polite": ["助走をつけて{f}してください。", "少し助走してから{f}してください。", "走りながら{f}してください。"],
    "kansai": ["助走つけて{f}して。", "ちょっと走って{f}して。", "走りながら{f}して!"],
    "short": ["助走して{f}。", "走りながら{f}。", "助走→{f}。"],
}
"""Same bargain as RUN_FLIP_RUN_VAGUE: no run-up length, so only for the 3 s one, and only
forwards -- 「助走」 and 「走りながら」 are not what a sideways shuffle is called."""

# --------------------------------------------------------------------- the same skill twice

FLIP_REPEAT: dict[str, list[str]] = {
    "polite": ["{f}を{n}回してください。", "{f}{n}回お願いします。", "{f}を{n}連続でお願いします。",
               "{f}{n}発いってください。", "続けて{f}を{n}回してください。"],
    "kansai": ["{f}{n}回して。", "{f}{n}連発!", "{f}を{n}発!", "連続で{f}{n}回!", "{f}{n}回いこか!",
               "{f}{n}連続でいって!"],
    "short": ["{f}{n}回。", "{f}{n}連発。", "{f}{n}発。", "連続{f}{n}回。"],
}

FLIP_TWICE: dict[str, list[str]] = {
    "polite": ["{f}して、もう一回してください。", "{f}をもう一丁お願いします。", "{f}を2回続けてください。"],
    "kansai": ["{f}して、もういっちょ!", "{f}、もう一回いっとこ!", "{f}ダブルでいって!", "{f}してもっかい!"],
    "short": ["{f}、もういっちょ。", "{f}×2。", "{f}2連。"],
}
"""Only for two: 「もういっちょ」 adds one, and three of them is not how anyone says three."""

# ----------------------------------------------------------------------- stances with no end

OPEN_STANCE_IDIOMS: dict[str, list[str]] = {
    "polite": ["ずっと{st}しててください。", "{st}したままでいてください。", "{st}をキープしてください。",
               "止めるまで{st}しててください。", "{st}のままでいてください。", "{st}キープでお願いします。"],
    "kansai": ["ずっと{st}しとって。", "{st}したままでおって。", "{st}キープ!", "止めるまで{st}してて。",
               "{st}のままでいてや。", "ずっと{st}してて。"],
    "short": ["ずっと{st}。", "{st}キープ。", "{st}のまま。", "{st}したまま。"],
}

STANCE_IDIOMS: dict[str, list[str]] = {
    "polite": ["{ae}だけ{st}してください。", "{st}を{a}キープしてください。", "{st}{a}お願いします。"],
    "kansai": ["{ae}だけ{st}して。", "{st}{a}キープ!", "{st}を{a}や。", "{a}{st}してみて。"],
    "short": ["{st}{a}。", "{a}{st}。", "{st}{a}キープ。"],
}

# ----------------------------------------------------------------------------- single skills

FLIP_SINGLE: dict[str, list[str]] = {
    "polite": ["{f}を1発お願いします。", "{f}いってください!", "{f}を決めてください!", "{f}お願いします!",
               "{f}、いけますか?", "{f}を見せてください!"],
    "kansai": ["{f}1発!", "{f}いったって!", "{f}決めて!", "{f}いこか!", "{f}いける?", "{f}見せて!",
               "{f}やったろ!", "{f}や!"],
    "short": ["{f}1発。", "{f}!", "{f}いって。", "{f}決めて。"],
}
"""One rotation, from standing. The composition can only write 「バク転を1回してください」; this is
how it is actually called for, and the sequential dialogues ask for one step at a time, so it is
the single most used table here."""

# ------------------------------------------------------------------------------ single move

MOVE_IDIOMS: dict[str, list[str]] = {
    "polite": ["{s}{d}{a}お願いします。", "{ae}だけ{s}{d}{v}ください。", "{s}{d}{ae}だけ{v}ください。"],
    "kansai": ["{s}{d}{a}!", "{ae}だけ{s}{d}{v}。", "{s}{d}{a}やって。", "{s}{d}{a}な。"],
    "short": ["{s}{ds}{a}。", "{a}{s}{ds}。", "{s}{ds}に{a}。"],
}

MOVE_OPEN_TAIL: dict[str, list[str]] = {
    "polite": ["{a}{s}{d}{v}から、あとはずっと{s2}{d2}{v2}ください。",
               "{s}{d}{a}{v}、そのあとは止めるまで{s2}{d2}{v2t}ください。",
               "{a}{s}{d}{v}、それからずっと{s2}{d2}{v2t}ください。"],
    "kansai": ["{a}{s}{d}{v}から、あとはずっと{s2}{d2}{v2k}。",
               "{s}{d}{a}{v}、そのあとは止めるまで{s2}{d2}{v2t}。",
               "{a}{s}{d}{v}、それからずっと{s2}{d2}{v2k}。"],
    "short": ["{s}{ds}{a}、あと{s2}{ds2}ずっと。", "{s}{ds}{a}から{s2}{ds2}ずっと。"],
}
"""A measured move and then one with no end -- 「3m進んでから、あとはずっと右に」."""

MOVE_STANCE: dict[str, list[str]] = {
    "polite": ["{a}{s}{d}{v}から{st}してください。", "{s}{d}{a}{v}、そのまま{st}お願いします。"],
    "kansai": ["{a}{s}{d}{v}から{st}して。", "{s}{d}{a}{v}、そのまま{st}や!"],
    "short": ["{s}{ds}{a}、{st}。", "{s}{ds}{a}から{st}。"],
}
STANCE_MOVE: dict[str, list[str]] = {
    "polite": ["{st}してから{a}{s}{d}{v}ください。", "{st}のあと{a}{s}{d}{v}ください。"],
    "kansai": ["{st}してから{a}{s}{d}{v}。", "{st}のあと{a}{s}{d}{v}。"],
    "short": ["{st}、{s}{ds}{a}。", "{st}してから{s}{ds}{a}。"],
}

# ------------------------------------------------------------------------- two-step combines

TURN_MOVE: dict[str, list[str]] = {
    "polite": ["{turn}{a}{s}{d}{v}ください。", "{turn}それから{a}{s}{d}{v}ください。"],
    "kansai": ["{turn}{a}{s}{d}{v}。", "{turn}ほんで{a}{s}{d}{v}。"],
    "short": ["{turn}{s}{ds}{a}。", "{turn}ほんで{s}{ds}{a}。"],
}
MOVE_TURN: dict[str, list[str]] = {
    "polite": ["{s}{d}{a}{v}から{turn}", "{a}{s}{d}{v}、そのあと{turn}"],
    "kansai": ["{s}{d}{a}{v}から{turn}", "{a}{s}{d}{v}、ほんで{turn}"],
    "short": ["{s}{ds}{a}、{turn}", "{s}{ds}{a}から{turn}"],
}
FLIP_MOVE: dict[str, list[str]] = {
    "polite": ["{f}してから{a}{s}{d}{v}ください。", "{f}を決めて、そのあと{a}{s}{d}{v}ください。"],
    "kansai": ["{f}してから{a}{s}{d}{v}。", "{f}決めて、そのあと{a}{s}{d}{v}。"],
    "short": ["{f}、{s}{ds}{a}。", "{f}してから{s}{ds}{a}。"],
}
"""The turn half of these is written by the turn tables, so 「回れ右して3m進んで」 exists as one
sentence rather than only as 「右に180度回って、3m進んで」."""


# ----------------------------------------------------------------------------- the machinery


def _exact_amount(step: dict) -> str:
    """The amount with no 「くらい」 on it -- 「3m」, 「5秒」.

    「だいたい8秒だけ歩いて」 is two hedges fighting: 〜だけ pins the number down and the
    approximator loosens it again. Templates that use だけ take this instead.
    """
    if step.get("distance_m") is not None:
        return f"{_n(step['distance_m'])}m"
    if step.get("duration_s") is not None:
        return f"{_n(step['duration_s'])}秒"
    return ""


def _move_fields(step: dict, style: str, rng: random.Random) -> dict:
    verb = rng.choice(MOVE_VERBS[step["dir"]])
    return {"s": _speed_word(step.get("speed", "normal"), style, rng),
            "d": rng.choice(DIR_WORDS[step["dir"]]),
            "ds": DIR_WORDS_SHORT[step["dir"]],
            "a": _amount_move(step, style, rng) if not _is_open_move(step) else "",
            "ae": _exact_amount(step),
            "v": verb.te, "vs": verb.stem, "vt": verb.teru, "vk": verb.toku}


def _is_open_move(step: dict) -> bool:
    return step["skill"] == "move" and step.get("duration_s") is None and step.get("distance_m") is None


def _is_open_stance(step: dict) -> bool:
    return step["skill"] == "stance" and step.get("duration_s") is None


def _stance_with_amount(step: dict, style: str, rng: random.Random, words: dict | None) -> str:
    """「倒立5秒」 as one piece, for the templates that carry a stance inside a longer sentence."""
    word = skill_word(step, rng, words, style)
    if _is_open_stance(step):
        return f"ずっと{word}"
    return f"{word}{_amount_stance(step, style, rng)}"


def _turn_tables(step: dict, rng: random.Random) -> list[dict] | None:
    """Which idiom tables can say this turn. None when no idiom covers the angle."""
    angle = step.get("angle_deg")
    if angle is None:
        return None
    if angle == 180.0:
        tables = [TURN_180_NEUTRAL, TURN_180_NEUTRAL]
        if step["dir"] == "right":
            tables += [TURN_180_RIGHT, TURN_180_RIGHT]
        return tables
    return {90.0: [TURN_90], 360.0: [TURN_360], 270.0: [TURN_270],
            45.0: [TURN_45], 30.0: [TURN_30]}.get(angle)


def _turn_idiom(step: dict, style: str, rng: random.Random) -> str | None:
    tables = _turn_tables(step, rng)
    if tables is None:
        return None
    dj = "左" if step["dir"] == "left" else "右"
    return rng.choice(rng.choice(tables)[style]).format(dj=dj)


def idiom_for(steps: list[dict], style: str, rng: random.Random, words: dict | None = None,
              context: dict | None = None) -> str | None:
    """The whole program in one idiomatic sentence, or None when no table fits its shape.

    Matching is on the shape of the *program*, not on the words, so an idiom can never say
    something the program does not: every template spells out the direction, the speed and the
    amount, except the two marked above that are drawn only at their default value.
    """
    kinds = [step["skill"] for step in steps]
    running = running_flags(steps, context)

    # --- one step ------------------------------------------------------------------------
    if len(steps) == 1:
        step = steps[0]
        if step["skill"] == "turn":
            return _turn_idiom(step, style, rng)
        if _is_open_move(step):
            return rng.choice(OPEN_MOVE_IDIOMS[style]).format(**_move_fields(step, style, rng))
        if _is_open_stance(step):
            return rng.choice(OPEN_STANCE_IDIOMS[style]).format(st=skill_word(step, rng, words, style))
        if step["skill"] == "stance":
            return rng.choice(STANCE_IDIOMS[style]).format(
                st=skill_word(step, rng, words, style), a=_amount_stance(step, style, rng),
                ae=_exact_amount(step))
        if step["skill"] == "move" and not running[0]:
            return rng.choice(MOVE_IDIOMS[style]).format(**_move_fields(step, style, rng))
        if step["skill"] == "flip" and not running[0]:
            return rng.choice(FLIP_SINGLE[style]).format(f=skill_word(step, rng, words, style))
        return None

    # --- a run-up, the rotation out of it, and what follows ---------------------------------
    if kinds[:2] == ["move", "flip"] and not _is_open_move(steps[0]):
        run, flip = steps[0], steps[1]
        fields = _move_fields(run, style, rng)
        fields["f"] = skill_word(flip, rng, words, style)
        tail = steps[2] if len(steps) > 2 else None
        carries_on = (tail is not None and len(steps) == 3 and _is_open_move(tail)
                      and tail["dir"] == run["dir"] and tail.get("speed") == run.get("speed"))
        # 「走りながら」 drops the speed word as well as the length, so it is only right for the
        # ordinary pace: 「ちょっと走って」 about a slow walk says the wrong thing twice.
        vague_ok = (run["dir"] == "forward" and run.get("duration_s") == 3.0
                    and run.get("speed", "normal") == "normal")
        if carries_on:
            tables = [RUN_FLIP_RUN]
            if run["dir"] == "forward":
                tables += [RUN_FLIP_RUN_FORWARD]
            if vague_ok:
                tables += [RUN_FLIP_RUN_VAGUE]
            return rng.choice(rng.choice(tables)[style]).format(**fields)
        if tail is None:
            tables = [RUN_FLIP] + ([RUN_FLIP_VAGUE] if vague_ok else [])
            return rng.choice(rng.choice(tables)[style]).format(**fields)
        return None

    # --- the same rotation twice or three times ---------------------------------------------
    if set(kinds) == {"flip"} and len({step["kind"] for step in steps}) == 1 and not running[0]:
        table = FLIP_TWICE if len(steps) == 2 and rng.random() < 0.5 else FLIP_REPEAT
        return rng.choice(table[style]).format(f=skill_word(steps[0], rng, words, style),
                                               n=len(steps))

    # --- a measured move and then one with no end ---------------------------------------------
    if len(steps) == 2 and kinds == ["move", "move"] and not _is_open_move(steps[0]) and _is_open_move(steps[1]):
        fields = _move_fields(steps[0], style, rng)
        tail = _move_fields(steps[1], style, rng)
        fields.update({"s2": tail["s"], "d2": tail["d"], "ds2": tail["ds"],
                       "v2": tail["v"], "v2t": tail["vt"], "v2k": tail["vk"]})
        return rng.choice(MOVE_OPEN_TAIL[style]).format(**fields)

    # --- a move and a stance, either way round -------------------------------------------------
    if len(steps) == 2 and kinds == ["move", "stance"] and not _is_open_move(steps[0]):
        return rng.choice(MOVE_STANCE[style]).format(
            st=_stance_with_amount(steps[1], style, rng, words), **_move_fields(steps[0], style, rng))
    if len(steps) == 2 and kinds == ["stance", "move"] and not _is_open_move(steps[1]):
        return rng.choice(STANCE_MOVE[style]).format(
            st=_stance_with_amount(steps[0], style, rng, words), **_move_fields(steps[1], style, rng))

    # --- a turn and a move, either way round -------------------------------------------------
    if len(steps) == 2 and kinds == ["turn", "move"] and not _is_open_move(steps[1]):
        turn = _turn_idiom(steps[0], style, rng)
        if turn is None:
            return None
        return rng.choice(TURN_MOVE[style]).format(turn=turn, **_move_fields(steps[1], style, rng))
    if len(steps) == 2 and kinds == ["move", "turn"] and not _is_open_move(steps[0]):
        turn = _turn_idiom(steps[1], style, rng)
        if turn is None:
            return None
        return rng.choice(MOVE_TURN[style]).format(turn=turn, **_move_fields(steps[0], style, rng))

    # --- a rotation from standing, then off it goes -------------------------------------------
    if len(steps) == 2 and kinds == ["flip", "move"] and not _is_open_move(steps[1]) and not running[0]:
        return rng.choice(FLIP_MOVE[style]).format(f=skill_word(steps[0], rng, words, style),
                                                   **_move_fields(steps[1], style, rng))
    return None
