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
V_TOBU = Verb("跳んで", "跳びます", "跳ぶ")
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
    "jump": ["ジャンプ", "垂直跳び", "ジャンプ"],
}

FLIP_VERBS: dict[str, list[Verb]] = {
    "backflip": [V_SURU, V_YARU, V_KIMERU],
    "frontflip": [V_SURU, V_YARU, V_KIMERU],
    "sideflip_left": [V_SURU, V_YARU],
    "sideflip_right": [V_SURU, V_YARU],
    "jump": [V_SURU, V_YARU, V_TOBU],
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

TOTAL_TIME: dict[str, list[str]] = {
    "polite": ["全部で約{t}秒です。", "所要時間は約{t}秒です。", "だいたい{t}秒かかります。"],
    "kansai": ["全部で{t}秒くらいやな。", "だいたい{t}秒で終わるわ。", "{t}秒くらいかかるで。"],
    "short": ["約{t}秒。", "{t}秒くらい。"],
}

# ------------------------------------------------------------------------------------- cautions
# Keyed by a risk label that ``build_dataset.py`` derives from the capability table, never from the
# wording. "descent" は降りるときだけ危ない、"landing" は着地だけ危ない。

CAUTIONS: dict[str, dict[str, list[str]]] = {
    "frontflip_landing": {
        "polite": [
            "まれに着地に失敗することがあります。",
            "前方回転は着地が乱れることがあります。",
            "着地でこけるかもしれませんが、やってみます。",
        ],
        "kansai": [
            "たまに着地こけることあるけどな。",
            "前転は着地が微妙なときあるから、そこは堪忍な。",
            "着地失敗するかもしれんけど、いっとくわ。",
        ],
        "short": ["たまに着地失敗するけど。", "着地危ないかも。"],
    },
    "handstand_descent": {
        "polite": [
            "ただ、降りるときに四足へうまく戻れないことが多いです。",
            "倒立から降りる動きがまだ苦手で、転ぶかもしれません。",
            "起き上がるところで体勢を崩すかもしれませんが、挑戦します。",
        ],
        "kansai": [
            "ただ、降りるとき転ぶかもしれんで。",
            "倒立から戻るんがまだ下手やから、こけたらごめんな。",
            "降りるとこがまだあかんねん。それでもやってみるわ。",
        ],
        "short": ["降りる時こけるかも。", "降りるの苦手やけど。"],
    },
    "running_frontflip_repeat": {
        "polite": ["走りながらの前方回転は連続すると失敗するので、1回だけ行います。", "走りながらの前方回転は1回までにします。"],
        "kansai": ["走りながらの前転は続けるとこけるから、1回だけにするで。", "走りながらの前転は1回までな。"],
        "short": ["走りながらの前転は1回まで。"],
    },
    "hindstand_descent": {
        "polite": ["降りるときにふらつくことがあります。", "戻るときに少し不安定です。"],
        "kansai": ["降りるときちょっとふらつくけどな。", "戻るとこが少し不安定やねん。"],
        "short": ["降りる時ふらつくかも。"],
    },
}

DECLINE_STEP: dict[str, list[str]] = {
    "polite": [
        "申し訳ありません、{skill}はまだうまく着地できないので、そこは行いません。",
        "{skill}はまだ成功しないため、その部分は省きます。",
        "すみません、{skill}だけはできないので飛ばします。",
    ],
    "kansai": [
        "ごめん、{skill}はまだようできひんくてこけてまうから、そこは飛ばすで。",
        "{skill}はまだ無理やねん。そこだけ抜かすわ。",
        "すまん、{skill}はでけへんから省かせてな。",
    ],
    "short": ["{skill}は無理やから飛ばす。", "{skill}はできひん。"],
}

DECLINE_ALL: dict[str, list[str]] = {
    "polite": [
        "申し訳ありません、{skill}はまだできません。バク転や側転でしたら行えます。",
        "すみません、{skill}は今のところ成功しないので行えません。ほかの動きなら任せてください。",
    ],
    "kansai": [
        "ごめんな、{skill}はまだでけへんねん。バク転や側転やったらいけるで。",
        "{skill}は今のところ無理やわ。ほかのアクロバットなら任しとき。",
    ],
    "short": ["{skill}はまだ無理。", "{skill}はできひん。バク転なら行ける。"],
}

REMAINING: dict[str, list[str]] = {
    "polite": ["残りは行います。", "そのほかは予定どおり行います。", "ほかの動きはこのまま実行します。"],
    "kansai": ["残りはやるで。", "ほかはそのままいくわ。", "あとの分はちゃんとやるからな。"],
    "short": ["あとはやる。", "残りはいくで。"],
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
    return seconds_word(step["duration_s"], style, rng)


def move_clause(step: dict, style: str, rng: random.Random) -> Clause:
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


def turn_clause(step: dict, style: str, rng: random.Random) -> Clause:
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


def stop_clause(step: dict, style: str, rng: random.Random) -> Clause:
    amount = seconds_word(step.get("duration_s", 1.0), style, rng)
    head = rng.choice(["", "", "そこで", "その場で"])  # not そのまま -- see IN_PLACE
    return Clause(f"{head}{amount}", rng.choice(STOP_VERBS))


def flip_clause(step: dict, style: str, rng: random.Random) -> Clause:
    word = rng.choice(FLIP_WORDS[step["kind"]])
    count = count_word(int(step.get("count", 1)), style, rng)
    if step.get("running"):
        place = rng.choice(RUNNING_WORDS[style])
    else:
        place = "その場で" if rng.random() < 0.2 else ""
    particle = "を" if count and rng.random() < 0.6 else ""
    verb = rng.choice(FLIP_VERBS[step["kind"]])
    return Clause(f"{place}{word}{particle}{count}", verb)


def stance_clause(step: dict, style: str, rng: random.Random) -> Clause:
    amount = seconds_word(step.get("duration_s", 5.0), style, rng)
    if step.get("dir"):
        word = rng.choice(STANCE_WORDS[step["kind"]][:2])
        dirw = rng.choice(DIR_WORDS[step["dir"]])
        return Clause(f"{word}のまま{dirw}{amount}", rng.choice([V_SUSUMU, V_ARUKU, V_IDOU]))
    choices = STANCE_WORDS[step["kind"]][:2] if style == "polite" else STANCE_WORDS[step["kind"]]
    word = rng.choice(choices)
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


def clause_for(step: dict, style: str, rng: random.Random) -> Clause:
    return _CLAUSE_BUILDERS[step["skill"]](step, style, rng)


# ------------------------------------------------------------------------------- short register


def short_phrase(step: dict, rng: random.Random) -> str:
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
        count = count_word(int(step.get("count", 1)), "short", rng)
        running = rng.choice(RUNNING_WORDS["short"]) if step.get("running") else ""
        return f"{running}{FLIP_WORDS[step['kind']][0]}{count}"
    word = STANCE_WORDS[step["kind"]][0]
    amount = seconds_word(step.get("duration_s", 5.0), "short", rng)
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


def instruction(steps: list[dict], style: str, rng: random.Random) -> str:
    if not steps:
        return ""
    if style == "short":
        body = rng.choice(["、", "、", "。"]).join(short_phrase(s, rng) for s in steps)
        return body + rng.choice(ENDINGS["short"])
    clauses = [clause_for(s, style, rng) for s in steps]
    out = ""
    for clause in clauses[:-1]:
        out += clause.te() + rng.choice(CONNECTIVES[style])
    if rng.random() < 0.15 and _nounable(steps[-1]):
        # Noun-ended: 「後ろに3秒下がって、バク転1回。」 The verb is dropped and nothing is lost --
        # people write this constantly and the bank never did.
        return out + clauses[-1].body + rng.choice(["。", "。", "!"])
    return out + clauses[-1].te() + rng.choice(ENDINGS[style])


def restate(steps: list[dict], style: str, rng: random.Random) -> str:
    """The reply's "what I will do" sentence, in the same register."""
    if not steps:
        return ""
    if style == "short":
        body = "、".join(short_phrase(s, rng) for s in steps)
        return body + rng.choice(["、いくで。", "、やるで。", "。いくで。", "な、いくで。"])
    clauses = [clause_for(s, style, rng) for s in steps]
    head = "".join(clause.te() + rng.choice(["、", "、", "から、"]) for clause in clauses[:-1])
    last = clauses[-1]
    if style == "polite":
        return head + last.masu() + "。"
    return head + last.plain() + rng.choice(["で。", "で。", "わ。", "な。"])
