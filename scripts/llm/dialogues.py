"""Multi-turn rows: one program, cut into a conversation.

A program is a plan; a conversation is that plan arriving in pieces, or being changed halfway. Both
are made here by cutting a sampled program, so every turn's target is decided by the program rather
than by reading the Japanese. There are two ways to cut, and they make different conversations:

**By time** -- compile the program, pick a moment, and speak over it. What the person sees is the
queue as it stands at that moment: finished steps gone, the step under way shortened to what is
left, the rest untouched. This is where interrupting, inserting, appending and small talk live,
because all of them need something to already be running.

    [右側転, 左回90°, 遅後3s]  at t=3.0s  ->  queue [左回38°, 遅後3s]
    「バク転して!」             ->  [バク転, 左回38°, 遅後3s]

**By step** -- hand the program over one piece at a time, each from a standstill. This is how most
real instructing goes, and it is the only way to build a reference to an earlier turn: the words
「バク転して」 stay in the history, so 「もう一回」 and 「2つ前の技」 have something to point at.

    [バク転, バク転, 倒立8s]
      「バク転して」    -> [バク転]
      「もう一回」      -> [バク転]        <- the repeat is in the program, not invented
      「倒立8秒」       -> [倒立8s]

Wording lives in the tables below; nothing in the scenarios invents Japanese. What the model is
being taught -- copy the queue back, put a step in front of it, add one after it, empty it -- is
decided from the cut, never from the text.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Callable

import negatives as ng
import phrasebank as pb
from unitree_rl_lab.program import CompilerConfig, ProgramError, compile_program, program_from_json
from unitree_rl_lab.program.grammar import JA

# ==================================================================================== wording

STOP_IN = {
    "polite": ["止まってください。", "ストップ!", "待ってください。", "そこで止まって。", "一旦止まってください。", "とまってください。"],
    "kansai": ["あ、ストップ!", "止まって!", "待って待って!", "ちょい待ち!", "やめて!", "そこでストップ。", "待った。",
               "待った待った!", "ストップや!", "止まれ止まれ!", "とまって!", "とまれ!", "はよとまって!", "とめて!"],
    "short": ["ストップ。", "止まれ。", "待て。", "とまれ。", "とめて。"],
}
STOP_RE = {
    "polite": ["止まります。", "はい、ここで止まります。", "停止します。残りは取り消しました。"],
    "kansai": ["止まるで。", "はいよ、ここで止まるわ。", "おっけー、やめとく。残りは無しな。"],
    "short": ["止まる。", "ストップ。"],
}
STOPPED_ALREADY_RE = {
    "polite": ["今は止まっています。", "もう止まっていますよ。", "はい、待機中です。"],
    "kansai": ["もう止まっとるで。", "今は動いてへんよ。", "止まってる止まってる。"],
    "short": ["止まってる。", "もう止まっとる。"],
}

NOW_IN = {
    "polite": ["{what}してください!", "そのまま{what}して!", "{what}お願いします!"],
    "kansai": ["{what}して!", "{what}!", "そのまま{what}いって!", "{what}やって!"],
    "short": ["{what}!", "{what}。"],
}
NOW_RE = {
    "polite": ["{what}します!", "はい、{what}。", "{what}、いきます。"],
    "kansai": ["{what}いくで!", "そのまま{what}や!", "{what}、いくわ!"],
    "short": ["{what}いくで。", "そのまま{what}。"],
}

APPEND_IN = {
    "polite": ["終わったら、{inst}", "それが済んだら、{inst}", "そのあとに{inst}"],
    "kansai": ["終わったら{inst}", "それ済んだら{inst}", "そのあと{inst}", "ついでに終わったら{inst}"],
    "short": ["終わったら{inst}", "そのあと{inst}"],
}
APPEND_RE = {
    "polite": ["はい、今の動きが終わってから{restate}", "承知しました。終わり次第、{restate}"],
    "kansai": ["おっけー、今のが終わったら{restate}", "了解、終わってから{restate}"],
    "short": ["終わったら{restate}"],
}

CORRECTION_IN = {
    "polite": ["やっぱり、{inst}", "変更です。{inst}", "それはやめて、{inst}"],
    "kansai": ["やっぱ{inst}", "ちゃう、{inst}", "それやめて、{inst}", "変更!{inst}"],
    "short": ["やっぱ{inst}", "変更、{inst}"],
}
CORRECTION_RE = {
    "polite": ["はい、切り替えます。{restate}", "承知しました。今の動きは止めて、{restate}"],
    "kansai": ["おっけー、切り替えるで。{restate}", "了解、今のはやめて{restate}"],
    "short": ["切り替え。{restate}"],
}

# Referring back to something already said. The referent is in the history as the person's own
# words, which is why this works without the past programs being repeated in the prompt.
RECALL_LAST_IN = {
    "polite": ["もう一回お願いします。", "もう一度同じものを。", "同じのをもう一回。", "さっきのをもう一度。",
               "もういっかいお願いします。", "今のをもう一回。", "今のをもう一度お願いします。",
               "いまのをもう一回お願いします。", "同じものをもう一度。", "それをもう一回。",
               "今のやつ、もう一回お願いします。", "リピートしてください。", "もう一回やってください。"],
    "kansai": ["もう一回やって。", "もっかい。", "さっきのもう一回!", "同じやつもう一回!", "もっぺん。",
               "アンコール!", "もういっかい!", "もういっかいやって!", "いまのまた!", "今のまた!",
               "いまのやつもう一回!", "今のをもっかい!", "それもう一回!", "同じのまた!", "おかわり!",
               "さっきのやつまた!", "もっかいやって!", "もう一丁!", "今のもっぺん!", "リピートして!"],
    "short": ["もう一回。", "もっかい。", "同じの。", "もういっかい。", "いまのまた。", "今のまた。",
              "リピート。", "もう一丁。", "同じのまた。", "それまた。"],
}
"""Pointing at the turn just before. Every one of these has to be in the data, because the operator
types whichever comes to hand and the surface is all the model has: 「もういっかい」 in kana and
「いまのまた」 were both absent from the first cut, and a form that is absent is a form that fails."""

RECALL_BACK_IN = {
    "polite": ["2つ前のをもう一度お願いします。", "その前のものをもう一回。", "2個前のをやってください。",
               "ひとつ前じゃなくて、その前のをお願いします。", "2つ前のやつをもう一度。"],
    "kansai": ["2つ前のやつもう一回。", "その前のやって。", "2個前のをもっかい。", "さっきの前のやつ!",
               "2つ前のやつまた!", "その前のやつもっかい!"],
    "short": ["2つ前の。", "その前の。", "2個前の。"],
}
RECALL_NAMED_IN = {
    "polite": ["さっきの{what}をもう一度お願いします。", "例の{what}をもう一回。"],
    "kansai": ["さっきの{what}もう一回。", "例の{what}やって。", "{what}、さっきのやつもっかい。"],
    "short": ["さっきの{what}。", "{what}もう一回。"],
}
RECALL_RE = {
    "polite": ["はい、もう一度行います。", "承知しました。同じものを。", "もう一度やります。"],
    "kansai": ["ほな、もう一回いくで。", "おっけー、同じやつな。", "了解、もっかいやるわ。"],
    "short": ["もう一回、いくで。", "同じの、いくで。"],
}

# Not everything said while standing still is an instruction.
FILLER_IN = {
    "polite": ["なるほど。", "そうですか。", "ふむ。", "へえ。", "ふーん。", "ああ。"],
    "kansai": ["なるほど。", "なるほどな。", "そうか。", "ふーん。", "へえ。", "ほんまか。", "おお。", "あー。"],
    "short": ["ふーん。", "へえ。", "なるほど。", "そうか。"],
}
FILLER_RE = {
    "polite": ["はい。次のご指示をお待ちしています。", "何かあればお申し付けください。", "待機しています。"],
    "kansai": ["せやろ。次どうする?", "まあな。次の指示待っとるで。", "ほんで、次は?", "待っとるで。"],
    "short": ["待機中。", "次どうする?", "待っとる。"],
}

# ==================================================================================== helpers


def _pick(table: dict, style: str, rng: random.Random) -> str:
    return rng.choice(table[style])


def _split_for(key: str) -> str:
    return "eval" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


MOVE_NOUN = {"forward": ["前進", "前に進むやつ"], "backward": ["後退", "バック", "下がるやつ"],
             "left": ["左移動", "左に寄るやつ"], "right": ["右移動", "右に寄るやつ"]}
"""A move named as a *thing*, for 「さっきの前進もう一回」. 「さっきの前」 is not a thing."""


def _what(step: dict, rng: random.Random, words: dict | None = None) -> str:
    """How a person names one step in a few words -- 「バク転」, 「左回り90度」, 「前進」.

    Through the row's memo, so 「さっきのバク転」 names it the way the earlier turn did.
    """
    if step["skill"] in ("flip", "stance"):
        return pb.skill_word(step, rng, words)
    if step["skill"] == "turn":
        return f"{JA[step['dir']]}回り{step['angle_deg']:g}度"
    return rng.choice(MOVE_NOUN[step["dir"]])


def queue_at(program: list[dict], timeline, at: float) -> list[dict]:
    """The queue as it stands ``at`` seconds in: finished steps gone, the current one shortened.

    Shortening is done here so the model never has to: it is handed 「あと38度」 and copies it
    straight back. A flip whose window has opened is dropped -- it has been committed to, and
    re-issuing it would fire it twice.
    """
    out: list[dict] = []
    for index, step in enumerate(program):
        parts = [s for s in timeline.segments if s.step_index == index]
        if not parts:
            continue
        t0, t1 = min(s.t0 for s in parts), max(s.t1 for s in parts)
        if t1 <= at:
            continue
        if t0 > at:
            out.append(dict(step))
            continue
        if step["skill"] == "flip":
            continue
        fraction = (t1 - at) / (t1 - t0)
        shortened = dict(step)
        for key in ("duration_s", "distance_m", "angle_deg"):
            if shortened.get(key) is not None:
                shortened[key] = round(shortened[key] * fraction, 2)
        out.append(shortened)
    return out


def moment(program: list[dict], timeline, rng: random.Random, want: str) -> float | None:
    """A time to speak at: inside a move or a turn with room left, or inside a flip window."""
    options = []
    for seg in timeline.segments:
        if seg.step_index is None:
            continue
        if want == "flip" and seg.kind == "flip":
            options.append((seg, 0.15, 0.15))
        elif want == "moving" and seg.kind in ("move", "turn") and seg.duration >= 1.5:
            options.append((seg, 0.4, 0.6))
        elif want == "move" and seg.kind == "move" and seg.duration >= 2.0:
            options.append((seg, 0.5, 1.0))
    if not options:
        return None
    seg, lo, hi = rng.choice(options)
    return round(rng.uniform(seg.t0 + lo, seg.t1 - hi), 1)


class Turns:
    """Accumulates one conversation as ``(user text, queue, reply, program)`` tuples."""

    def __init__(self) -> None:
        self.items: list[tuple] = []

    def add(self, text: str, queue: list[dict], reply: str, program: list[dict],
            roundtrip: bool = False) -> None:
        self.items.append((text, queue, reply, program, roundtrip))


# ==================================================================================== scenarios
#
# Each takes the program, its timeline, a style and the shared helpers, and returns the turns or
# None when the program does not suit it (nothing long enough to interrupt, no repeat to refer to).

def _opening(program, style, rng, ctx) -> Turns:
    """Turn one: the whole program asked for from a standstill."""
    turns = Turns()
    turns.add(ctx.say(program, style, rng), [], ctx.reply(program, style, rng), program, True)
    return turns


def sc_stop(program, timeline, style, rng, ctx):
    at = moment(program, timeline, rng, "moving")
    if at is None:
        return None
    turns = _opening(program, style, rng, ctx)
    turns.add(_pick(STOP_IN, style, rng), queue_at(program, timeline, at), _pick(STOP_RE, style, rng), [])
    return turns


def sc_now_flip(program, timeline, style, rng, ctx):
    """A skill called for mid-run: it goes in front of what was queued, and the run carries on."""
    at = moment(program, timeline, rng, "move")
    if at is None:
        return None
    queue = queue_at(program, timeline, at)
    if not queue:
        return None
    step = {"skill": "flip", "kind": rng.choice(ctx.flip_kinds)}
    turns = _opening(program, style, rng, ctx)
    word = pb.skill_word(step, rng, ctx.words)
    turns.add(_pick(NOW_IN, style, rng).format(what=word), queue,
              _pick(NOW_RE, style, rng).format(what=word), [step] + queue)
    return turns


def sc_now_turn(program, timeline, style, rng, ctx):
    at = moment(program, timeline, rng, "moving")
    if at is None:
        return None
    queue = queue_at(program, timeline, at)
    if not queue:
        return None
    step = {"skill": "turn", "dir": rng.choice(("left", "right")), "angle_deg": rng.choice((45.0, 90.0, 180.0))}
    word = f"{JA[step['dir']]}に{step['angle_deg']:g}度"
    turns = _opening(program, style, rng, ctx)
    turns.add(_pick(NOW_IN, style, rng).format(what=word), queue,
              _pick(NOW_RE, style, rng).format(what=word), [step] + queue)
    return turns


def sc_append(program, timeline, style, rng, ctx):
    at = moment(program, timeline, rng, "moving")
    if at is None:
        return None
    queue = queue_at(program, timeline, at)
    if not queue:
        return None
    last = queue[-1]
    if last["skill"] in ("move", "stance") and last.get("duration_s") is None and last.get("distance_m") is None:
        # Nothing can be queued behind a step with no end: it runs until something replaces it, so
        # 「終わったら〜も」 has no "afterwards" to attach to and the appended program would not
        # compile. Another program gets the scenario instead.
        return None
    extra = ctx.other(rng, 1)
    turns = _opening(program, style, rng, ctx)
    # The extra steps run after the queue, so the queue's last step is what they follow: a flip
    # appended behind a move is fired out of that move, and must not be described as a standing one.
    turns.add(_pick(APPEND_IN, style, rng).format(inst=ctx.say(extra, style, rng, queue[-1])), queue,
              _pick(APPEND_RE, style, rng).format(restate=ctx.restate(extra, style, rng, queue[-1])),
              queue + extra)
    return turns


def sc_correction(program, timeline, style, rng, ctx):
    at = moment(program, timeline, rng, "moving")
    if at is None:
        return None
    new = ctx.other(rng, 2)
    turns = _opening(program, style, rng, ctx)
    turns.add(_pick(CORRECTION_IN, style, rng).format(inst=ctx.say(new, style, rng)),
              queue_at(program, timeline, at),
              _pick(CORRECTION_RE, style, rng).format(restate=ctx.restate(new, style, rng)), new)
    return turns


def sc_chitchat(program, timeline, style, rng, ctx):
    """Small talk while the robot runs. The queue comes back untouched -- this is the row that
    teaches the difference between "leave it alone" and "stop", which look alike when idle."""
    at = moment(program, timeline, rng, "moving")
    if at is None:
        return None
    queue = queue_at(program, timeline, at)
    if not queue:
        return None
    casual_in, polite_in, casual_re, polite_re = rng.choice(ng.CHITCHAT)
    text, reply = ((ng.pick(polite_in, rng), ng.pick(polite_re, rng)) if style == "polite"
                   else (ng.pick(casual_in, rng), ng.pick(casual_re, rng)))
    turns = _opening(program, style, rng, ctx)
    turns.add(text, queue, reply, queue)
    return turns


def sc_sequential(program, timeline, style, rng, ctx):
    """The program handed over a step at a time, each from a standstill."""
    if len(program) < 2:
        return None
    turns = Turns()
    for step in program:
        piece = [step]
        turns.add(ctx.say(piece, style, rng), [], ctx.reply(piece, style, rng), piece, True)
    return turns


def sc_recall(program, timeline, style, rng, ctx):
    """Step by step, with one turn pointing back at an earlier one instead of naming it again."""
    if len(program) < 2:
        return None
    repeat = [i for i in range(1, len(program)) if program[i] == program[i - 1]]
    # 「2つ前のやつ」 only means something when the one *just* before is a different thing. With
    # [前転, 前転, 前転] the third turn is 「もう一回」, and pointing two back at an identical step
    # would be teaching a reference nobody makes.
    apart = [i for i in range(2, len(program))
             if program[i] in program[:i - 1] and program[i] != program[i - 1]]
    turns = Turns()
    if repeat and rng.random() < 0.6:
        cut = rng.choice(repeat)
        table = RECALL_LAST_IN
    elif apart:
        cut = rng.choice(apart)
        table = RECALL_BACK_IN if program[cut] == program[cut - 2] else RECALL_NAMED_IN
    elif repeat:
        cut = rng.choice(repeat)
        table = RECALL_LAST_IN
    else:
        return None
    for index, step in enumerate(program):
        piece = [step]
        if index == cut:
            text = _pick(table, style, rng)
            if "{what}" in text:
                text = text.format(what=_what(step, rng, ctx.words))
            turns.add(text, [], _pick(RECALL_RE, style, rng), piece)
        else:
            turns.add(ctx.say(piece, style, rng), [], ctx.reply(piece, style, rng), piece, True)
    return turns


def sc_stopped_already(program, timeline, style, rng, ctx):
    turns = _opening(program, style, rng, ctx)
    turns.add(_pick(STOP_IN, style, rng), [], _pick(STOPPED_ALREADY_RE, style, rng), [])
    return turns


def sc_filler(program, timeline, style, rng, ctx):
    turns = _opening(program, style, rng, ctx)
    turns.add(_pick(FILLER_IN, style, rng), [], _pick(FILLER_RE, style, rng), [])
    return turns


SCENARIOS: dict[str, Callable] = {
    "dlg_stop": sc_stop,
    "dlg_now_flip": sc_now_flip,
    "dlg_now_turn": sc_now_turn,
    "dlg_append": sc_append,
    "dlg_correction": sc_correction,
    "dlg_chitchat": sc_chitchat,
    "dlg_sequential": sc_sequential,
    "dlg_recall": sc_recall,
    "dlg_stopped": sc_stopped_already,
    "dlg_filler": sc_filler,
}

SHARE = {
    "dlg_stop": 0.12, "dlg_now_flip": 0.16, "dlg_now_turn": 0.06, "dlg_append": 0.10,
    "dlg_correction": 0.08, "dlg_chitchat": 0.09, "dlg_sequential": 0.10, "dlg_recall": 0.20,
    "dlg_stopped": 0.03, "dlg_filler": 0.06,
}
"""Sums to 1.0. `dlg_recall` carries the most, and 「もう一回」 is why: in real operation the same
skill is asked for again constantly, and the referent is nowhere in the prompt except the person's
own words a turn or two back. `dlg_now_flip` is next -- firing a skill out of a run."""


class Context:
    """What the scenarios need from the builder: a reply writer and a pool of other programs.

    It also carries ``words``, the row's memo of what each skill is called. Every line of one
    conversation is written through it, so the person and the robot use one name per skill for the
    length of the conversation -- which is also what makes 「さっきの{what}」 point at something.
    """

    def __init__(self, reply: Callable, programs: list[list[dict]]):
        self._reply = reply
        self.programs = programs
        self.flip_kinds = list(pb.FLIP_WORDS)
        self.words: dict = {}

    def new_row(self) -> None:
        self.words = {}

    def reply(self, steps: list[dict], style: str, rng: random.Random) -> str:
        return self._reply(steps, style, rng, self.words)

    def say(self, steps: list[dict], style: str, rng: random.Random,
            context: dict | None = None) -> str:
        return pb.instruction(steps, style, rng, self.words, context)

    def restate(self, steps: list[dict], style: str, rng: random.Random,
                context: dict | None = None) -> str:
        return pb.restate(steps, style, rng, self.words, context)

    def other(self, rng: random.Random, max_steps: int) -> list[dict]:
        for _ in range(20):
            candidate = rng.choice(self.programs)
            if 1 <= len(candidate) <= max_steps:
                return candidate
        return [{"skill": "flip", "kind": "backflip"}]


def build(records: list[dict], total: int, compiler: CompilerConfig,
          reply: Callable, rng: random.Random) -> list[dict]:
    """``total`` multi-turn rows drawn across the scenarios."""
    from build_dataset import row  # imported here to keep the module importable on its own

    ctx = Context(reply, [r["program"] for r in records])
    out: list[dict] = []
    seen: set[str] = set()
    for name, share in SHARE.items():
        quota = round(total * share)
        made = tries = 0
        order = list(records)
        rng.shuffle(order)
        while made < quota and tries < quota * 8:
            record = order[tries % len(order)]
            style = pb.STYLES[tries % len(pb.STYLES)]
            tries += 1
            try:
                timeline = compile_program(program_from_json(record["program"]), compiler)
            except ProgramError:
                continue
            ctx.new_row()
            turns = SCENARIOS[name](record["program"], timeline, style, rng, ctx)
            if turns is None:
                continue
            key = json.dumps(turns.items, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(row(f"{name}-{record['id']}-{made:04d}#{style}",
                           _split_for(record["id"]), name, style, turns.items))
            made += 1
        if made < quota:
            print(f"  note: {name} produced {made}/{quota}")
    return out
