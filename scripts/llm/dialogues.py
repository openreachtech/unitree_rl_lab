"""Multi-turn rows: what the model has to do when the conversation continues while the robot moves.

Each scenario below is a small script -- a first instruction, then one or two follow-ups spoken at
a definite moment (idle after the program finished, or ``t`` seconds into it) -- rendered into the
same ``turns`` shape ``build_dataset.py`` writes for single-turn rows. The state block on every user
turn is computed from the compiled timeline (``chat_format.state_from_timeline``), so the times,
the step marked as under way and the 中断可/不可 flag are the ones an executor would report.

    repeat          idle    「もう一回」                     -> replace, same program (or slowed down)
    interrupt       running 「ストップ」                     -> cancel   (during a flip: after it)
    resume          idle    「続けて」 after a cancel         -> replace, the steps from the cut
    insert          running 「ハンドスプリングして！」         -> insert, the heading's running flip
                    running 「左に曲がって」                 -> insert, a turn
                    running (fast) any flip                  -> insert, from standing, says so
    insert_propose  running a flip the heading forbids       -> none + proposal, then yes/no
    append          running 「終わったら〜も」               -> append
    correction      running 「やっぱり〜」                    -> replace, another program
    chitchat        running small talk                        -> none (the robot keeps going)
    status          running 「今なにしてる？」                -> none, reads the state block
    stopped_already idle    「ストップ」                     -> none, already standing
    jump_propose    idle    「ジャンプして」                 -> none + alternative, then yes/no
    running_propose idle    「前に走りながらバク転して」       -> none + the heading's flip, then yes/no
    running_fast    idle    「速く走りながら前転して」         -> replace at normal speed, says so
    missing_amount  idle    「バックで戻って」(数量なし)        -> none + ask how far, then the number
    idle_filler     idle+直前 「なるほど」「ふーん」               -> none (not everything is a command)

Wording lives in the tables at the top; nothing below them invents Japanese. Decisions -- which
action, which program, whether a warning is due -- come from the scenario, the grammar and the
capability rates, exactly as for the single-turn rows.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from typing import Callable

import negatives as ng
import phrasebank as pb
from chat_format import RobotState, render_user_turn, state_from_timeline
from unitree_rl_lab.program import CompilerConfig, ProgramError, compile_program, program_from_json
from unitree_rl_lab.program.grammar import FLIP_KINDS, JA, RUNNING_FLIP_FOR, RUNNING_FLIP_SPEEDS

# ==================================================================================== wording

_S = {"polite": "polite", "kansai": "kansai", "short": "kansai"}  # the short register speaks Kansai

REPEAT_IN = {
    "polite": ["もう一回お願いします。", "もう一度同じことをしてください。", "同じのをもう一回。", "さっきのをもう一度お願いします。"],
    "kansai": ["もう一回やって。", "もっかい。", "さっきのもう一回やってくれる?", "同じやつもう一回!", "もう一回いこ。", "もっぺん。", "もっぺんやって。", "リピート。", "アンコール!"],
    "short": ["もう一回。", "もっかい。", "同じの。"],
}
REPEAT_SLOW_IN = {
    "polite": ["もう一回、今度はゆっくりでお願いします。", "同じものをゆっくりめでもう一度。"],
    "kansai": ["もう一回、今度はゆっくりで。", "もっかい、ゆっくりめにやって。", "同じやつ、ゆっくりバージョンで。"],
    "short": ["もう一回、ゆっくり。", "同じの、ゆっくりで。"],
}
REPEAT_RE = {
    "polite": ["はい、もう一度行います。", "承知しました。同じ動きをもう一回。", "もう一度やります。"],
    "kansai": ["ほな、もう一回いくで。", "おっけー、同じやつな。", "了解、もっかいやるわ。"],
    "short": ["もう一回、いくで。", "同じの、いくで。"],
}
REPEAT_SLOW_RE = {
    "polite": ["はい、今度はゆっくりめで行います。", "承知しました。速さを落としてもう一度。"],
    "kansai": ["おっけー、今度はゆっくりめでいくわ。", "了解、ゆっくりバージョンな。"],
    "short": ["ゆっくりで、いくで。"],
}

# The kana spellings are here because a person typing in a hurry does not reach for the kanji.
# Measured on v6: 「とまれ!」 was answered by re-running the last program -- 0 of 10164 training turns
# spelled it that way, so it was simply an unknown word, and an unknown short word in this state
# falls into the 84.5% of them that mean 「もう一回」.
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
STOP_DURING_FLIP_RE = {
    "polite": ["技の途中なので、着地してから止まります。", "今は空中です。降りたらすぐ止まります。"],
    "kansai": ["技の途中やから、着地してから止まるで。", "今飛んでる最中や。降りたらすぐ止まるわ。"],
    "short": ["着地したら止まる。", "技終わったら止まる。"],
}
STOPPED_ALREADY_RE = {
    "polite": ["今は止まっています。", "もう止まっていますよ。", "はい、待機中です。"],
    "kansai": ["もう止まっとるで。", "今は動いてへんよ。", "止まってる止まってる。"],
    "short": ["止まってる。", "もう止まっとる。"],
}

RESUME_IN = {
    "polite": ["続けてください。", "やっぱり続きをお願いします。", "止めたところから再開してください。"],
    "kansai": ["続けて。", "やっぱ続きやって。", "止めたとこからいって。", "ごめん、続けてええよ。"],
    "short": ["続けて。", "再開。"],
}
RESUME_RE = {
    "polite": ["はい、止めたところから続けます。", "再開します。"],
    "kansai": ["おっけー、止めたとこからいくで。", "ほな続きな。", "了解、再開するわ。"],
    "short": ["続きいくで。", "再開。"],
}

# insert: a flip called for by name while running
INSERT_FLIP_IN = {
    "polite": ["{flip}してください!", "そのまま{flip}して!", "{flip}お願いします!"],
    "kansai": ["{flip}して!", "{flip}!", "そのまま{flip}いって!", "{flip}やって!"],
    "short": ["{flip}!", "{flip}。"],
}
INSERT_FLIP_RE = {
    "polite": ["走りながら{flip}します!", "はい、そのまま{flip}。", "{flip}、いきます。"],
    "kansai": ["走りながら{flip}いくで!", "そのまま{flip}や!", "{flip}、いくわ!"],
    "short": ["{flip}いくで。", "そのまま{flip}。"],
}
INSERT_FLIP_FAST_RE = {
    "polite": ["速く走っている間は技ができないので、一度止まって{flip}してから続けます。",
               "この速さでは技が出せません。止まって{flip}をして、また走ります。"],
    "kansai": ["速いと技できひんから、一回止まって{flip}してから続けるで。", "この速さやと無理やねん。止まって{flip}して、また走るわ。"],
    "short": ["速いから止まって{flip}する。", "一回止まって{flip}。"],
}
INSERT_TURN_RE = {
    "polite": ["{turn}、そのあと続けます。", "はい、{turn}。それから元の動きに戻ります。"],
    "kansai": ["{turn}、ほんで続きな。", "おっけー、{turn}。そっから元に戻るで。"],
    "short": ["{turn}、いくで。"],
}

# A flip the heading does not allow while running: propose the one it does.
#
# Every variant used to open with 「走りながら」, and that one word then carried the whole decision:
# measured on the v3 dataset, a reply containing 走りながら was a proposal 48% of the time while a
# reply containing any other running word (そのまま, 止まらずに, その勢いで, 走った勢いで) was a
# proposal 0.0% of the time. The model reuses the word the person used, so 「走りながら前転して」
# reliably produced a question and 「そのまま前転して」 -- the same request -- produced the program.
# The word is spread across the variants here so it cannot stand in for the actual signal, which is
# whether the asked rotation matches the heading.
PROPOSE_RUNNING_RE = {
    "polite": ["{dir}へ{run}出せるのは{allowed}だけです。{allowed}でよいですか。",
               "{dir}に{run}やる場合は{allowed}になります。それでよろしいですか。",
               "{run}できるのは、{dir}なら{allowed}だけです。{allowed}でよいですか。"],
    "kansai": ["{dir}に{run}出せるんは{allowed}だけやねん。{allowed}でええ?",
               "{dir}に{run}やるんやったら{allowed}になるけど、それでええ?",
               "{run}できるんは、{dir}やと{allowed}だけや。それでええ?"],
    "short": ["{dir}に{run}出せるんは{allowed}だけ。ええ?",
              "{run}できるんは{dir}やと{allowed}だけ。それでええ?"],
}

ALL_RUNNING_WORDS = sorted({w for words in pb.RUNNING_WORDS.values() for w in words}, key=len, reverse=True)


def _running_word(asked_text: str, style: str, rng: random.Random) -> str:
    """The word the person just used for "without stopping", to say it back to them.

    Echoed rather than chosen, because choosing is what broke v3. Every variant of this reply used
    to open with 「走りながら」, and since the model reuses the person's word, that one word ended up
    deciding the action: measured on the v3 dataset a reply containing 走りながら was a proposal 48%
    of the time and a reply containing any other running word 0.0% of the time, so
    「走りながら前転して」 got a question and 「そのまま前転して」 -- the same request -- got the
    program. Echoing makes the word's distribution here identical to its distribution in the
    instructions, which leaves the model nothing to read off it but the thing that actually decides:
    whether the rotation asked for matches the heading.
    """
    for word in ALL_RUNNING_WORDS:
        if word in asked_text:
            return word
    return rng.choice(pb.RUNNING_WORDS[style])


PROPOSE_YES_RE = {
    "polite": ["はい、{allowed}でいきます。", "承知しました、{allowed}にします。"],
    "kansai": ["ほな{allowed}でいくで!", "おっけー、{allowed}な!"],
    "short": ["{allowed}、いくで。"],
}
PROPOSE_NO_RE = {
    "polite": ["わかりました。そのまま続けます。", "了解です、やめておきます。"],
    "kansai": ["おっけー、そのまま走るわ。", "了解、やめとくな。"],
    "short": ["了解、やめとく。"],
}
PROPOSE_NO_IDLE_RE = {
    "polite": ["わかりました。何もしないでおきます。", "了解です、やめておきます。"],
    "kansai": ["おっけー、やめとくわ。", "了解、なしな。"],
    "short": ["了解、やめとく。"],
}
# Reactions and fillers -- the things a person says that are not instructions at all. The existing
# chit-chat rows put these at a plain idle state (nothing run yet) or mid-run; neither covers the
# state you are in most of the time, which is standing still with a program just finished. Measured
# on the v6 dataset, a short utterance in *that* state was `replace` 84.5% of the time, so anything
# the model did not recognise -- 「なるほど」, 「そうか」, 「とまれ」 -- re-ran the last program.
# Deliberately no 「うん」「ええよ」「オッケー」: those are how a proposal is accepted, and they have
# to keep meaning yes when one is pending.
IDLE_FILLER_IN = {
    "polite": ["なるほど。", "そうですか。", "ふむ。", "へえ。", "なるほどですね。", "ふーん。", "ああ。", "すごいですね。"],
    "kansai": ["なるほど。", "なるほどな。", "そうか。", "ふーん。", "へえ。", "ほんまか。", "やるやん。", "おお。", "すごいな。", "あー。"],
    "short": ["ふーん。", "へえ。", "なるほど。", "そうか。"],
}
IDLE_FILLER_RE = {
    "polite": ["はい。次のご指示をお待ちしています。", "何かあればお申し付けください。", "待機しています。"],
    "kansai": ["せやろ。次どうする?", "まあな。次の指示待っとるで。", "ほんで、次は?", "待っとるで。"],
    "short": ["待機中。", "次どうする?", "待っとる。"],
}

# A direction with no amount. Every instruction the phrase bank writes names a distance or a
# duration, so 「バックで戻って」 sat outside the data and the model answered it out of the decline
# templates instead ("人を乗せるところがないねん"). Asking is the honest answer: the alternative is
# guessing a number, and the number is how far a 15 kg robot actually travels.
MISSING_AMOUNT_IN = {
    "polite": ["{dir}に進んでください。", "{dir}へ行ってください。", "{dir}に動いてください。", "{dir}へお願いします。"],
    "kansai": ["{dir}に行って。", "{dir}へ進んで。", "{dir}に動いて。", "{dir}やって。", "{dir}に行ってくれる?"],
    "short": ["{dir}に。", "{dir}へ。", "{dir}。"],
}
MISSING_AMOUNT_RE = {
    # No example distance in the wording: check_dataset's numerals rule counts any number in a
    # reply that the program does not justify, and it is right to -- that rule is what stops
    # 「ぴったり5m」. The question works without one.
    "polite": ["どのくらい{dir}へ進みましょうか。距離か秒数を教えてください。",
               "{dir}ですね。どのくらい進みましょうか。メートルでも秒でも構いません。",
               "{dir}に進みます。距離か時間を決めてもらえますか。"],
    "kansai": ["どんくらい{dir}行く?距離か秒数を言うてくれたらいくで。",
               "{dir}やな。どんくらい行く?メートルでも秒でもええで。",
               "{dir}やね。距離か時間だけ決めてくれる?"],
    "short": ["どんくらい?距離か秒数を。", "{dir}やな。どんくらい?", "距離は?"],
}
AMOUNT_IN = {
    "polite": ["{amount}です。", "{amount}でお願いします。", "{amount}で。", "{amount}くらいで。"],
    "kansai": ["{amount}や。", "あ、{amount}やで。", "{amount}で。", "{amount}くらいやな。"],
    "short": ["{amount}。", "{amount}で。"],
}

YES_IN = {
    "polite": ["はい、それでお願いします。", "お願いします。", "それでいいです。", "はい。"],
    "kansai": ["うん、それで。", "ええよ。", "それでええ。", "おk。", "頼む。", "うん。"],
    "short": ["うん。", "それで。", "おk。"],
}
NO_IN = {
    "polite": ["いいえ、結構です。", "やめておきます。", "いえ、大丈夫です。"],
    "kansai": ["いや、ええわ。", "やめとく。", "ううん、いらん。"],
    "short": ["いや。", "やめとく。"],
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

STATUS_IN = {
    "polite": ["今何をしていますか。", "今どこまで進みましたか。", "あと何秒くらいですか。", "何が残っていますか。"],
    "kansai": ["今なにしてる?", "今どこまでいった?", "あと何秒?", "残りなに?", "今なんの途中?"],
    "short": ["今なに?", "あと何秒?", "残り?"],
}
STATUS_RUNNING_RE = {
    "polite": ["いま{step}の途中です。この手順はあと{remain}秒ほど、全体では残り{left}秒くらいです。{rest}",
               "{step}をしているところです。全体の残りは約{left}秒。{rest}"],
    "kansai": ["いま{step}の途中や。この手順はあと{remain}秒くらい、全部で残り{left}秒ほどやな。{rest}",
               "{step}してるとこ。あと{left}秒くらいで終わるで。{rest}"],
    "short": ["{step}の途中。残り{left}秒くらい。{rest}"],
}
STATUS_REST = {
    "polite": {0: "これが最後の手順です。", 1: "あと{n}つ手順が残っています。"},
    "kansai": {0: "これで最後や。", 1: "あと{n}つ残っとる。"},
    "short": {0: "これで最後。", 1: "あと{n}つ。"},
}
STATUS_IDLE_RE = {
    "polite": ["今は何もしていません。待機中です。", "止まって待っています。"],
    "kansai": ["今はなんもしてへんよ。待機中や。", "止まって待っとるで。"],
    "short": ["待機中。", "止まってる。"],
}

JUMP_IN = {
    "polite": ["ジャンプしてください。", "その場でジャンプをお願いします。", "垂直跳びしてください。", "ジャンプを2回してください。"],
    "kansai": ["ジャンプして。", "ジャンプしてみて。", "その場でジャンプ!", "垂直跳びやって。", "ジャンプ2回やって。"],
    "short": ["ジャンプ。", "ジャンプ2回。", "垂直跳び。"],
}
JUMP_PROPOSE_RE = {
    "polite": ["すみません、ジャンプはまだ着地に成功しません。代わりに{alt}ならできますが、いかがですか。",
               "ジャンプはまだできないんです。{alt}に変えてもよろしいですか。"],
    "kansai": ["ごめん、ジャンプはまだ着地できひんねん。代わりに{alt}ならできるけど、それでええ?",
               "ジャンプはまだ無理やわ。{alt}に変えてええ?"],
    "short": ["ジャンプは無理。{alt}でええ?"],
}
JUMP_YES_RE = {
    "polite": ["はい、{alt}をします。", "では{alt}でいきます。"],
    "kansai": ["ほな{alt}いくで!", "おっけー、{alt}な!"],
    "short": ["{alt}、いくで。"],
}

RUNNING_FAST_RE = {
    "polite": ["速く走りながらだと技が出せないので、普通の速さで{dir}に走って{flip}します。",
               "速いと{flip}ができません。速さを普通に落として走り、そのまま{flip}します。"],
    "kansai": ["速く走りながらやと技出せへんから、普通の速さで{dir}に走って{flip}するで。",
               "速いと{flip}できひんねん。普通の速さに落として走って、そのまま{flip}するわ。"],
    "short": ["速いと無理。普通の速さで走って{flip}する。"],
}

# ================================================================================== helpers


def _pick(table: dict, style: str, rng: random.Random) -> str:
    return rng.choice(table[style])


def _dir_ja(direction: str) -> str:
    return JA[direction]


def _flip_ja(kind: str) -> str:
    return JA[kind]


def _slowed(program: list[dict]) -> list[dict] | None:
    """The same program with every move and turn at "slow"; None if that changes nothing."""
    out, changed = [], False
    for step in program:
        step = dict(step)
        if step["skill"] in ("move", "turn") and step.get("speed", "normal") != "slow":
            step["speed"] = "slow"
            changed = True
        out.append(step)
    return out if changed else None


def _valid(program: list[dict], context: dict | None = None) -> bool:
    try:
        ctx = program_from_json([context])[0] if context else None
        compile_program(program_from_json(program), context=ctx, resume=context is not None)
    except ProgramError:
        return False
    return True


def _split_for(key: str) -> str:
    return "eval" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


class Turns:
    """Accumulates one dialogue; ``user`` renders the state block, ``bot`` records the target."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def user(self, state: RobotState, text: str) -> None:
        self.items.append({"role": "user", "content": render_user_turn(state, text), "state": asdict(state)})

    def bot(self, reply: str, action: str, program: list[dict], **extra) -> None:
        self.items.append({"role": "assistant", "reply": reply, "action": action, "program": program, **extra})


def _moment(timeline, program: list[dict], rng: random.Random, *, want: str) -> tuple[float, int] | None:
    """A time to speak at. ``want`` is "interruptible" (inside a move/turn), "flip" (inside a flip
    window), or "move" (inside a move at slow/normal, with 2 s still to go). -> (t, step_index)."""
    options = []
    for seg in timeline.segments:
        if seg.step_index is None:
            continue
        step = program[seg.step_index]
        if want == "flip" and seg.kind == "flip":
            options.append((seg, 0.15, 0.15))
        elif want == "interruptible" and seg.kind in ("move", "turn") and seg.duration >= 1.0:
            options.append((seg, 0.3, 0.3))
        elif want == "move" and seg.kind == "move" and step.get("speed", "normal") in RUNNING_FLIP_SPEEDS and seg.duration >= 3.0:
            options.append((seg, 0.5, 2.5))
        elif want == "fast_move" and seg.kind == "move" and step.get("speed") == "fast" and seg.duration >= 2.0:
            options.append((seg, 0.5, 1.0))
    if not options:
        return None
    seg, lo, hi = rng.choice(options)
    t = round(rng.uniform(seg.t0 + lo, seg.t1 - hi), 1)
    return t, seg.step_index


# ================================================================================== scenarios
#
# Every scenario takes the same arguments and returns the turns, or None when the base program
# does not suit it (too short to interrupt, no move to flip out of, ...).

Scenario = Callable[[dict, list[dict], object, str, str, list[str], random.Random, "Context"], "Turns | None"]


class Context:
    """What the scenarios need from the dataset builder: replies, rates, and other programs."""

    def __init__(self, first_reply: Callable, cautions_for: Callable, other_programs: list[dict], compiler: CompilerConfig,
                 all_records: list[dict] | None = None):
        self.first_reply = first_reply      # (steps, style, duration_s, rng) -> reply text for an instruction
        self.cautions_for = cautions_for    # (steps, style, rng) -> caution sentences for these steps
        self.other_programs = other_programs
        self.compiler = compiler
        self.all_records = all_records or []
        """Every validated program, passed or not: the proposal scenarios are about the *request*,
        and a running frontflip that then fails in the sim is still the right thing to propose."""


def _open(record, steps, timeline, style, rng, ctx) -> Turns:
    """Turn 1: the instruction for the base program, answered as a single-turn row would."""
    turns = Turns()
    turns.user(RobotState.idle(), pb.instruction(steps, style, rng))
    turns.bot(ctx.first_reply(steps, style, timeline.duration, rng), "replace", steps, roundtrip=True)
    return turns


def sc_repeat(record, steps, timeline, style, rng, ctx) -> Turns | None:
    turns = _open(record, steps, timeline, style, rng, ctx)
    slow = _slowed(steps) if rng.random() < 0.3 else None
    state = RobotState.idle(steps, "completed")
    if slow is not None and _valid(slow):
        turns.user(state, _pick(REPEAT_SLOW_IN, style, rng))
        turns.bot(_pick(REPEAT_SLOW_RE, style, rng) + ctx.cautions_for(slow, style, rng), "replace", slow)
    else:
        turns.user(state, _pick(REPEAT_IN, style, rng))
        turns.bot(_pick(REPEAT_RE, style, rng) + ctx.cautions_for(steps, style, rng), "replace", steps)
    return turns


def sc_interrupt(record, steps, timeline, style, rng, ctx) -> Turns | None:
    during_flip = rng.random() < 0.3
    moment = _moment(timeline, steps, rng, want="flip" if during_flip else "interruptible")
    if moment is None:
        return None
    t, index = moment
    turns = _open(record, steps, timeline, style, rng, ctx)
    state = state_from_timeline(steps, timeline, t)
    turns.user(state, _pick(STOP_IN, style, rng))
    turns.bot(_pick(STOP_DURING_FLIP_RE if not state.interruptible else STOP_RE, style, rng), "cancel", [])
    return turns


def sc_resume(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None or len(steps) < 2:
        return None
    t, index = moment
    remaining = steps[index:]
    if remaining and remaining[0].get("running"):
        remaining = steps[index - 1:]
    if not _valid(remaining):
        return None
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), _pick(STOP_IN, style, rng))
    turns.bot(_pick(STOP_RE, style, rng), "cancel", [])
    turns.user(RobotState.idle(steps, "cancelled", at_step=index), _pick(RESUME_IN, style, rng))
    turns.bot(_pick(RESUME_RE, style, rng) + ctx.cautions_for(remaining, style, rng), "replace", remaining)
    return turns


def sc_insert_flip(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="move")
    if moment is None:
        return None
    t, index = moment
    kind = RUNNING_FLIP_FOR[steps[index]["dir"]]
    inserted = [{"skill": "flip", "kind": kind, "count": 1, "running": True}]
    if not _valid(inserted, context=steps[index]):
        return None
    word = rng.choice(pb.FLIP_WORDS[kind])
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), _pick(INSERT_FLIP_IN, style, rng).format(flip=word))
    turns.bot(_pick(INSERT_FLIP_RE, style, rng).format(flip=_flip_ja(kind)) + ctx.cautions_for(inserted, style, rng),
              "insert", inserted)
    return turns


def sc_insert_flip_fast(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="fast_move")
    if moment is None:
        return None
    t, index = moment
    kind = rng.choice([k for k in FLIP_KINDS if k != "jump"])
    inserted = [{"skill": "flip", "kind": kind, "count": 1}]
    word = rng.choice(pb.FLIP_WORDS[kind])
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), _pick(INSERT_FLIP_IN, style, rng).format(flip=word))
    turns.bot(_pick(INSERT_FLIP_FAST_RE, style, rng).format(flip=_flip_ja(kind)) + ctx.cautions_for(inserted, style, rng),
              "insert", inserted)
    return turns


def sc_insert_turn(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None or steps[moment[1]]["skill"] != "move":
        return None
    t, index = moment
    turn = {"skill": "turn", "dir": rng.choice(["left", "right"]), "speed": "normal",
            "angle_deg": rng.choice([45.0, 90.0, 180.0])}
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), pb.instruction([turn], style, rng))
    turns.bot(_pick(INSERT_TURN_RE, style, rng).format(turn=pb.restate([turn], style, rng).rstrip("。でわな")), "insert", [turn])
    return turns


def sc_insert_propose(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="move")
    if moment is None:
        return None
    t, index = moment
    direction = steps[index]["dir"]
    allowed = RUNNING_FLIP_FOR[direction]
    asked = rng.choice([k for k in FLIP_KINDS if k not in (allowed, "jump")])
    inserted = [{"skill": "flip", "kind": allowed, "count": 1, "running": True}]
    if not _valid(inserted, context=steps[index]):
        return None
    turns = _open(record, steps, timeline, style, rng, ctx)
    asked_text = _pick(INSERT_FLIP_IN, style, rng).format(flip=rng.choice(pb.FLIP_WORDS[asked]))
    turns.user(state_from_timeline(steps, timeline, t), asked_text)
    turns.bot(_pick(PROPOSE_RUNNING_RE, style, rng).format(
        dir=_dir_ja(direction), allowed=_flip_ja(allowed), run=_running_word(asked_text, style, rng)),
        "none", [], proposed=inserted)
    later = state_from_timeline(steps, timeline, t + 1.5)
    if rng.random() < 0.7:
        turns.user(later, _pick(YES_IN, style, rng))
        turns.bot(_pick(PROPOSE_YES_RE, style, rng).format(allowed=_flip_ja(allowed)) + ctx.cautions_for(inserted, style, rng),
                  "insert", inserted, confirms=True)
    else:
        turns.user(later, _pick(NO_IN, style, rng))
        turns.bot(_pick(PROPOSE_NO_RE, style, rng), "none", [])
    return turns


def sc_append(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None:
        return None
    t, index = moment
    extra = rng.choice([p for p in ctx.other_programs if 1 <= len(p) <= 2 and not p[0].get("running")])
    if not _valid(extra):
        return None
    turns = _open(record, steps, timeline, style, rng, ctx)
    inst = pb.instruction(extra, style, rng)
    turns.user(state_from_timeline(steps, timeline, t), _pick(APPEND_IN, style, rng).format(inst=inst))
    turns.bot(_pick(APPEND_RE, style, rng).format(restate=pb.restate(extra, style, rng)) + ctx.cautions_for(extra, style, rng),
              "append", extra)
    return turns


def sc_correction(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None:
        return None
    t, index = moment
    new = rng.choice([p for p in ctx.other_programs if 1 <= len(p) <= 2 and p != steps])
    if not _valid(new):
        return None
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), _pick(CORRECTION_IN, style, rng).format(inst=pb.instruction(new, style, rng)))
    turns.bot(_pick(CORRECTION_RE, style, rng).format(restate=pb.restate(new, style, rng)) + ctx.cautions_for(new, style, rng),
              "replace", new)
    return turns


def sc_chitchat(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None:
        return None
    t, index = moment
    casual_in, polite_in, casual_re, polite_re = rng.choice(ng.CHITCHAT)
    text, reply = (rng.choice(polite_in), rng.choice(polite_re)) if style == "polite" else (rng.choice(casual_in), rng.choice(casual_re))
    turns = _open(record, steps, timeline, style, rng, ctx)
    turns.user(state_from_timeline(steps, timeline, t), text)
    turns.bot(reply, "none", [])
    return turns


def _status_reply(state: RobotState, style: str, rng: random.Random) -> str:
    from chat_format import describe_step
    step = describe_step(state.program[state.step_index])
    left_steps = len(state.program) - state.step_index - 1
    rest = STATUS_REST[style][0] if left_steps == 0 else STATUS_REST[style][1].format(n=left_steps)
    return _pick(STATUS_RUNNING_RE, style, rng).format(
        step=step, remain=int(round(state.step_remaining_s)), left=int(round(state.duration_s - state.elapsed_s)), rest=rest)


def sc_status(record, steps, timeline, style, rng, ctx) -> Turns | None:
    moment = _moment(timeline, steps, rng, want="interruptible")
    if moment is None:
        return None
    t, index = moment
    turns = _open(record, steps, timeline, style, rng, ctx)
    state = state_from_timeline(steps, timeline, t)
    turns.user(state, _pick(STATUS_IN, style, rng))
    turns.bot(_status_reply(state, style, rng), "none", [], status=True)
    return turns


def sc_stopped_already(record, steps, timeline, style, rng, ctx) -> Turns | None:
    turns = _open(record, steps, timeline, style, rng, ctx)
    state = RobotState.idle(steps, "completed")
    if rng.random() < 0.5:
        turns.user(state, _pick(STOP_IN, style, rng))
        turns.bot(_pick(STOPPED_ALREADY_RE, style, rng), "none", [])
    else:
        turns.user(state, _pick(STATUS_IN, style, rng))
        turns.bot(_pick(STATUS_IDLE_RE, style, rng), "none", [])
    return turns


def sc_jump_propose(record, steps, timeline, style, rng, ctx) -> Turns | None:
    alt = rng.choice(["backflip", "backflip", "sideflip_left", "sideflip_right"])
    program = [{"skill": "flip", "kind": alt, "count": 1}]
    turns = Turns()
    turns.user(RobotState.idle(), _pick(JUMP_IN, style, rng))
    turns.bot(_pick(JUMP_PROPOSE_RE, style, rng).format(alt=_flip_ja(alt)), "none", [], proposed=program)
    if rng.random() < 0.7:
        turns.user(RobotState.idle(), _pick(YES_IN, style, rng))
        turns.bot(_pick(JUMP_YES_RE, style, rng).format(alt=_flip_ja(alt)), "replace", program, confirms=True)
    else:
        turns.user(RobotState.idle(), _pick(NO_IN, style, rng))
        turns.bot(_pick(PROPOSE_NO_IDLE_RE, style, rng), "none", [])
    return turns


def sc_idle_filler(record, steps, timeline, style, rng, ctx) -> Turns | None:
    """Something that is not an instruction, said while standing after a program finished.

    The state that produced 課題1 in MuJoCo: the robot is idle, `last` holds what it just did, and
    the person says something -- a reaction, a stop word the bank never spelled that way, anything.
    Every short utterance the model had seen in this state meant "do it again", so it did it again.
    These rows are the counterweight: in this state, not everything is a command.
    """
    turns = _open(record, steps, timeline, style, rng, ctx)
    state = RobotState.idle(steps, "completed")
    if rng.random() < 0.45:
        casual_in, polite_in, casual_re, polite_re = rng.choice(ng.CHITCHAT)
        text, reply = ((rng.choice(polite_in), rng.choice(polite_re)) if style == "polite"
                       else (rng.choice(casual_in), rng.choice(casual_re)))
    else:
        text, reply = _pick(IDLE_FILLER_IN, style, rng), _pick(IDLE_FILLER_RE, style, rng)
    turns.user(state, text)
    turns.bot(reply, "none", [])
    return turns


def sc_missing_amount(record, steps, timeline, style, rng, ctx) -> Turns | None:
    """A direction with no amount: ask how far, then take the number and go.

    Not answered with a guess. A distance the person did not give is a distance the robot actually
    travels, and the two turns here are what they did anyway -- 「バックで戻って」 followed by
    「あ、5mやで」 -- so the only thing missing was the robot asking instead of declining.
    """
    move = next((s for s in steps if s["skill"] == "move" and s.get("distance_m") is not None), None)
    if move is None:
        return None
    # Normal speed, whatever the base program used. The person gave a direction and then a distance
    # and nothing else, so a reply that says 「ゆっくり」 is inventing the same way a guessed distance
    # would -- and this scenario exists precisely to stop the model filling in what it was not told.
    program = [{"skill": "move", "dir": move["dir"], "speed": "normal", "distance_m": move["distance_m"]}]
    line = compile_program(program_from_json(program), ctx.compiler)
    where = _dir_ja(move["dir"])
    turns = Turns()
    turns.user(RobotState.idle(), _pick(MISSING_AMOUNT_IN, style, rng).format(dir=where))
    turns.bot(_pick(MISSING_AMOUNT_RE, style, rng).format(dir=where), "none", [])
    turns.user(RobotState.idle(), _pick(AMOUNT_IN, style, rng).format(
        amount=pb.distance_word(move["distance_m"], style, rng)))
    turns.bot(ctx.first_reply(program, style, line.duration, rng), "replace", program)
    return turns


def _first_running_pair(steps: list[dict]) -> int | None:
    for i in range(len(steps) - 1):
        if steps[i]["skill"] == "move" and steps[i + 1].get("running"):
            return i
    return None


def sc_running_propose(record, steps, timeline, style, rng, ctx) -> Turns | None:
    """A running flip asked for by name from idle -- sometimes the one the heading allows, sometimes
    not, and the answer differs on exactly that.

    A third of these ask for the rotation the heading *does* allow, and are answered with the
    program rather than a question. They are the minimal pair: same generator, same wording, same
    two-turn shape, one word different. Without them the model has no example where a running flip
    named from idle is simply carried out in this shape, and it learns "running flip from idle ->
    ask" -- which is what v3 and v4 both did, v3 keying off the word 走りながら and v4, once that was
    decorrelated, off whether the sentence looked generated at all.
    """
    i = _first_running_pair(steps)
    if i is None or len(steps) > 3:
        return None
    allowed = steps[i + 1]["kind"]
    if rng.random() < 0.35:
        turns = Turns()
        turns.user(RobotState.idle(), pb.instruction(steps, style, rng))
        turns.bot(ctx.first_reply(steps, style, timeline.duration, rng), "replace", steps, roundtrip=True)
        turns.user(RobotState.idle(steps, "completed"), _pick(REPEAT_IN, style, rng))
        turns.bot(_pick(REPEAT_RE, style, rng) + ctx.cautions_for(steps, style, rng), "replace", steps)
        return turns
    asked = rng.choice([k for k in FLIP_KINDS if k not in (allowed, "jump")])
    wrong = [dict(s) for s in steps]
    wrong[i + 1]["kind"] = asked
    turns = Turns()
    asked_text = pb.instruction(wrong, style, rng)
    turns.user(RobotState.idle(), asked_text)
    turns.bot(_pick(PROPOSE_RUNNING_RE, style, rng).format(
        dir=_dir_ja(steps[i]["dir"]), allowed=_flip_ja(allowed), run=_running_word(asked_text, style, rng)),
        "none", [], proposed=steps)
    if rng.random() < 0.7:
        turns.user(RobotState.idle(), _pick(YES_IN, style, rng))
        turns.bot(_pick(PROPOSE_YES_RE, style, rng).format(allowed=_flip_ja(allowed)) + ctx.cautions_for(steps, style, rng),
                  "replace", steps, confirms=True)
    else:
        turns.user(RobotState.idle(), _pick(NO_IN, style, rng))
        turns.bot(_pick(PROPOSE_NO_IDLE_RE, style, rng), "none", [])
    return turns


def sc_running_fast(record, steps, timeline, style, rng, ctx) -> Turns | None:
    i = _first_running_pair(steps)
    if i is None or len(steps) > 3:
        return None
    asked = [dict(s) for s in steps]
    asked[i]["speed"] = "fast"
    out = [dict(s) for s in steps]
    out[i]["speed"] = "normal"
    if not _valid(out):
        return None
    turns = Turns()
    turns.user(RobotState.idle(), pb.instruction(asked, style, rng))
    kind = steps[i + 1]["kind"]
    turns.bot(_pick(RUNNING_FAST_RE, style, rng).format(dir=_dir_ja(steps[i]["dir"]), flip=_flip_ja(kind))
              + ctx.cautions_for(out, style, rng), "replace", out)
    return turns


SCENARIOS: dict[str, Scenario] = {
    "dlg_repeat": sc_repeat,
    "dlg_interrupt": sc_interrupt,
    "dlg_resume": sc_resume,
    "dlg_insert": sc_insert_flip,
    "dlg_insert_fast": sc_insert_flip_fast,
    "dlg_insert_turn": sc_insert_turn,
    "dlg_insert_propose": sc_insert_propose,
    "dlg_append": sc_append,
    "dlg_correction": sc_correction,
    "dlg_chitchat": sc_chitchat,
    "dlg_status": sc_status,
    "dlg_stopped": sc_stopped_already,
    "dlg_jump_propose": sc_jump_propose,
    "dlg_running_propose": sc_running_propose,
    "running_fast": sc_running_fast,
    "dlg_missing_amount": sc_missing_amount,
    "dlg_idle_filler": sc_idle_filler,
}

# Scenarios whose first turn is a proposal, not an execution: any validated program will do.
FROM_ALL_RECORDS = {"dlg_running_propose", "running_fast"}

# Share of the dialogue budget per scenario.
SHARE: dict[str, float] = {
    "dlg_repeat": 0.06, "dlg_interrupt": 0.07, "dlg_resume": 0.06, "dlg_insert": 0.12, "dlg_insert_fast": 0.05,
    "dlg_insert_turn": 0.04, "dlg_insert_propose": 0.10, "dlg_append": 0.07, "dlg_correction": 0.07,
    "dlg_chitchat": 0.06, "dlg_status": 0.06, "dlg_stopped": 0.02, "dlg_jump_propose": 0.04,
    "dlg_running_propose": 0.05, "running_fast": 0.02, "dlg_missing_amount": 0.05,
    "dlg_idle_filler": 0.06,
}
"""Sums to 1.0. The 0.05 for dlg_missing_amount came off dlg_repeat and dlg_chitchat, which score
highest of the lot and are the least sensitive to a smaller share."""


def build_dialogue_rows(records: list[dict], total: int, styles: tuple[str, ...], ctx: Context,
                        rng: random.Random, seen: set[tuple]) -> list[dict]:
    """``total`` dialogue rows from base programs (validated, nothing declined), across scenarios."""
    programs = [r["program"] for r in records]
    ctx.other_programs = programs
    rows: list[dict] = []
    for name, share in SHARE.items():
        quota = round(total * share)
        made = tries = 0
        order = list(ctx.all_records or records) if name in FROM_ALL_RECORDS else list(records)
        if name in FROM_ALL_RECORDS:  # only programs with a move + running flip can be misasked
            order = [r for r in order if _first_running_pair(r["program"]) is not None and len(r["program"]) <= 3]
        rng.shuffle(order)
        while made < quota and tries < quota * 6:
            record = order[tries % len(order)]
            style = styles[tries % len(styles)]
            tries += 1
            steps = record["program"]
            timeline = compile_program(program_from_json(steps), ctx.compiler)
            turns = SCENARIOS[name](record, steps, timeline, style, rng, ctx)
            if turns is None:
                continue
            key = tuple((t.get("content"), t.get("reply"), json.dumps(t.get("program"), ensure_ascii=False)) for t in turns.items)
            if key in seen:
                continue
            seen.add(key)
            rid = f"{name}-{record['id']}-{made:04d}#{style}"
            rows.append({
                "id": rid, "split": _split_for(record["id"]), "category": name, "style": style,
                "turns": turns.items,
                "source": {"program_id": record["id"], "passed": record.get("passed"), "kind": name},
            })
            made += 1
        if made < quota:
            print(f"  note: {name} produced {made}/{quota}")
    return rows
