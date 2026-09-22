"""Build the fine-tuning dataset from the sampled programs and the phrase bank.

    python scripts/llm/build_dataset.py --programs data/llm/programs.jsonl --out data/llm/dataset.jsonl

One row is one conversation. A row is made by putting Japanese on a program:

    programs.jsonl   {"id", "program"}          what the robot should do
    phrasebank       program -> 指示文           how a person asks for it
    phrasebank       program -> 返事             how the robot answers
    chat_format      -> the user turn, the target

Nothing here judges whether the robot can do it. That question belonged to an earlier design in
which the dataset was filtered by simulation and the model was taught to decline, warn and propose;
it is gone. The model translates, the compiler makes the result executable, and what the compiler
had to change is logged for whoever tunes the policy. See DATASET.md.

The split between this file and ``phrasebank.py`` is the one that makes the checks meaningful:
every Japanese *word* comes from the phrase bank, every *decision* is made here or in
``dialogues.py``, and ``parse_instruction.py`` reads the result back with a third, independent
lexicon. Nothing may invent Japanese outside the phrase bank.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

import chat_format as cf  # noqa: E402
import dialogues  # noqa: E402
import negatives as ng  # noqa: E402
import phrasebank as pb  # noqa: E402
from unitree_rl_lab.program import CapabilityTable, CompilerConfig, compile_program, program_from_json  # noqa: E402

SYSTEM_PROMPT = """あなたは四足歩行ロボット Go2 です。人の日本語を聞き、短い返事と、これから実行する手順を返します。
会話は続きます。前のやりとりを覚えていて、「もう一回」「やめて」「それでいい」のような言葉は前の流れから解釈します。

人の発言のあとに「queued programs:」の行が付きます。これはいま実行中の手順と、そのあとに実行する手順です。
数字は残りの量です(5m進む途中なら「残り1.8m」と書いてあります)。何も動いていないときは [] です。

出力は次の形式ひとつだけ:

<日本語の返事>

program: [<手順>]

The reply is one line (no newlines inside it), then a blank line, then the program line.

The program you return becomes the queue. Whatever was in "queued programs" is discarded, so:

  to change nothing    copy the given list back unchanged -- small talk, a question, a suggestion,
                       waiting for an answer: all of these leave the robot alone
  to stop              []
  to act now           [<new step>, <the given list>...] -- the first step runs immediately, then
                       what was queued carries on
  to add at the end    [<the given list>..., <new step>] -- 「終わったら〜も」
  to replace           a different list -- a new instruction while idle, or 「やっぱり〜」

When you copy the given list back, copy it exactly, numbers included.

{grammar}

返事は1〜2文。相手の口調(丁寧語/関西弁/短文)に合わせる。program の中身は返事に書かない。
"""


def write_system_prompt(path: str) -> None:
    """The prompt each row is trained under. Stored once, not repeated on every line."""
    from unitree_rl_lab.program import describe_grammar

    Path(path).write_text(SYSTEM_PROMPT.format(grammar=describe_grammar()))


# ------------------------------------------------------------------------------------ replies


def build_reply(steps: list[dict], style: str, rng: random.Random, words: dict | None = None) -> str:
    """The robot's answer to an instruction: an acknowledgement and what it is about to do.

    No duration, no warning, no refusal. The total time is a number the model cannot work out --
    it depends on the measured speed per direction -- and saying it would be parroting; the
    conductor prints it from the compiled timeline instead.
    """
    parts = [rng.choice(pb.OPENERS[style])]
    if steps:
        parts.append(pb.restate(steps, style, rng, words))
    tail = rng.choice(pb.REPLY_TAIL[style])
    if tail and tail.rstrip("!。") not in parts[-1]:  # "…いくで。いくで。"
        parts.append(tail)
    return "".join(parts)


# ------------------------------------------------------------------------------------ rows


def split_for(key: str) -> str:
    """Hash the program id, so the same program never straddles train and eval."""
    import hashlib

    return "eval" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


def is_templated(key: str, share: float) -> bool:
    """Whether this program's instruction is written by the phrase bank or by hand.

    By a hash of the id rather than by shuffling and slicing, so the same program lands on the same
    side of the line whatever the seed is, whatever order the file is in, and however many programs
    there are. The hand-written half is the complement of this, and it is written once: a split that
    moved with the seed would silently hand some programs two instructions and others none.

    Salted so it is independent of ``split_for`` -- the train/eval line and the written-by line
    should not correlate.
    """
    import hashlib

    return int(hashlib.sha1(("templated:" + key).encode()).hexdigest(), 16) % 1000 < share * 1000


def row(row_id: str, split: str, category: str, style: str,
        turns: list[tuple]) -> dict:
    """One conversation. ``turns`` is ``(user text, queue shown, reply, program returned)`` each,
    optionally with a fifth element marking the turn for the round trip: the user text is a whole
    rendered instruction, so ``parse_instruction.py`` can be asked whether it still says what the
    program does. Turns that quote an instruction inside a frame, or point back at an earlier one,
    say less than their program on purpose and are left out of that check."""
    items = []
    for turn in turns:
        user_text, queue, reply, program = turn[:4]
        items.append({"role": "user", "content": user_text, "queue": queue})
        assistant = {"role": "assistant", "reply": reply, "program": program}
        if len(turn) > 4 and turn[4]:
            assistant["roundtrip"] = True
        items.append(assistant)
    return {"id": row_id, "split": split, "category": category, "style": style, "turns": items}


def instruction_row(record: dict, style: str, rng: random.Random) -> dict:
    """A single instruction given while the robot is standing still."""
    steps = record["program"]
    words: dict = {}   # one name per skill for this row -- the reply echoes what the ask called it
    return row(f"{record['id']}#{style}", split_for(record["id"]), "normal", style,
               [(pb.instruction(steps, style, rng, words), [],
                 build_reply(steps, style, rng, words), steps, True)])


def load_handwritten(path: str) -> list[dict]:
    """The hand-written half: whole conversations, written by hand in the dataset's own row shape.

    See HANDWRITTEN.md. Unlike the templated half nothing is rendered here -- the person's words
    and the robot's replies were both written, so this only stamps on what the writer must not
    choose: the row id, and the train/eval side, which comes from the program id so a program never
    straddles the split. Nothing is checked; ``check_handwritten.py`` is a separate pass, kept out
    of the writing so that no tool's idea of a readable sentence shapes the Japanese.
    """
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        rows.append({**item, "id": f"{item['id']}#hw", "split": split_for(item["id"])})
    return rows


def topic_rows(topics: list[ng.Topic], category: str, quota: int, rng: random.Random) -> list[dict]:
    """Small talk and vague asks: the robot answers and leaves the queue alone.

    The queue is empty here, so "leave it alone" and "stop" look the same from the outside. Rows
    with something *running* are the ones that teach the difference, and those are dialogues.
    """
    out = []
    for index in range(quota):
        casual_in, polite_in, casual_re, polite_re = topics[index % len(topics)]
        style = pb.STYLES[index % len(pb.STYLES)]
        text = ng.pick(polite_in if style == "polite" else casual_in, rng)
        reply = ng.pick(polite_re if style == "polite" else casual_re, rng)
        out.append(row(f"neg-{category}-{index:04d}", split_for(f"{category}{index}"), category, style,
                       [(text, [], reply, [])]))
    return out


TE_FORM_EXCEPTIONS = {
    "階段を登る": "階段を登っ", "階段を降りる": "階段を降り", "ドアを開ける": "ドアを開け",
    "荷物を運ぶ": "荷物を運ん", "ボールを取ってくる": "ボールを取ってき", "泳ぐ": "泳い",
    "空を飛ぶ": "空を飛ん", "坂道を登る": "坂道を登っ", "壁を登る": "壁を登っ",
    "お手をする": "お手をし", "握手する": "握手し", "写真を撮る": "写真を撮っ",
    "掃除する": "掃除し", "料理する": "料理し", "音楽をかける": "音楽をかけ", "歌う": "歌っ",
    "犬と遊ぶ": "犬と遊ん", "ついてくる": "ついてき", "持ち主を探す": "持ち主を探し",
    "部屋の隅まで行く": "部屋の隅まで行っ", "地図を見て移動する": "地図を見て移動し",
    "障害物をよけて進む": "障害物をよけて進ん", "穴を掘る": "穴を掘っ", "寝転がる": "寝転がっ",
    "お座りする": "お座りし", "背中に人を乗せる": "背中に乗せ", "抱っこする": "抱っこし",
    "空中で2回転する": "空中で2回転し", "5m跳ぶ": "5m跳ん", "バク転しながら前に進む": "バク転しながら前に進ん",
    "倒立したまま階段を登る": "倒立したまま階段を登っ", "車を運転する": "車を運転し",
    "買い物に行く": "買い物に行っ", "電気を消す": "電気を消し", "見えている色を答える": "見えてる色を教え",
    "落ちている物を拾う": "落ちてる物を拾っ", "跳んで壁を越える": "跳んで壁を越え",
    "一輪車に乗る": "一輪車に乗っ", "逆立ちで階段を降りる": "逆立ちで階段を降り",
    "回転しながらジャンプする": "回転しながらジャンプし", "水たまりを飛び越える": "水たまりを飛び越え",
    "誰か呼んでくる": "誰か呼んでき", "荷物を押して動かす": "荷物を押して動かし",
    "足を1本上げたまま歩く": "足を1本上げたまま歩い", "後ろ向きに階段を降りる": "後ろ向きに階段を降り",
}


def te_form(action: str) -> str:
    """The 連用形 stem the input templates attach "て/てください" to."""
    return TE_FORM_EXCEPTIONS[action]


def impossible_rows(quota: int, rng: random.Random) -> list[dict]:
    """Asks the robot has no skill for at all -- climbing stairs, fetching things, singing.

    Answered with a plain "no, and here is what I can do". There is no program to return, so the
    queue comes back empty; while something is running the same ask is a dialogue row instead, and
    the queue comes back untouched.
    """
    combos = [(style, action, missing, template)
              for action, missing in ng.IMPOSSIBLE
              for style in ("kansai", "polite")
              for template in ng.IMPOSSIBLE_INPUT[style]]
    rng.shuffle(combos)
    out = []
    for index, (style, action, missing, template) in enumerate(combos[:quota]):
        text = template.format(a=te_form(action))
        reply = rng.choice(ng.IMPOSSIBLE_REPLY[style]).format(
            a=action, m=missing, f=rng.choice(ng.IMPOSSIBLE_FALLBACK[style]))
        out.append(row(f"neg-impossible-{index:04d}", split_for(f"imp{index}"), "impossible", style,
                       [(text, [], reply, [])]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--programs", default="data/llm/programs.jsonl")
    parser.add_argument("--handwritten", default="data/llm/handwritten.jsonl",
                        help="the hand-written half's instructions; see HANDWRITTEN.md. Pass an "
                             "empty string to build without it, before it exists.")
    parser.add_argument("--capability", default="data/llm/capability.json")
    parser.add_argument("--out", default="data/llm/dataset.jsonl")
    parser.add_argument("--prompt-out", default="data/llm/system_prompt.txt")
    parser.add_argument("--share", type=float, default=0.5,
                        help="fraction of the programs to render with the phrase bank; the rest are "
                             "left for hand-written instructions")
    parser.add_argument("--dialogues", type=int, default=3200, help="multi-turn rows")
    parser.add_argument("--single", type=float, default=0.25,
                        help="share of the templated programs that also get a one-turn row of their "
                             "own. The rest appear only inside dialogues, which is how they arrive "
                             "in practice: an instruction is almost always said to a robot that has "
                             "been spoken to before.")
    parser.add_argument("--negatives", type=float, default=0.08, help="share of all rows that are negatives")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = [json.loads(line) for line in open(args.programs)]
    templated = [r for r in records if is_templated(r["id"], args.share)]
    compiler = CompilerConfig(calibration=CapabilityTable.load(args.capability).calibration())

    rows: list[dict] = []
    for record in templated:
        # A one-turn row is a standing start with nothing said before it -- the rarest thing that
        # happens in operation, so only a quarter of the programs get one. Every program still
        # reaches the data through the dialogues, which draw from the same pool.
        if not is_templated(record["id"] + ":single", args.single):
            continue
        style = pb.STYLES[rng.randrange(len(pb.STYLES))]
        rows.append(instruction_row(record, style, rng))

    if args.handwritten and Path(args.handwritten).exists():
        rows += load_handwritten(args.handwritten)

    quota = round((len(rows) + args.dialogues) * args.negatives / (1 - args.negatives))
    share = {"chitchat": 0.45, "impossible": 0.33, "ambiguous": 0.22}
    rows += topic_rows(ng.CHITCHAT, "chitchat", round(quota * share["chitchat"]), rng)
    rows += impossible_rows(round(quota * share["impossible"]), rng)
    rows += topic_rows(ng.AMBIGUOUS, "ambiguous", round(quota * share["ambiguous"]), rng)

    rows += dialogues.build(templated, args.dialogues, compiler, build_reply, rng)

    write_system_prompt(args.prompt_out)
    rng.shuffle(rows)
    with open(args.out, "w") as out:
        for item in rows:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    turns = sum(len(r["turns"]) // 2 for r in rows)
    print(f"wrote {len(rows)} rows ({turns} assistant turns) to {args.out}")
    import collections
    for name, counter in (("category", collections.Counter(r["category"] for r in rows)),
                          ("style", collections.Counter(r["style"] for r in rows)),
                          ("split", collections.Counter(r["split"] for r in rows))):
        print(f"  {name:9}", "  ".join(f"{k}={v}" for k, v in counter.most_common()))
    written = pb.IDIOM_USE["idiom"] + pb.IDIOM_USE["composed"]
    if written:
        share = pb.IDIOM_USE["idiom"] / written
        print(f"  {'wording':9} idiom={pb.IDIOM_USE['idiom']} composed={pb.IDIOM_USE['composed']}"
              f"  ({share:.0%} idiomatic; IDIOM_RATE={pb.IDIOM_RATE:g} of the shapes a table covers)")


if __name__ == "__main__":
    main()
