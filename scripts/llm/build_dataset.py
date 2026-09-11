"""Build the fine-tuning dataset from validated programs, the capability table and the phrase bank.

    python scripts/llm/build_dataset.py --out data/llm/dataset.jsonl --total 5000

Every row is one (instruction, reply + program) pair:

    {"id": "s0-000123#kansai", "split": "train", "category": "normal", "style": "kansai",
     "input": "前へ3mくらい進んでから、バク転2回してや。",
     "output": {"reply": "...", "program": [...]},
     "source": {"program_id": "s0-000123", "passed": true, "kind": "paraphrase"}}

Two things decide what a row looks like, and they are kept apart on purpose:

* **wording** comes from ``phrasebank.py`` / ``negatives.py`` -- hand-written Japanese, no numbers
  of its own;
* **decisions** (decline this step / warn about that one / how long it takes) come from the
  capability table and the sim measurements, never from the wording.

Reliability thresholds follow DATASET.md 2.1: decline a skill below 10% population success rate,
warn below 80%, say nothing at or above 80%. Those rates are *population* rates over the whole
2000-program run, not the n=8 replicas of the program at hand -- an instruction says
「前方回転して」 and cannot know which run it will get.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

import negatives as ng
import phrasebank as pb

DECLINE_BELOW = 0.10
WARN_BELOW = 0.80

CAUTION_LABEL = {
    "frontflip": "frontflip_landing",
    "handstand": "handstand_descent",
    "hindstand": "hindstand_descent",
}

SKILL_JA = {
    "backflip": "バク転", "frontflip": "前方回転", "sideflip_left": "左側転",
    "sideflip_right": "右側転", "jump": "ジャンプ", "handstand": "倒立", "hindstand": "後ろ足立ち",
}


# ------------------------------------------------------------------------------------ reliability


def load_rates(capability_path: str, validated: list[dict]) -> dict[str, float]:
    """Population success rate per flip kind, and per stance kind the rate of *getting back down*.

    The capability table's stance rate covers the rise and the hold only; the descent is scored in
    ``validated.jsonl`` as "did this replica recover to four feet", so it is counted here.
    """
    skills = json.load(open(capability_path))["skills"]
    rates = {kind: skills[f"flip:{kind}"]["rate"] for kind in SKILL_JA if f"flip:{kind}" in skills}

    total: dict[str, int] = {}
    recovered: dict[str, int] = {}
    for record in validated:
        for stance in record["stances"]:
            total[stance["kind"]] = total.get(stance["kind"], 0) + len(stance["recover_s"])
            recovered[stance["kind"]] = recovered.get(stance["kind"], 0) + sum(1 for x in stance["recover_s"] if x is not None)
    for kind, n in total.items():
        rates[kind] = recovered[kind] / n if n else 1.0
    return rates


def step_kind(step: dict) -> str | None:
    return step.get("kind") if step["skill"] in ("flip", "stance") else None


def classify(program: list[dict], rates: dict[str, float]) -> tuple[list[dict], list[str], list[str]]:
    """-> (steps to keep, kinds declined, caution labels for the kept steps)."""
    keep, declined, cautions = [], [], []
    for step in program:
        kind = step_kind(step)
        rate = rates.get(kind, 1.0) if kind else 1.0
        if rate < DECLINE_BELOW:
            if kind not in declined:
                declined.append(kind)
            continue
        keep.append(step)
        if rate < WARN_BELOW:
            label = CAUTION_LABEL.get(kind)
            if label and label not in cautions:
                cautions.append(label)
    return keep, declined, cautions


# ------------------------------------------------------------------------------------- replies


def total_time_sentence(seconds: float, style: str, rng: random.Random) -> str:
    return rng.choice(pb.TOTAL_TIME[style]).format(t=int(round(seconds)))


def build_reply(steps: list[dict], declined: list[str], cautions: list[str], style: str,
                duration_s: float | None, rng: random.Random) -> str:
    parts = [rng.choice(pb.OPENERS[style])]

    if declined:
        names = "と".join(SKILL_JA[kind] for kind in declined)
        if not steps:
            return rng.choice(pb.DECLINE_ALL[style if style != "short" else "short"]).format(skill=names)
        parts = [rng.choice(pb.DECLINE_STEP[style]).format(skill=names)]

    if steps:
        parts.append(pb.restate(steps, style, rng))
    for label in cautions[:2]:
        parts.append(rng.choice(pb.CAUTIONS[label][style]))
    if duration_s is not None and not declined and len(steps) >= 2 and rng.random() < 0.35:
        parts.append(total_time_sentence(duration_s, style, rng))
    tail = rng.choice(pb.REPLY_TAIL[style])
    if tail:
        parts.append(tail)
    return "".join(parts)


# ------------------------------------------------------------------------------------ rows


def split_for(key: str) -> str:
    return "eval" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


def program_row(record: dict, style: str, rates: dict[str, float], rng: random.Random) -> dict:
    program = record["program"]
    keep, declined, cautions = classify(program, rates)

    if declined:
        category = "partial_decline" if keep else "declined"
    elif "handstand_descent" in cautions:
        category = "risky_attempt"
    elif cautions:
        category = "minor_caution"
    else:
        category = "normal"

    instruction = pb.instruction(program, style, rng)
    reply = build_reply(keep, declined, cautions, style, record["timeline"]["duration_s"], rng)
    return {
        "id": f"{record['id']}#{style}",
        "split": split_for(record["id"]),
        "category": category,
        "style": style,
        "input": instruction,
        "output": {"reply": reply, "program": keep},
        "source": {"program_id": record["id"], "passed": record["passed"], "kind": "paraphrase"},
    }


def negative_row(index: int, category: str, style: str, input_text: str, reply: str,
                 program: list[dict]) -> dict:
    key = f"neg-{category}-{index:04d}"
    return {
        "id": f"{key}#{style}",
        "split": split_for(key),
        "category": category,
        "style": style,
        "input": input_text,
        "output": {"reply": reply, "program": program},
        "source": {"program_id": None, "passed": None, "kind": category},
    }


# ------------------------------------------------------------------------------------ negatives


def topic_rows(topics: list[ng.Topic], category: str, quota: int, rng: random.Random) -> list[dict]:
    combos = []
    for casual_in, polite_in, casual_re, polite_re in topics:
        combos += [("kansai", text, reply) for text in casual_in for reply in casual_re]
        combos += [("polite", text, reply) for text in polite_in for reply in polite_re]
    rng.shuffle(combos)
    return [negative_row(i, category, style, text, reply, [])
            for i, (style, text, reply) in enumerate(combos[:quota])]


def impossible_rows(quota: int, rng: random.Random) -> list[dict]:
    combos = []
    for action, missing in ng.IMPOSSIBLE:
        for style in ("kansai", "polite"):
            for template in ng.IMPOSSIBLE_INPUT[style]:
                combos.append((style, action, missing, template))
    rng.shuffle(combos)
    rows = []
    for i, (style, action, missing, template) in enumerate(combos[:quota]):
        text = template.format(a=te_form(action))
        reply = rng.choice(ng.IMPOSSIBLE_REPLY[style]).format(
            a=action, m=missing, f=rng.choice(ng.IMPOSSIBLE_FALLBACK[style]))
        rows.append(negative_row(i, "impossible", style, text, reply, []))
    return rows


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


def over_ask_combos(rng: random.Random) -> list[tuple]:
    combos = []
    for style in ("kansai", "polite", "short"):
        for kind, name in ng.OVER_ASK_FLIPS:
            combos += [("flip", style, kind, name, count) for count in (6, 7, 8, 10, 12, 15, 20, 30)]
        for kind, name in ng.OVER_ASK_STANCES:
            combos += [("stance", style, kind, name, seconds) for seconds in (15, 20, 30, 45, 60, 90, 120)]
        for direction in ("forward", "backward", "left", "right"):
            combos += [("move", style, direction, None, metres) for metres in (50, 80, 100, 150, 200, 300)]
    rng.shuffle(combos)
    return combos


def over_ask_rows(quota: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for index, (shape, style, key, name, value) in enumerate(over_ask_combos(rng)[:quota]):
        if shape == "flip":
            kind, count = key, value
            program = [{"skill": "flip", "kind": kind, "count": count}]
            text = {"kansai": f"{name}を{count}回連続でやって。", "polite": f"{name}を{count}回続けてください。",
                    "short": f"{name}{count}回。"}[style]
            reply = {
                "kansai": f"{count}回は多いから{ng.FLIP_LIMIT}回までにするで。{name}を{ng.FLIP_LIMIT}回いくわ。",
                "polite": f"{count}回は多いので{ng.FLIP_LIMIT}回までにします。{name}を{ng.FLIP_LIMIT}回行います。",
                "short": f"{count}回は多い。{ng.FLIP_LIMIT}回までにするで。",
            }[style]
        elif shape == "stance":
            kind, seconds = key, value
            program = [{"skill": "stance", "kind": kind, "duration_s": float(seconds), "speed": "slow"}]
            text = {"kansai": f"{name}を{seconds}秒キープして。", "polite": f"{name}を{seconds}秒続けてください。",
                    "short": f"{name}{seconds}秒。"}[style]
            limit = int(ng.STANCE_MAX_S)
            reply = {
                "kansai": f"{seconds}秒は長すぎるから{limit}秒までにするわ。降りるとき転ぶかもしれんで。",
                "polite": f"{seconds}秒は長すぎるため{limit}秒までにします。降りるときに転ぶことがあります。",
                "short": f"{seconds}秒は長い。{limit}秒までな。",
            }[style]
        else:
            direction, metres = key, value
            label = {"forward": "前", "backward": "後ろ", "left": "左", "right": "右"}[direction]
            program = [{"skill": "move", "dir": direction, "speed": "normal", "distance_m": float(metres)}]
            text = {"kansai": f"{label}に{metres}m進んで。", "polite": f"{label}へ{metres}m進んでください。",
                    "short": f"{label}{metres}m。"}[style]
            cap = int(ng.PROGRAM_MAX_S)
            reply = {
                "kansai": f"{metres}mは遠すぎるわ。一度に動けるんは{cap}秒までやから、そこまで進んで止まるで。",
                "polite": f"{metres}mは長すぎます。一度の指示で動けるのは{cap}秒までなので、そこまで進んで止まります。",
                "short": f"{metres}mは遠い。{cap}秒までしか進まれへん。",
            }[style]
        rows.append(negative_row(index, "over_ask", style, text, reply, program))
    return rows


# ------------------------------------------------------------------------------------------ main


def assign_styles(records: list[dict], rng: random.Random) -> list[tuple[dict, str]]:
    """Every program once in each style, ordered so a first pass covers every program."""
    pairs: list[tuple[dict, str]] = []
    order = list(records)
    rng.shuffle(order)
    styles = list(pb.STYLES)
    for round_index in range(len(styles)):
        for offset, record in enumerate(order):
            pairs.append((record, styles[(offset + round_index) % len(styles)]))
    return pairs


def take_rows(records: list[dict], quota: int, rates: dict[str, float], rng: random.Random,
              seen: set[tuple]) -> list[dict]:
    """Fill a quota from a pool, skipping rows that would duplicate one already written.

    Duplicates happen because the sampler drew the same short program twice, and a terse instruction
    for it ("バク転2回。") renders identically both times.
    """
    rows: list[dict] = []
    for record, style in assign_styles(records, rng):
        if len(rows) >= quota:
            break
        row = program_row(record, style, rates, rng)
        key = (row["input"], row["output"]["reply"], json.dumps(row["output"]["program"], ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


SYSTEM_PROMPT = """あなたは四足歩行ロボット Go2 です。日本語の指示を聞き、短い返事と実行するプログラムを返します。

出力は次の JSON ひとつだけ:
{{"reply": "<日本語の返事>", "program": [<手順>]}}

{grammar}

守ること:
- 指示に無い手順を足さない。指示された順番を変えない。
- 雑談・質問・ロボットにできない依頼には program を空リストにして、返事だけ返す。
- ジャンプ (flip kind "jump") は着地に成功しないので program に入れない。ほかの手順が
  あるときはジャンプだけを外し、断ったことを返事で言う。
- 前方回転 (frontflip) は約4回に1回着地に失敗する。倒立 (handstand) は降りるときに
  半分以上の確率で転ぶ。どちらも実行はするが、返事で一言ことわる。
- 実測にない約束をしない(「必ず成功する」「ぴったり5m」など)。
- 上限を超える要求はそのまま program に入れ、どこまでしか実行できないかを返事で言う
  (アクロバットは{flip_limit}回まで、二足立ちは{stance_max}秒まで、プログラム全体は{program_max}秒まで)。
- 返事は1〜2文。相手の口調(丁寧語/関西弁/短文)に合わせる。
"""


def write_system_prompt(path: str) -> None:
    """The prompt each row is trained under. Stored once, not repeated on every line."""
    from unitree_rl_lab.program import describe_grammar

    open(path, "w").write(SYSTEM_PROMPT.format(
        grammar=describe_grammar(), flip_limit=ng.FLIP_LIMIT,
        stance_max=int(ng.STANCE_MAX_S), program_max=int(ng.PROGRAM_MAX_S)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validated", default="data/llm/validated.jsonl")
    parser.add_argument("--capability", default="data/llm/capability.json")
    parser.add_argument("--out", default="data/llm/dataset.jsonl")
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--rates-out", default="data/llm/rates.json")
    parser.add_argument("--prompt-out", default="data/llm/system_prompt.txt")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    validated = [json.loads(line) for line in open(args.validated)]
    rates = load_rates(args.capability, validated)

    has_declined = [r for r in validated if any(rates.get(step_kind(s), 1.0) < DECLINE_BELOW for s in r["program"])]
    declined_ids = {r["id"] for r in has_declined}
    passed = [r for r in validated if r["passed"] and r["id"] not in declined_ids]
    failed = [r for r in validated if not r["passed"] and r["id"] not in declined_ids]

    quota_negatives = round(args.total * 0.156)
    quota_decline = round(args.total * 0.09)
    quota_failed = round(args.total * 0.09)
    quota_passed = args.total - quota_negatives - quota_decline - quota_failed

    seen: set[tuple] = set()
    rows: list[dict] = []
    rows += take_rows(passed, quota_passed, rates, rng, seen)
    rows += take_rows(failed, quota_failed, rates, rng, seen)
    rows += take_rows(has_declined, quota_decline, rates, rng, seen)

    share = {"chitchat": 0.385, "impossible": 0.282, "over_ask": 0.218, "ambiguous": 0.115}
    rows += topic_rows(ng.CHITCHAT, "chitchat", round(quota_negatives * share["chitchat"]), rng)
    rows += impossible_rows(round(quota_negatives * share["impossible"]), rng)
    rows += over_ask_rows(round(quota_negatives * share["over_ask"]), rng)
    rows += topic_rows(ng.AMBIGUOUS, "ambiguous", round(quota_negatives * share["ambiguous"]), rng)

    json.dump(rates, open(args.rates_out, "w"), ensure_ascii=False, indent=2, sort_keys=True)
    write_system_prompt(args.prompt_out)

    rng.shuffle(rows)
    with open(args.out, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    evals = sum(1 for row in rows if row["split"] == "eval")
    print(f"wrote {len(rows)} rows -> {args.out}   (train {len(rows) - evals} / eval {evals})")
    for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category:<16} {n}")
    print("  rates used:", {k: round(v, 3) for k, v in sorted(rates.items())})


if __name__ == "__main__":
    main()
