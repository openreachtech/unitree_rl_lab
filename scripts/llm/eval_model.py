"""Score a fine-tuned adapter the way the robot will use it: one program per turn.

    source /home/tak/isaacsim/env_llm/bin/activate
    python scripts/llm/eval_model.py logs/llm/qwen3-1.7b-dora/adapter
    python scripts/llm/eval_model.py logs/llm/qwen3-1.7b-dora/checkpoint-444 --show 20

Every assistant turn of the eval split is one question. The prompt is built by
``chat_format.conversation_text`` -- byte-identical to what ``conductor`` sends -- the model
generates freely, ``parse_output`` reads the answer back, and the program is compared with the one
the dataset holds. Three numbers come out of that:

    完全一致   the program is the dataset's program, field for field
    ≈一致      the same steps and directions, amounts within ``roundtrip_check``'s 20%
              ("3m" answered as "2.8m" is a translation this system accepts)
    形式エラー  no ``program:`` line, or not JSON. llama.cpp constrains sampling with the GBNF so
              these cannot happen in service -- here they are a symptom, not an outage

Some sentences do not name a side. 「後ろ向いて」 and 「一周」 are obeyed by turning either way, and
the dataset had to write one of them down; answering with the other is right and would score wrong.
So where ``reparse.jsonl`` holds an independent reading of that turn -- it encodes the freedom as
``dir: "any"`` -- an answer matching it counts as 一致 too.

**History is the dataset's replies, not the model's own.** A wrong reply on turn 2 would otherwise
decide turn 3, and the score would stop being per-turn and start being per-conversation. So this
measures each turn given a correct past; the conductor feeds back what the model actually said, and
a long session can drift below this. Read it as the ceiling.

Sampling is greedy and unconstrained. The grammar is deliberately not applied: a model that needs
the grammar to stay in format has not learned the format.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import (  # noqa: E402
    FormatError, conversation_text, parse_output, render_user_turn, row_turns,
)
from roundtrip_check import programs_match  # noqa: E402

try:
    from unitree_rl_lab.program.grammar import program_from_json, program_to_json
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))
    from unitree_rl_lab.program.grammar import program_from_json, program_to_json


def canonical(program: list[dict]) -> str:
    """The program as the grammar writes it, so field order and absent fields cannot cause a miss."""
    return program_to_json(program_from_json(program))


def questions(dataset: str, split: str, system_prompt: str, limit: int | None) -> list[dict]:
    """One entry per assistant turn: the exact prompt for it, and what the dataset answers."""
    out = []
    for line in open(dataset):
        row = json.loads(line)
        if row["split"] != split:
            continue
        user_texts, replies = [], []
        for turn, (user_text, queue, reply, program) in enumerate(row_turns(row), start=1):
            user_texts.append(render_user_turn(user_text, queue))
            out.append({"id": row["id"], "category": row["category"], "turn": turn,
                        "prompt": conversation_text(system_prompt, user_texts, replies),
                        "said": user_text, "program": program})
            replies.append(reply)
        if limit and len(out) >= limit:
            break
    return out


def generate(model, tokenizer, prompts: list[str], max_new: int) -> list[str]:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
    width = batch["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, width:], skip_special_tokens=True)


def judge(item: dict, text: str, reparse: list[dict] | None = None) -> dict:
    """What the generation was worth, and why -- one record, so failures can be printed or written."""
    result = {**{k: item[k] for k in ("id", "category", "turn", "said")}, "raw": text}
    try:
        output = parse_output(text)
    except FormatError as exc:
        return {**result, "verdict": "format", "reason": str(exc)}
    result["reply"] = output.reply
    try:
        got = canonical(output.program)
    except Exception as exc:  # a parseable list that is not a program: wrong fields, bad enum
        return {**result, "verdict": "grammar", "reason": str(exc)}
    want = canonical(item["program"])
    if got == want:
        return {**result, "verdict": "exact"}
    reason = programs_match(item["program"], output.program)
    if reason is not None and reparse is not None and programs_match(reparse, output.program) is None:
        reason = None  # the sentence named no side, and the answer took the other one
    return {**result, "verdict": "close" if reason is None else "wrong",
            "reason": reason or f"{want} -> {got}", "got": got, "want": want}


def report(records: list[dict], show: int) -> None:
    order = ["exact", "close", "wrong", "format", "grammar"]
    total = len(records)
    counts = Counter(r["verdict"] for r in records)
    ok = counts["exact"] + counts["close"]
    print(f"\n{'=' * 78}\n{total} ターン   完全一致 {counts['exact']} ({counts['exact'] / total:.1%})"
          f"   ≈一致込み {ok} ({ok / total:.1%})")
    print("  " + "   ".join(f"{name} {counts[name]}" for name in order))

    def table(title: str, key) -> None:
        groups = defaultdict(list)
        for record in records:
            groups[key(record)].append(record)
        print(f"\n{title:<16}{'n':>6}{'完全':>8}{'≈込み':>8}   内訳")
        for name, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            c = Counter(r["verdict"] for r in group)
            good = c["exact"] + c["close"]
            bad = "  ".join(f"{k}:{c[k]}" for k in order[2:] if c[k])
            print(f"  {str(name):<14}{len(group):>6}{c['exact'] / len(group):>8.0%}"
                  f"{good / len(group):>8.0%}   {bad}")

    table("カテゴリ", lambda r: r["category"])
    table("ターン目", lambda r: r["turn"])

    failures = [r for r in records if r["verdict"] != "exact"]
    if show and failures:
        print(f"\n--- 外したもの {min(show, len(failures))}/{len(failures)} ---")
        for record in failures[:show]:
            print(f"  [{record['verdict']}] {record['id']} turn {record['turn']} ({record['category']})")
            print(f"      人   {record['said']}")
            print(f"      Go2  {record.get('reply', record['raw'][:80])!r}")
            print(f"      → {record.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", help="an adapter directory, or a checkpoint-N inside the run")
    parser.add_argument("--base", help="override the base model recorded in adapter_config.json")
    parser.add_argument("--dataset", default="data/llm/dataset.jsonl")
    parser.add_argument("--reparse", default="data/llm/reparse.jsonl",
                        help="independent readings, to forgive a turn whose sentence named no side")
    parser.add_argument("--system-prompt", help="default: the copy saved beside the adapter")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=320,
                        help="a 5-step program plus the reply runs past 200 tokens; a cut-off answer scores as a format error that is the harness's fault, not the model's")
    parser.add_argument("--limit", type=int, help="first N turns only, to check the wiring")
    parser.add_argument("--show", type=int, default=12, help="print this many misses")
    parser.add_argument("--out", help="write every judged turn here as jsonl")
    args = parser.parse_args()

    adapter = Path(args.adapter)
    config = json.load(open(adapter / "adapter_config.json"))
    base = args.base or config["base_model_name_or_path"]
    # The prompt is part of the trained artefact: train_sft.py saves it next to the adapter, and a
    # checkpoint sits one level below that.
    candidates = [adapter / "system_prompt.txt", adapter.parent / "system_prompt.txt"]
    prompt_path = Path(args.system_prompt) if args.system_prompt else next(
        (p for p in candidates if p.exists()), None)
    if prompt_path is None or not prompt_path.exists():
        sys.exit(f"no system prompt beside the adapter (looked in {[str(p) for p in candidates]}); "
                 f"pass --system-prompt, and it must be the one the run trained under")
    system_prompt = prompt_path.read_text()
    print(f"base    {base}\nadapter {adapter}\nprompt  {prompt_path}")

    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.padding_side = "left"  # generation reads from the right edge of the batch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    items = questions(args.dataset, args.split, system_prompt, args.limit)
    print(f"{len(items)} turns in the {args.split} split\n")

    reparse = {}
    if Path(args.reparse).exists():
        for line in open(args.reparse):
            entry = json.loads(line)
            reparse[(entry["id"], entry["turn"] + 1)] = entry["reparsed_program"]  # reparse counts from 0

    # Batched greedy decoding pays for its longest member twice over -- padding on the prompt and
    # waiting on the generation -- so neighbours should be the same size. Order is restored after.
    items.sort(key=lambda i: len(i["prompt"]))

    records = []
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        for item, text in zip(chunk, generate(model, tokenizer, [i["prompt"] for i in chunk], args.max_new)):
            records.append(judge(item, text, reparse.get((item["id"], item["turn"]))))
        done = len(records)
        exact = sum(1 for r in records if r["verdict"] == "exact")
        print(f"\r  {done}/{len(items)}  完全一致 {exact / done:.1%}", end="", flush=True)

    records.sort(key=lambda r: (r["id"], r["turn"]))
    report(records, args.show)
    if args.out:
        with open(args.out, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n{args.out} に全ターン書き出した")


if __name__ == "__main__":
    main()
