"""Score a fine-tuned model on the held-out split, on the things the task is actually judged by.

    source /home/tak/isaacsim/env_llm/bin/activate
    python scripts/llm/eval_model.py --adapter logs/llm/qwen3-1.7b-dora/checkpoint-284
    python scripts/llm/eval_model.py --base-only          # what the model does before fine-tuning

Eval loss is not the number that matters here. Writing ``forward_left`` where the instruction said
``forward`` costs one token of loss and the whole task; emitting a jump the robot cannot land reads
as perfectly fluent Japanese. So every metric below is computed on the *program*, or on whether a
required sentence is present in the reply.

The scoring functions are imported, not rewritten: ``programs_match`` comes from
``roundtrip_check.py`` (the same 20% / rounding tolerance the dataset was verified under) and the
grammar check from ``unitree_rl_lab.program``. A separately written scorer would drift towards being
kinder than the standard the data was built to.

One thing is deliberately looser than ``check_dataset.py``: that script requires a caution sentence
to be one of the phrase-bank's own, which is right for generated data and wrong for a model that is
supposed to paraphrase. Here a caution counts if the reply carries the *meaning*, matched on
keywords -- a fuzzy test, so both numbers are reported and disagreements are worth reading by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phrasebank as pb  # noqa: E402
from chat_format import FormatError, parse_output  # noqa: E402
from roundtrip_check import programs_match  # noqa: E402
from unitree_rl_lab.program import ProgramError, program_from_json  # noqa: E402

NO_PROGRAM = {"chitchat", "impossible", "ambiguous"}
FORBIDDEN = ["必ず", "ぴったり", "絶対", "100%", "確実に"]
DECLINE_BELOW = 0.10
WARN_BELOW = 0.80
CAUTION_FOR_KIND = {"frontflip": "frontflip_landing", "handstand": "handstand_descent", "hindstand": "hindstand_descent"}

# The phrase bank's own sentences (strict) and what the caution has to *mean* (loose). A fine-tuned
# model paraphrases, so the loose test is the one the score is read from.
STRICT_CAUTIONS = {label: set(sum(by_style.values(), [])) for label, by_style in pb.CAUTIONS.items()}
TROUBLE = ["失敗", "乱れ", "こけ", "転ぶ", "転ぶかも", "危", "ミス", "苦手", "ふらつ", "不安定",
           "戻れな", "微妙", "堪忍", "ごめん", "下手", "自信な"]
LOOSE_CAUTIONS = {
    "frontflip_landing": ["着地", "前転", "前方回転"],
    "handstand_descent": ["降り", "戻る", "戻り", "倒立"],
    "hindstand_descent": ["降り", "戻る", "戻り", "後ろ足", "二足"],
}
# handstand and hindstand cautions are the same sentence about coming down; keywords cannot tell
# them apart, so an unearned descent warning is only counted when no stance was asked for at all.
CAUTION_KIND = {"frontflip_landing": "frontflip", "handstand_descent": "handstand",
                "hindstand_descent": "hindstand"}


def has_caution(reply: str, label: str, strict: bool) -> bool:
    """Strict: one of the phrase bank's own sentences. Loose: that, or the meaning in any wording."""
    if any(sentence in reply for sentence in STRICT_CAUTIONS[label]):
        return True
    if strict:
        return False
    return any(word in reply for word in LOOSE_CAUTIONS[label]) and any(word in reply for word in TROUBLE)


def generate(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int) -> tuple[list[str], float]:
    """Greedy decoding, batched. Greedy so two runs of the same adapter give the same score."""
    outputs, started, new_tokens = [], time.time(), 0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for row, prompt_len in zip(ids, [inputs.input_ids.shape[1]] * len(batch)):
            grown = row[prompt_len:]
            new_tokens += int((grown != tokenizer.pad_token_id).sum())
            outputs.append(tokenizer.decode(grown, skip_special_tokens=True))
        done = start + len(batch)
        print(f"\r  generated {done}/{len(prompts)}", end="", flush=True)
    elapsed = time.time() - started
    print(f"\r  generated {len(prompts)} replies in {elapsed:.0f}s  ({new_tokens / elapsed:.0f} tok/s)")
    return outputs, elapsed


def score(rows: list[dict], generations: list[str], rates: dict[str, float]) -> tuple[dict, list[dict]]:
    counts: Counter = Counter()
    by_category: dict[str, list[int]] = defaultdict(list)
    details = []

    for row, text in zip(rows, generations):
        verdict = {"id": row["id"], "category": row["category"], "input": row["input"], "generated": text}
        counts["rows"] += 1

        try:
            reply, program = parse_output(text)
        except FormatError as exc:
            verdict["fail"] = f"format: {exc}"
            details.append(verdict)
            by_category[row["category"]].append(0)
            continue
        counts["parse_ok"] += 1
        verdict["reply"], verdict["program"] = reply, program

        try:
            program_from_json(program)
            counts["grammar_ok"] += 1
        except ProgramError as exc:
            verdict["fail"] = f"grammar: {exc}"
            details.append(verdict)
            by_category[row["category"]].append(0)
            continue

        want = row["output"]["program"]
        reason = programs_match(want, program)
        matched = reason is None
        counts["program_match"] += matched
        counts["program_exact"] += program == want
        by_category[row["category"]].append(int(matched))
        if not matched:
            verdict["fail"] = f"program: {reason}"

        if row["category"] in NO_PROGRAM:
            counts["no_program_rows"] += 1
            counts["no_program_ok"] += program == []

        kinds = [step.get("kind") for step in program if step.get("kind")]
        if any(rates.get(kind, 1.0) < DECLINE_BELOW for kind in kinds):
            counts["declined_leak"] += 1
            verdict["fail"] = (verdict.get("fail", "") + " | emits a skill that must be declined").strip(" |")

        wanted = {CAUTION_FOR_KIND[k] for k in kinds if rates.get(k, 1.0) < WARN_BELOW and k in CAUTION_FOR_KIND}
        if row["category"] != "over_ask":
            for label in wanted:
                counts["caution_wanted"] += 1
                counts["caution_loose"] += has_caution(reply, label, strict=False)
                counts["caution_strict"] += has_caution(reply, label, strict=True)
            stance_asked = bool({"handstand", "hindstand"} & set(kinds))
            for label in set(LOOSE_CAUTIONS) - wanted:
                if not has_caution(reply, label, strict=False):
                    continue
                if CAUTION_KIND[label] in ("handstand", "hindstand") and stance_asked:
                    continue  # the same sentence, credited to the stance that was actually asked for
                counts["caution_unearned"] += 1

        if any(word in reply for word in FORBIDDEN):
            counts["forbidden"] += 1
            verdict["fail"] = (verdict.get("fail", "") + " | forbidden wording").strip(" |")

        counts["reply_chars"] += len(reply)
        if "fail" in verdict:
            details.append(verdict)

    return {"counts": counts, "by_category": by_category}, details


def report(result: dict) -> None:
    counts, by_category = result["counts"], result["by_category"]
    n = counts["rows"]

    def line(label: str, hits: int, total: int) -> None:
        print(f"  {label:<26} {hits:>4}/{total:<5} {hits / total:6.1%}" if total else f"  {label:<26}    n/a")

    print("\n=== eval ===")
    line("format parses", counts["parse_ok"], n)
    line("grammar valid", counts["grammar_ok"], n)
    line("program matches", counts["program_match"], n)
    line("program byte-identical", counts["program_exact"], n)
    line("empty when it must be", counts["no_program_ok"], counts["no_program_rows"])
    line("caution present (loose)", counts["caution_loose"], counts["caution_wanted"])
    line("caution present (strict)", counts["caution_strict"], counts["caution_wanted"])
    print(f"  {'declined skill emitted':<26} {counts['declined_leak']:>4}        (must be 0)")
    print(f"  {'unearned caution':<26} {counts['caution_unearned']:>4}")
    print(f"  {'forbidden wording':<26} {counts['forbidden']:>4}        (must be 0)")
    print(f"  {'mean reply length':<26} {counts['reply_chars'] / max(1, counts['parse_ok']):>6.1f} chars")

    print("\n  program matches by category")
    for category, hits in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"    {category:<18} {sum(hits):>3}/{len(hits):<4} {sum(hits) / len(hits):6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="/home/tak/models/Qwen3-1.7B")
    parser.add_argument("--adapter", help="a DoRA adapter directory, or a Trainer checkpoint-N")
    parser.add_argument("--base-only", action="store_true", help="score the base model, for comparison")
    parser.add_argument("--dataset", default="data/llm/dataset.jsonl")
    parser.add_argument("--system-prompt", default="data/llm/system_prompt.txt")
    parser.add_argument("--rates", default="data/llm/rates.json")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", help="per-row generations and verdicts, as JSONL")
    args = parser.parse_args()

    if not args.adapter and not args.base_only:
        parser.error("give --adapter, or --base-only to score the untuned model")

    system_prompt = Path(args.system_prompt).read_text()
    if args.adapter:  # the prompt the adapter was trained under wins over the current one
        trained_with = Path(args.adapter).parent / "system_prompt.txt"
        if trained_with.exists() and trained_with.read_text() != system_prompt:
            print(f"! {args.system_prompt} differs from the prompt this adapter was trained under; using the latter")
            system_prompt = trained_with.read_text()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"adapter: {args.adapter}")
    model.eval()

    rows = [json.loads(line) for line in open(args.dataset)]
    rows = [row for row in rows if row["split"] == args.split][: args.limit]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": row["input"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for row in rows
    ]
    print(f"{len(rows)} rows from split '{args.split}'")

    generations, _ = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)
    result, details = score(rows, generations, json.load(open(args.rates)))
    report(result)

    if args.out:
        with open(args.out, "w") as out:
            for verdict in details:
                out.write(json.dumps(verdict, ensure_ascii=False) + "\n")
        print(f"\n{len(details)} failing rows -> {args.out}")


if __name__ == "__main__":
    main()
