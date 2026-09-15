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
from chat_format import FormatError, conversation_text, parse_output, render_output, row_turns, strip_state  # noqa: E402
from roundtrip_check import programs_match  # noqa: E402
from unitree_rl_lab.program import ProgramError, compile_program, program_from_json  # noqa: E402

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
    "running_frontflip_repeat": ["1回"],
    "frontflip_landing": ["着地", "前転", "前方回転"],
    "handstand_descent": ["降り", "戻る", "戻り", "倒立"],
    "hindstand_descent": ["降り", "戻る", "戻り", "後ろ足", "二足"],
}
# handstand and hindstand cautions are the same sentence about coming down; keywords cannot tell
# them apart, so an unearned descent warning is only counted when no stance was asked for at all.
CAUTION_KIND = {"frontflip_landing": "frontflip", "handstand_descent": "handstand",
                "hindstand_descent": "hindstand", "running_frontflip_repeat": "frontflip"}


def has_caution(reply: str, label: str, strict: bool) -> bool:
    """Strict: one of the phrase bank's own sentences. Loose: that, or the meaning in any wording."""
    if any(sentence in reply for sentence in STRICT_CAUTIONS[label]):
        return True
    if strict:
        return False
    if label == "running_frontflip_repeat":  # a clamp notice, not a warning: "1回" is the content
        return "1回" in reply
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


def items_of(rows: list[dict]) -> list[dict]:
    """One scoring item per assistant turn, with the gold history before it (teacher-forced)."""
    items = []
    for row in rows:
        user_texts, answers = [], []
        for k, (user_text, reply, action, program) in enumerate(row_turns(row)):
            user_texts.append(user_text)
            state = row["turns"][2 * k].get("state", {}) if "turns" in row else {}
            any_program = bool(row["turns"][2 * k + 1].get("any_program")) if "turns" in row else False
            items.append({"id": f"{row['id']} t{k}", "row": row, "turn": k, "category": row["category"],
                          "user_texts": list(user_texts), "answers": list(answers), "state": state,
                          "want_action": action, "want_program": program, "any_program": any_program,
                          "input": strip_state(user_text)})
            try:
                answers.append(render_output(reply, action, program))
            except FormatError:  # a hand-written "any program" turn has no gold program to show as history
                answers.append(render_output(reply, "none", []))
    return items


def score(items: list[dict], generations: list[str], rates: dict[str, float]) -> tuple[dict, list[dict]]:
    counts: Counter = Counter()
    by_category: dict[str, list[int]] = defaultdict(list)
    action_confusion: Counter = Counter()
    details = []

    for item, text in zip(items, generations):
        row, category = item["row"], item["category"]
        verdict = {"id": item["id"], "category": category, "input": item["input"], "generated": text,
                   "want_action": item["want_action"], "want_program": item["want_program"]}
        counts["rows"] += 1

        try:
            reply, action, program = parse_output(text)
        except FormatError as exc:
            verdict["fail"] = f"format: {exc}"
            details.append(verdict)
            by_category[category].append(0)
            continue
        counts["parse_ok"] += 1
        verdict["reply"], verdict["action"], verdict["program"] = reply, action, program

        action_ok = action == item["want_action"]
        counts["action_ok"] += action_ok
        action_confusion[(item["want_action"], action)] += 1
        if item["want_action"] in ("none", "cancel") or action in ("none", "cancel"):
            counts["stop_decisions"] += 1
            counts["stop_decisions_ok"] += action_ok

        state = item["state"]
        try:
            context = None
            if action == "insert" and state.get("running"):
                context = program_from_json([state["program"][state["step_index"]]])[0]
            compile_program(program_from_json(program), context=context, resume=context is not None)
            counts["grammar_ok"] += 1
        except (ProgramError, IndexError) as exc:
            verdict["fail"] = f"grammar: {exc}"
            details.append(verdict)
            by_category[category].append(0)
            continue

        want = item["want_program"]
        # a hand-written case may accept any valid program for its action (the ask had no numbers)
        reason = None if item["any_program"] and program else programs_match(want, program)
        matched = reason is None and action_ok
        counts["program_match"] += matched
        counts["program_exact"] += program == want and action_ok
        by_category[category].append(int(matched))
        if reason is not None:
            verdict["fail"] = f"program: {reason}"
        elif not action_ok:
            verdict["fail"] = f"action: wanted {item['want_action']}, got {action}"

        if category in NO_PROGRAM or item["want_action"] in ("none", "cancel"):
            counts["no_program_rows"] += 1
            counts["no_program_ok"] += program == []

        keys = [f"{s['kind']}:running" if s.get("running") else s["kind"] for s in program if s.get("kind")]
        kinds = [key.split(":")[0] for key in keys]
        if any(rates.get(key, 1.0) < DECLINE_BELOW for key in keys):
            counts["declined_leak"] += 1
            verdict["fail"] = (verdict.get("fail", "") + " | emits a skill that must be declined").strip(" |")

        wanted = {CAUTION_FOR_KIND[k] for k, key in zip(kinds, keys)
                  if rates.get(key, 1.0) < WARN_BELOW and k in CAUTION_FOR_KIND}
        if any(s.get("running") and s.get("kind") == "frontflip" and s.get("count", 1) > 1 for s in program):
            wanted.add("running_frontflip_repeat")
        if category != "over_ask":
            for label in wanted:
                counts["caution_wanted"] += 1
                counts["caution_loose"] += has_caution(reply, label, strict=False)
                counts["caution_strict"] += has_caution(reply, label, strict=True)
            stance_asked = bool({"handstand", "hindstand"} & set(kinds))
            for label in (set(LOOSE_CAUTIONS) - wanted) if program else ():  # a proposal may well say 着地

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

    return {"counts": counts, "by_category": by_category, "action_confusion": action_confusion}, details


def report(result: dict) -> None:
    counts, by_category = result["counts"], result["by_category"]
    n = counts["rows"]

    def line(label: str, hits: int, total: int) -> None:
        print(f"  {label:<26} {hits:>4}/{total:<5} {hits / total:6.1%}" if total else f"  {label:<26}    n/a")

    print("\n=== eval ===")
    line("format parses", counts["parse_ok"], n)
    line("action correct", counts["action_ok"], n)
    line("stop-or-not correct", counts["stop_decisions_ok"], counts["stop_decisions"])
    line("grammar valid", counts["grammar_ok"], n)
    line("action+program match", counts["program_match"], n)
    line("action+program identical", counts["program_exact"], n)
    line("empty when it must be", counts["no_program_ok"], counts["no_program_rows"])
    line("caution present (loose)", counts["caution_loose"], counts["caution_wanted"])
    line("caution present (strict)", counts["caution_strict"], counts["caution_wanted"])
    print(f"  {'declined skill emitted':<26} {counts['declined_leak']:>4}        (must be 0)")
    print(f"  {'unearned caution':<26} {counts['caution_unearned']:>4}")
    print(f"  {'forbidden wording':<26} {counts['forbidden']:>4}        (must be 0)")
    print(f"  {'mean reply length':<26} {counts['reply_chars'] / max(1, counts['parse_ok']):>6.1f} chars")

    print("\n  action+program matches by category")
    for category, hits in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"    {category:<20} {sum(hits):>3}/{len(hits):<4} {sum(hits) / len(hits):6.1%}")
    wrong = [(pair, n) for pair, n in result["action_confusion"].items() if pair[0] != pair[1]]
    if wrong:
        print("\n  action confusions (wanted -> got)")
        for (want, got), n in sorted(wrong, key=lambda kv: -kv[1]):
            print(f"    {want:<8} -> {got:<8} {n}")


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
    items = items_of(rows)
    # Every assistant turn is scored with the *gold* history before it, so a wrong second turn is
    # the model's own mistake and not the echo of a wrong first one.
    prompts = [conversation_text(system_prompt, item["user_texts"], item["answers"]) for item in items]
    print(f"{len(rows)} rows, {len(items)} assistant turns from split '{args.split}'")

    generations, _ = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)
    result, details = score(items, generations, json.load(open(args.rates)))
    report(result)

    if args.out:
        with open(args.out, "w") as out:
            for verdict in details:
                out.write(json.dumps(verdict, ensure_ascii=False) + "\n")
        print(f"\n{len(details)} failing turns -> {args.out}")


if __name__ == "__main__":
    main()
