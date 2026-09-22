"""DoRA fine-tune of Qwen3-1.7B on the instruction dataset. Runs in the ``env_llm`` venv.

    source /home/tak/isaacsim/env_llm/bin/activate
    python scripts/llm/train_sft.py --out logs/llm/qwen3-1.7b-dora

What the model is taught, per row:

    <|im_start|>system\\n{system prompt}<|im_end|>          <- masked, loss is not taken here
    <|im_start|>user\\n{instruction}<|im_end|>              <- masked
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n      <- masked: the empty think block the
    {reply}\\n\\nprogram: [...]<|im_end|>                    non-thinking template always inserts
    <|im_start|>user\\n{next turn}<|im_end|>                 <- masked; the conversation continues as a
    <|im_start|>assistant\\n<think>...  {next answer}<|im_end|>   raw stream (see chat_format)

Two details decide whether this works at all.

**The empty think block.** Qwen3 is a hybrid model; ``enable_thinking=False`` does not remove the
reasoning channel, it pre-fills it with an empty ``<think></think>``. That block therefore belongs
to the *prompt*, not to the target -- if it is left in the target the model learns to open a think
block itself and will sit there reasoning at inference, on a robot, for no reason.

**Separate tokenisation.** The prompt and the target are tokenised apart and concatenated, never as
one string. At inference the model is handed exactly the prompt tokens, so that is the boundary it
must be trained on; tokenising the two together can merge the last prompt character with the first
reply character into a token the model will never see when it matters.

The trainer is ``transformers.Trainer`` with a hand-written collator rather than TRL's ``SFTTrainer``:
the masking above is the whole point of the run, and it is worth seeing it done in twenty lines
instead of inferring it from a config flag.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import conversation_text, first_prompt, render_output, render_user_turn, row_turns  # noqa: E402

IGNORE = -100


def build_rows(dataset: str, system_prompt: str, tokenizer, max_len: int, limit: int | None):
    """Tokenise every row into ``input_ids`` and ``labels``, masking everything but the answers.

    A row is a conversation; the loss is taken on every assistant turn, each one tokenised apart
    from the prompt text before it so the boundary is the one inference has. The prompt text is
    the raw stream `chat_format.conversation_text` describes, checked here against the template.
    """
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos != tokenizer.eos_token_id:
        print(f"  note: eos_token_id is {tokenizer.eos_token_id}, using <|im_end|> = {eos}")
    templated = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "x"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert templated == first_prompt(system_prompt, "x"), "chat_format.first_prompt drifted from the tokenizer's template"

    train, evals, dropped = [], [], 0
    for line in open(dataset):
        row = json.loads(line)
        ids, labels, n_answer = [], [], 0
        user_texts, answers = [], []
        for user_text, queue, reply, program in row_turns(row):
            user_texts.append(render_user_turn(user_text, queue))
            prompt = conversation_text(system_prompt, user_texts, answers)
            answer = render_output(reply, program)
            answers.append(answer)
            # Only the part of the prompt not yet tokenised: the first turn whole, then each
            # "<|im_end|>\n<|im_start|>user..." continuation after the previous answer.
            delta = prompt if not ids else prompt[len(prompt_so_far):]
            delta_ids = tokenizer(delta, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
            ids += delta_ids + answer_ids + [eos]
            labels += [IGNORE] * len(delta_ids) + answer_ids + [eos]
            n_answer += len(answer_ids) + 1
            # The <|im_end|> was appended as a token above; the "\n" after it is part of the next delta.
            prompt_so_far = prompt + answer + "<|im_end|>"
        if len(ids) > max_len:
            dropped += 1
            continue
        item = {"input_ids": ids, "labels": labels, "n_answer": n_answer, "turns": len(answers)}
        (evals if row["split"] == "eval" else train).append(item)

    if dropped:
        print(f"  dropped {dropped} rows longer than --max-len {max_len}")
    if limit:
        train, evals = train[:limit], evals[: max(1, limit // 8)]
    return train, evals


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        pad = width - len(item["input_ids"])
        out["input_ids"].append(item["input_ids"] + [pad_id] * pad)
        out["labels"].append(item["labels"] + [IGNORE] * pad)
        out["attention_mask"].append([1] * len(item["input_ids"]) + [0] * pad)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in out.items()}


def show_example(tokenizer, item: dict) -> None:
    """Print one tokenised row with the masked boundary visible -- read this before a long run."""
    ids, labels = item["input_ids"], item["labels"]
    print("\n--- one training row ---")
    starts = [i for i in range(len(ids)) if labels[i] != IGNORE and (i == 0 or labels[i - 1] == IGNORE)]
    for k, cut in enumerate(starts):
        print(f"  turn {k + 1} prompt ends : ...{tokenizer.decode(ids[max(0, cut - 24):cut])!r}")
        print(f"  turn {k + 1} loss starts : {tokenizer.decode(ids[cut:cut + 20])!r}")
    print(f"  loss ends   : ...{tokenizer.decode(ids[-12:])!r}")
    print(f"  tokens      : {len(ids)} total, {item['n_answer']} supervised over {item['turns']} turn(s)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="/home/tak/models/Qwen3-1.7B")
    parser.add_argument("--dataset", default="data/llm/dataset.jsonl")
    parser.add_argument("--system-prompt", default="data/llm/system_prompt.txt")
    parser.add_argument("--out", default="logs/llm/qwen3-1.7b-dora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4, help="DoRA diverges above ~1e-4")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lora", action="store_true", help="plain LoRA instead of DoRA, for comparison")
    parser.add_argument("--batch-size", type=int, default=4, help="8 fitted 1280-token rows; the ~2000-token conversations need 4")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=2304, help="prompt ~1400 tokens + up to three turns")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="recompute activations in the backward pass instead of storing them. "
                             "About 30%% slower and several times smaller: the 1.7B run holds every "
                             "activation for 4 x 2304 tokens and peaks near 90 GB, which a 4B model "
                             "cannot do on one card. Off by default so the 1.7B numbers stay comparable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="tiny run, to check the wiring")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    system_prompt = Path(args.system_prompt).read_text()

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    train, evals = build_rows(args.dataset, system_prompt, tokenizer, args.max_len, args.limit)
    lengths = [len(item["input_ids"]) for item in train]
    print(f"  train {len(train)} / eval {len(evals)}   tokens per row: "
          f"mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}   "
          f"supervised {sum(item['n_answer'] for item in train) / len(train):.0f} on average")
    show_example(tokenizer, train[0])

    peft_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules="all-linear",  # q,k,v,o,gate,up,down in every block; lm_head is excluded
        bias="none", use_dora=not args.lora, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if args.grad_checkpoint:
        # Without this the checkpointed blocks see inputs that do not require grad -- every base
        # weight is frozen -- and the backward pass finds nothing to recompute through.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(total_steps * 0.03)),
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_checkpoint else None,
        logging_steps=10,
        eval_strategy="epoch" if evals else "no",
        # One adapter per epoch. Eval loss says when the model stops improving on *token* prediction;
        # which epoch actually writes the best programs is decided after the run, so keep them all.
        save_strategy="epoch",
        save_total_limit=4,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=evals or None,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id or tokenizer.eos_token_id),
        processing_class=tokenizer,
    )
    print(f"\n{total_steps} optimiser steps "
          f"(batch {args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum})")
    trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    # The prompt is part of the trained artefact: a model fine-tuned under one prompt and served
    # under another is a silent quality loss, so the exact text travels with the weights.
    shutil.copy(args.system_prompt, out / "system_prompt.txt")
    json.dump({**vars(args), "base_model": args.model, "dora": not args.lora,
               "train_rows": len(train), "eval_rows": len(evals), "steps": total_steps},
              open(out / "train_config.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nadapter -> {out / 'adapter'}")


if __name__ == "__main__":
    main()
