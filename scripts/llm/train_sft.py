"""DoRA fine-tune of Qwen3-1.7B on the instruction dataset. Runs in the ``env_llm`` venv.

    source /home/tak/isaacsim/env_llm/bin/activate
    python scripts/llm/train_sft.py --out logs/llm/qwen3-1.7b-dora

What the model is taught, per row:

    <|im_start|>system\\n{system prompt}<|im_end|>                            <- masked
    <|im_start|>user\\n{turn 1}<|im_end|>                                     <- masked, queue dropped
    <|im_start|>assistant\\n{reply 1}<|im_end|>                               <- masked, program dropped
    <|im_start|>user\\n{turn 2}\\n\\nqueued programs: [...]<|im_end|>          <- masked
    <|im_start|>assistant\\n{reply 2}\\n\\nprogram: [...]<|im_end|>            <- the target

Three details decide whether this works at all.

**One example per assistant turn, not per conversation.** A conversation cannot be a single growing
sequence. History drops the queue line the moment a turn is past and drops the program line off the
answer (``chat_format.strip_queue``, and what ``conductor`` keeps) -- so the prompt for turn 3 is
not the prompt for turn 2 with text appended, it is the same conversation re-rendered shorter. A
5-turn row therefore becomes 5 examples, each one the exact text the server will send for that turn.
Turns of one row share its split, so nothing leaks between train and eval.

**No think block.** Qwen3 is a hybrid model; ``enable_thinking=False`` does not remove the reasoning
channel, it pre-fills the generation prompt with an empty ``<think></think>``. We fine-tune, so the
format is ours: ``chat_format`` drops that block everywhere and the GBNF forbids a reply opening
with ``<``. Training has to drop it too -- if the model is trained behind a block the server never
sends, every turn starts off-distribution. The assert below still compares against the tokenizer's
template, with that one block removed, so a template change is caught.

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
EMPTY_THINK = "<think>\n\n</think>\n\n"
"""What ``enable_thinking=False`` still inserts. The wire format has no think block; see above."""


def build_rows(dataset: str, system_prompt: str, tokenizer, max_len: int, limit: int | None):
    """Tokenise every assistant turn into ``input_ids`` and ``labels``, masking everything but the answer.

    The prompt and the answer are tokenised apart and concatenated; the prompt text is the raw
    stream `chat_format.conversation_text` describes, checked here against the template.
    """
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos != tokenizer.eos_token_id:
        print(f"  note: eos_token_id is {tokenizer.eos_token_id}, using <|im_end|> = {eos}")
    templated = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "x"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert templated.replace(EMPTY_THINK, "") == first_prompt(system_prompt, "x"), \
        "chat_format.first_prompt drifted from the tokenizer's template"

    train, evals, dropped, rows = [], [], 0, 0
    for line in open(dataset):
        row = json.loads(line)
        rows += 1
        user_texts, answers = [], []
        for user_text, queue, reply, program in row_turns(row):
            user_texts.append(render_user_turn(user_text, queue))
            prompt = conversation_text(system_prompt, user_texts, answers)
            answers.append(reply)  # history is the words; the program line is not carried forward
            answer = render_output(reply, program)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
            if len(prompt_ids) + len(answer_ids) + 1 > max_len:
                dropped += 1
                continue
            item = {"input_ids": prompt_ids + answer_ids + [eos],
                    "labels": [IGNORE] * len(prompt_ids) + answer_ids + [eos],
                    "n_answer": len(answer_ids) + 1, "turn": len(answers)}
            (evals if row["split"] == "eval" else train).append(item)

    print(f"  {rows} conversations -> {len(train) + len(evals)} turns")
    if dropped:
        print(f"  dropped {dropped} turns longer than --max-len {max_len}")
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
    """Print one tokenised turn with the masked boundary visible -- read this before a long run."""
    ids, labels = item["input_ids"], item["labels"]
    cut = next(i for i in range(len(ids)) if labels[i] != IGNORE)
    print("\n--- one training turn ---")
    print(f"  prompt ends : ...{tokenizer.decode(ids[max(0, cut - 40):cut])!r}")
    print(f"  loss starts : {tokenizer.decode(ids[cut:cut + 24])!r}")
    print(f"  loss ends   : ...{tokenizer.decode(ids[-12:])!r}")
    print(f"  tokens      : {len(ids)} total, {item['n_answer']} supervised (turn {item['turn']})\n")


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
    parser.add_argument("--batch-size", type=int, default=8, help="16 already OOMs on a 96 GB card: what fills it is the 151k-vocab logits for every position, not the weights")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1792, help="system prompt 640 tokens + the longest conversation; nothing in the dataset reaches 1100")
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
    print(f"  train {len(train)} / eval {len(evals)}   tokens per turn: "
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
    # The prompt is part of the trained artefact and eval_model.py reads it from here, so it is
    # written before the run rather than after: the per-epoch checkpoints are evaluated while the
    # run is still going.
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.system_prompt, out / "system_prompt.txt")

    print(f"\n{total_steps} optimiser steps "
          f"(batch {args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum})")
    trainer.train()

    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    # A model fine-tuned under one prompt and served under another is a silent quality loss, so the
    # exact text travels with the weights, beside the adapter as well as beside the checkpoints.
    shutil.copy(args.system_prompt, out / "adapter" / "system_prompt.txt")
    json.dump({**vars(args), "base_model": args.model, "dora": not args.lora,
               "train_rows": len(train), "eval_rows": len(evals), "steps": total_steps},
              open(out / "train_config.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nadapter -> {out / 'adapter'}")


if __name__ == "__main__":
    main()
