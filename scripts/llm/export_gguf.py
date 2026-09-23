"""Fold the DoRA adapter into the base weights and produce the GGUF the robot runs.

    source /home/tak/isaacsim/env_llm/bin/activate
    python scripts/llm/export_gguf.py --adapter logs/llm/qwen3-1.7b-dora-v3/adapter --out logs/llm/gguf/v3

Three steps, each skipped when its output already exists:

1. merge     PEFT ``merge_and_unload`` -> a plain HF model in ``<out>/merged`` (bf16). DoRA's
             magnitude vectors are folded into the weights here; the served model is a 1.7B dense
             model with no adapter machinery.
2. convert   ``llama.cpp/convert_hf_to_gguf.py`` -> ``<out>/model-f16.gguf``
3. quantize  ``llama-quantize`` -> ``<out>/model-<quant>.gguf`` (default Q8_0, which keeps the
             trained weights nearly intact and still fits the Orin; Q4_K_M only if speed demands it)

The system prompt the adapter was trained under and the GBNF grammar are written next to the
weights, so the serving side has everything it needs in one directory. ``--test`` then runs
``llama-completion`` on the quantized model with the grammar and a sample conversation, on this machine,
which is the first time the grammar meets the real sampler.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chat_format import conversation_text, gbnf_grammar, parse_output, render_user_turn  # noqa: E402

LLAMA_CPP = Path("/home/tak/isaacsim/llama.cpp")


def merge(base: str, adapter: str, out: Path) -> Path:
    merged = out / "merged"
    if (merged / "config.json").exists():
        print(f"merged model exists: {merged}")
        return merged
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"merging {adapter} into {base} (cpu)")
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    print(f"  merged in {time.time() - started:.0f}s -> {merged}")
    return merged


def convert(merged: Path, out: Path) -> Path:
    f16 = out / "model-f16.gguf"
    if f16.exists():
        print(f"f16 gguf exists: {f16}")
        return f16
    script = LLAMA_CPP / "convert_hf_to_gguf.py"
    print(f"converting -> {f16}")
    subprocess.run([sys.executable, str(script), str(merged), "--outfile", str(f16), "--outtype", "f16"],
                   check=True, env={"PYTHONPATH": str(LLAMA_CPP / "gguf-py"), "PATH": "/usr/bin:/bin"})
    return f16


def quantize(f16: Path, out: Path, quant: str) -> Path:
    q = out / f"model-{quant}.gguf"
    if q.exists():
        print(f"{quant} gguf exists: {q}")
        return q
    print(f"quantizing -> {q}")
    subprocess.run([str(LLAMA_CPP / "build" / "bin" / "llama-quantize"), str(f16), str(q), quant], check=True,
                   stdout=subprocess.DEVNULL)
    return q


def test(gguf: Path, out: Path, threads: int) -> None:
    """A few turns through llama-completion with the grammar on: what came back, and how fast.

    Four shapes, because they fail differently: a standing start, a two-step instruction, an
    interruption while a queue is running (the model has to copy the remainder back), and chitchat
    while running (the queue must come back untouched). The queues are written out by hand -- a
    queue is just what is left of the program when the reply lands.
    """
    system_prompt = (out / "system_prompt.txt").read_text()
    walking = [{"skill": "move", "dir": "forward", "speed": "normal", "duration_s": 7.0}]
    cases = [
        (["前に10秒歩いて。"], [[]], []),
        (["5mくらい前に走ってから、そのまま前転して"], [[]], []),
        (["前に10秒歩いて。", "バク転して！"], [[], walking], ["了解、前に10秒歩くで。"]),
        (["前に10秒歩いて。", "ええ天気やなあ"], [[], walking], ["了解、前に10秒歩くで。"]),
    ]

    for texts, queues, answers in cases:
        user_texts = [render_user_turn(text, queue) for text, queue in zip(texts, queues)]
        prompt = conversation_text(system_prompt, user_texts, answers)
        prompt_file = out / "_prompt.txt"
        prompt_file.write_text(prompt)
        cmd = [str(LLAMA_CPP / "build" / "bin" / "llama-completion"), "-m", str(gguf), "-f", str(prompt_file),
               "--grammar-file", str(out / "output.gbnf"), "-n", "200", "--temp", "0", "-t", str(threads),
               "-no-cnv", "--no-display-prompt", "-r", "<|im_end|>"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        text = result.stdout.replace("[end of text]", "").strip()
        perf = " ".join(re.sub(r".*common_perf_print:\s*", "", line).strip()
                        for line in result.stderr.splitlines() if "prompt eval time" in line or " eval time" in line)
        print(f"\n--- {texts[-1]}   queue {json.dumps(queues[-1], ensure_ascii=False)}")
        print(text)
        try:
            parsed = parse_output(text)
            print(f"  parsed: steps={len(parsed.program)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  PARSE FAILED: {exc}")
            print(result.stderr[-800:])
        print(f"  {perf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="/home/tak/models/Qwen3-1.7B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", default="Q8_0", help="Q8_0 keeps the trained weights nearly intact and fits the Orin easily; Q4_K_M only if speed demands it")
    parser.add_argument("--test", action="store_true", help="run llama-completion with the grammar on a few turns")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The prompt the adapter was trained under, not whatever data/llm holds today: train_sft.py
    # saves a copy beside the adapter and another beside the checkpoints.
    adapter = Path(args.adapter)
    trained_prompt = next((p for p in (adapter / "system_prompt.txt", adapter.parent / "system_prompt.txt")
                           if p.exists()), None)
    if trained_prompt is None:
        sys.exit(f"no system_prompt.txt beside {adapter} -- serving under a different prompt than the "
                 f"one trained is a silent quality loss, so this is not guessed")
    shutil.copy(trained_prompt, out / "system_prompt.txt")
    (out / "output.gbnf").write_text(gbnf_grammar())

    merged = merge(args.base, args.adapter, out)
    f16 = convert(merged, out)
    q = quantize(f16, out, args.quant)
    for path in (f16, q):
        print(f"  {path.name:<22} {path.stat().st_size / 2**20:8.0f} MB")
    if args.test:
        test(q, out, args.threads)


if __name__ == "__main__":
    main()
