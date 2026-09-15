"""A small hand-written test set in wording the phrase bank never produces.

The 99.x% on the generated eval split says the model learned the generator's Japanese. This file
is the other question -- does it understand *people* -- and is deliberately written without
looking at ``phrasebank.py``: different verbs (前進 / 後退 / 一周 / 左向け左), casual particles, typos
a person actually makes, and follow-ups phrased as they come out of a mouth. Add your own; the
Kansai lines in particular should come from a Kansai speaker.

    python scripts/llm/handwritten_eval.py --out data/llm/handwritten_eval.jsonl
    python scripts/llm/eval_model.py --adapter ... --dataset data/llm/handwritten_eval.jsonl

Each case is (category, [ (user words, state, action, program[, history reply]), ... ]) where ``state`` is
``idle`` / ``done`` / ``cut`` / a float ``t`` seconds into the *first* program of the case. The
JSONL rows come out in the dataset's ``turns`` shape with ``split: eval`` so ``eval_model.py``
scores them as it scores the generated split.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

from chat_format import RobotState, render_user_turn, state_from_timeline
from unitree_rl_lab.program import CapabilityTable, CompilerConfig, compile_program, program_from_json


def mv(d, amount, speed="normal", by="m"):
    step = {"skill": "move", "dir": d, "speed": speed}
    step["distance_m" if by == "m" else "duration_s"] = float(amount)
    return step


def tn(d, amount, speed="normal", by="deg"):
    step = {"skill": "turn", "dir": d, "speed": speed}
    step["angle_deg" if by == "deg" else "duration_s"] = float(amount)
    return step


def fl(kind, count=1, running=False):
    step = {"skill": "flip", "kind": kind, "count": count}
    if running:
        step["running"] = True
    return step


def st(kind, seconds=5.0):
    return {"skill": "stance", "kind": kind, "duration_s": float(seconds), "speed": "slow"}


def stop(seconds=1.0):
    return {"skill": "stop", "duration_s": float(seconds)}


CASES = [
    # --- single instructions, unfamiliar wording ---
    ("hw_single", [("3メートルくらい前進して、そのあと1メートル後退して。", "idle", "replace", [mv("forward", 3), mv("backward", 1)])]),
    ("hw_single", [("左向け左。", "idle", "replace", [tn("left", 90)])]),
    ("hw_single", [("その場でぐるっと一周してから、バク転をひとつ。", "idle", "replace", [tn("left", 360), fl("backflip")])]),
    ("hw_single", [("真後ろ向いて。", "idle", "replace", [tn("left", 180)])]),
    ("hw_single", [("前にちょっと出て、それから逆立ち5秒。", "idle", "replace", None)]),   # None: any valid program
    ("hw_single", [("ダッシュで前に5メートル!", "idle", "replace", [mv("forward", 5, "fast")])]),
    ("hw_single", [("右にカニ歩きで2メートルほど。", "idle", "replace", [mv("right", 2)])]),
    ("hw_single", [("ゆっくりバックで3秒ね。", "idle", "replace", [mv("backward", 3, "slow", by="s")])]),
    ("hw_single", [("バク転を三回連続。", "idle", "replace", [fl("backflip", 3)])]),
    ("hw_single", [("走ってる勢いのまま前転して。2メートルくらい走ってから。", "idle", "replace", [mv("forward", 2), fl("frontflip", running=True)])]),
    ("hw_single", [("後ろに下がりながらバク転。3秒くらい下がって。", "idle", "replace", [mv("backward", 3, by="s"), fl("backflip", running=True)])]),
    ("hw_single", [("左に走って、走りながら左の側転やってみて。", "idle", "replace", None)]),
    # --- must not run anything ---
    ("hw_none", [("お腹すいたわー。", "idle", "none", [])]),
    ("hw_none", [("君って何歳?", "idle", "none", [])]),
    ("hw_none", [("階段のぼれる?", "idle", "none", [])]),
    ("hw_none", [("なんかテキトーに動いてみて。", "idle", "none", [])]),
    # --- jump: proposal, then the answer ---
    ("hw_jump", [("ぴょんって跳んで。", "idle", "none", [], "ジャンプはまだ着地できひんねん。代わりにバク転ならできるけど、それでええ?"),
                 ("うん、それで頼むわ。", "idle", "replace", [fl("backflip")])]),
    ("hw_jump", [("ジャンプ!", "idle", "none", [], "ジャンプはまだ無理やわ。バク転に変えてええ?"),
                 ("いや、いいや。", "idle", "none", [])]),
    # --- running-flip mismatch: proposal ---
    ("hw_mismatch", [("前に3メートル走りながらバク転して。", "idle", "none", [], "前に走りながらやと前転になるけど、それでええ?"),
                     ("ええよ、それで。", "idle", "replace", [mv("forward", 3), fl("frontflip", running=True)])]),
    # --- fast + running flip: any valid program at replace; the grammar refuses fast + running ---
    ("hw_fast", [("全速力で前に走って、そのまま前転!", "idle", "replace", None)]),
    # --- follow-ups while running: base program forward 10 s ---
    ("hw_stop", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                 ("あっ、ストップストップ!", 3.0, "cancel", [])]),
    ("hw_stop", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                 ("待った。", 5.0, "cancel", [])]),
    ("hw_insert", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                   ("ハンドスプリングして!", 3.0, "insert", [fl("frontflip", running=True)])]),
    ("hw_insert", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                   ("そこで前転!", 4.0, "insert", [fl("frontflip", running=True)])]),
    ("hw_insert", [("後ろに8秒下がって。", "idle", "replace", [mv("backward", 8, by="s")]),
                   ("バク転いっとこ。", 3.0, "insert", [fl("backflip", running=True)])]),
    ("hw_insert_mismatch", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                            ("バク転して!", 3.0, "none", [], "前に走りながらやと前転になるけど、それでええ?"),
                            ("ok", 4.5, "insert", [fl("frontflip", running=True)])]),
    ("hw_insert_turn", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                        ("ちょい右向いて。", 3.0, "insert", None)]),   # None: any right turn
    ("hw_chat_running", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                         ("今日めっちゃ暑いな。", 3.0, "none", [])]),
    ("hw_status", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                   ("今どんな感じ?", 6.0, "none", [])]),
    ("hw_append", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                   ("それ終わったらバク転もな。", 3.0, "append", [fl("backflip")])]),
    ("hw_correction", [("前に10秒歩いて。", "idle", "replace", [mv("forward", 10, by="s")]),
                       ("ちゃうちゃう、後ろに3メートル。", 3.0, "replace", [mv("backward", 3)])]),
    # --- after it finished ---
    ("hw_repeat", [("前に2メートル行って、バク転。", "idle", "replace", [mv("forward", 2), fl("backflip")]),
                   ("もっぺん。", "done", "replace", [mv("forward", 2), fl("backflip")])]),
    ("hw_repeat", [("前に2メートル行って、バク転。", "idle", "replace", [mv("forward", 2), fl("backflip")]),
                   ("同じのゆっくりでもう一回。", "done", "replace", [mv("forward", 2, "slow"), fl("backflip")])]),
    ("hw_stopped", [("前に2メートル行って、バク転。", "idle", "replace", [mv("forward", 2), fl("backflip")]),
                    ("止まって!", "done", "none", [])]),
    ("hw_resume", [("前に5メートル行って、バク転して、左に90度。", "idle", "replace", [mv("forward", 5), fl("backflip"), tn("left", 90)]),
                   ("ストップ!", 2.0, "cancel", []),
                   ("ごめん、続けて。", "cut", "replace", [mv("forward", 5), fl("backflip"), tn("left", 90)])]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/llm/handwritten_eval.jsonl")
    parser.add_argument("--capability", default="data/llm/capability.json")
    args = parser.parse_args()
    compiler = CompilerConfig(calibration=CapabilityTable.load(args.capability).calibration())

    rows = []
    for n, (category, script) in enumerate(CASES):
        first = script[0][3]
        timeline = compile_program(program_from_json(first), compiler) if first else None
        turns = []
        cut_at = 0
        for words, when, action, program, *reply in script:
            if when == "idle":
                state = RobotState.idle()
            elif when == "done":
                state = RobotState.idle(first, "completed")
            elif when == "cut":
                state = RobotState.idle(first, "cancelled", at_step=cut_at)
            else:
                state = state_from_timeline(first, timeline, float(when))
                cut_at = state.step_index
            turns.append({"role": "user", "content": render_user_turn(state, words), "state": dataclasses.asdict(state)})
            # The gold history needs a reply the next turn can make sense of; the text itself is not scored.
            turns.append({"role": "assistant", "reply": reply[0] if reply else "了解。", "action": action,
                          "program": program if program is not None else [],
                          **({"any_program": True} if program is None else {})})
        rows.append({"id": f"hw-{n:03d}", "split": "eval", "category": category, "style": "hand",
                     "turns": turns, "source": {"program_id": None, "passed": None, "kind": "handwritten"}})
    with open(args.out, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases, {sum(len(r['turns']) // 2 for r in rows)} turns -> {args.out}")


if __name__ == "__main__":
    main()
