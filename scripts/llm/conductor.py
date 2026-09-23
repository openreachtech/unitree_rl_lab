#!/usr/bin/env python3
"""conductor.py -- the language side of the robot, between the console and the controller.

    you type Japanese  ->  llama-server writes a reply + a skill program  ->  the compiler renders
    it to a 50 Hz command stream  ->  UDP  ->  go2_ctrl (State_Multitask)  ->  DDS  ->  MuJoCo or Go2

Everything above the command stream lives here rather than in C++ because it already exists here:
``unitree_rl_lab.program`` compiled the programs the policy was validated against, and
``chat_format`` wrote the state blocks the model was trained to read. Running the robot from the
same code is the only way to be sure the model is reading, at 3 a.m. on hardware, exactly the kind
of state block it saw in training. The controller gains one UDP port and nothing else; see
``deploy/robots/go2/include/ProgramLink.h`` and ``scripts/llm/SYSTEM.md``.

Four threads:

    console  (main)  reads a line and hands it to the LLM
    llm              one request at a time; applies the action it gets back to the queue
    executor (50 Hz) walks the compiled timeline and sends VEL / FLIP / STANCE
    receiver         reads the controller's state line

The queue is the only shared state, and it is small: the program under way plus its elapsed time,
a stack of programs an ``insert`` interrupted, and a list waiting behind an ``append``.

Run it without ``--llm`` to drive the same queue by hand (``/fwd 3``, ``/flip backflip`` ...), which
is how the executor was brought up in MuJoCo before the model was in the loop.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import readline  # noqa: F401 -- gives input() a line buffer Console.say can redraw
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

import chat_format as cf  # noqa: E402
from unitree_rl_lab.program import (  # noqa: E402
    CapabilityTable,
    CompilerConfig,
    Move,
    ProgramError,
    compile_program,
    program_from_json,
)
from unitree_rl_lab.program.compiler import COL_FLIP, COL_STANCE, COL_VX, COL_VY, COL_WZ  # noqa: E402
from unitree_rl_lab.program.grammar import FLIP_MOTION  # noqa: E402

FLIP_BY_CODE = {code: kind for kind, (code, _, _) in FLIP_MOTION.items()}

class Console:
    """One terminal, four threads -- and an operator typing into the middle of it.

    A line printed while someone is mid-word does not merely look untidy. With a Japanese IME the
    half-converted text can be dropped, and the sentence that reaches the model is not the one that
    was typed: 「手前に 戻ってきて」 arrived as 「手戻ってきて」, which the model dutifully answered.
    A corrupted prompt reads exactly like a model failure, so it is worth handling.

    The obvious fix -- hold other threads' output until the line is submitted -- is worse than the
    problem: the answer to what you just asked then appears only when you type the *next* line, so
    the transcript runs a turn behind. So output is never delayed. Instead the prompt line is erased
    before writing and redrawn after, with whatever had been typed into it, which ``readline`` still
    holds. Importing ``readline`` is what puts that buffer within reach, and is why it is imported
    for its side effect alone.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt = ""
        """Non-empty while ``input()`` is waiting; also what has to be drawn back."""

    def say(self, text: str) -> None:
        """Print from any thread, stepping around the half-typed line rather than through it."""
        with self.lock:
            if not self.prompt:
                print(text, flush=True)
                return
            typed = readline.get_line_buffer()
            sys.stdout.write(f"\r\x1b[2K{text}\n{self.prompt}{typed}")
            sys.stdout.flush()

    def read(self, prompt: str) -> str:
        with self.lock:
            self.prompt = prompt
        try:
            return input(prompt)
        finally:
            with self.lock:
                self.prompt = ""


console = Console()

STANCE_WORD = {1: "front", -1: "hind", 0: "off"}

QUEUE_LABEL = "queued programs: "
"""How the queue is labelled in the prompt. Pairs with the ``program:`` line the model writes back,
so the thing it reads and the thing it returns are named and shaped alike."""

MAX_HISTORY_TURNS = 5
"""How many user/assistant pairs are kept. The state block carries the queue, so older turns only
matter for 「もう一回」-style references, and those look back one or two turns at most. Dropping the
oldest pair costs a prompt re-read on the server (the KV cache is a prefix), which is ~0.2 s at the
measured 890 tok/s prefill -- cheaper than carrying a prompt that grows without bound."""


# =================================================================================================
# The link to the controller
# =================================================================================================


@dataclass
class RobotStatus:
    """The controller's state line, parsed. All defaults mean "nothing heard from yet"."""

    heard_at: float = 0.0
    fsm: str = "none"
    link: bool = False
    flip_enabled: bool = False
    flip_elapsed: float = -1.0
    flip_window: float = 1.0
    stance: int = 0
    stance_elapsed: float = -1.0
    manual_stop: bool = False

    @property
    def alive(self) -> bool:
        """True while the controller is in Multitask and talking to us."""
        return self.fsm == "Multitask" and (time.monotonic() - self.heard_at) < 0.5


class RobotLink:
    """UDP to go2_ctrl: commands out on ``port``, the state line in on ``state_port``."""

    def __init__(self, host: str, port: int, state_port: int, verbose: bool = False):
        self.addr = (host, port)
        self.verbose = verbose
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", state_port))
        self.rx.settimeout(0.2)
        self.status = RobotStatus()
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def send(self, line: str) -> None:
        if self.verbose and not line.startswith("VEL"):
            console.say(f"  -> {line}")
        try:
            self.tx.sendto((line + "\n").encode(), self.addr)
        except OSError:
            pass  # the controller is not up; the executor keeps time regardless

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.tx.close()
        self.rx.close()

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self.rx.recvfrom(1024)
            except (socket.timeout, OSError):
                continue
            for line in data.decode(errors="replace").splitlines():
                if line.startswith("STATE "):
                    self._parse_state(line)

    def _parse_state(self, line: str) -> None:
        fields = {}
        for token in line.split()[1:]:
            key, _, value = token.partition("=")
            fields[key] = value
        status = RobotStatus(heard_at=time.monotonic(), fsm=fields.get("fsm", "none"))
        status.link = fields.get("link") == "1"
        status.manual_stop = fields.get("manual_stop") == "1"
        flip = fields.get("flip", "0,0,0").split(",")
        stance = fields.get("stance", "0,0,0").split(",")
        try:
            status.flip_enabled = flip[0] == "1"
            status.flip_elapsed = float(flip[1])
            status.flip_window = float(flip[2])
            status.stance = int(stance[0])
            status.stance_elapsed = float(stance[1])
        except (IndexError, ValueError):
            pass
        self.status = status


# =================================================================================================
# The queue
# =================================================================================================


@dataclass
class Job:
    """One compiled program, and how far into it the robot is."""

    program: list[dict]
    timeline: object
    array: np.ndarray
    elapsed: float = 0.0
    open_ended: bool = False
    """The last step is a move with no length: hold its command past the end of the array rather
    than finishing. The timeline is only drawn out to ``open_move_s``; the move itself has no end."""
    fired_flip_at: int = -1
    """Array row of the last flip trigger sent, so a flip fires once and not every tick."""

    @property
    def step(self) -> int:
        return int(round(self.elapsed / self.timeline.dt))

    def spans(self, index: int) -> tuple[float, float] | None:
        """``(start, end)`` of a program step across all the segments the compiler gave it."""
        parts = [s for s in self.timeline.segments if s.step_index == index]
        if not parts:
            return None
        return min(s.t0 for s in parts), max(s.t1 for s in parts)

    def flip_open_at(self, at: float) -> bool:
        """True while a flip window is running: the move is on the policy's clock, not ours."""
        return any(s.kind == "flip" and s.t0 <= at < s.t1 for s in self.timeline.segments)

    def flips_fired(self, index: int, at: float) -> int:
        """How many windows of this flip step have already been triggered by ``at``."""
        return sum(1 for s in self.timeline.segments
                   if s.step_index == index and s.kind == "flip" and s.t0 <= at)


def _shrink(step: dict, fraction: float) -> dict | None:
    """The part of a step still to come, as a step in its own right.

    The conductor does this arithmetic so the model never has to. It is handed 「あと5.5秒」 and can
    copy it straight back, and because the executor replaces rather than resumes, copying it back is
    what makes the run continue. A flip is not shrunk but counted: a window already fired is a move
    the robot has committed to, and re-issuing it would fire it twice.
    """
    out = dict(step)
    for key in ("duration_s", "distance_m", "angle_deg"):
        if out.get(key) is not None:
            out[key] = round(out[key] * fraction, 2)
    return out


class Executor:
    """The queue and the 50 Hz loop that walks it.

    One rule: whatever the model returns *is* the queue from now on. There is no merge, no diff and
    no resume -- the five actions the old format carried (none / cancel / replace / insert / append)
    all reduce to replacement, because the model is handed the queue as it will stand when its reply
    lands and hands back the version it wants. Echoing it unchanged continues the run; returning []
    stops; putting a flip in front of it flips and carries on.

    That only works because :meth:`upcoming` reports what is *left* of each step rather than what it
    was originally asked for. Copy back 「あと5.5秒前へ」 and the robot runs 5.5 more seconds, which
    is the same thing as not being interrupted.

    The one thing that is not replaceable is a flip already in the air.
    """

    def __init__(self, link: RobotLink, cfg: CompilerConfig):
        self.link = link
        self.cfg = cfg
        self.lock = threading.RLock()

        self.current: Job | None = None
        self.deferred: Job | None = None
        """Waiting for a flip window to close before taking over."""
        self.stance_sent = 0
        self.driving = False
        self.resumed_at = 0.0
        """When RESUME was last sent, so the controller's lagging manual_stop does not wipe it."""

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    # -- what the model is shown -------------------------------------------------------------

    def upcoming(self, lookahead_s: float = 0.0) -> list[dict]:
        """The queue as it will stand when the reply lands: the step under way, then the rest.

        ``lookahead_s`` is the expected generation time. Describing the queue as it is *now* would
        hand the model a step that has finished by the time it answers, and under replacement that
        means the robot re-runs it.
        """
        with self.lock:
            job = self.current
            if job is None:
                return []
            at = min(job.elapsed + lookahead_s, job.timeline.duration)
            out: list[dict] = []
            for index, step in enumerate(job.program):
                span = job.spans(index)
                if span is None:
                    continue
                t0, t1 = span
                last_and_open = job.open_ended and index == len(job.program) - 1
                if t1 <= at and not last_and_open:
                    continue                      # finished
                if last_and_open:
                    out.append(dict(step))        # no end, so nothing to report but itself
                    continue
                if t0 > at:
                    out.append(dict(step))        # not started: as asked for
                    continue
                if step["skill"] == "flip":
                    # A window already fired is a move the robot has committed to; re-issuing it
                    # would fire it twice. One step is one rotation, so it is in or it is gone.
                    if not job.flips_fired(index, at):
                        out.append(dict(step))
                    continue
                out.append(_shrink(step, (t1 - at) / (t1 - t0)))
            return out

    # -- the only action ---------------------------------------------------------------------

    def apply(self, program: list[dict]) -> str:
        """Run this instead of whatever was queued. Returns a one-line note for the console."""
        with self.lock:
            # A new instruction is the operator asking for the robot back, which is the only thing
            # that clears the [Space] latch.
            if self.link.status.manual_stop:
                self.link.send("RESUME")
                self.resumed_at = time.monotonic()

            if not program:
                self.deferred = None
                if self.current is None:
                    return "(already idle)"
                self.current = None
                self._stop_robot()
                return "停止しました"

            job = self.compile(program, context=self._step_under_way())
            note = self._adjustment_note(job)
            if self.current is not None and self.current.flip_open_at(self.current.elapsed):
                # Mid-air. The window is at most flip_window_s, and firing anything now would either
                # re-trigger the move or command a velocity while the robot is upside down.
                self.deferred = job
                return f"技が終わってから差し替えます ({job.timeline.duration:.1f}s)" + note
            self.deferred = None
            self.current = job
            return f"差し替え: {job.timeline.duration:.1f}s" + note

    def compile(self, steps: list[dict], context: dict | None = None) -> Job:
        ctx = program_from_json([context])[0] if context else None
        parsed = program_from_json(steps)
        timeline = compile_program(parsed, self.cfg, context=ctx)
        open_ended = bool(parsed) and isinstance(parsed[-1], Move) and parsed[-1].open_ended
        return Job(program=steps, timeline=timeline, array=timeline.to_array(), open_ended=open_ended)

    def _step_under_way(self) -> dict | None:
        """The step the robot is in, for a program that opens with a flip.

        「そのままハンドスプリングして」 comes back as ``[{flip}, {move ...}]`` -- flip now, then
        carry on -- and whether that flip comes out of a run is decided by what precedes it. Nothing
        does, in the list; the move it flips out of is the one already under way. Handing that step
        to the compiler as context is what tells it so.
        """
        job = self.current
        if job is None:
            return None
        for index, step in enumerate(job.program):
            span = job.spans(index)
            if span and span[0] <= job.elapsed < span[1]:
                return step if step["skill"] == "move" else None
        return None

    def _adjustment_note(self, job: Job) -> str:
        if not job.timeline.adjustments:
            return ""
        return "\n    " + "\n    ".join(job.timeline.adjustments)

    def _stop_robot(self) -> None:
        if self.stance_sent:
            self.link.send("STANCE off")
            self.stance_sent = 0
        self.link.send("STOP")
        self.driving = False

    def describe(self) -> str:
        with self.lock:
            steps = self.upcoming()
            if not steps:
                return QUEUE_LABEL + "[]"
            lines = [QUEUE_LABEL + json.dumps(steps, ensure_ascii=False)]
            if self.current is not None:
                lines.append(self.current.timeline.describe())
            if self.deferred is not None:
                lines.append("  (技の終了待ちの差し替えが1件)")
            return "\n".join(lines)

    # -- the loop ----------------------------------------------------------------------------

    def _loop(self) -> None:
        dt = self.cfg.dt
        next_tick = time.monotonic()
        while self._running:
            next_tick += dt
            with self.lock:
                self._tick(dt)
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # fell behind; do not try to catch up by running fast

    def _tick(self, dt: float) -> None:
        # [Space] wins over anything queued: the controller has already zeroed the command and
        # latched the link off, so carrying on would mean shouting at a closed port.
        if (self.link.status.manual_stop
                and time.monotonic() - self.resumed_at > 0.3
                and (self.current or self.deferred)):
            self.current = self.deferred = None
            self.stance_sent = 0
            self.driving = False
            console.say("\n[操作] スペースで停止されました。予定を捨てます。")
            return

        job = self.current
        if job is not None and self.deferred is not None and not job.flip_open_at(job.elapsed):
            job = self.current = self.deferred      # the window closed: take the replacement
            self.deferred = None

        if job is None:
            # Idle: stop sending. The target goes stale in the controller after timeout_s and the
            # keyboard has the robot back, which is what "not driving" should mean.
            self.driving = False
            return

        index = job.step
        if index >= len(job.array) and job.open_ended:
            index = len(job.array) - 1        # hold the last command; the move has no end
        if index >= len(job.array):
            self.current = None
            if self.stance_sent:
                self.link.send("STANCE off")
                self.stance_sent = 0
            self.driving = False
            console.say("\n[実行] 完了。")
            return

        row = job.array[index]
        self.link.send(f"VEL {row[COL_VX]:.3f} {row[COL_VY]:.3f} {row[COL_WZ]:.3f}")
        self.driving = True

        code = int(round(float(row[COL_FLIP])))
        if code and job.fired_flip_at != index:
            job.fired_flip_at = index
            kind = FLIP_BY_CODE.get(code)
            if kind:
                self.link.send(f"FLIP {kind}")

        stance = int(round(float(row[COL_STANCE])))
        if stance != self.stance_sent:
            self.link.send(f"STANCE {STANCE_WORD[stance]}")
            self.stance_sent = stance

        job.elapsed += dt

# =================================================================================================
# The model
# =================================================================================================


class LlmClient:
    """llama-server's ``/completion``, driven with a raw prompt rather than the chat endpoint.

    The chat endpoint would re-render the history through Qwen3's template, and that template drops
    the ``<think></think>`` block from past assistant turns -- so every turn would rewrite the text
    the server has already cached and the KV cache would miss from the system prompt onward. Raw
    text built by ``conversation_text`` is byte-identical to what training saw and grows only at the
    end, which is exactly what ``cache_prompt`` wants.
    """

    def __init__(self, url: str, system_prompt: str, grammar: str, timeout: float = 60.0,
                 trace: bool = False, trace_file: Path | None = None):
        self.url = url.rstrip("/") + "/completion"
        self.system_prompt = system_prompt
        self.grammar = grammar
        self.timeout = timeout
        self.user_texts: list[str] = []
        self.answers: list[str] = []
        self.latency_ema = 1.5
        """Seconds a generation takes, smoothed. Feeds the state block's lookahead."""
        self.trace = trace
        self.trace_file = trace_file
        self.last_prompt = ""
        """The exact text the model was last given, for /prompt."""

    def next_prompt(self, queue: list[dict], text: str) -> str:
        """What would be sent for this turn. Building it is free, so /prompt can show it."""
        return cf.conversation_text(
            self.system_prompt, self.user_texts + [cf.render_user_turn(text, queue)], self.answers)

    def ask(self, queue: list[dict], text: str) -> tuple[cf.Output, float]:
        user_turn = cf.render_user_turn(text, queue)
        prompt = cf.conversation_text(self.system_prompt, self.user_texts + [user_turn], self.answers)
        payload = {
            "prompt": prompt,
            "grammar": self.grammar,
            "temperature": 0.0,
            "n_predict": 512,
            "cache_prompt": True,
            "stop": ["<|im_end|>"],
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        took = time.monotonic() - started
        self.latency_ema = 0.7 * self.latency_ema + 0.3 * took

        self.last_prompt = prompt
        self._report(prompt, user_turn, body, took)
        output = cf.parse_output(body["content"])
        # Remembered only once it parsed: a malformed turn in the history would teach the cache a
        # shape the model never produced in training.
        self.user_texts.append(user_turn)
        self.answers.append(output.reply)
        if len(self.user_texts) > MAX_HISTORY_TURNS:
            self.user_texts = self.user_texts[-MAX_HISTORY_TURNS:]
            self.answers = self.answers[-MAX_HISTORY_TURNS:]
        return output, took

    def _report(self, prompt: str, user_turn: str, body: dict, took: float) -> None:
        """Show what went in and what came out, when asked.

        Only the turn just added is printed, not the whole prompt: the system prompt is 1.6k tokens
        and is identical every time, and burying the one line that changed under it is how you stop
        reading traces. `/prompt` prints the whole thing when that is what you need, and
        --trace-file keeps every byte.

        The number to watch is `timings.prompt_n` -- tokens the server actually had to read this
        call, against `cache_n` reused from the KV cache. Tens against thousands means the cache is
        holding; if prompt_n ever climbs into the thousands the history stopped being a prefix of
        what was cached, and every turn is paying the full 1.7k-token prefill.
        (`tokens_evaluated` is the whole prompt length and says nothing about the cache.)
        """
        raw = body.get("content", "")
        if self.trace:
            timings = body.get("timings") or {}
            console.say("\n" + "-" * 72
                        + "\n--- モデルへ (この1ターンぶん。全文は /prompt) ---\n" + user_turn.rstrip()
                        + "\n--- モデルから (生の出力) ---\n" + raw.rstrip()
                        + f"\n--- {took:.2f}s  プロンプト {body.get('tokens_evaluated', '?')} tok "
                        + f"(新規に読んだ {timings.get('prompt_n', '?')} / キャッシュ再利用 "
                        + f"{timings.get('cache_n', '?')}) / 生成 {body.get('tokens_predicted', '?')} tok"
                        + (f" @ {timings.get('predicted_per_second', 0):.1f} tok/s" if timings else "")
                        + " ---\n" + "-" * 72)
        if self.trace_file is not None:
            record = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "user_turn": user_turn,
                "raw": raw,
                "took_s": round(took, 3),
                "tokens_evaluated": body.get("tokens_evaluated"),
                "tokens_predicted": body.get("tokens_predicted"),
                "prompt_n": (body.get("timings") or {}).get("prompt_n"),
                "cache_n": (body.get("timings") or {}).get("cache_n"),
            }
            with self.trace_file.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def forget(self) -> None:
        self.user_texts.clear()
        self.answers.clear()


# =================================================================================================
# The console
# =================================================================================================


HELP = """\
  そのまま日本語で指令  (--llm があるときだけ)
  /fwd <m|-> [speed]  /back  /left  /right      "-" で止まるまで進む
  /turn left|right <deg>
  /flip <kind>        backflip frontflip sideflip_left sideflip_right
  /stance front|hind <s>
  /stop               キューを空にして止まる
  /prog <json>        実行予定そのものを書く
  /state  /forget  /resume  /help  /quit
  /prompt [文]        モデルに渡すプロンプト全文（引数なしなら直前に送ったもの）
"""


def shorthand(line: str) -> list[dict] | None:
    """The hand-driving commands, as a whole queue. None if the line is not one.

    Only enough grammar to bring the executor up in MuJoCo without the model in the loop -- the
    real vocabulary is the model's job.
    """
    parts = line.split()
    verb = parts[0][1:]
    rest = parts[1:]
    directions = {"fwd": "forward", "back": "backward", "left": "left", "right": "right"}
    if verb in directions:
        step = {"skill": "move", "dir": directions[verb], "speed": rest[1] if len(rest) > 1 else "normal"}
        if rest and rest[0] != "-":          # "-" for an open-ended move
            step["distance_m"] = float(rest[0])
        return [step]
    if verb == "turn":
        return [{"skill": "turn", "dir": rest[0] if rest else "left",
                 "angle_deg": float(rest[1]) if len(rest) > 1 else 90.0}]
    if verb == "flip":
        return [{"skill": "flip", "kind": rest[0] if rest else "backflip"}]
    if verb == "stance":
        kind = {"front": "handstand", "hind": "hindstand"}.get(rest[0] if rest else "front", "handstand")
        return [{"skill": "stance", "kind": kind, "duration_s": float(rest[1]) if len(rest) > 1 else 5.0}]
    if verb == "stop":
        return []
    if verb == "prog":
        return json.loads(line.split(None, 1)[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf-dir", default="logs/llm/gguf/v1",
                        help="where system_prompt.txt and output.gbnf were written by export_gguf.py")
    parser.add_argument("--llm", default=None, help="llama-server base URL, e.g. http://localhost:8080")
    parser.add_argument("--capability", default="data/llm/capability.json")
    parser.add_argument("--host", default="127.0.0.1", help="where go2_ctrl is listening")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--state-port", type=int, default=7778)
    parser.add_argument("--pre-flip-settle-s", type=float, default=None,
                        help="override the standing settle before a flip; longer on hardware, where "
                             "there is no measured base velocity to wait on")
    parser.add_argument("--verbose", action="store_true", help="print every non-VEL datagram")
    parser.add_argument("--trace", action="store_true",
                        help="print the turn given to the model and the raw text it returned")
    parser.add_argument("--trace-file", metavar="PATH",
                        help="append every exchange (full prompt included) to this file as JSONL")
    args = parser.parse_args()

    cfg = CompilerConfig(calibration=CapabilityTable.load(args.capability).calibration())
    if args.pre_flip_settle_s is not None:
        cfg.pre_flip_settle_s = args.pre_flip_settle_s

    link = RobotLink(args.host, args.port, args.state_port, verbose=args.verbose)
    executor = Executor(link, cfg)
    executor.start()

    llm = None
    if args.llm:
        gguf_dir = Path(args.gguf_dir)
        llm = LlmClient(
            args.llm,
            (gguf_dir / "system_prompt.txt").read_text(),
            (gguf_dir / "output.gbnf").read_text(),
            trace=args.trace,
            trace_file=Path(args.trace_file) if args.trace_file else None,
        )
        print(f"LLM: {args.llm}  ({gguf_dir})")
        if args.trace_file:
            print(f"trace: {args.trace_file} に全文を追記します")
    else:
        print("LLM: 未接続 (--llm で繋ぐ)。/ で始まる手動コマンドだけ使えます。")

    print(f"robot: {args.host}:{args.port} -> state :{args.state_port}")
    print(HELP)

    # One request at a time, off the console thread: a generation takes seconds and the console has
    # to stay free for 「ストップ」.
    asks: queue.Queue[str] = queue.Queue()

    def llm_worker() -> None:
        while True:
            text = asks.get()
            if text is None:
                return
            try:
                queue = executor.upcoming(lookahead_s=llm.latency_ema)
                output, took = llm.ask(queue, text)
            except (urllib.error.URLError, TimeoutError) as exc:
                console.say(f"\n[LLM] 届かへん: {exc}")
                continue
            except cf.FormatError as exc:
                console.say(f"\n[LLM] 返答が読めへん: {exc}")
                continue
            console.say(f"\n🤖 {output.reply}")
            try:
                note = executor.apply(output.program)
            except ProgramError as exc:
                # The grammar keeps the shape right and the compiler substitutes what it can, so
                # this is a program that is well-formed and still impossible -- too long, mostly.
                console.say(f"[実行] できひん: {exc}")
                continue
            console.say(f"[実行] {note}")

    if llm is not None:
        threading.Thread(target=llm_worker, daemon=True).start()

    try:
        while True:
            try:
                line = console.read("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            if line.startswith("/"):
                verb = line.split()[0][1:]
                if verb in ("quit", "q", "exit"):
                    break
                if verb == "help":
                    print(HELP)
                    continue
                if verb == "state":
                    print(executor.describe())
                    status = link.status
                    print(f"  controller: fsm={status.fsm} alive={status.alive} "
                          f"link={status.link} manual_stop={status.manual_stop}")
                    continue
                if verb == "prompt":
                    if llm is None:
                        print("LLM に繋がってへんので、渡すプロンプトもありません。")
                        continue
                    rest = line.split(None, 1)
                    if len(rest) > 1:
                        # Build the prompt for a line without sending it: the state is the live one,
                        # so this shows exactly what that sentence would be asked against.
                        text = llm.next_prompt(executor.upcoming(lookahead_s=llm.latency_ema), rest[1])
                        print(f"--- 「{rest[1]}」をいま送ったら、こうなります ---")
                    elif llm.last_prompt:
                        text = llm.last_prompt
                        print("--- 直前に送ったプロンプト（全文） ---")
                    else:
                        print("まだ一度も送ってません。/prompt <文> で送る前の全文が見られます。")
                        continue
                    print(text)
                    print(f"--- ここまで ({len(text)} 文字) ---")
                    continue
                if verb == "forget":
                    if llm:
                        llm.forget()
                    print("会話履歴を消しました。")
                    continue
                if verb == "resume":
                    link.send("RESUME")
                    executor.resumed_at = time.monotonic()
                    print("RESUME を送りました（スペース停止の解除）。")
                    continue
                try:
                    parsed = shorthand(line)
                except (ValueError, IndexError, json.JSONDecodeError) as exc:
                    print(f"読めへん: {exc}")
                    continue
                if parsed is None:
                    print(f"知らんコマンド: {line.split()[0]}")
                    continue
                try:
                    print(executor.apply(parsed))
                except ProgramError as exc:
                    print(f"できひん: {exc}")
                continue

            if llm is None:
                print("LLM に繋がってへんので、/help のコマンドだけ使えます。")
                continue
            if not link.status.alive:
                print("[注意] コントローラから状態が来てません。Multitask に入ってますか？")
            asks.put(line)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n止めます。")
        executor.apply([])
        time.sleep(0.1)
        link.send("STOP")
        executor.close()
        link.close()


if __name__ == "__main__":
    main()
