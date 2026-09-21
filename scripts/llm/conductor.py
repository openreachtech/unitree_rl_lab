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

    console  (main)  reads a line, answers emergencies itself, hands the rest to the LLM
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
    ProgramError,
    compile_program,
    program_from_json,
)
from unitree_rl_lab.program.compiler import COL_FLIP, COL_STANCE, COL_VX, COL_VY, COL_WZ  # noqa: E402
from unitree_rl_lab.program.grammar import FLIP_MOTION  # noqa: E402

FLIP_BY_CODE = {code: kind for kind, (code, _, _) in FLIP_MOTION.items()}
STANCE_WORD = {1: "front", -1: "hind", 0: "off"}

# Words that must not wait for a language model. A generation is 1-3 s on CPU, and the robot covers
# four metres in that time; "止まって" has to reach the controller in the time it takes to send a
# datagram. The line still goes to the model afterwards, so the conversation and the reply stay
# consistent with what the robot did -- it is told, in the state block, that it has been stopped.
# The lookahead keeps 「止まれというまで走って」 from reading as a stop: the word is there, but it is
# the *subject* of the sentence rather than the request. Anything followed by という / と言 / まで is
# someone talking about stopping, not asking for it.
EMERGENCY = re.compile(
    r"(?:ストップ|すとっぷ|とまっ|止まっ|止まれ|とまれ|停止|やめ|止めて|とめて|待って|まって|STOP|stop)"
    r"(?!という|と言|っていう|まで)", re.IGNORECASE)

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
            print(f"  -> {line}", flush=True)
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
    fired_flip_at: int = -1
    """Step index of the last flip trigger sent, so a flip fires once and not every tick."""

    @property
    def step(self) -> int:
        return int(round(self.elapsed / self.timeline.dt))

    def segment(self):
        """The segment under way, or None past the end.

        Used for "can the robot be stopped right now", which is a narrower question than the
        ``interruptible`` flag the model reads. That flag covers a whole step -- a two-turn flip
        reads as uninterruptible from its settle to its last landing, which is the right thing to
        tell a person ("I can't cut in") -- but the executor only has to wait out the *window it is
        in*, and then the stop takes effect before the repeat rather than after it.
        """
        for seg in self.timeline.segments:
            if seg.t0 <= self.elapsed < seg.t1:
                return seg
        return None


class Executor:
    """The queue and the 50 Hz loop that walks it.

    The five actions land here:

        none      nothing
        cancel    stop now -- or at the end of the move, if one is under way that cannot be cut
        replace   drop everything and run the new program
        insert    suspend what is running, run the new program, then carry on where it left off
        append    run it after what is already queued

    ``insert`` is why the jobs form a stack rather than a list. A skill asked for mid-run is not a
    request to stop: 「走りながらハンドスプリング！」 flips out of the run and the run continues,
    and the compiler is told that with ``context=`` (the step under way) and ``resume=True`` (do not
    append a final stop -- the interrupted program's own stop is still coming).
    """

    def __init__(self, link: RobotLink, cfg: CompilerConfig):
        self.link = link
        self.cfg = cfg
        self.lock = threading.RLock()

        self.current: Job | None = None
        self.stack: list[Job] = []      # suspended by an insert, newest last
        self.pending: list[Job] = []    # waiting behind an append
        self.last: list[dict] = []
        self.last_outcome: str | None = None
        self.last_step_index: int = 0
        self.cancel_when_free = False
        self.stop_at: float | None = None
        """Monotonic time the robot may go limp at, while a commanded stance comes down."""
        self.stance_sent = 0
        self.driving = False
        self.resumed_at = 0.0
        """When RESUME was last sent. The controller's state line lags by up to a tick, so the
        manual-stop latch is ignored for a moment after that -- otherwise the stale ``manual_stop=1``
        that is already in flight would wipe the program that just cleared it."""

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    # -- compiling ---------------------------------------------------------------------------

    def compile(self, steps: list[dict], *, context: dict | None = None, resume: bool = False) -> Job:
        program = program_from_json(steps)
        ctx = program_from_json([context])[0] if context else None
        timeline = compile_program(program, self.cfg, context=ctx, resume=resume)
        return Job(program=steps, timeline=timeline, array=timeline.to_array())

    # -- the actions -------------------------------------------------------------------------

    def apply(self, action: str, steps: list[dict], *, stop_requested: bool = False) -> str:
        """Apply one model output to the queue. Returns a one-line note for the console.

        ``stop_requested`` marks a turn where the person used a stop word, which the console has
        already acted on without waiting for a generation. Nothing the model returns for that turn
        may start the robot moving again: it has been stopped, and a reply that re-runs the last
        program -- which is what v6 does when it reads 「とまれ」 against an idle state -- would have
        the robot drive off in answer to "stop". The action is refused here rather than argued with,
        because this is the one invariant that cannot depend on a language model being right.
        """
        with self.lock:
            if stop_requested and action in cf.QUEUE_ACTIONS:
                return f"[安全] 「止まって」と言われたターンなので {action} は実行しません"
            if action == "none":
                return "(no change)"
            if action == "cancel":
                return self.cancel()
            # A new instruction is the operator asking for the robot back, which is the only thing
            # that clears the [Space] latch. Sent here rather than left to the person: they have
            # already said what they want, and a silent refusal to move is the worst failure this
            # system can have.
            if self.link.status.manual_stop:
                self.link.send("RESUME")
                self.resumed_at = time.monotonic()
            if action == "replace":
                self._retire("cancelled")
                self.stack.clear()
                self.pending.clear()
                job = self.compile(steps)
                self.current = job
                return f"replace: {job.timeline.duration:.1f}s" + self._adjustment_note(job)
            if action == "insert":
                if self.current is None:
                    job = self.compile(steps)
                    self.current = job
                    return f"insert (idle, so just run it): {job.timeline.duration:.1f}s" + self._adjustment_note(job)
                # The step under way is the context: a running flip written first flips out of
                # *this* move, at its speed and heading, without stopping.
                state = self._state_locked(0.0)
                context = None
                if state.running and state.step_index < len(self.current.program):
                    context = self.current.program[state.step_index]
                job = self.compile(steps, context=context, resume=True)
                self.stack.append(self.current)
                self.current = job
                return f"insert: {job.timeline.duration:.1f}s, then back to the rest" + self._adjustment_note(job)
            if action == "append":
                return self._append(steps)
            raise ValueError(f"unknown action {action!r}")

    def _append(self, steps: list[dict]) -> str:
        """Extend the queue.

        Merged into the running program and recompiled rather than queued behind it, because the
        state block the model reads *is* the queue -- `RobotState.program` is "steps already done,
        the one under way, the ones still to come". Queue them as two programs and the next state
        block would hide everything after the current one, and 「終わったらバク転も」 followed by
        「やっぱりやめて」 would cancel something the model cannot see.

        The prefix of the recompiled timeline is identical to the old one (the compiler walks the
        steps in order and only the trailing stop moves), so `elapsed` carries over unchanged.
        """
        target = self.current if self.current is not None else None
        if target is None:
            job = self.compile(steps)
            self.current = job
            return f"append (idle, so just run it): {job.timeline.duration:.1f}s" + self._adjustment_note(job)
        if self.stack:
            # An insert is under way: the queue being extended is the one it interrupted, not the
            # interruption itself, which is over in a second or two.
            target = self.stack[0]
        try:
            job = self.compile(target.program + steps)
        except ProgramError as exc:
            # Nearly always the 40 s program limit. Run it as a separate job rather than refusing;
            # the state block loses sight of it, which is the lesser problem.
            self.pending.append(self.compile(steps))
            return f"append: queued separately ({exc})"
        job.elapsed = target.elapsed
        job.fired_flip_at = target.fired_flip_at
        if target is self.current:
            self.current = job
        else:
            self.stack[0] = job
        return f"append: {len(steps)} step(s), queue now {job.timeline.duration:.1f}s" + self._adjustment_note(job)

    def cancel(self) -> str:
        """Stop -- now if that means anything, otherwise at the first moment it does."""
        with self.lock:
            self.stack.clear()
            self.pending.clear()
            if self.current is None and self.stop_at is None:
                return "(already idle)"
            self.cancel_when_free = True
            if not self._cancel_step(announce=False):
                return "cancel: 技が終わってから止めます（いま中断できません）"
            if self.stop_at is not None:
                return "cancel: 立ちを降ろしてから止まります"
            return "cancel: 止めました"

    def _cancel_step(self, announce: bool = True) -> bool:
        """Carry a pending cancel one tick forward. False = still inside a flip, keep running.

        Zeroing the velocity does not stop a flip: the move runs on the policy's own clock and the
        command is ignored until the window closes, so a "stop" issued mid-air would only land the
        robot and then look like it had been ignored. A stance is worse -- going limp leaves the
        robot to fall out of it -- so the descent is commanded and given the settle the policy
        learned before anything else happens.
        """
        now = time.monotonic()
        if self.stop_at is not None:
            self.link.send("VEL 0.000 0.000 0.000")
            if now < self.stop_at:
                return True
            self.stop_at = None
            self._commit_cancel(announce)
            return True

        segment = self.current.segment() if self.current is not None else None
        if segment is not None and segment.kind == "flip":
            return False
        if segment is not None and segment.stance != 0.0:
            self.link.send("STANCE off")
            self.stance_sent = 0
            self.link.send("VEL 0.000 0.000 0.000")
            self.stop_at = now + self.cfg.post_stance_settle_s
            return True
        self._commit_cancel(announce)
        return True

    def _commit_cancel(self, announce: bool) -> None:
        self.cancel_when_free = False
        self.stack.clear()
        self.pending.clear()
        self._retire("cancelled")
        self._stop_robot()
        if announce:
            print("\n[実行] 止まりました。", flush=True)

    def _adjustment_note(self, job: Job) -> str:
        if not job.timeline.adjustments:
            return ""
        return "\n    " + "\n    ".join(job.timeline.adjustments)

    def _retire(self, outcome: str) -> None:
        """Move the running job into ``last`` so 「もう一回」 still has a referent."""
        if self.current is not None:
            state = self._state_locked(0.0)
            self.last = self.current.program
            self.last_outcome = outcome
            self.last_step_index = state.step_index if state.running else len(self.current.program)
        self.current = None

    def _stop_robot(self) -> None:
        if self.stance_sent:
            self.link.send("STANCE off")
            self.stance_sent = 0
        self.link.send("STOP")
        self.driving = False

    # -- state -------------------------------------------------------------------------------

    def state(self, lookahead_s: float = 0.0) -> cf.RobotState:
        with self.lock:
            return self._state_locked(lookahead_s)

    def _state_locked(self, lookahead_s: float) -> cf.RobotState:
        """What the model reads. ``lookahead_s`` moves it forward by the expected generation time.

        The model's answer arrives one to three seconds after the person spoke, and the robot has
        moved on by then; a state block describing the moment of asking would have it planning
        against a step that is already finished. Predicting it is a guess, but a guess in the right
        direction beats a fact about the past.
        """
        if self.current is None:
            return cf.RobotState.idle(self.last, self.last_outcome, self.last_step_index)
        at = min(self.current.elapsed + lookahead_s, self.current.timeline.duration)
        return cf.state_from_timeline(
            self.current.program, self.current.timeline, at,
            last=self.last, last_outcome=self.last_outcome, last_step_index=self.last_step_index,
        )

    def describe(self) -> str:
        with self.lock:
            lines = [cf.render_state(self._state_locked(0.0))]
            if self.current is not None:
                lines.append(self.current.timeline.describe())
            if self.stack:
                lines.append(f"  (+{len(self.stack)} suspended by an insert)")
            if self.pending:
                lines.append(f"  (+{len(self.pending)} appended)")
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
        # The operator's [Space] wins over anything queued: the controller has already zeroed the
        # command and latched the link off, so carrying on would only mean shouting at a closed
        # port and then lurching when RESUME is sent.
        if (self.link.status.manual_stop
                and time.monotonic() - self.resumed_at > 0.3
                and (self.current or self.pending or self.stack)):
            self.pending.clear()
            self.stack.clear()
            self._retire("cancelled")
            self.stance_sent = 0
            self.driving = False
            print("\n[操作] スペースで停止されました。キューを捨てます。", flush=True)
            return

        # Before anything is advanced: a cancel that has been waiting for a window to close has to
        # be taken at the first tick it can be, not at the end of the program.
        if self.cancel_when_free and self._cancel_step():
            return

        if self.current is None:
            if self.stack:
                self.current = self.stack.pop()   # an insert finished: back to what it interrupted
            elif self.pending:
                self.current = self.pending.pop(0)
            else:
                # Idle: stop sending. The target goes stale in the controller after timeout_s and
                # the keyboard has the robot back -- which is what "the conductor is not driving"
                # should mean, rather than a stream of zeros that blocks the operator.
                self.driving = False
                return

        job = self.current
        index = job.step
        if index >= len(job.array):
            self._finish()
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

    def _finish(self) -> None:
        if self.stack or self.pending:
            # An inserted program is over: hand back to what it interrupted rather than reporting
            # a completion, which would tell the model the queue is empty when it is not.
            self.current = None
            return
        self._retire("completed")
        if self.stance_sent:
            self.link.send("STANCE off")
            self.stance_sent = 0
        self.driving = False
        print("\n[実行] 完了。", flush=True)


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

    def next_prompt(self, state: cf.RobotState, text: str) -> str:
        """What would be sent for this turn. Building it is free, so /prompt can show it."""
        return cf.conversation_text(
            self.system_prompt, self.user_texts + [cf.render_user_turn(state, text)], self.answers)

    def ask(self, state: cf.RobotState, text: str) -> tuple[cf.Output, float]:
        user_turn = cf.render_user_turn(state, text)
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
        self.answers.append(cf.render_output(output.reply, output.action, output.program))
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
            print("\n" + "-" * 72)
            print("--- モデルへ (この1ターンぶん。全文は /prompt) ---")
            print(user_turn.rstrip())
            print("--- モデルから (生の出力) ---")
            print(raw.rstrip())
            print(f"--- {took:.2f}s  プロンプト {body.get('tokens_evaluated', '?')} tok "
                  f"(新規に読んだ {timings.get('prompt_n', '?')} / キャッシュ再利用 "
                  f"{timings.get('cache_n', '?')}) / 生成 {body.get('tokens_predicted', '?')} tok"
                  + (f" @ {timings.get('predicted_per_second', 0):.1f} tok/s" if timings else "") + " ---")
            print("-" * 72, flush=True)
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
  /fwd <m> [speed]    /back <m>   /left <m>   /right <m>     まっすぐ
  /turn left|right <deg>
  /flip <kind> [running]   backflip frontflip sideflip_left sideflip_right
  /stance front|hind <s>
  /prog <json>        action replace で生プログラム
  /insert <json>      実行中のものに割り込む
  /append <json>      キューの後ろに足す
  /cancel  /state  /forget  /resume  /help  /quit
  /prompt [文]        モデルに渡すプロンプト全文（引数なしなら直前に送ったもの）
  /state-block        いまモデルに見せる状態ブロックの1行だけ
"""


def shorthand(line: str) -> tuple[str, list[dict]] | None:
    """The hand-driving commands. Returns ``(action, program)`` or None if the line is not one.

    Only enough grammar to bring the executor up in MuJoCo without the model in the loop -- the
    real vocabulary is the model's job, and anything this parser accepts it also accepts.
    """
    parts = line.split()
    verb = parts[0][1:]
    rest = parts[1:]
    directions = {"fwd": "forward", "back": "backward", "left": "left", "right": "right"}
    if verb in directions:
        distance = float(rest[0]) if rest else 2.0
        speed = rest[1] if len(rest) > 1 else "normal"
        return "replace", [{"skill": "move", "dir": directions[verb], "speed": speed, "distance_m": distance}]
    if verb == "turn":
        side = rest[0] if rest else "left"
        angle = float(rest[1]) if len(rest) > 1 else 90.0
        return "replace", [{"skill": "turn", "dir": side, "speed": "normal", "angle_deg": angle}]
    if verb == "flip":
        kind = rest[0] if rest else "backflip"
        running = len(rest) > 1 and rest[1] == "running"
        step = {"skill": "flip", "kind": kind, "count": 1}
        if running:
            step["running"] = True
        return ("insert" if running else "replace"), [step]
    if verb == "stance":
        kind = {"front": "handstand", "hind": "hindstand"}.get(rest[0] if rest else "front", "handstand")
        duration = float(rest[1]) if len(rest) > 1 else 5.0
        return "replace", [{"skill": "stance", "kind": kind, "duration_s": duration}]
    if verb in ("prog", "insert", "append"):
        steps = json.loads(line.split(None, 1)[1])
        return {"prog": "replace", "insert": "insert", "append": "append"}[verb], steps
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf-dir", default="logs/llm/gguf/v3",
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
    asks: queue.Queue[tuple[str, bool]] = queue.Queue()

    def llm_worker() -> None:
        while True:
            item = asks.get()
            if item is None:
                return
            text, stop_requested = item
            try:
                state = executor.state(lookahead_s=llm.latency_ema)
                output, took = llm.ask(state, text)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"\n[LLM] 届かへん: {exc}", flush=True)
                continue
            except cf.FormatError as exc:
                print(f"\n[LLM] 返答が読めへん: {exc}", flush=True)
                continue
            print(f"\n🤖 {output.reply}", flush=True)
            try:
                note = executor.apply(output.action, output.program, stop_requested=stop_requested)
            except ProgramError as exc:
                # The grammar keeps the shape right and the compiler substitutes what it can, so
                # this is a program that is well-formed and still impossible -- too long, mostly.
                print(f"[実行] できひん: {exc}", flush=True)
                continue
            print(f"[実行] {note}", flush=True)

    if llm is not None:
        threading.Thread(target=llm_worker, daemon=True).start()

    try:
        while True:
            try:
                line = input("> ").strip()
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
                if verb == "cancel":
                    print(executor.cancel())
                    continue
                if verb == "prompt":
                    if llm is None:
                        print("LLM に繋がってへんので、渡すプロンプトもありません。")
                        continue
                    rest = line.split(None, 1)
                    if len(rest) > 1:
                        # Build the prompt for a line without sending it: the state is the live one,
                        # so this shows exactly what that sentence would be asked against.
                        text = llm.next_prompt(executor.state(lookahead_s=llm.latency_ema), rest[1])
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
                if verb == "state-block":
                    print(cf.render_state(executor.state(lookahead_s=llm.latency_ema if llm else 0.0)))
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
                    print(executor.apply(*parsed))
                except ProgramError as exc:
                    print(f"できひん: {exc}")
                continue

            # A plain line. Emergencies never wait for a generation.
            stop_requested = bool(EMERGENCY.search(line))
            if stop_requested:
                print(executor.cancel())
            if llm is None:
                print("LLM に繋がってへんので、/help のコマンドだけ使えます。")
                continue
            if not link.status.alive:
                print("[注意] コントローラから状態が来てません。Multitask に入ってますか？")
            asks.put((line, stop_requested))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n止めます。")
        executor.cancel()
        time.sleep(0.1)
        link.send("STOP")
        executor.close()
        link.close()


if __name__ == "__main__":
    main()
