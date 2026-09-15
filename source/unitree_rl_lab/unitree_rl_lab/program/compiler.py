"""Program -> command timeline.

The compiler owns every constraint the policy has and the model should not need to know:

* A flip or a stance is started from standing unless asked otherwise. ``Go2-Multitask-v2`` offers
  the moves below a commanded 1.0 m/s and the experts learned them from rest, so the compiler
  brings the robot to a stop and lets it settle before each one.
* A *running* flip (``Flip.running``) keeps the preceding move's velocity through the flip window
  and through the gaps between repeats -- training fired its moves into a running gait. What
  follows the last landing is whatever the program says next: the final stop if nothing does, so
  「5m走って前転」 ends with the flip and not with a stroll. Training itself mostly ran on after
  landing (the velocity command was resampled every 10 s regardless of flips), but stopping on a
  zero command is the locomotion expert's everyday transition; whether it copes with one arriving
  while it is still collecting itself from a landing is measured by ``--calibrate``, and
  ``running_recover_s`` is the knob to turn if it does not.
* A flip occupies a fixed window (``rearm_after_s`` = 1.0 s in the merged environment) and the
  training schedule never fired two closer than 1.5 s after the window, so repeated flips are
  spaced the same way.
* A stance is held between 2 s and 12 s -- the expert trained on 6-12 s holds, the merged policy on
  10 s -- and is followed by a settle while the robot comes back down onto four legs.
* Distances and angles become durations through a *calibration*: per skill, the length actually
  covered in ``t`` seconds is modelled as ``a * t + b`` (``b`` is negative and absorbs the start-up).
  Without a measured table the nominal command speed is used with ``b = 0``.

The output is a :class:`Timeline` of :class:`Segment` objects. ``to_array`` renders it at the
control rate for the simulator; ``events`` lists the same thing as timestamped commands for the
robot-side executor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grammar import (
    DIRECTION_VECTORS,
    FLIP_MOTION,
    RUNNING_FLIP_FOR,
    RUNNING_FLIP_SPEEDS,
    STANCE_SIGN,
    Flip,
    Move,
    Program,
    ProgramError,
    Stance,
    Step,
    Stop,
    Turn,
    validate_program,
)

# Columns of the array `Timeline.to_array` produces.
COL_VX, COL_VY, COL_WZ, COL_FLIP, COL_STANCE = range(5)


def _default_speeds(slow: float, normal: float, fast: float) -> dict[str, float]:
    return {"slow": slow, "normal": normal, "fast": fast}


@dataclass
class CompilerConfig:
    """Everything the compiler needs to know about the policy. Defaults describe ``Go2-Multitask-v2``."""

    dt: float = 0.02
    """Control period the timeline is rendered at (50 Hz)."""

    # -- nominal command magnitudes per speed word. The policy's command range is forward
    #    [0, 2.5], backward [0, 1.0], lateral [0, 1.0] m/s and yaw [0, 1.0] rad/s; the words sit
    #    comfortably inside it, "fast" forward deliberately short of the ceiling.
    #    "slow" sits where the policy still tracks: measured on Go2-Multitask-v2, a 0.3 m/s lateral
    #    command produced 0.10-0.30 m/s and a 0.4 rad/s yaw command 0.06-0.17 rad/s, both under the
    #    tracking reward's dead zone, while 0.5 m/s and 0.7 rad/s tracked at 0.85-1.0x.
    forward_speed: dict[str, float] = field(default_factory=lambda: _default_speeds(0.5, 1.0, 2.0))
    backward_speed: dict[str, float] = field(default_factory=lambda: _default_speeds(0.4, 0.6, 0.9))
    lateral_speed: dict[str, float] = field(default_factory=lambda: _default_speeds(0.4, 0.5, 0.8))
    yaw_rate: dict[str, float] = field(default_factory=lambda: _default_speeds(0.6, 0.8, 1.0))
    stance_speed: dict[str, float] = field(default_factory=lambda: _default_speeds(0.2, 0.3, 0.4))
    """Walking speed while on two legs. Small: the bipedal expert tracks slow commands only."""

    min_move_s: float = 0.5
    max_move_s: float = 15.0
    max_turn_s: float = 10.0

    pre_flip_settle_s: float = 0.5
    flip_window_s: float = 1.0
    """``rearm_after_s`` of the merged environment's jump command: the move is over after this."""
    flip_gap_s: float = 1.5
    """Standing time between consecutive flips, after the window. Training's shortest retrigger."""
    post_flip_settle_s: float = 0.5
    running_recover_s: float = 0.0
    """Running time kept after the last landing of a running flip before the next command. Zero:
    the flip ends the move, and the stop (or the next step) follows as soon as the window closes.
    Raise it only if calibration shows the robot falling when a stop arrives while it is still
    collecting itself -- the policy's own diagnostics allow up to 3 s from the trigger for that."""
    max_flip_count: int = 5
    running_count_limit: dict[str, int] = field(default_factory=lambda: {"frontflip": 1})
    """Per kind, how many times a *running* flip may repeat. Measured on Go2-Multitask-v2: a
    running frontflip lands 0.88 the first time at normal speed and 0.38 / 0.25 the second and
    third, while running backflips and sideflips hold 0.95-1.0 through three repeats."""

    pre_stance_settle_s: float = 0.5
    min_stance_s: float = 2.0
    max_stance_s: float = 12.0
    post_stance_settle_s: float = 1.5

    final_stop_s: float = 1.0
    """Appended when a program would otherwise end while moving."""
    max_program_s: float = 40.0

    calibration: dict[str, tuple[float, float]] = field(default_factory=dict)
    """``skill key -> (a, b)`` with ``length = a * duration + b``. Keys are
    ``move:<dir>:<speed>`` (metres) and ``turn:<dir>:<speed>`` (radians). See
    :meth:`CapabilityTable.calibration`."""

    # -- lookups -------------------------------------------------------------------------------

    def velocity(self, direction: str, speed: str) -> tuple[float, float]:
        """Body-frame ``(vx, vy)`` for a direction word at a speed word."""
        ux, uy = DIRECTION_VECTORS[direction]
        if ux != 0.0 and uy != 0.0:
            # A diagonal uses the lateral table on both axes so the heading is a true 45 degrees;
            # the forward table would skew it toward the nose.
            magnitude = self.lateral_speed[speed]
            return ux * magnitude, uy * magnitude
        if ux > 0:
            return self.forward_speed[speed], 0.0
        if ux < 0:
            return -self.backward_speed[speed], 0.0
        return 0.0, uy * self.lateral_speed[speed]

    def stance_velocity(self, direction: str | None, speed: str) -> tuple[float, float]:
        if direction is None:
            return 0.0, 0.0
        ux, uy = DIRECTION_VECTORS[direction]
        norm = math.hypot(ux, uy)
        magnitude = self.stance_speed[speed]
        return ux / norm * magnitude, uy / norm * magnitude

    def duration_for_length(self, key: str, nominal_rate: float, length: float) -> float:
        """Seconds needed to cover ``length`` under the calibration for ``key``."""
        a, b = self.calibration.get(key, (nominal_rate, 0.0))
        if a <= 0:
            raise ProgramError(f"{key} is not something the robot can do (measured rate {a:.3f})")
        return max((length - b) / a, 0.0)


@dataclass
class Segment:
    """One span of constant command."""

    kind: str
    """``move``, ``turn``, ``stop``, ``settle``, ``flip``, ``stance``, or for a running flip ``gap``
    (running between two repeats) and ``recover`` (running after the last landing)."""
    t0: float
    t1: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stance: float = 0.0
    """``+1`` handstand, ``-1`` hindstand, ``0`` on four legs."""
    flip: str | None = None
    """Flip kind, fired at ``t0``; the window runs to ``t1``."""
    step_index: int | None = None
    """Which program step produced this segment; ``None`` for inserted settles and the final stop."""
    label: str = ""

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def running_flip(self) -> bool:
        """A flip fired while the velocity command is still non-zero."""
        return self.flip is not None and self.moving

    @property
    def moving(self) -> bool:
        return self.vx != 0.0 or self.vy != 0.0 or self.wz != 0.0

    def commanded_delta(self) -> tuple[float, float, float]:
        """Open-loop displacement ``(dx, dy, dyaw)`` in the frame at ``t0`` if tracking were perfect."""
        return self.vx * self.duration, self.vy * self.duration, self.wz * self.duration


@dataclass
class Timeline:
    segments: list[Segment]
    adjustments: list[str] = field(default_factory=list)
    """Human-readable notes on what the compiler changed (clamped durations, capped counts)."""
    dt: float = 0.02

    @property
    def duration(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    @property
    def num_steps(self) -> int:
        return int(round(self.duration / self.dt))

    def to_array(self, num_steps: int | None = None) -> np.ndarray:
        """``(T, 5)`` float array of ``[vx, vy, wz, flip_code, stance]`` per control step.

        ``flip_code`` is non-zero only on the step a flip fires. Rows past the timeline's end (when
        ``num_steps`` pads it) are zero: stand still.
        """
        steps = self.num_steps if num_steps is None else num_steps
        out = np.zeros((steps, 5), dtype=np.float32)
        for seg in self.segments:
            start = int(round(seg.t0 / self.dt))
            stop = max(int(round(seg.t1 / self.dt)), start + 1)
            start, stop = min(start, steps), min(stop, steps)
            if start >= stop:
                continue
            out[start:stop, COL_VX] = seg.vx
            out[start:stop, COL_VY] = seg.vy
            out[start:stop, COL_WZ] = seg.wz
            out[start:stop, COL_STANCE] = seg.stance
            if seg.flip is not None:
                out[start, COL_FLIP] = FLIP_MOTION[seg.flip][0]
        return out

    def events(self) -> list[dict[str, Any]]:
        """Timestamped commands for an executor: velocity changes, flip triggers, stance on/off."""
        out: list[dict[str, Any]] = []
        velocity = (None, None, None)
        stance = 0.0
        for seg in self.segments:
            if seg.flip is not None:
                out.append({"t": round(seg.t0, 3), "type": "flip", "kind": seg.flip})
            if seg.stance != stance:
                if seg.stance != 0.0:
                    kind = next(name for name, sign in STANCE_SIGN.items() if sign == seg.stance)
                    out.append({"t": round(seg.t0, 3), "type": "stance_on", "kind": kind, "duration_s": round(seg.duration, 3)})
                else:
                    out.append({"t": round(seg.t0, 3), "type": "stance_off"})
                stance = seg.stance
            current = (seg.vx, seg.vy, seg.wz)
            if current != velocity:
                out.append({"t": round(seg.t0, 3), "type": "velocity", "vx": seg.vx, "vy": seg.vy, "wz": seg.wz})
                velocity = current
        out.append({"t": round(self.duration, 3), "type": "end"})
        return out

    def describe(self) -> str:
        """One line per segment, for a person or for the model that writes the reply."""
        lines = []
        for seg in self.segments:
            span = f"{seg.t0:5.1f}-{seg.t1:5.1f} s"
            if seg.kind == "move":
                dx, dy, _ = seg.commanded_delta()
                lines.append(f"{span}  move {seg.label} at ({seg.vx:+.2f}, {seg.vy:+.2f}) m/s, ~{math.hypot(dx, dy):.1f} m")
            elif seg.kind == "turn":
                lines.append(f"{span}  turn {seg.label} at {seg.wz:+.2f} rad/s, ~{math.degrees(abs(seg.wz) * seg.duration):.0f} deg")
            elif seg.kind == "flip":
                running = f" while moving at ({seg.vx:+.2f}, {seg.vy:+.2f}) m/s" if seg.running_flip else ""
                lines.append(f"{span}  {seg.flip}{running}")
            elif seg.kind == "stance":
                moving = f", walking at ({seg.vx:+.2f}, {seg.vy:+.2f}) m/s" if seg.moving else ""
                lines.append(f"{span}  {seg.label}{moving}")
            else:
                lines.append(f"{span}  {seg.kind}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration, 3),
            "adjustments": list(self.adjustments),
            "events": self.events(),
        }


# =================================================================================================
# The compiler
# =================================================================================================


class _Builder:
    def __init__(self, cfg: CompilerConfig):
        self.cfg = cfg
        self.segments: list[Segment] = []
        self.adjustments: list[str] = []
        self.t = 0.0

    def add(self, kind: str, duration: float, **fields: Any) -> Segment:
        # Snap to the control period so segment boundaries land on steps and event times stay tidy.
        duration = max(round(duration / self.cfg.dt), 1) * self.cfg.dt
        seg = Segment(kind=kind, t0=self.t, t1=round(self.t + duration, 6), **fields)
        self.segments.append(seg)
        self.t = seg.t1
        return seg

    def still_for(self) -> float:
        """How long the robot has already been standing still at the current end of the timeline."""
        total = 0.0
        for seg in reversed(self.segments):
            if seg.moving or seg.flip is not None or seg.stance != 0.0:
                break
            total += seg.duration
        return total

    def settle(self, needed: float, step_index: int) -> None:
        """Insert standing time so the next skill starts from rest, counting any stop already there."""
        remaining = needed - self.still_for()
        if remaining > 1e-9:
            self.add("settle", remaining, step_index=step_index)

    def clamp(self, value: float, low: float, high: float, what: str) -> float:
        if value < low:
            self.adjustments.append(f"{what} raised from {value:g} to the minimum {low:g}")
            return low
        if value > high:
            self.adjustments.append(f"{what} reduced from {value:g} to the maximum {high:g}")
            return high
        return value


def normalise_running_flips(program: Program, context: Step | None, adjustments: list[str]) -> Program:
    """Make every running flip one the robot can actually do, noting what changed.

    The policy chooses the rotation from the heading whatever code it is handed
    (``_select_motion_for_direction``), and above the acrobatics speed ceiling it defers the move
    until the robot slows -- so a program asking for the wrong kind, or a flip out of a fast run,
    would be executed *as* the matching kind, or from standing, anyway. Doing the same here, before
    validation, means a language model's slip in this one rule is carried out the way the robot
    would have carried it out, with a note, instead of being refused. Everything else the grammar
    checks still raises.
    """
    out: Program = []
    previous = context
    for index, step in enumerate(program):
        if isinstance(step, Flip) and step.running:
            step = Flip(kind=step.kind, count=step.count, running=True)
            if not isinstance(previous, Move) or previous.speed not in RUNNING_FLIP_SPEEDS:
                why = "no move to flip out of" if not isinstance(previous, Move) else f"the move is {previous.speed}"
                adjustments.append(f"step {index} (running {step.kind}): done from standing instead -- {why}")
                step.running = False
            elif step.kind != RUNNING_FLIP_FOR[previous.dir]:
                wanted = RUNNING_FLIP_FOR[previous.dir]
                adjustments.append(f"step {index} (running {step.kind}): {wanted} instead -- the only flip possible while moving {previous.dir}")
                step.kind = wanted
        out.append(step)
        previous = step
    return out


def compile_program(program: Program, cfg: CompilerConfig | None = None, *, context: Step | None = None,
                    resume: bool = False) -> Timeline:
    """Turn a validated program into a timeline. Raises :class:`ProgramError` when it cannot be run.

    ``context`` is the step under way when this program is inserted into a running one (the
    ``insert`` action): a running flip written first then flips out of that move's velocity. With
    ``resume`` the timeline does not end in a stop -- the executor goes back to the interrupted
    step -- so an inserted running flip hands the gait straight back at the same speed.
    """
    cfg = cfg or CompilerConfig()
    b = _Builder(cfg)
    program = normalise_running_flips(program, context, b.adjustments)
    validate_program(program, context)

    for index, step in enumerate(program):
        if isinstance(step, Move):
            vx, vy = cfg.velocity(step.dir, step.speed)
            key = f"move:{step.dir}:{step.speed}"
            if step.distance_m is not None:
                duration = cfg.duration_for_length(key, math.hypot(vx, vy), step.distance_m)
                what = f"step {index} ({step.dir} {step.distance_m:g} m): duration {duration:.1f} s"
            else:
                duration = step.duration_s
                what = f"step {index} ({step.dir}): duration"
            duration = b.clamp(duration, cfg.min_move_s, cfg.max_move_s, what)
            b.add("move", duration, vx=vx, vy=vy, step_index=index, label=f"{step.dir} {step.speed}")

        elif isinstance(step, Turn):
            sign = 1.0 if step.dir == "left" else -1.0
            rate = cfg.yaw_rate[step.speed]
            key = f"turn:{step.dir}:{step.speed}"
            if step.angle_deg is not None:
                duration = cfg.duration_for_length(key, rate, math.radians(step.angle_deg))
                what = f"step {index} (turn {step.dir} {step.angle_deg:g} deg): duration {duration:.1f} s"
            else:
                duration = step.duration_s
                what = f"step {index} (turn {step.dir}): duration"
            duration = b.clamp(duration, cfg.min_move_s, cfg.max_turn_s, what)
            b.add("turn", duration, wz=sign * rate, step_index=index, label=f"{step.dir} {step.speed}")

        elif isinstance(step, Stop):
            b.add("stop", step.duration_s, step_index=index)

        elif isinstance(step, Flip):
            count = step.count
            if count > cfg.max_flip_count:
                b.adjustments.append(f"step {index} ({step.kind}): count reduced from {count} to the maximum {cfg.max_flip_count}")
                count = cfg.max_flip_count
            running_limit = cfg.running_count_limit.get(step.kind) if step.running else None
            if running_limit is not None and count > running_limit:
                b.adjustments.append(f"step {index} (running {step.kind}): count reduced from {count} to {running_limit}")
                count = running_limit
            if step.running:
                # validate_program has made sure the step before is a move (in the program, or the
                # context it is inserted into), and a move is one segment, so the segment just
                # written -- or the context -- carries the velocity to flip out of.
                if b.segments:
                    vx, vy = b.segments[-1].vx, b.segments[-1].vy
                else:
                    vx, vy = cfg.velocity(context.dir, context.speed)
                gap_kind = "gap"  # keep running between repeats, as the training schedule did
            else:
                vx = vy = 0.0
                gap_kind = "settle"
                b.settle(cfg.pre_flip_settle_s, index)
            for k in range(count):
                if k > 0:
                    b.add(gap_kind, cfg.flip_gap_s, vx=vx, vy=vy, step_index=index)
                b.add("flip", cfg.flip_window_s, vx=vx, vy=vy, flip=step.kind, step_index=index, label=step.kind)
            if step.running:
                if cfg.running_recover_s > 0.0:
                    b.add("recover", cfg.running_recover_s, vx=vx, vy=vy, step_index=index)
            else:
                b.add("settle", cfg.post_flip_settle_s, step_index=index)

        elif isinstance(step, Stance):
            duration = b.clamp(step.duration_s, cfg.min_stance_s, cfg.max_stance_s, f"step {index} ({step.kind}): hold")
            vx, vy = cfg.stance_velocity(step.dir, step.speed)
            b.settle(cfg.pre_stance_settle_s, index)
            b.add("stance", duration, vx=vx, vy=vy, stance=STANCE_SIGN[step.kind], step_index=index, label=step.kind)
            b.add("settle", cfg.post_stance_settle_s, step_index=index)

        else:  # pragma: no cover - the grammar already rejects this
            raise ProgramError(f"unknown step type {type(step).__name__}")

    if b.segments and b.segments[-1].moving and not resume:
        b.add("stop", cfg.final_stop_s)

    if b.t > cfg.max_program_s:
        raise ProgramError(f"program runs {b.t:.1f} s, longer than the {cfg.max_program_s:g} s limit -- shorten or split it")

    return Timeline(segments=b.segments, adjustments=b.adjustments, dt=cfg.dt)
