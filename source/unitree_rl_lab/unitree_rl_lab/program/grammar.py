"""The skill vocabulary a language model writes programs in.

Five skills, each a dataclass with a small set of fields, serialised as a JSON list of objects
keyed by ``"skill"``. The vocabulary is closed on purpose: every kind here maps to something the
``Go2-Multitask-v2`` policy was trained to do, and the compiler knows how to sequence them. A
language model asked for anything outside it should decline, not improvise.

Directions are in the robot's own frame (``forward`` is where the head points, ``left`` is the
robot's left), at the eight compass points. Finer headings are not offered because people do not
give them in words; a curve is a ``move`` followed by a ``turn``.

Flip kinds are named by what they look like from outside, and mapped to the acrobatics command's
motion codes in :data:`FLIP_MOTION`:

    backflip        rotates backwards (nose up first)           MOTION_BACKFLIP,       pitch -1 turn
    frontflip       rotates forwards  (nose down first)         MOTION_HANDSPRING,     pitch +1 turn
    sideflip_left   rolls to the robot's left                   MOTION_SIDEFLIP,       roll  -1 turn
    sideflip_right  rolls to the robot's right                  MOTION_SIDEFLIP_RIGHT, roll  +1 turn

There is deliberately no vertical jump. ``Go2-Multitask-v2`` sets ``enable_jump = False`` -- the
jump existed only as the substitute fired when a sampled move fought the commanded direction, and
with all four rotations trained there was nothing left for it to stand in for. Keeping it in the
grammar meant every layer above carried a skill the robot cannot do: a decline rule in the system
prompt, an alternative-proposal dialogue, a rate of 0.0 in the capability table. Removed 2026-09-21.

A flip is normally done from standing. With ``"running": true`` it is fired at the end of the
preceding ``move`` without stopping first. The policy then chooses the rotation itself from the
commanded heading (``JumpCommand._select_motion_for_direction``): forward -> frontflip, backward ->
backflip, left -> sideflip_left, right -> sideflip_right, the dominant axis deciding on a diagonal
and ties going fore-aft. The grammar makes that choice explicit rather than silent -- a running
flip whose kind does not match the move's direction is an error, so the model has to propose the
matching one instead of asking for something the robot would quietly replace. Running flips are
only offered under the acrobatics speed ceiling (1.0 m/s), which rules out ``"fast"``.

Stance kinds follow ``HandstandCommand``'s sign convention: ``handstand`` stands on the front legs
(hind legs lifted, ``STANCE_FRONT = +1``); ``hindstand`` stands on the hind legs (front lifted,
``STANCE_HIND = -1``).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union

DIRECTIONS: tuple[str, ...] = ("forward", "backward", "left", "right")
"""The four headings. Diagonals were dropped 2026-09-21: they doubled the direction vocabulary and
the running-flip table for a heading nobody asks for by name, and the policy reaches any of them
as a turn followed by a move."""
TURN_DIRECTIONS: tuple[str, ...] = ("left", "right")
SPEEDS: tuple[str, ...] = ("slow", "normal", "fast")
FLIP_KINDS: tuple[str, ...] = ("backflip", "frontflip", "sideflip_left", "sideflip_right")
STANCE_KINDS: tuple[str, ...] = ("handstand", "hindstand")
OPEN_HOLD_S: float = 10.0
"""How much of an open-ended stance the compiler draws. Not a limit: the step has no end, and the
executor holds it until something replaces it. The policy's own flag drops at BIPED_HOLD_S and the
robot comes down there whatever anyone writes, which is the policy's business and not the
grammar's."""

# Unit vectors in the body frame, (x forward, y left).
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}

# Motion code, pitch turns, roll turns -- the values `JumpCommand` reads. The codes are restated
# here rather than imported so this package stays free of Isaac Lab; `validate_programs.py`
# asserts they match `JumpCommand.MOTION_*` at start-up.
FLIP_MOTION: dict[str, tuple[int, float, float]] = {
    "backflip": (2, -1.0, 0.0),
    "sideflip_left": (3, 0.0, -1.0),
    "frontflip": (4, 1.0, 0.0),
    "sideflip_right": (5, 0.0, 1.0),
}

STANCE_SIGN: dict[str, float] = {"handstand": 1.0, "hindstand": -1.0}



# Japanese labels, for prompts and the generated replies.
JA: dict[str, str] = {
    "forward": "前",
    "backward": "後ろ",
    "left": "左",
    "right": "右",
    "forward_left": "左斜め前",
    "forward_right": "右斜め前",
    "backward_left": "左斜め後ろ",
    "backward_right": "右斜め後ろ",
    "slow": "ゆっくり",
    "normal": "普通の速さで",
    "fast": "速く",
    "backflip": "バク転",
    "frontflip": "前方回転",
    "sideflip_left": "左側転",
    "sideflip_right": "右側転",
    "handstand": "倒立",
    "hindstand": "後ろ足立ち",
    "move": "移動",
    "turn": "旋回",
    "flip": "アクロバット",
    "stance": "二足立ち",
}


class ProgramError(ValueError):
    """A program that does not fit the grammar. The message is meant to be shown to the model."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProgramError(message)


def _positive(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number")
    _require(value > 0, f"{name} must be positive, got {value}")
    return float(value)


@dataclass
class Move:
    """Walk or run in one of the four directions.

    A length is optional. With neither field the move is *open-ended*: it runs until something
    replaces it, which is what 「前に進んで」 with no amount means. Nothing can follow an open-ended
    move -- see :func:`validate_program`.
    """

    dir: str
    speed: str = "normal"
    duration_s: float | None = None
    distance_m: float | None = None
    skill: str = field(default="move", init=False)

    @property
    def open_ended(self) -> bool:
        return self.duration_s is None and self.distance_m is None

    def validate(self) -> None:
        _require(self.dir in DIRECTIONS, f"move.dir must be one of {list(DIRECTIONS)}, got {self.dir!r}")
        _require(self.speed in SPEEDS, f"move.speed must be one of {list(SPEEDS)}, got {self.speed!r}")
        _require(self.duration_s is None or self.distance_m is None,
                 "move takes duration_s or distance_m, not both")
        if self.duration_s is not None:
            self.duration_s = _positive(self.duration_s, "move.duration_s")
        if self.distance_m is not None:
            self.distance_m = _positive(self.distance_m, "move.distance_m")


@dataclass
class Turn:
    """Rotate in place through an angle, at the robot's fastest rate.

    There is no speed to choose. Waiting out a slow turn is time the operator spends staring at a
    robot rotating, and the acrobatics never depended on the rate.
    """

    dir: str
    angle_deg: float = 90.0
    skill: str = field(default="turn", init=False)

    def validate(self) -> None:
        _require(self.dir in TURN_DIRECTIONS, f"turn.dir must be one of {list(TURN_DIRECTIONS)}, got {self.dir!r}")
        self.angle_deg = _positive(self.angle_deg, "turn.angle_deg")


@dataclass
class Flip:
    """One acrobatic rotation.

    One step is one rotation -- two in a row is the step written twice. Whether it is done out of a
    run or from standing is not a field: it follows from what precedes it, a move or a stop. The
    kind is not checked against the heading either; the policy does what it does with a mismatch,
    and guessing on its behalf was worth more confusion than it saved.
    """

    kind: str
    skill: str = field(default="flip", init=False)

    def validate(self) -> None:
        _require(self.kind in FLIP_KINDS, f"flip.kind must be one of {list(FLIP_KINDS)}, got {self.kind!r}")


@dataclass
class Stance:
    """Rise onto two legs and hold it.

    With no ``duration_s`` the hold is open-ended -- 「ずっと立ってて」 -- and, like an open-ended
    move, it can only be the last step because nothing after it would ever run.
    """

    kind: str
    duration_s: float | None = None
    skill: str = field(default="stance", init=False)

    @property
    def open_ended(self) -> bool:
        return self.duration_s is None

    def validate(self) -> None:
        _require(self.kind in STANCE_KINDS, f"stance.kind must be one of {list(STANCE_KINDS)}, got {self.kind!r}")
        if self.duration_s is not None:
            self.duration_s = _positive(self.duration_s, "stance.duration_s")


Step = Union[Move, Turn, Flip, Stance]
Program = list  # list[Step]

_STEP_TYPES: dict[str, type] = {"move": Move, "turn": Turn, "flip": Flip, "stance": Stance}


def step_from_dict(data: dict[str, Any]) -> Step:
    """Build one step from its JSON object, validating it."""
    _require(isinstance(data, dict), f"each step must be an object, got {type(data).__name__}")
    kind = data.get("skill")
    _require(kind in _STEP_TYPES, f"skill must be one of {list(_STEP_TYPES)}, got {kind!r}")
    cls = _STEP_TYPES[kind]
    allowed = {name for name in cls.__dataclass_fields__ if name != "skill"}
    unknown = set(data) - allowed - {"skill"}
    _require(not unknown, f"{kind} does not take {sorted(unknown)}; allowed fields: {sorted(allowed)}")
    try:
        step = cls(**{key: value for key, value in data.items() if key != "skill"})
    except TypeError as exc:  # a required field is missing; say which, in the form a model can act on
        required = [name for name, f in cls.__dataclass_fields__.items()
                    if name != "skill" and f.default is dataclasses.MISSING]
        raise ProgramError(f"{kind} is missing {sorted(set(required) - set(data))}: {exc}") from exc
    step.validate()
    return step


def step_to_dict(step: Step) -> dict[str, Any]:
    """The JSON object for one step, ``skill`` first and ``None`` fields dropped."""
    data = asdict(step)
    out: dict[str, Any] = {"skill": data.pop("skill")}
    out.update({key: value for key, value in data.items() if value is not None})
    return out


def program_from_json(source: str | list[dict[str, Any]]) -> Program:
    """Parse a program from a JSON string or an already-decoded list. Raises :class:`ProgramError`."""
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ProgramError(f"program is not valid JSON: {exc}") from exc
    if isinstance(source, dict) and "program" in source:
        source = source["program"]
    _require(isinstance(source, list), "program must be a JSON list of steps")
    return [step_from_dict(item) for item in source]


def program_to_json(program: Program, indent: int | None = None) -> str:
    return json.dumps([step_to_dict(step) for step in program], ensure_ascii=False, indent=indent)


def validate_program(program: Program) -> None:
    """Check every step, plus the one rule that spans steps.

    That rule is the open-ended move: it never ends, so anything written after it would never run.
    Everything else a step needs to know is inside the step.
    """
    for index, step in enumerate(program):
        step.validate()
        if getattr(step, "open_ended", False) and index != len(program) - 1:
            raise ProgramError(
                f"step {index} ({step.skill}) has no length, so nothing after it would ever run; "
                "put it last, or give it a length")


def _alt(values) -> str:
    return "|".join(f'"{value}"' for value in values)


def describe_grammar() -> str:
    """The grammar as text, for a system prompt. Kept in one place so prompt and code agree."""
    return f"""A program is a JSON list of steps. Each step is an object with a "skill" field and its parameters:

  {{"skill": "move",   "dir": {_alt(DIRECTIONS)}, "speed": {_alt(SPEEDS)},
                      "duration_s": <s>   (or "distance_m": <m>, or leave both out)}}
  {{"skill": "turn",   "dir": "left"|"right", "angle_deg": <deg>}}
  {{"skill": "flip",   "kind": <flip>}}
  {{"skill": "stance", "kind": <stance>, "duration_s": <s>   (or leave it out)}}

  <flip>   : {" | ".join(FLIP_KINDS)}
  <stance> : {" | ".join(STANCE_KINDS)}   (handstand = on the front legs, hindstand = on the hind legs)

Leave the length out of a "move" or a "stance" and it carries on until something replaces it --
that is 「前に進んで」 and 「ずっと立ってて」. Only the last step can be written that way. To stop
the robot, return an empty program; there is no step for standing still."""
