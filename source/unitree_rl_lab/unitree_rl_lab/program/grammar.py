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
    jump            straight up, no rotation                    MOTION_JUMP,           0.20 m

Stance kinds follow ``HandstandCommand``'s sign convention: ``handstand`` stands on the front legs
(hind legs lifted, ``STANCE_FRONT = +1``); ``hindstand`` stands on the hind legs (front lifted,
``STANCE_HIND = -1``).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union

DIRECTIONS: tuple[str, ...] = (
    "forward",
    "backward",
    "left",
    "right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
)
TURN_DIRECTIONS: tuple[str, ...] = ("left", "right")
SPEEDS: tuple[str, ...] = ("slow", "normal", "fast")
FLIP_KINDS: tuple[str, ...] = ("backflip", "frontflip", "sideflip_left", "sideflip_right", "jump")
STANCE_KINDS: tuple[str, ...] = ("handstand", "hindstand")

# Unit vectors in the body frame, (x forward, y left).
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward_left": (1.0, 1.0),
    "forward_right": (1.0, -1.0),
    "backward_left": (-1.0, 1.0),
    "backward_right": (-1.0, -1.0),
}

# Motion code, pitch turns, roll turns -- the values `JumpCommand` reads. The codes are restated
# here rather than imported so this package stays free of Isaac Lab; `validate_programs.py`
# asserts they match `JumpCommand.MOTION_*` at start-up.
FLIP_MOTION: dict[str, tuple[int, float, float]] = {
    "jump": (1, 0.0, 0.0),
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
    "jump": "ジャンプ",
    "handstand": "倒立",
    "hindstand": "後ろ足立ち",
    "move": "移動",
    "turn": "旋回",
    "stop": "停止",
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
    """Walk or run in one of the eight directions, for a time or a distance (exactly one)."""

    dir: str
    speed: str = "normal"
    duration_s: float | None = None
    distance_m: float | None = None
    skill: str = field(default="move", init=False)

    def validate(self) -> None:
        _require(self.dir in DIRECTIONS, f"move.dir must be one of {list(DIRECTIONS)}, got {self.dir!r}")
        _require(self.speed in SPEEDS, f"move.speed must be one of {list(SPEEDS)}, got {self.speed!r}")
        _require(
            (self.duration_s is None) != (self.distance_m is None),
            "move needs exactly one of duration_s or distance_m",
        )
        if self.duration_s is not None:
            self.duration_s = _positive(self.duration_s, "move.duration_s")
        if self.distance_m is not None:
            self.distance_m = _positive(self.distance_m, "move.distance_m")


@dataclass
class Turn:
    """Rotate in place, left or right, for a time or through an angle (exactly one)."""

    dir: str
    speed: str = "normal"
    duration_s: float | None = None
    angle_deg: float | None = None
    skill: str = field(default="turn", init=False)

    def validate(self) -> None:
        _require(self.dir in TURN_DIRECTIONS, f"turn.dir must be one of {list(TURN_DIRECTIONS)}, got {self.dir!r}")
        _require(self.speed in SPEEDS, f"turn.speed must be one of {list(SPEEDS)}, got {self.speed!r}")
        _require(
            (self.duration_s is None) != (self.angle_deg is None),
            "turn needs exactly one of duration_s or angle_deg",
        )
        if self.duration_s is not None:
            self.duration_s = _positive(self.duration_s, "turn.duration_s")
        if self.angle_deg is not None:
            self.angle_deg = _positive(self.angle_deg, "turn.angle_deg")


@dataclass
class Stop:
    """Stand still."""

    duration_s: float = 1.0
    skill: str = field(default="stop", init=False)

    def validate(self) -> None:
        self.duration_s = _positive(self.duration_s, "stop.duration_s")


@dataclass
class Flip:
    """One of the acrobatic moves, ``count`` times in a row. Always performed from standing."""

    kind: str
    count: int = 1
    skill: str = field(default="flip", init=False)

    def validate(self) -> None:
        _require(self.kind in FLIP_KINDS, f"flip.kind must be one of {list(FLIP_KINDS)}, got {self.kind!r}")
        _require(isinstance(self.count, int) and not isinstance(self.count, bool), "flip.count must be an integer")
        _require(self.count >= 1, f"flip.count must be at least 1, got {self.count}")


@dataclass
class Stance:
    """Rise onto two legs and hold it. Optionally walk while up, in a direction at a speed."""

    kind: str
    duration_s: float = 5.0
    dir: str | None = None
    speed: str = "slow"
    skill: str = field(default="stance", init=False)

    def validate(self) -> None:
        _require(self.kind in STANCE_KINDS, f"stance.kind must be one of {list(STANCE_KINDS)}, got {self.kind!r}")
        self.duration_s = _positive(self.duration_s, "stance.duration_s")
        if self.dir is not None:
            _require(self.dir in DIRECTIONS, f"stance.dir must be one of {list(DIRECTIONS)}, got {self.dir!r}")
        _require(self.speed in SPEEDS, f"stance.speed must be one of {list(SPEEDS)}, got {self.speed!r}")


Step = Union[Move, Turn, Stop, Flip, Stance]
Program = list  # list[Step]

_STEP_TYPES: dict[str, type] = {"move": Move, "turn": Turn, "stop": Stop, "flip": Flip, "stance": Stance}


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
    for step in program:
        step.validate()


def describe_grammar() -> str:
    """The grammar as text, for a system prompt. Kept in one place so prompt and code agree."""
    return f"""A program is a JSON list of steps. Each step is an object with a "skill" field and its parameters:

  {{"skill": "move",   "dir": <direction>, "speed": <speed>, "duration_s": <s> | "distance_m": <m>}}
  {{"skill": "turn",   "dir": "left"|"right", "speed": <speed>, "duration_s": <s> | "angle_deg": <deg>}}
  {{"skill": "stop",   "duration_s": <s>}}
  {{"skill": "flip",   "kind": <flip>, "count": <n>}}
  {{"skill": "stance", "kind": <stance>, "duration_s": <s>, "dir": <direction> (optional), "speed": <speed>}}

  <direction> : {" | ".join(DIRECTIONS)}   (robot frame; forward = where the head points)
  <speed>     : {" | ".join(SPEEDS)}   (default "normal"; for stance default "slow")
  <flip>      : {" | ".join(FLIP_KINDS)}
  <stance>    : {" | ".join(STANCE_KINDS)}   (handstand = on the front legs, hindstand = on the hind legs)

Rules: "move" and "turn" take exactly one of the two length fields. Flips and stances are always done
from standing; the executor stops the robot first, so do not add a stop before them. An empty list is a
valid program and means "do nothing". Steps run in order, one after another."""
