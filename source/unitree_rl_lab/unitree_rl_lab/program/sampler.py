"""Random programs shaped like the ones people ask for.

A uniform draw over the grammar produces "backward_right 2.3 s, frontflip, hindstand 7.1 s" -- valid,
but nobody says that, and a model trained on it learns the wrong prior. The sampler biases toward
what instructions actually contain: one to four steps, round numbers, whole-number repeat counts,
forward more often than backward, a stop at the end sometimes. The bias is data, in
:class:`SamplerConfig`, so the distribution can be tuned without touching the code.

Every sampled program is compiled, and one that fails or runs longer than ``max_total_s`` is
redrawn, so what comes out is always executable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .compiler import CompilerConfig, Timeline, compile_program
from .grammar import DIRECTIONS, FLIP_KINDS, STANCE_KINDS, Flip, Move, Program, ProgramError, Stance, Stop, Turn


def _weighted(rng: random.Random, table: dict) -> object:
    keys = list(table)
    return rng.choices(keys, weights=[table[k] for k in keys], k=1)[0]


@dataclass
class SamplerConfig:
    max_total_s: float = 20.0
    """Programs longer than this are redrawn. Matches the policy's training episode."""

    num_steps: dict[int, float] = field(default_factory=lambda: {1: 0.30, 2: 0.35, 3: 0.25, 4: 0.10})
    skills: dict[str, float] = field(
        default_factory=lambda: {"move": 0.42, "turn": 0.15, "flip": 0.23, "stance": 0.12, "stop": 0.08}
    )
    directions: dict[str, float] = field(
        default_factory=lambda: {
            "forward": 0.40,
            "backward": 0.12,
            "left": 0.10,
            "right": 0.10,
            "forward_left": 0.08,
            "forward_right": 0.08,
            "backward_left": 0.06,
            "backward_right": 0.06,
        }
    )
    speeds: dict[str, float] = field(default_factory=lambda: {"slow": 0.25, "normal": 0.55, "fast": 0.20})
    move_by_distance: float = 0.5
    """Probability a move is given as a distance rather than a time."""
    move_durations_s: dict[float, float] = field(
        default_factory=lambda: {1.0: 0.10, 2.0: 0.25, 3.0: 0.25, 5.0: 0.25, 8.0: 0.10, 10.0: 0.05}
    )
    move_distances_m: dict[float, float] = field(
        default_factory=lambda: {0.5: 0.10, 1.0: 0.30, 2.0: 0.25, 3.0: 0.20, 5.0: 0.15}
    )
    turn_by_angle: float = 0.6
    turn_angles_deg: dict[float, float] = field(default_factory=lambda: {45.0: 0.15, 90.0: 0.45, 180.0: 0.30, 360.0: 0.10})
    turn_durations_s: dict[float, float] = field(default_factory=lambda: {1.0: 0.2, 2.0: 0.4, 3.0: 0.3, 5.0: 0.1})
    flip_kinds: dict[str, float] = field(
        default_factory=lambda: {"backflip": 0.35, "frontflip": 0.2, "sideflip_left": 0.15, "sideflip_right": 0.15, "jump": 0.15}
    )
    flip_counts: dict[int, float] = field(default_factory=lambda: {1: 0.60, 2: 0.25, 3: 0.15})
    stance_kinds: dict[str, float] = field(default_factory=lambda: {"handstand": 0.55, "hindstand": 0.45})
    stance_durations_s: dict[float, float] = field(default_factory=lambda: {3.0: 0.30, 5.0: 0.40, 8.0: 0.15, 10.0: 0.15})
    stance_walks: float = 0.15
    """Probability a stance also walks (slowly) in some direction."""
    stop_durations_s: dict[float, float] = field(default_factory=lambda: {1.0: 0.3, 2.0: 0.4, 3.0: 0.3})
    trailing_stop: float = 0.15
    """Probability of an explicit stop at the end, on top of the compiler's own."""
    max_redraws: int = 200

    def __post_init__(self) -> None:
        assert set(self.directions) <= set(DIRECTIONS)
        assert set(self.flip_kinds) <= set(FLIP_KINDS)
        assert set(self.stance_kinds) <= set(STANCE_KINDS)


def sample_step(rng: random.Random, cfg: SamplerConfig, previous: str | None) -> Move | Turn | Stop | Flip | Stance:
    skills = dict(cfg.skills)
    if previous in ("stop", None):
        skills.pop("stop", None)  # two stops in a row, or a program that opens with one, is noise
    skill = _weighted(rng, skills)

    if skill == "move":
        direction = _weighted(rng, cfg.directions)
        speed = _weighted(rng, cfg.speeds)
        if rng.random() < cfg.move_by_distance:
            return Move(dir=direction, speed=speed, distance_m=_weighted(rng, cfg.move_distances_m))
        return Move(dir=direction, speed=speed, duration_s=_weighted(rng, cfg.move_durations_s))
    if skill == "turn":
        direction = rng.choice(("left", "right"))
        speed = _weighted(rng, cfg.speeds)
        if rng.random() < cfg.turn_by_angle:
            return Turn(dir=direction, speed=speed, angle_deg=_weighted(rng, cfg.turn_angles_deg))
        return Turn(dir=direction, speed=speed, duration_s=_weighted(rng, cfg.turn_durations_s))
    if skill == "flip":
        return Flip(kind=_weighted(rng, cfg.flip_kinds), count=_weighted(rng, cfg.flip_counts))
    if skill == "stance":
        step = Stance(kind=_weighted(rng, cfg.stance_kinds), duration_s=_weighted(rng, cfg.stance_durations_s))
        if rng.random() < cfg.stance_walks:
            step.dir = _weighted(rng, cfg.directions)
            step.speed = "slow"
        return step
    return Stop(duration_s=_weighted(rng, cfg.stop_durations_s))


def sample_program(
    rng: random.Random,
    cfg: SamplerConfig | None = None,
    compiler: CompilerConfig | None = None,
) -> tuple[Program, Timeline]:
    """Draw one executable program and its compiled timeline."""
    cfg = cfg or SamplerConfig()
    compiler = compiler or CompilerConfig()
    for _ in range(cfg.max_redraws):
        program: Program = []
        previous: str | None = None
        for _ in range(_weighted(rng, cfg.num_steps)):
            step = sample_step(rng, cfg, previous)
            program.append(step)
            previous = step.skill
        if previous != "stop" and rng.random() < cfg.trailing_stop:
            program.append(Stop(duration_s=_weighted(rng, cfg.stop_durations_s)))
        try:
            timeline = compile_program(program, compiler)
        except ProgramError:
            continue
        if timeline.duration <= cfg.max_total_s:
            return program, timeline
    raise RuntimeError(f"could not draw a program under {cfg.max_total_s} s in {cfg.max_redraws} tries")
