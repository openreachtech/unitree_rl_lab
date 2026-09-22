"""Random programs shaped like the ones people ask for.

A uniform draw over the grammar produces "backward_right 2.3 s, frontflip, hindstand 7.1 s" -- valid,
but nobody says that, and a model trained on it learns the wrong prior. The sampler biases toward
what instructions actually contain: one to four steps, round numbers, whole-number repeat counts,
forward more often than backward, a stop at the end sometimes. The bias is data, in
:class:`SamplerConfig`, so the distribution can be tuned without touching the code.

Every sampled program is compiled, and one that fails to compile is redrawn, so what comes out is
always executable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .compiler import CompilerConfig, Timeline, compile_program
from .grammar import (
    DIRECTIONS,
    FLIP_KINDS,
    STANCE_KINDS,
    TURN_DIRECTIONS,
    Flip,
    Move,
    Program,
    ProgramError,
    Stance,
    Turn,
)


def _weighted(rng: random.Random, table: dict) -> object:
    keys = list(table)
    return rng.choices(keys, weights=[table[k] for k in keys], k=1)[0]


@dataclass
class SamplerConfig:
    num_steps: dict[int, float] = field(default_factory=lambda: {1: 0.30, 2: 0.35, 3: 0.25, 4: 0.10})
    skills: dict[str, float] = field(
        default_factory=lambda: {"move": 0.46, "turn": 0.17, "flip": 0.24, "stance": 0.13}
    )
    directions: dict[str, float] = field(
        default_factory=lambda: {"forward": 0.52, "backward": 0.16, "left": 0.16, "right": 0.16}
    )
    speeds: dict[str, float] = field(default_factory=lambda: {"slow": 0.25, "normal": 0.55, "fast": 0.20})
    move_by_distance: float = 0.5
    """Probability a move is given as a distance rather than a time, when it is given at all."""
    open_move: float = 0.20
    """Share of programs that end in a move with no length -- 「前に進んで」, which runs until
    something replaces it. Drawn as the last step whatever the skill weights would have said, so
    the number means what it reads as; nothing can follow one, which is why it is the last."""
    move_durations_s: dict[float, float] = field(
        default_factory=lambda: {1.0: 0.08, 2.0: 0.20, 3.0: 0.22, 5.0: 0.22, 8.0: 0.12, 10.0: 0.08,
                                 15.0: 0.05, 20.0: 0.03}
    )
    move_distances_m: dict[float, float] = field(
        default_factory=lambda: {0.5: 0.08, 1.0: 0.25, 2.0: 0.22, 3.0: 0.18, 5.0: 0.15,
                                 8.0: 0.07, 10.0: 0.05}
    )
    turn_angles_deg: dict[float, float] = field(
        default_factory=lambda: {30.0: 0.10, 45.0: 0.15, 90.0: 0.35, 180.0: 0.20, 270.0: 0.10, 360.0: 0.10}
    )
    flip_kinds: dict[str, float] = field(
        default_factory=lambda: {"backflip": 0.4, "frontflip": 0.25, "sideflip_left": 0.175, "sideflip_right": 0.175}
    )
    flip_repeat: float = 0.25
    """Probability a flip is written twice in a row -- one step is one rotation, so that is what
    「2回」 looks like."""
    repeat_earlier: float = 0.15
    """Probability that a step is a *copy of an earlier one* rather than a fresh draw.

    Two of the same thing next to each other only makes sense for a rotation -- 「前に3秒、前に3秒」
    is 「前に6秒」 with extra words, which is what :func:`_redundant` throws away. But the same
    thing again *later* is ordinary: 「3m進んで、右向いて、また3m進んで」. Without this the only
    programs that repeat anything are the flips, and 「さっきの倒立もう一回」 has nothing to be
    built from: 4000 programs carried 446 repeated rotations and 18 of everything else."""
    running_flip_after_move: float = 0.18
    """Probability that a move is followed straight away by a flip, which makes it a running one.
    The kind is the one the heading was trained with: the grammar no longer insists, but there is
    no reason to draw a pairing the policy was never shown."""
    flip_and_continue: float = 0.10
    """Share of programs that are 「歩きながら前転して」 in full: a run-up, the flip out of it, and
    the same move again with no length so the robot carries on. Three steps, and the only shape
    that says "do it while moving and keep moving" -- a flip cannot follow an open-ended move, so
    the run-up has to be given a length."""
    flip_run_up_s: dict[float, float] = field(default_factory=lambda: {2.0: 0.5, 3.0: 0.5})
    """How long the robot walks before the flip. Not arbitrary: the gait has to be up to speed, and
    the measured start-up loss is about 0.16 m, so two seconds is already well clear of it."""

    running_pair: dict[str, str] = field(
        default_factory=lambda: {"forward": "frontflip", "backward": "backflip",
                                 "left": "sideflip_left", "right": "sideflip_right"}
    )
    running_flip_speeds: tuple[str, ...] = ("slow", "normal")
    """Above this the policy defers the move until the robot slows, so a fast run is not drawn."""
    stance_kinds: dict[str, float] = field(default_factory=lambda: {"handstand": 0.55, "hindstand": 0.45})
    stance_durations_s: dict[float, float] = field(default_factory=lambda: {3.0: 0.35, 5.0: 0.40, 10.0: 0.25})
    """The hold lengths the training programs use. Any length is writable -- 「8秒立ってて」 is a
    thing to say -- these are just what gets drawn."""
    open_stance: float = 0.20
    """Share of stances with no length at all -- 「ずっと立ってて」. Like an open-ended move it can
    only be the last step, so it is only drawn there."""
    max_redraws: int = 200

    def __post_init__(self) -> None:
        assert set(self.directions) <= set(DIRECTIONS)
        assert set(self.running_pair) <= set(DIRECTIONS)
        assert set(self.flip_kinds) <= set(FLIP_KINDS)
        assert set(self.stance_kinds) <= set(STANCE_KINDS)


def sample_step(rng: random.Random, cfg: SamplerConfig, previous: str | None) -> Move | Turn | Flip | Stance:
    skill = _weighted(rng, cfg.skills)

    if skill == "move":
        direction = _weighted(rng, cfg.directions)
        speed = _weighted(rng, cfg.speeds)
        if rng.random() < cfg.move_by_distance:
            return Move(dir=direction, speed=speed, distance_m=_weighted(rng, cfg.move_distances_m))
        return Move(dir=direction, speed=speed, duration_s=_weighted(rng, cfg.move_durations_s))
    if skill == "turn":
        return Turn(dir=rng.choice(TURN_DIRECTIONS), angle_deg=_weighted(rng, cfg.turn_angles_deg))
    if skill == "flip":
        return Flip(kind=_weighted(rng, cfg.flip_kinds))
    return Stance(kind=_weighted(rng, cfg.stance_kinds), duration_s=_weighted(rng, cfg.stance_durations_s))


def _open_tail(rng: random.Random, cfg: SamplerConfig) -> Move | Stance:
    """A last step with no length: 「前に進んで」 or 「ずっと立ってて」."""
    if rng.random() < cfg.open_stance:
        return Stance(kind=_weighted(rng, cfg.stance_kinds))
    return Move(dir=_weighted(rng, cfg.directions), speed=_weighted(rng, cfg.speeds))


def _repeat_of_earlier(rng: random.Random, cfg: SamplerConfig, program: Program):
    """A step already in this program, asked for again -- or None to draw a new one.

    Never the one immediately before (that is :data:`SamplerConfig.flip_repeat`'s business, and for
    anything else it is redundant), and never one that would land a rotation the heading in front of
    it was not trained with.
    """
    if len(program) < 2 or rng.random() >= cfg.repeat_earlier:
        return None
    previous = program[-1]
    candidates = [step for step in program[:-1] if not _redundant(previous, step)]
    if isinstance(previous, Move):
        # A copy placed here comes out of the run, so only a rotation the heading pairs with.
        candidates = [step for step in candidates
                      if not isinstance(step, Flip) or step.kind == cfg.running_pair[previous.dir]]
    return rng.choice(candidates) if candidates else None


def _redundant(a, b) -> bool:
    """Whether ``b`` says nothing ``a`` did not already say.

    「前に5秒、そのあと前に3秒」 is 「前に8秒」 with extra words, and 「倒立3秒、そのあと倒立5秒」 has
    the robot come down and go back up for no reason. Nobody asks for either, so neither belongs in
    the prior. Two flips in a row are *not* redundant -- that is how two rotations are written.
    """
    if isinstance(a, Move) and isinstance(b, Move):
        return a.dir == b.dir and a.speed == b.speed
    if isinstance(a, Stance) and isinstance(b, Stance):
        return a.kind == b.kind
    if isinstance(a, Turn) and isinstance(b, Turn):
        return a.dir == b.dir
    return False


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
        steps = _weighted(rng, cfg.num_steps)
        if rng.random() < cfg.flip_and_continue:
            direction = _weighted(rng, cfg.directions)
            speed = rng.choice(cfg.running_flip_speeds)
            program = [Move(dir=direction, speed=speed, duration_s=_weighted(rng, cfg.flip_run_up_s)),
                       Flip(kind=cfg.running_pair[direction]),
                       Move(dir=direction, speed=speed)]
            try:
                return program, compile_program(program, compiler)
            except ProgramError:
                continue

        open_last = rng.random() < cfg.open_move
        for index in range(steps):
            if open_last and index == steps - 1:
                tail = _open_tail(rng, cfg)
                # 「3m進んで、そのあと前に進んで」 is one instruction with a redundant half; the
                # open-ended tail has to clear the same bar as any other step.
                if not (program and _redundant(program[-1], tail)):
                    program.append(tail)
                    previous = tail.skill
                break
            step = _repeat_of_earlier(rng, cfg, program) or sample_step(rng, cfg, previous)
            # A flip drawn straight after a move *is* a running one -- that is what the order means
            # now that "running" is not a field. Make it one the policy was trained on: the kind its
            # heading pairs with, or, out of a run too fast for the acrobatics, a stop first.
            if isinstance(step, Flip) and program and isinstance(program[-1], Move):
                if program[-1].open_ended:
                    pass                                   # nothing follows an open-ended move
                elif program[-1].speed in cfg.running_flip_speeds:
                    step = Flip(kind=cfg.running_pair[program[-1].dir])
                else:
                    continue                               # too fast to flip out of; draw again
            if program and _redundant(program[-1], step):
                continue                                   # drawn again next time round
            program.append(step)
            previous = step.skill
            if isinstance(step, Flip) and rng.random() < cfg.flip_repeat:
                program.append(Flip(kind=step.kind))       # 「2回」 is the step written twice
            elif (isinstance(step, Move) and not step.open_ended
                  and step.speed in cfg.running_flip_speeds
                  and rng.random() < cfg.running_flip_after_move):
                program.append(Flip(kind=cfg.running_pair[step.dir]))
                previous = "flip"
        try:
            timeline = compile_program(program, compiler)
        except ProgramError:
            continue
        return program, timeline
    raise RuntimeError(f"could not draw a compilable program in {cfg.max_redraws} tries")
