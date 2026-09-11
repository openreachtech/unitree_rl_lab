"""Skill programs: the interface between a language model and the Go2 multi-task policy.

A language model does not drive the robot at 50 Hz. It emits a short *program* -- a list of skills
such as "move forward 5 m", "backflip twice", "handstand for 5 s" -- and a deterministic compiler
turns that into the velocity / flip / stance command stream the policy was trained on, inserting
the settle gaps and speed limits the policy needs. Everything that has to be exactly right lives in
the compiler; the model only has to get the intent right.

- :mod:`.grammar`    -- the skill vocabulary, its JSON form, and validation.
- :mod:`.compiler`   -- program -> timeline of commands, with a calibration table.
- :mod:`.sampler`    -- human-like random programs for building the training set.
- :mod:`.capability` -- what the policy can actually do, measured in simulation.

This package deliberately imports nothing from Isaac Lab, so it can be used on the robot and in a
data-generation job without a simulator installed.
"""

from .capability import CapabilityTable
from .compiler import CompilerConfig, Timeline, compile_program
from .grammar import (
    DIRECTIONS,
    FLIP_KINDS,
    SPEEDS,
    STANCE_KINDS,
    TURN_DIRECTIONS,
    Flip,
    Move,
    Program,
    ProgramError,
    Stance,
    Step,
    Stop,
    Turn,
    describe_grammar,
    program_from_json,
    program_to_json,
)
from .sampler import SamplerConfig, sample_program

__all__ = [
    "CapabilityTable",
    "CompilerConfig",
    "DIRECTIONS",
    "FLIP_KINDS",
    "Flip",
    "Move",
    "Program",
    "ProgramError",
    "SPEEDS",
    "STANCE_KINDS",
    "SamplerConfig",
    "Stance",
    "Step",
    "Stop",
    "TURN_DIRECTIONS",
    "Timeline",
    "Turn",
    "compile_program",
    "describe_grammar",
    "program_from_json",
    "program_to_json",
    "sample_program",
]
