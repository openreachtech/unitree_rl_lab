"""Single source of truth for the unified multi-task observation layout.

The multi-task policy blends a locomotion expert (initialised from ``Go2-Gallop-Phase2``) with an
acrobatics expert (initialised from ``Go2-Jump-Phase2``). Those two tasks were trained on different
observation vectors, so this module defines one superset layout that both can be expressed in, and
derives -- programmatically, never by hand-written indices -- the column mapping needed to widen a
single-task checkpoint's first layer into the unified width.

Layouts are described as an ordered list of :class:`Block`. The order here **must** match the order
of the ``ObsTerm`` declarations in the corresponding ``ObsGroup`` config class, because Isaac Lab
concatenates observation terms in class-body declaration order. :func:`assert_layout_matches` checks
that at runtime so a reordering cannot silently corrupt the mapping.

History convention
------------------
Isaac Lab's ``CircularBuffer.buffer`` returns ``(batch, history_length, dim)`` with the **oldest
entry first and the most recent entry last**, then flattens it. So a block with ``dim=12`` and
``history=3`` occupies 36 columns laid out as ``[t-2 | t-1 | t]`` and the *current* frame is the
final 12. A source layout with fewer history frames than the unified layout therefore maps onto the
unified layout's **trailing** frames; the older frames are zero-filled. Getting this backwards
produces a policy that acts on stale observations without raising any error, which is why the
mapping is derived rather than written out.
"""

from __future__ import annotations

from dataclasses import dataclass

# Number of height-scanner rays: GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
# -> (1.6 / 0.1 + 1) * (1.0 / 0.1 + 1) = 17 * 11.
HEIGHT_SCAN_DIM = 187

NUM_JOINTS = 12
HISTORY_LENGTH = 3
"""History length of the per-joint observation terms, inherited from ``PolicyCfgGo2``."""


@dataclass(frozen=True)
class Block:
    """One observation term's contribution to a flattened observation vector."""

    name: str
    dim: int
    history: int = 1

    @property
    def width(self) -> int:
        return self.dim * self.history


# =================================================================================================
# Policy (actor) layouts
# =================================================================================================

POLICY_UNIFIED: tuple[Block, ...] = (
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("velocity_commands", 3),
    Block("jump_command", 4),
    Block("jump_time", 1),
    Block("handstand_command", 2),
    Block("joint_pos_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_vel_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("last_action", NUM_JOINTS, HISTORY_LENGTH),
)
"""Unified actor observation: 124 columns.

``handstand_command`` is ``(enabled, stance)``: a 0/1 flag saying a bipedal stance is commanded,
and its sign -- +1 for the front-leg stance, -1 for the hind-leg one, 0 while disabled. Two columns
rather than one so the mirror stance can be trained later without moving every column after it
again; a layout change costs a re-widen of every checkpoint that exists.

Adding it took this layout from 122 columns to 124, and the critic's from 330 to 335. The
checkpoints trained on the old widths were carried across with ``widen_checkpoint.py`` -- the new
columns start at zero weight, so a widened network computes exactly the function it computed
before -- and the snapshot of the old layout was deleted once they had been. Recover it from git
history if another 122-column checkpoint ever turns up, rather than reconstructing it by hand.
"""

POLICY_LOCOMOTION: tuple[Block, ...] = (
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("velocity_commands", 3),
    Block("joint_pos_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_vel_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("last_action", NUM_JOINTS, HISTORY_LENGTH),
)
"""``Go2-Gallop-Phase2`` actor observation: 117 columns (verified against ``actor.0.weight``)."""

POLICY_JUMP: tuple[Block, ...] = (
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("jump_command", 4),
    Block("jump_time", 1),
    Block("joint_pos_rel", NUM_JOINTS),
    Block("joint_vel_rel", NUM_JOINTS),
    Block("last_action", NUM_JOINTS),
)
"""``Go2-Jump-Phase2`` actor observation: 47 columns (verified against ``actor.0.weight``)."""


# =================================================================================================
# Critic layouts
# =================================================================================================

CRITIC_UNIFIED: tuple[Block, ...] = (
    Block("base_lin_vel", 3),
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("velocity_commands", 3),
    Block("jump_command", 4),
    Block("jump_time", 1),
    Block("handstand_command", 2),
    Block("root_height", 1),
    Block("root_roll_angle", 1),
    Block("root_pitch_angle", 1),
    Block("maximum_jump_height", 1),
    Block("accumulated_root_pitch", 1),
    Block("accumulated_root_roll", 1),
    Block("com_cop", 3),
    Block("joint_pos_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_vel_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_effort", NUM_JOINTS),
    Block("last_action", NUM_JOINTS, HISTORY_LENGTH),
    Block("height_scan", HEIGHT_SCAN_DIM),
)
"""Unified critic observation: 335 columns.

``com_cop`` is the vector from the centre of pressure to the centre of mass, the state variable the
bipedal balance rewards are written in (TumblerNet). Privileged, so it costs nothing at deployment,
and computed over all four feet -- the force-weighted centre of pressure collapses onto whichever
feet are actually loaded, so one term serves the quadruped gait and either bipedal stance.
"""

CRITIC_LOCOMOTION: tuple[Block, ...] = (
    Block("base_lin_vel", 3),
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("velocity_commands", 3),
    Block("joint_pos_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_vel_rel", NUM_JOINTS, HISTORY_LENGTH),
    Block("joint_effort", NUM_JOINTS),
    Block("last_action", NUM_JOINTS, HISTORY_LENGTH),
    Block("height_scan", HEIGHT_SCAN_DIM),
)
"""``Go2-Gallop-Phase2`` critic observation: 319 columns (verified against ``critic.0.weight``)."""

CRITIC_JUMP: tuple[Block, ...] = (
    Block("base_lin_vel", 3),
    Block("base_ang_vel", 3),
    Block("projected_gravity", 3),
    Block("jump_command", 4),
    Block("jump_time", 1),
    Block("root_height", 1),
    Block("root_roll_angle", 1),
    Block("root_pitch_angle", 1),
    Block("maximum_jump_height", 1),
    Block("accumulated_root_pitch", 1),
    Block("accumulated_root_roll", 1),
    Block("joint_pos_rel", NUM_JOINTS),
    Block("joint_vel_rel", NUM_JOINTS),
    Block("last_action", NUM_JOINTS),
)
"""``Go2-Jump-Phase2`` critic observation: 56 columns (verified against ``critic.0.weight``)."""


# =================================================================================================
# Derived quantities
# =================================================================================================


def layout_dim(layout: tuple[Block, ...]) -> int:
    """Total flattened width of a layout."""
    return sum(block.width for block in layout)


def block_offsets(layout: tuple[Block, ...]) -> dict[str, int]:
    """Map each block name to its starting column in the flattened vector."""
    offsets: dict[str, int] = {}
    cursor = 0
    for block in layout:
        if block.name in offsets:
            raise ValueError(f"Duplicate block name in layout: {block.name!r}")
        offsets[block.name] = cursor
        cursor += block.width
    return offsets


POLICY_DIM = layout_dim(POLICY_UNIFIED)  # 122
CRITIC_DIM = layout_dim(CRITIC_UNIFIED)  # 330



def source_to_unified_map(
    source: tuple[Block, ...], unified: tuple[Block, ...]
) -> list[tuple[int, int, int]]:
    """Derive the column mapping that widens ``source`` onto ``unified``.

    Returns a list of ``(source_start, source_end, unified_start)`` contiguous copy instructions.
    Columns of ``unified`` not covered by any instruction receive zero weight, which makes them
    inert: a linear layer's output is unchanged by inputs whose weights are zero, so the widened
    network computes exactly the same function as the original.

    Blocks present in ``source`` but absent from ``unified`` are an error -- that means the unified
    layout is missing information the checkpoint depends on.
    """
    unified_blocks = {block.name: block for block in unified}
    unified_offsets = block_offsets(unified)
    source_offsets = block_offsets(source)

    mapping: list[tuple[int, int, int]] = []
    for block in source:
        target = unified_blocks.get(block.name)
        if target is None:
            raise ValueError(
                f"Block {block.name!r} exists in the source layout but not in the unified layout;"
                " the unified layout cannot represent this checkpoint."
            )
        if target.dim != block.dim:
            raise ValueError(
                f"Block {block.name!r} has dim {block.dim} in the source layout but {target.dim} in"
                " the unified layout."
            )
        if target.history < block.history:
            raise ValueError(
                f"Block {block.name!r} has history {block.history} in the source layout, which does"
                f" not fit the unified layout's history {target.history}."
            )
        source_start = source_offsets[block.name]
        # The source's frames are the most recent ones, so they land on the unified layout's
        # trailing frames -- see the history convention in this module's docstring.
        frame_shift = (target.history - block.history) * target.dim
        mapping.append((source_start, source_start + block.width, unified_offsets[block.name] + frame_shift))
    return mapping


POLICY_MAP_LOCOMOTION = source_to_unified_map(POLICY_LOCOMOTION, POLICY_UNIFIED)
POLICY_MAP_JUMP = source_to_unified_map(POLICY_JUMP, POLICY_UNIFIED)
CRITIC_MAP_LOCOMOTION = source_to_unified_map(CRITIC_LOCOMOTION, CRITIC_UNIFIED)
CRITIC_MAP_JUMP = source_to_unified_map(CRITIC_JUMP, CRITIC_UNIFIED)


def assert_layout_matches(layout: tuple[Block, ...], term_dims: dict[str, int], group_name: str) -> None:
    """Check a layout against the observation manager's actual term dimensions.

    Call this once at env startup with ``env.observation_manager``'s term names and dimensions. The
    column mapping is only correct if the config declares the same terms, in the same order, with
    the same widths -- and none of that is enforced by Isaac Lab itself.
    """
    expected = [(block.name, block.width) for block in layout]
    actual = list(term_dims.items())
    if expected != actual:
        raise RuntimeError(
            f"Observation group {group_name!r} does not match the layout declared in obs_spec.py.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Reorder or resize the ObsTerm declarations to match, or update obs_spec.py -- the"
            " expert weight mapping depends on this order."
        )
