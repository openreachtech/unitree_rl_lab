"""Observation layout of the recovery policy, and how each expert's checkpoint maps onto it.

Kept free of Isaac Lab imports so the checkpoint builder can use it from a plain interpreter.

The actor observation is exactly the blind locomotion policy's (``Unitree-Go2-Velocity-v1``,
117 columns): that expert loads unchanged. Isaac Lab flattens a term's history oldest-first
(``CircularBuffer.buffer``), so the newest frame of each history block is its last 12 columns, and
the get-up expert's single-frame inputs go there.
"""

from __future__ import annotations

HISTORY = 3

POLICY_LAYOUT: list[tuple[str, int]] = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos_rel", 12 * HISTORY),
    ("joint_vel_rel", 12 * HISTORY),
    ("last_action", 12 * HISTORY),
]

CRITIC_LAYOUT: list[tuple[str, int]] = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos_rel", 12),
    ("joint_vel_rel", 12),
    ("joint_effort", 12),
    ("last_action", 12),
    ("base_height", 1),
]


def block_offsets(layout: list[tuple[str, int]]) -> dict[str, int]:
    offsets, start = {}, 0
    for name, width in layout:
        offsets[name] = start
        start += width
    return offsets


def layout_dim(layout: list[tuple[str, int]]) -> int:
    return sum(width for _, width in layout)


POLICY_DIM = layout_dim(POLICY_LAYOUT)
CRITIC_DIM = layout_dim(CRITIC_LAYOUT)

_P = block_offsets(POLICY_LAYOUT)
POLICY_GRAVITY_Z = _P["projected_gravity"] + 2
CRITIC_GRAVITY_Z = block_offsets(CRITIC_LAYOUT)["projected_gravity"] + 2


def _newest(name: str) -> int:
    return _P[name] + 12 * (HISTORY - 1)


GETUP_POLICY_DIM = 42
"""``Unitree-Go2-GetUp-v0``: ang_vel(3), gravity(3), joint_pos(12), joint_vel(12), last_action(12)."""

GETUP_POLICY_MAP: list[tuple[int, int, int]] = [
    (0, 3, _P["base_ang_vel"]),
    (3, 6, _P["projected_gravity"]),
    (6, 18, _newest("joint_pos_rel")),
    (18, 30, _newest("joint_vel_rel")),
    (30, 42, _newest("last_action")),
]
"""``(source_start, source_end, target_start)`` for the get-up actor's input layer."""

GETUP_LAST_ACTION_COLUMNS = (_newest("last_action"), _newest("last_action") + 12)
"""Where the get-up expert reads its previous action in the unified layout."""

GETUP_ACTION_SCALE = 0.5
ACTION_SCALE = 0.25
"""The get-up policy was trained at action scale 0.5, the locomotion one (and this task) at 0.25.
The builder rescales the get-up expert's output and its last-action input by the ratio so it
commands exactly the same joint targets it did standalone."""
