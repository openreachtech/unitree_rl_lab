"""Left-right mirror augmentation for the unified-observation Go2.

The Go2 is bilaterally symmetric and its bipedal task is symmetric with it: a stance held with the
weight on the left leg is the mirror of one held on the right, and a policy that has learned only
one of them has learned half the problem. Both stances trained here show that it does exactly
that. The unified policy's hind half settles with 94% of its steps on the left hind foot against
37% on the right, and props itself up with the *left* front foot 10% of the time against 1% on the
right. The hind stance's own first run put its left shin on the floor 8.7% of the time and its
right shin 0.2%. Nothing in the task prefers a side; the policy invents the preference.

Mirroring the training batch removes the asymmetry from the hypothesis space rather than
penalising it after the fact. This is the tool the acrobatics side identified for the same problem
-- a left sideflip at 0.272 against a right at 0.851 -- and never applied.

Everything here is derived from the environment at construction rather than written out. The
observation is 124 columns of eleven blocks and the critic 335 of seventeen, each needing its own
sign and permutation, and a single wrong entry is silent: training continues, the augmented half of
every batch is simply wrong. :func:`_build` therefore reads the joint order and the height-scanner
ray offsets from the running scene, and :func:`_check` verifies the result is an involution before
a single batch is augmented.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ...multitask.obs_spec import CRITIC_UNIFIED, HISTORY_LENGTH, NUM_JOINTS, POLICY_UNIFIED, Block, block_offsets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# How each block transforms under a left-right mirror, as a per-column sign. Blocks whose columns
# are per-joint are handled separately, since they permute as well as flip.
#
#   base_ang_vel        roll and yaw rates reverse; pitch rate does not
#   projected_gravity   only its y component reverses
#   velocity_commands   (v_x, v_y, w_z): lateral speed and yaw rate reverse
#   jump_command        (enabled, height, pitch turns, roll turns): only the roll target reverses
#   handstand_command   (enabled, stance): stance names an *end* of the robot, not a side
#   root_roll_angle     reverses; root_pitch_angle and the accumulated pitch do not
#   com_cop             a vector in the base frame: y reverses
_BLOCK_SIGNS: dict[str, tuple[float, ...]] = {
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
    "jump_command": (1.0, 1.0, 1.0, -1.0),
    "jump_time": (1.0,),
    "handstand_command": (1.0, 1.0),
    "root_height": (1.0,),
    "root_roll_angle": (-1.0,),
    "root_pitch_angle": (1.0,),
    "maximum_jump_height": (1.0,),
    "accumulated_root_pitch": (1.0,),
    "accumulated_root_roll": (-1.0,),
    "com_cop": (1.0, -1.0, 1.0),
}

_JOINT_BLOCKS = {"joint_pos_rel", "joint_vel_rel", "joint_effort", "last_action"}


def _joint_map(env) -> tuple[list[int], list[float]]:
    """Per-joint permutation and sign, read from the articulation's own joint order.

    Read rather than written out because the order is the articulation's, not the config's, and a
    transposition here would swap two legs' commands with nothing to catch it. Hip joints reverse
    because abduction is measured about the forward axis; thigh and calf rotate about the lateral
    axis and keep their sign.
    """
    names = env.scene["robot"].joint_names
    index = {name: i for i, name in enumerate(names)}
    order, signs = [], []
    for name in names:
        mirrored = name.replace("FR_", "F@_").replace("FL_", "FR_").replace("F@_", "FL_")
        mirrored = mirrored.replace("RR_", "R@_").replace("RL_", "RR_").replace("R@_", "RL_")
        if mirrored not in index:
            raise ValueError(f"No mirror partner for joint {name!r} among {names}")
        order.append(index[mirrored])
        signs.append(-1.0 if "_hip_" in name else 1.0)
    return order, signs


def _height_scan_map(env, width: int) -> list[int]:
    """Permutation that reflects the height-scanner grid about the robot's fore-aft axis.

    Derived from the sensor's own ray offsets: for each ray at ``(x, y)`` find the ray at
    ``(x, -y)``. Deriving it beats assuming a row-major convention -- on flat ground a wrong
    permutation is invisible, and it would stay invisible until the first run on terrain.
    """
    sensor = env.scene.sensors["height_scanner"]
    offsets = sensor.ray_starts[0, :, :2].clone()
    if offsets.shape[0] != width:
        raise ValueError(f"Height scanner has {offsets.shape[0]} rays, layout expects {width}")
    mirrored = offsets.clone()
    mirrored[:, 1] *= -1.0
    distance = torch.cdist(mirrored, offsets)
    partner = distance.argmin(dim=1)

    # No distance threshold. Three attempts at one all compared a float32 rounding error against
    # some other float32 rounding error; the condition below is the one that actually matters and
    # is exact.
    # Every ray must be claimed exactly once. A mapping that sends two rays to the same partner is
    # not a reflection -- it would quietly duplicate half the scan and drop the other half -- and
    # this catches that regardless of how the float error happens to fall.
    if sorted(partner.tolist()) != list(range(width)):
        raise ValueError("Height-scanner mirror is not a permutation: some rays share a partner")
    return partner.tolist()


def _build_layout(env, layout: tuple[Block, ...], total: int) -> tuple[torch.Tensor, torch.Tensor]:
    order, joint_signs = _joint_map(env)
    offsets = block_offsets(layout)
    index = list(range(total))
    sign = [1.0] * total

    for block in layout:
        start = offsets[block.name]
        if block.name in _JOINT_BLOCKS:
            for frame in range(block.history):
                base = start + frame * block.dim
                for joint in range(NUM_JOINTS):
                    index[base + joint] = base + order[joint]
                    sign[base + joint] = joint_signs[joint]
        elif block.name == "height_scan":
            for ray, partner in enumerate(_height_scan_map(env, block.dim)):
                index[start + ray] = start + partner
        elif block.name in _BLOCK_SIGNS:
            signs = _BLOCK_SIGNS[block.name]
            if len(signs) != block.dim:
                raise ValueError(f"Block {block.name!r} has dim {block.dim}, {len(signs)} signs given")
            for column in range(block.dim):
                sign[start + column] = signs[column]
        else:
            raise ValueError(f"No mirror rule for observation block {block.name!r}")

    device = env.device
    return torch.tensor(index, device=device), torch.tensor(sign, device=device)


def _check(index: torch.Tensor, sign: torch.Tensor, label: str) -> None:
    """A mirror is its own inverse. Anything that is not is a permutation bug."""
    twice_index = index[index]
    twice_sign = sign * sign[index]
    if not torch.equal(twice_index, torch.arange(len(index), device=index.device)):
        raise ValueError(f"{label} mirror is not an involution: applying it twice permutes columns")
    if not torch.allclose(twice_sign, torch.ones_like(twice_sign)):
        raise ValueError(f"{label} mirror is not an involution: applying it twice flips signs")


class _Mirror:
    """Lazily built index/sign tensors for one environment."""

    def __init__(self, env) -> None:
        inner = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = _build_layout(inner, POLICY_UNIFIED, sum(b.width for b in POLICY_UNIFIED))
        self.critic = _build_layout(inner, CRITIC_UNIFIED, sum(b.width for b in CRITIC_UNIFIED))
        order, signs = _joint_map(inner)
        self.action = (
            torch.tensor(order, device=inner.device),
            torch.tensor(signs, device=inner.device),
        )
        _check(*self.policy, "policy")
        _check(*self.critic, "critic")
        _check(*self.action, "action")
        print(f"[INFO] left-right mirror built and verified: policy {len(self.policy[0])} columns,"
              f" critic {len(self.critic[0])}, actions {len(self.action[0])}")


_MIRRORS: dict[int, _Mirror] = {}


def _apply(tensor: torch.Tensor, index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return tensor[..., index] * sign


def mirror_left_right(env, obs=None, actions=None, obs_type: str = "policy"):
    """Return each batch concatenated with its left-right mirror image.

    The signature is rsl_rl's: it calls this with ``obs`` and ``actions`` during the update, and the
    returned batch is twice as long. Every other per-sample quantity in the update is repeated to
    match, so the two halves stay aligned.
    """
    mirror = _MIRRORS.get(id(env))
    if mirror is None:
        mirror = _MIRRORS[id(env)] = _Mirror(env)

    mirrored_obs = None
    if obs is not None:
        mirrored_obs = obs.clone()
        for group, (index, sign) in (("policy", mirror.policy), ("critic", mirror.critic)):
            if group in obs.keys():
                mirrored_obs[group] = _apply(obs[group], index, sign)
        mirrored_obs = torch.cat([obs, mirrored_obs], dim=0)

    mirrored_actions = None
    if actions is not None:
        index, sign = mirror.action
        mirrored_actions = torch.cat([actions, _apply(actions, index, sign)], dim=0)

    return mirrored_obs, mirrored_actions
