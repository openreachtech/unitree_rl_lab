"""The frozen terrain encoder, run as an observation.

Feeds the policy a *denoised* height grid: a noisy elevation map passed through the
belief encoder of ``sandbox/TERRAIN_ENCODER.md``, whose weights are frozen. This term
is where that encoder runs.

Nothing in the repo currently instantiates it. The ``Go2-HM-*`` arms that did, the
LiDAR fan that fed them and the encoder's own training stage have all been removed;
this is kept because the encoder earned a small but real gain in terrain level, so it
is worth being able to put back in front of a map without rebuilding it. What a caller
has to supply is a ``noisy_map`` term whose width matches
:data:`BELIEF_CROP_HALF_EXTENT_X` / ``_Y`` below and a ``checkpoint`` --
:data:`BELIEF_ENCODER_CHECKPOINT` is the one that was adopted.

Why here and not inside the policy network
------------------------------------------
Putting a frozen front-end on the actor looks tidier, and it would have solved the
awkwardness below. It breaks on PPO's update pass. The encoder is recurrent, so its
output depends on a hidden state that advanced during the rollout; when PPO replays a
stored batch it would have to replay that state exactly, and rsl_rl's storage keeps
observations and the actor's own recurrent state, not a third one belonging to a
module inside the network. Replay it wrongly and the gradient is computed against
inputs the policy never actually acted on.

Making the denoised grid an *observation* removes the problem entirely: the storage
already saves observations, so the update sees exactly the values the rollout did.

The awkward part
----------------
The encoder needs the 45-dim proprioceptive vector, and in an environment shaped like
the one it was trained in that vector is the first half of the policy group -- which is
the group this term's output goes into. A term cannot read the group it is part of, so the
proprioception is rebuilt here from the same six ``mdp`` functions, in the same order,
with the same scales, noise and clips, and in the same order the observation manager
applies them (noise, then clip, then scale).

That duplication is a real hazard: reorder or rescale a term in the policy group and
this silently feeds the encoder something it was never trained on, with no error --
just a worse map. :data:`GO2_BLIND_PROPRIO_SPEC` is the single description both are
meant to follow, and the width is asserted at construction.

The noise draws differ from the policy group's, since each is sampled independently.
That is fine: the encoder was trained on this distribution, and a different draw from
the same distribution is exactly what it saw at every training step.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import isaaclab.envs.mdp as isaac_mdp

from unitree_rl_lab.assets.models.terrain_encoder_belief import BeliefTerrainEncoder

from .observations import _height_scan_indices

# One entry per block of the blind policy's 45-dim
# proprioception, in order. Mirrors ``ObservationsCfg.PolicyCfg`` in
# ``robots/go2/velocity_env_cfg.py``; the two have to agree or the frozen encoder is
# fed a vector it was not trained on.
GO2_BLIND_PROPRIO_SPEC = (
    (isaac_mdp.base_ang_vel, {}, 0.2, 0.2, (-100.0, 100.0)),
    (isaac_mdp.projected_gravity, {}, 0.05, 1.0, (-100.0, 100.0)),
    (isaac_mdp.generated_commands, {"command_name": "base_velocity"}, 0.0, 1.0, (-100.0, 100.0)),
    (isaac_mdp.joint_pos_rel, {}, 0.01, 1.0, (-100.0, 100.0)),
    (isaac_mdp.joint_vel_rel, {}, 1.5, 0.05, (-100.0, 100.0)),
    (isaac_mdp.last_action, {}, 0.0, 1.0, (-100.0, 100.0)),
)
"""``(func, kwargs, noise_half_width, scale, clip)``. Noise is uniform on
``+-noise_half_width``; 0 means the policy group applies none to that term."""

GO2_BLIND_PROPRIO_DIM = 45

# The crop the encoder was trained against, widened from the top-down tasks' 0.30/0.20
# because the fan that fed it had a blind cone that size. It lived in the fan's config
# until that was removed; it is frozen here rather than re-derived, because the
# checkpoint's input width is 388 cells and only this pair reproduces it.
BELIEF_CROP_HALF_EXTENT_X = 0.40
BELIEF_CROP_HALF_EXTENT_Y = 0.30

BELIEF_ENCODER_CHECKPOINT = "logs/terrain_encoder/2026-08-29_22-24-19/encoder_4000.pt"
"""The adopted terrain encoder: belief architecture, Phase 2 -> 3 -> 4 then extended,
4000 iterations. Chosen over the convolutional one because accuracy was comparable while
training was 3.8x faster in a 17.8th of the memory -- see ``sandbox/TERRAIN_ENCODER.md``."""


def _blind_proprio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rebuild the blind policy's 45-dim observation, post-processing included."""
    parts = []
    for func, kwargs, noise_half, scale, clip in GO2_BLIND_PROPRIO_SPEC:
        value = func(env, **kwargs)
        if noise_half > 0.0:
            # Same order the observation manager uses: noise, then clip, then scale.
            value = value + (torch.rand_like(value) * 2.0 - 1.0) * noise_half
        value = value.clip(min=clip[0], max=clip[1])
        if scale != 1.0:
            value = value * scale
        parts.append(value)
    return torch.cat(parts, dim=-1)


class BeliefHeightMap(ManagerTermBase):
    """Noisy LiDAR grid in, denoised grid out, through a frozen recurrent encoder.

    Holds one hidden state per environment and zeroes it on reset -- terrain
    remembered across a reset belongs to a patch of ground the robot is no longer on,
    the same rule the encoder was trained under.

    The encoder is loaded from a training checkpoint, which carries its own widths, so
    a run trained at a different size still loads. It is put in ``eval()`` and its
    parameters have ``requires_grad_(False)``; nothing here is part of the policy's
    graph.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        checkpoint = cfg.params["checkpoint"]
        state = torch.load(checkpoint, map_location=env.device, weights_only=False)
        self._encoder = build_terrain_encoder(
            torch.device(env.device), belief_latent=state.get("belief_latent", 96)
        )
        self._encoder.load_state_dict(state["encoder"])
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

        self._hidden = self._encoder.init_hidden(env.num_envs, torch.device(env.device))
        print(
            f"[INFO] BeliefHeightMap: {checkpoint} (iteration={state.get('iteration', '?')})",
            flush=True,
        )

        # The LiDAR grid is itself a stateful term (it holds the previous frame), so it
        # has to be instantiated, not called. Building our own instance rather than
        # reaching for the observation manager's keeps this term self-contained -- and
        # gives the encoder a fan whose noise draws are its own, exactly as in training.
        noisy_cfg = cfg.params["noisy_map"]
        self._noisy_map = (
            noisy_cfg.func(noisy_cfg, env)
            if isinstance(noisy_cfg.func, type) and issubclass(noisy_cfg.func, ManagerTermBase)
            else None
        )

        width = _blind_proprio(env).shape[-1]
        if width != GO2_BLIND_PROPRIO_DIM:
            raise ValueError(
                f"the frozen encoder expects {GO2_BLIND_PROPRIO_DIM} proprioceptive inputs, "
                f"GO2_BLIND_PROPRIO_SPEC produced {width}. It has drifted from the policy "
                "observation group it is meant to mirror -- see this module's docstring."
            )

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if self._noisy_map is not None:
            self._noisy_map.reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self._hidden.zero_()
            return
        # mask_hidden knows where the batch axis is; the two encoders lay their state
        # out differently and this term should not have to care which one it holds.
        ids = torch.as_tensor(env_ids, device=self._hidden.device, dtype=torch.long)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self._hidden.device)
        done[ids] = True
        self._hidden = self._encoder.mask_hidden(self._hidden, done)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        noisy_map: ObservationTermCfg,
        checkpoint: str,
    ) -> torch.Tensor:
        """Returns the reconstruction on the observation's cells, in metres.

        Args:
            noisy_map: The LiDAR term whose output is the encoder's exteroceptive
                input. Passed as a config rather than a tensor because observation
                terms cannot read each other's results; this term evaluates it.
            checkpoint: Read at construction, ignored here.
        """
        raw = (
            self._noisy_map(env, **noisy_map.params)
            if self._noisy_map is not None
            else noisy_map.func(env, **noisy_map.params)
        )
        if noisy_map.clip:
            raw = raw.clip(min=noisy_map.clip[0], max=noisy_map.clip[1])

        # no_grad rather than inference_mode: the hidden state outlives this call, and a
        # tensor created under inference_mode cannot be used outside one.
        with torch.no_grad():
            grid = self._encoder.scatter_observation(raw)
            mean, _, self._hidden = self._encoder(grid, _blind_proprio(env), self._hidden)
            return self._encoder.gather_prediction(mean)


def build_terrain_encoder(device: torch.device, belief_latent: int = 96) -> BeliefTerrainEncoder:
    """Miki et al.'s belief encoder, at the width its checkpoints were trained on.

    Trained by the (since removed) terrain-encoder stage and kept only to be loaded;
    ``sandbox/TERRAIN_ENCODER.md`` records how it was built and why the convolutional
    alternative was dropped.

    The crop is :data:`BELIEF_CROP_HALF_EXTENT_X` / ``_Y``, not the top-down tasks'
    0.30 / 0.20 -- that keeps 388 cells where theirs keeps 492, and 388 is what the
    checkpoint expects. Taking the indices from the same helper the observation term
    uses is what holds the flat observation vector and the 29 x 21 image in register.
    """
    # Imported inside the function: these live in the go2 task configs, which import the
    # mdp package this module belongs to. At module scope that is a cycle.
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
        GO2_HEIGHT_SCAN_CENTER_X,
        GO2_HEIGHT_SCAN_CENTER_Y,
        HEIGHT_SCAN_RESOLUTION,
        HEIGHT_SCAN_SIZE,
    )

    keep_index, _ = _height_scan_indices(
        resolution=HEIGHT_SCAN_RESOLUTION,
        size_x=HEIGHT_SCAN_SIZE[0],
        size_y=HEIGHT_SCAN_SIZE[1],
        scanner_offset_x=GO2_HEIGHT_SCAN_CENTER_X,
        scanner_offset_y=GO2_HEIGHT_SCAN_CENTER_Y,
        exclude_half_extent_x=BELIEF_CROP_HALF_EXTENT_X,
        exclude_half_extent_y=BELIEF_CROP_HALF_EXTENT_Y,
        device=device,
    )
    num_x = round(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1
    num_y = round(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1
    # Level ground on zero, so the scaled input is signed terrain deviation.
    common = dict(grid_shape=(num_x, num_y), proprio_dim=45, height_offset=0.0, keep_index=keep_index)
    return BeliefTerrainEncoder(
        extero_dim=int(keep_index.numel()), extero_latent=belief_latent, **common
    ).to(device)
