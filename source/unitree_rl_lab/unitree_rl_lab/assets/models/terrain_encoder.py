"""Recurrent terrain encoder: noisy LiDAR elevation grid -> clean terrain + uncertainty.

The walking policy this runs alongside is blind and frozen. It drives the robot
across the terrain; this module watches the LiDAR grid the robot would actually
measure and learns to reconstruct the terrain that is really there. Nothing here
feeds back into the policy, so the whole module is supervised -- there is no
behaviour-cloning term, only reconstruction.

Why this shape rather than the reference architecture
-----------------------------------------------------
Miki et al. 2022 use an MLP encoder and a flat 2x50 GRU. That works because
their exteroception is 208 foot-centred polar samples drawn from a *registered*
elevation map: a Kalman-filtered, drift-compensated accumulator already answers
"which cell is which" before the network sees anything, so the GRU only has to
remember corrections, and 100 units is ample for that.

Here the registration is deliberately left to the network, which moves the
accumulation itself into the recurrent state. Two consequences follow:

* The state has to be spatial. Shifting remembered terrain by however far the
  robot moved is a local operation for a convolution and a dense permutation for
  a fully connected layer -- see ``modules/conv_gru.py``.
* The state has to be big enough to hold a map. 100 units cannot store 388
  cells; the ``(32, 29, 21)`` state here is 19488.

No pooling. A 3x3 recurrent kernel propagates information one cell per step,
while the robot covers 0.6 cells per step at the 1.5 m/s command ceiling
(3 cm at the 20 ms control period, 5 cm cells). Propagation outruns the robot,
so time supplies the receptive field that down-sampling would otherwise buy, and
information crosses the 29-cell grid in 29 steps (0.6 s).

Inputs and the exteroceptive gate
---------------------------------
The gate is Miki et al.'s, lifted to the grid::

    alpha = sigmoid(g_a(h))
    b     = g_b(h) + l_e * alpha

``l_e`` is a skip path carrying the *current* measurement past the recurrent
bottleneck, so a freshly measured cell keeps its precision instead of being
encoded and decoded. ``alpha`` is what stops that path from also carrying
garbage: unobserved cells (36% on flat ground, 49% with a wall in view) and
outliers (up to 3% of rays, displaced by up to 0.6 m) would otherwise be added
straight into the output. Per-cell rather than per-vector because reliability
varies *within* one frame here -- the front band is measured, the body footprint
never is, the shadow behind a wall is not.

``l_e`` is computed from the exteroceptive channels alone, not from the
proprioceptive ones. Gating a mixture would gate proprioception too, which is
the opposite of what the gate is for; proprioception reaches the state only
through the recurrent path, as in the paper.

Output
------
Two channels, mean and log sigma, trained with the Gaussian negative
log-likelihood of Lee et al. 2020 (Eq. 8). Miki et al. use a plain squared error
and note in their discussion that the missing uncertainty is what makes the
controller step off occluded ledges; the second channel costs 33 parameters.

Training window
---------------
Activation memory over the truncation window, not the parameter count or the
state, is what this module costs; ``check_terrain_encoder.py --bench`` prints the
table. Truncating at ``num_steps_per_env`` (24) keeps one window per rollout, so
the detach lands on the boundary the state would be carried across anyway.

24 steps is 0.48 s, and a cell takes about 42 steps to travel from the front band
out the back of the body footprint. Gradients therefore do not span a full
traverse -- the state does, since only the gradient is cut, but credit for
holding terrain that long is weak. If the body-footprint reconstruction error
plateaus while the observed band keeps improving, that asymmetry is the first
thing to suspect, and the fix is to accumulate two rollouts per backward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unitree_rl_lab.assets.models.modules.conv_gru import ConvGRUCell


class TerrainEncoder(nn.Module):
    """Belief encoder/decoder over a robot-centred elevation grid.

    The grid is the full rectangle, body footprint included. The observation the
    policy receives drops the footprint cells -- no beam can reach them -- but the
    reconstruction covers them anyway: they are the only cells that are *never*
    directly measurable, so a correct value there can only have come from memory.
    That makes their error the cleanest available measure of whether the
    recurrence is doing anything, and it is free, since the top-down scan that
    supplies the targets sees straight through the robot.

    Args:
        grid_shape: ``(H, W)`` of the elevation grid. ``(29, 21)`` is the Go2's
            1.4 x 1.0 m grid at 5 cm, with x as rows to match the ``"ij"``
            flattening used by ``_height_scan_indices``.
        proprio_dim: Width of the proprioceptive vector. 45 for the blind Go2
            policy: base angular velocity, projected gravity, velocity command,
            joint positions, joint velocities, previous action.
        proprio_channels: Width the proprioceptive vector is compressed to before
            being broadcast across the grid.
        extero_channels: Width of ``l_e``, the encoded measurement.
        hidden_channels: Width of the recurrent state.
        kernel_size: Convolution kernel, odd.
        log_std_bounds: Clamp on the predicted log sigma. The lower bound stops
            a confident cell from driving the likelihood to infinity early in
            training; the upper bound keeps a hopeless cell from being ignored
            outright.
        keep_index: Optional flat indices of the cells the policy observation
            keeps, in observation order. Supplying it enables
            :meth:`scatter_observation` and :meth:`gather_prediction`, which
            convert between the flat observation vector and the grid. Obtain it
            from ``mdp.observations._height_scan_indices`` with the same crop the
            LiDAR term uses -- passing it in rather than recomputing it here
            keeps this package free of a dependency on the task configs.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (29, 21),
        proprio_dim: int = 45,
        proprio_channels: int = 8,
        extero_channels: int = 32,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        log_std_bounds: tuple[float, float] = (-7.0, 2.0),
        keep_index: torch.Tensor | None = None,
    ):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.proprio_dim = proprio_dim
        self.hidden_channels = hidden_channels
        self.log_std_bounds = log_std_bounds
        padding = kernel_size // 2

        # Exteroception: height and the observed-this-step mask. Kept separate from
        # proprioception so the gate below acts on the measurement alone.
        self.extero_conv = nn.Conv2d(2, extero_channels, kernel_size, padding=padding)

        # Proprioception has no spatial extent, so compressing it with a linear layer
        # and broadcasting is identical to the 1x1 convolution it stands in for, and
        # avoids materialising 45 full-grid channels to do it.
        self.proprio_fc = nn.Linear(proprio_dim, proprio_channels)

        self.rnn = ConvGRUCell(
            input_channels=extero_channels + proprio_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # g_a and g_b of the gate. g_b is left linear; the activation comes after
        # the skip has been added, so the measurement path is not squashed twice.
        self.gate_conv = nn.Conv2d(hidden_channels, extero_channels, 1)
        self.belief_conv = nn.Conv2d(
            hidden_channels, extero_channels, kernel_size, padding=padding
        )

        self.head_conv = nn.Conv2d(
            extero_channels, extero_channels, kernel_size, padding=padding
        )
        self.head_out = nn.Conv2d(extero_channels, 2, 1)

        self.activation = nn.LeakyReLU()

        if keep_index is not None:
            self.register_buffer("keep_index", keep_index.long(), persistent=False)
        else:
            self.keep_index = None

    # -- state ---------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero state for ``batch_size`` environments."""
        return self.rnn.init_hidden(batch_size, self.grid_shape, device)

    def reset_hidden(self, hidden: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Zero the state of the given environments, in place.

        Call this on episode boundaries. Terrain memory carried across a reset is
        memory of a different patch of ground.
        """
        hidden[env_ids] = 0.0
        return hidden

    # -- flat observation <-> grid ------------------------------------------

    def scatter_observation(
        self, flat: torch.Tensor, fill: float = 0.0
    ) -> torch.Tensor:
        """Place a flat observation vector into the full grid.

        Cells outside ``keep_index`` -- the body footprint -- take ``fill``. For the
        height channel that is a neutral value the mask marks as unobserved; the
        network is not meant to read anything into it.
        """
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to TerrainEncoder")
        height, width = self.grid_shape
        grid = flat.new_full((flat.shape[0], height * width), fill)
        grid[:, self.keep_index] = flat
        return grid.view(-1, height, width)

    def gather_prediction(self, grid: torch.Tensor) -> torch.Tensor:
        """Take the kept cells out of a full grid, in observation order."""
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to TerrainEncoder")
        return grid.flatten(start_dim=1).index_select(1, self.keep_index)

    def body_mask(self, device: torch.device) -> torch.Tensor:
        """``(H, W)`` boolean, true on the cells the observation drops.

        These are the never-measurable cells; report their reconstruction error
        separately from the observed band's.
        """
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to TerrainEncoder")
        height, width = self.grid_shape
        mask = torch.ones(height * width, dtype=torch.bool, device=device)
        mask[self.keep_index] = False
        return mask.view(height, width)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        height: torch.Tensor,
        mask: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step.

        Args:
            height: ``(N, H, W)`` noisy elevation, any value where ``mask`` is 0.
            mask: ``(N, H, W)`` 1 where a beam returned this step, else 0.
            proprio: ``(N, proprio_dim)``.
            hidden: ``(N, hidden_channels, H, W)`` from the previous step.

        Returns:
            ``(mean, log_std, hidden)`` -- the first two ``(N, H, W)``, the third
            the state to carry forward. Detach it at truncation boundaries.
        """
        # A masked-out cell must not leak its arbitrary height into the convolution,
        # and the mask channel alone cannot undo that once it has been summed in.
        extero = torch.stack([height * mask, mask], dim=1)
        l_e = self.activation(self.extero_conv(extero))

        proprio_feat = self.activation(self.proprio_fc(proprio))
        proprio_grid = proprio_feat[..., None, None].expand(
            -1, -1, *self.grid_shape
        )

        hidden = self.rnn(torch.cat([l_e, proprio_grid], dim=1), hidden)

        alpha = torch.sigmoid(self.gate_conv(hidden))
        belief = self.belief_conv(hidden) + l_e * alpha

        out = self.head_out(self.activation(self.head_conv(belief)))
        mean = out[:, 0]
        log_std = out[:, 1].clamp(*self.log_std_bounds)
        return mean, log_std, hidden

    def gate(self, hidden: torch.Tensor) -> torch.Tensor:
        """The gate's opening, for introspection. ``(N, extero_channels, H, W)``.

        Averaging over channels gives a per-cell picture of where the encoder is
        trusting the current measurement and where it has fallen back on memory.
        """
        return torch.sigmoid(self.gate_conv(hidden))


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lee et al. 2020 Eq. 8: ``(m - m_gt)^2 / (2 sigma^2) + log sigma``.

    Weighted mean over every element. ``weight`` is the place to trade the
    never-observable body footprint against the measured band -- pass a grid of
    per-cell weights, not a hard mask. Zeroing a region entirely removes the only
    gradient that would teach the encoder to fill it, which for the occluded
    cells is the whole point of the exercise.
    """
    inv_var = torch.exp(-2.0 * log_std)
    loss = 0.5 * (mean - target).pow(2) * inv_var + log_std
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
