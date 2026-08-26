"""A GRU whose hidden state is a feature map instead of a vector.

Written for the terrain encoder, where the recurrent state has to *hold an
elevation map* rather than a summary of one. The distinction matters because of
what the terrain task asks of memory.

``nn.GRU`` carries a flat hidden vector with no notion of "cell (i, j)". It can
remember "the ground ahead is rough"; it cannot remember "cell (14, 7) is at
12 cm" and then move that memory to cell (13, 7) because the robot advanced one
cell. That shift is the single most common operation a robot-centric map
performs, and expressing it over a flat vector means learning a dense,
velocity-conditioned permutation of the whole state.

Here the gates are convolutions, so the shift is local and a 3x3 kernel
represents it directly. The cost is that the state grows from a vector to
``(channels, height, width)``.

Reference systems avoid this by registering the map with odometry before the
network sees it (Miki et al. 2022 run a Kalman-filtered elevation map upstream
of a 2x50 GRU; Hoeller et al. 2022 transform the previous estimate by the pose
delta). This cell is what replaces that stage when the registration is left to
the network instead.

The blend follows ``nn.GRU`` (``h' = (1 - z) * n + z * h``). The reset gate does
not: this is the original placement,

    n = tanh(W_x * x + W_h * (r . h) + b)

which gates the remembered map *before* convolving it, where ``nn.GRU`` uses the
cuDNN-friendly reordering ``r . (W_h * h + b_h)`` that gates the result after.
Both are GRUs and they agree when ``r == 1``; the original is what the ConvGRU
literature uses and is the apter one here, since the point is to suppress
specific cells of stale terrain before mixing them with their neighbours.
``scripts/tools/check_terrain_encoder.py`` pins this down by forcing ``r`` open
and checking the rest against ``nn.GRUCell`` exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """One ConvGRU step over a ``(N, C, H, W)`` state.

    Args:
        input_channels: Channels of the per-step input ``x``.
        hidden_channels: Channels of the recurrent state.
        kernel_size: Odd, so ``padding`` keeps the grid size. The state's
            receptive field grows by ``kernel_size // 2`` cells per step, which
            is what lets memory keep up with a moving robot -- see
            ``TerrainEncoder`` for the speed comparison that sets this to 3.
        bias: Passed to the gate convolutions.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # A convolution over [x, h] is the sum of one over x and one over h. The
        # split form is preferred because concatenating creates a tensor autograd
        # keeps alive for every step of a truncated-BPTT window, while the split
        # branches only reference x and h, which the graph already holds. Measured
        # on the 29x21 grid, that is 14% off the peak of a training step (512
        # environments over 16 steps: 9.60 -> 8.21 GiB) for the same parameter
        # count and the same arithmetic. Activation memory is this module's binding
        # constraint, so the 14% is worth the extra layer;
        # ``scripts/tools/check_terrain_encoder.py --bench`` prints the full table.
        #
        # z and r share their output projection: they read the same inputs and
        # differ only in which slice they take.
        self.conv_zr_x = nn.Conv2d(
            input_channels, 2 * hidden_channels, kernel_size, padding=padding, bias=bias
        )
        self.conv_zr_h = nn.Conv2d(
            hidden_channels, 2 * hidden_channels, kernel_size, padding=padding, bias=False
        )
        # n reads r * h rather than h, so it cannot share the projection above.
        self.conv_n_x = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, padding=padding, bias=bias
        )
        self.conv_n_h = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False
        )

    def init_hidden(
        self, batch_size: int, grid_shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """A zero state. Zero rather than learned: at reset the robot has seen
        nothing, and a learned prior would assert terrain it has no evidence for."""
        height, width = grid_shape
        return torch.zeros(
            batch_size, self.hidden_channels, height, width, device=device
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Advance the state by one step.

        Args:
            x: ``(N, input_channels, H, W)``.
            hidden: ``(N, hidden_channels, H, W)``.

        Returns:
            The new hidden state, same shape as ``hidden``.
        """
        z, r = torch.sigmoid(self.conv_zr_x(x) + self.conv_zr_h(hidden)).chunk(2, dim=1)
        n = torch.tanh(self.conv_n_x(x) + self.conv_n_h(r * hidden))
        return (1.0 - z) * n + z * hidden
