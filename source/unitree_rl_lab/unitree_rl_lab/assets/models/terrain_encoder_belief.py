"""The paper's belief encoder/decoder, as a terrain encoder, for comparison.

This is the other arm of the experiment in
``tasks/locomotion/robots/go2/sandbox/TERRAIN_ENCODER.md``: the same reconstruction
task as ``terrain_encoder.py``, with Miki et al. 2022's flat recurrent architecture
in place of the convolutional one. Everything outside the network -- the 388-cell
noisy input, the 45-dim proprioception, the 609-cell target, the Gaussian NLL, the
training schedule -- is held identical, so what the comparison isolates is the shape
of the recurrence and not the input representation.

Built on ``student_actor.py``'s ``BeliefEncoder`` / ``BeliefDecoder`` rather than a
fresh transcription of the paper, so "close to the paper" means something a reader
can check: those classes and ``config.yaml``'s dimensions are this project's existing
implementation of that architecture, and only the decoder's output width changes
here.

    l_e            = g_e(noisy grid)                       388 -> [80, 60] -> 24
    b', h          = GRU([proprio, l_e], h)                69 -> 50, two layers
    belief         = g_b(b') + sigmoid(g_a(b')) * l_e      24, no padding needed
    reconstruction = g_dec(belief) + sigmoid(g_a_dec(belief)) * noisy grid

Three things about that shape are worth knowing before reading any result off it.

**The state is 100 numbers.** Two GRU layers of 50. The convolutional arm carries
9,744 per environment, so this is not a big-versus-small comparison -- the parameter
count runs the other way, about nine to one -- but a question of where the capacity
sits. Whether 100 numbers can hold a 609-cell map is the thing being measured.

**Everything the terrain is known by passes through 96 numbers.** That is the paper's
own total exteroceptive latent -- 24 per foot, four feet -- carried over whole because
the Go2 map is one body-centred grid where the paper had four foot-centred sets.
``config.yaml`` instead keeps 24 for the single encoder, which would put 609 cells
through a 25:1 squeeze against the paper's 1.7:1; that is a bottleneck the paper never
imposed, and losing under it would say nothing about the architecture. Selectable
either way; 96 is what the comparison runs.

**The decoder's skip carries the raw scan.** Not the encoded latent -- the noisy 388
values themselves, gated per cell. Where the sensor is trustworthy the decoder can
pass it straight through and spend its capacity on the corrections and the fill-in,
which is the right division of labour for a reconstruction task and is what the
existing implementation does. It also settles a reading the paper leaves open: "the
same gate is used in the decoder" turns out to mean a separate gate of the same
shape, not the encoder's alpha reused. Figure 7D would confirm this directly; the
parsed markdown in ``doc/papers/`` carries the figures only as image references.

Three deviations from the existing implementation, all forced by the task rather than
chosen.

**The decoder reads the belief state, not the GRU output.** In the paper the belief
state is what the policy consumes and the decoder hangs off the recurrent output; take
the policy away and the belief state has no consumer at all, which leaves ``g_a`` and
``g_b`` -- the attentional gate that is the paper's headline contribution, and the
subject of its S9 ablation -- receiving no gradient and doing nothing. Verified: with
the decoder fed from ``recurrent_output``, twelve parameter tensors come back with
``grad is None``. Feeding it the belief state instead puts the gate back in the only
path there is, which is the version of this architecture worth comparing against.

``BeliefDecoder`` adds its gated skip to its reconstruction elementwise, so
the two have to be the same width; the skip carries the scan, which means the
reconstruction is the scan's width. So the mean is produced on the full 609-cell grid
with the skip scattered into it -- the 221 body cells no beam reaches contribute zero,
which is the paper's zero-padding idea in another place -- and the log sigma comes
from a small separate head off the same GRU output. The paper has no log sigma at all
(it trains on squared error), so some addition was unavoidable; a separate head is the
least invasive one, and it leaves ``BeliefDecoder`` used exactly as written.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unitree_rl_lab.assets.models.student_actor import StudentPolicy


def _mlp_cfg(input_dim: int, output_dim: int, shape=(64, 64)) -> dict:
    """A ``BaseNet`` MLP entry at ``config.yaml``'s student dimensions."""
    return {
        "input": input_dim,
        "output": output_dim,
        "shape": list(shape),
        "activation": "leakyrelu",
        "dropout": 0.0,
    }


class BeliefTerrainEncoder(nn.Module):
    """Miki et al.'s belief encoder/decoder, wired for terrain reconstruction.

    The interface matches :class:`~unitree_rl_lab.assets.models.terrain_encoder.TerrainEncoder`
    so the training loop and the viewer can drive either arm without knowing which
    one they have: ``init_hidden`` / ``forward(height, proprio, hidden)`` returning
    ``(mean, log_std, hidden)`` on the 29 x 21 grid, plus the flat-vector helpers.

    The grid shape is carried for that interface's sake, not because this network
    uses it -- the 388 values enter a fully connected layer, where which cell
    neighbours which stops being represented at all. That is the point of the
    comparison.

    Args:
        grid_shape: ``(H, W)`` of the full elevation grid, for reshaping only.
        proprio_dim: Width of the proprioceptive vector.
        extero_dim: Width of the observation the fan produces (the kept cells).
        extero_latent: ``l_e``, and with it the belief state's width. 96 is the
            paper's *total* exteroceptive latent -- 24 per foot across four feet --
            and is the default here. ``config.yaml`` carries 24, which reads that
            number as per-encoder and keeps it when four foot-encoders collapse into
            one map encoder; the trouble is that it then asks 24 numbers to stand for
            609 cells, a 25:1 squeeze where the paper's own was 1.7:1. Losing under
            that would say nothing about the architecture. 24 remains selectable.
        gru_hidden: Hidden width per GRU layer. 50 is the paper's.
        gru_layers: 2 is the paper's.
        height_scale, height_offset: Same input conditioning as the other arm, so a
            difference in result cannot come from one network seeing metres and the
            other seeing something scaled.
        log_std_bounds: Same clamp as the other arm.
        keep_index: Flat indices of the cells the observation keeps, in observation
            order. Needed both for the helpers and to scatter the decoder's gated
            skip into the full grid.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (29, 21),
        proprio_dim: int = 45,
        extero_dim: int = 388,
        extero_latent: int = 96,
        gru_hidden: int = 50,
        gru_layers: int = 2,
        height_scale: float = 10.0,
        height_offset: float = 0.0,
        log_std_bounds: tuple[float, float] = (-7.0, 2.0),
        keep_index: torch.Tensor | None = None,
    ):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.num_cells = self.grid_shape[0] * self.grid_shape[1]
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.hidden_channels = gru_hidden  # named for interface parity
        self.gru_layers = gru_layers
        self.height_scale = height_scale
        self.height_offset = height_offset
        self.log_std_bounds = log_std_bounds

        self.extero_encoder = nn.Sequential(
            nn.Linear(extero_dim, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 60),
            nn.LeakyReLU(),
            nn.Linear(60, extero_latent),
        )
        self.belief_encoder = StudentPolicy.BeliefEncoder(
            {
                "GRU": {
                    "recurrent_encoder": {
                        "input": proprio_dim + extero_latent,
                        "hidden": gru_hidden,
                        "num_layers": gru_layers,
                        "batch_first": False,
                        "dropout": 0.0,
                    }
                },
                "MLP": {
                    "attention_encoder": _mlp_cfg(gru_hidden, extero_latent),
                    "state_encoder": _mlp_cfg(gru_hidden, extero_latent),
                },
            }
        )
        # Both entries are the grid's width, not the observation's: BeliefDecoder adds
        # the gated skip to the reconstruction elementwise, and the skip is the scan
        # placed back on the grid it was cut from.
        self.belief_decoder = StudentPolicy.BeliefDecoder(
            {
                "MLP": {
                    "attention_encoder": _mlp_cfg(extero_latent, self.num_cells),
                    "extero_decoder": _mlp_cfg(extero_latent, self.num_cells),
                }
            }
        )
        # The paper reconstructs a point estimate; the likelihood both arms are trained
        # against needs a spread as well.
        self.log_std_head = nn.Sequential(
            nn.Linear(extero_latent, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.num_cells),
        )

        if keep_index is not None:
            self.register_buffer("keep_index", keep_index.long(), persistent=False)
        else:
            self.keep_index = None

    # -- state ---------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """``(layers, N, hidden)`` -- ``nn.GRU``'s own layout, not the other arm's."""
        return torch.zeros(self.gru_layers, batch_size, self.hidden_channels, device=device)

    def mask_hidden(self, hidden: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Zero the state of the environments flagged in ``done``, out of place.

        ``nn.GRU``'s layout puts the batch in the middle, so the mask broadcasts over
        dimension 1 rather than dimension 0.
        """
        return torch.where(done.view(1, -1, 1), torch.zeros_like(hidden), hidden)


    # -- flat observation <-> grid -------------------------------------------

    def scatter_observation(self, flat: torch.Tensor, fill: float | None = None) -> torch.Tensor:
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to BeliefTerrainEncoder")
        if fill is None:
            fill = self.height_offset
        grid = flat.new_full((flat.shape[0], self.num_cells), fill)
        grid[:, self.keep_index] = flat
        return grid.view(-1, *self.grid_shape)

    def gather_prediction(self, grid: torch.Tensor) -> torch.Tensor:
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to BeliefTerrainEncoder")
        return grid.flatten(start_dim=1).index_select(1, self.keep_index)

    def body_mask(self, device: torch.device) -> torch.Tensor:
        if self.keep_index is None:
            raise RuntimeError("keep_index was not supplied to BeliefTerrainEncoder")
        mask = torch.ones(self.num_cells, dtype=torch.bool, device=device)
        mask[self.keep_index] = False
        return mask.view(*self.grid_shape)

    # -- forward -------------------------------------------------------------

    def forward(
        self, height: torch.Tensor, proprio: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step, same signature and units as the convolutional arm.

        Args:
            height: ``(N, H, W)`` noisy elevation in metres. Flattened and cropped to
                the observation's cells on the way in -- this network never sees the
                grid as a grid.
            proprio: ``(N, proprio_dim)``.
            hidden: ``(layers, N, hidden)``.
        """
        # Everything inside the network is in scaled units; the divide at the end puts
        # the answer back in metres. That also makes the decoder's skip an identity
        # path: pass the gate wide open and the reconstruction *is* the measurement.
        scaled = (height - self.height_offset) * self.height_scale
        # Body cells arrive at height_offset, so scaling leaves them at exactly zero --
        # the skip contributes nothing where nothing was measured.
        skip = scaled.flatten(start_dim=1)
        obs = self.gather_prediction(scaled)  # (N, extero_dim), the cells that exist

        l_e = self.extero_encoder(obs)
        # BeliefEncoder/Decoder come from a policy that runs its GRU with a leading
        # sequence axis; one step is a sequence of length one.
        out = self.belief_encoder(proprio.unsqueeze(0), l_e.unsqueeze(0), hidden)
        belief = out["belief_state"]

        mean_scaled = self.belief_decoder(skip.unsqueeze(0), belief).squeeze(0)
        mean = mean_scaled.view(-1, *self.grid_shape) / self.height_scale + self.height_offset
        log_std = (
            self.log_std_head(belief.squeeze(0))
            .view(-1, *self.grid_shape)
            .clamp(*self.log_std_bounds)
        )
        return mean, log_std, out["recurrent_hidden"]
