"""Check the terrain encoder without launching Isaac Sim.

``assets/models/terrain_encoder.py`` is a plain torch module, so everything worth
checking before it meets the simulator can be checked here in seconds: that the
ConvGRU really is a GRU, that the flat observation round-trips through the grid,
that memory actually carries terrain across steps, and what the thing costs.

    python3 scripts/tools/check_terrain_encoder.py
    python3 scripts/tools/check_terrain_encoder.py --bench
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from unitree_rl_lab.assets.models.modules.conv_gru import ConvGRUCell  # noqa: E402
from unitree_rl_lab.assets.models.terrain_encoder import (  # noqa: E402
    TerrainEncoder,
    gaussian_nll_loss,
)

# Mirrors velocity_env_cfg_go2.py / velocity_env_cfg_lidar.py.
RESOLUTION = 0.05
SIZE = (1.4, 1.0)
EXCLUDE = (0.40, 0.30)
CONTROL_DT = 0.02
MAX_SPEED = 1.5


def keep_index(device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Reproduce ``mdp.observations._height_scan_indices`` ordering ("ij" flatten)."""
    num_x = round(SIZE[0] / RESOLUTION) + 1
    num_y = round(SIZE[1] / RESOLUTION) + 1
    x = torch.linspace(-SIZE[0] / 2, SIZE[0] / 2, num_x, device=device)
    y = torch.linspace(-SIZE[1] / 2, SIZE[1] / 2, num_y, device=device)
    xg, yg = torch.meshgrid(x, y, indexing="ij")
    eps = RESOLUTION * 1.0e-4
    under = (xg.abs() <= EXCLUDE[0] + eps) & (yg.abs() <= EXCLUDE[1] + eps)
    return (~under).flatten().nonzero(as_tuple=False).squeeze(-1), num_x, num_y


def check_conv_gru_matches_nn_gru() -> None:
    """Cross-check the cell against ``nn.GRUCell``.

    With a 1x1 kernel on a 1x1 grid the convolutions are matrix multiplies, so the
    two implementations coincide -- except for where the reset gate is applied
    (see ``conv_gru.py``). Forcing ``r`` open removes that one difference and
    leaves an exact comparison of everything else: the z gate, the candidate, the
    blend, and the channel ordering of the fused zr convolution.
    """
    torch.manual_seed(0)
    in_ch, hid = 5, 7
    cell = ConvGRUCell(in_ch, hid, kernel_size=1)
    ref = torch.nn.GRUCell(in_ch, hid)

    # conv_zr_* emit z then r; nn.GRUCell orders its gates r, z, n. Only the x
    # branches carry a bias, so the reference's split bias goes entirely on
    # bias_ih and bias_hh is zeroed.
    with torch.no_grad():
        r_open = 20.0  # sigmoid(20) == 1 to float32 precision
        cell.conv_zr_x.weight[hid:].zero_()
        cell.conv_zr_h.weight[hid:].zero_()
        cell.conv_zr_x.bias[hid:] = r_open
        z_ih = cell.conv_zr_x.weight[:hid].view(hid, in_ch)
        z_hh = cell.conv_zr_h.weight[:hid].view(hid, hid)
        z_b = cell.conv_zr_x.bias[:hid]
        ref.weight_ih.copy_(
            torch.cat(
                [torch.zeros(hid, in_ch), z_ih, cell.conv_n_x.weight.view(hid, in_ch)], 0
            )
        )
        ref.weight_hh.copy_(
            torch.cat(
                [torch.zeros(hid, hid), z_hh, cell.conv_n_h.weight.view(hid, hid)], 0
            )
        )
        ref.bias_ih.copy_(
            torch.cat([torch.full((hid,), r_open), z_b, cell.conv_n_x.bias], 0)
        )
        ref.bias_hh.zero_()

    x = torch.randn(4, in_ch)
    h = torch.randn(4, hid)
    mine = cell(x[..., None, None], h[..., None, None])[..., 0, 0]
    theirs = ref(x, h)
    err = (mine - theirs).abs().max().item()
    status = "OK" if err < 1e-5 else "MISMATCH"
    print(f"  vs nn.GRUCell, reset gate open   max abs diff {err:.2e}  [{status}]")

    # And the blend itself: z saturated at 1 holds the state unchanged.
    with torch.no_grad():
        held = ConvGRUCell(in_ch, hid, kernel_size=1)
        held.conv_zr_x.weight.zero_()
        held.conv_zr_h.weight.zero_()
        held.conv_zr_x.bias[:hid] = 20.0  # z -> 1
        held.conv_zr_x.bias[hid:] = 0.0
        out = held(x[..., None, None], h[..., None, None])[..., 0, 0]
    err = (out - h).abs().max().item()
    status = "OK" if err < 1e-5 else "MISMATCH"
    print(f"  z -> 1 holds the state           max abs diff {err:.2e}  [{status}]")


def bench(grid: tuple[int, int], hidden: int, extero: int, proprio_dim: int) -> None:
    """Peak GPU and wall time for one truncated-BPTT training step.

    Activation memory over the window, not the parameter count or the state, is
    what limits this module: every step in the window keeps its convolution
    inputs alive. Since the rollouts come from the simulator, whatever this takes
    is taken away from Isaac on the same card -- so pick the operating point here
    before committing to an environment count.
    """
    import gc

    if not torch.cuda.is_available():
        print("  no CUDA device; skipping")
        return
    device = torch.device("cuda")
    height, width = grid

    def one(envs: int, window: int, checkpointing: bool) -> str:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        model = TerrainEncoder(
            grid_shape=grid,
            proprio_dim=proprio_dim,
            extero_channels=extero,
            hidden_channels=hidden,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), 5.0e-4)
        h_in = torch.randn(envs, height, width, device=device)
        m_in = (torch.rand(envs, height, width, device=device) > 0.4).float()
        p_in = torch.randn(envs, proprio_dim, device=device)
        target = torch.randn(envs, height, width, device=device)
        state = model.init_hidden(envs, device)
        try:
            torch.cuda.synchronize()
            start = time.time()
            loss = 0.0
            for _ in range(window):
                if checkpointing:
                    mean, log_std, state = torch.utils.checkpoint.checkpoint(
                        model, h_in, m_in, p_in, state, use_reentrant=False
                    )
                else:
                    mean, log_std, state = model(h_in, m_in, p_in, state)
                loss = loss + gaussian_nll_loss(mean, log_std, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            elapsed = time.time() - start
            return f"{torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB {elapsed * 1e3:6.0f} ms"
        except torch.OutOfMemoryError:
            return "     OOM        "
        finally:
            del model, opt, h_in, m_in, p_in, target, state
            torch.cuda.empty_cache()
            gc.collect()

    windows = (10, 24, 32, 48)
    for label, checkpointing, env_counts in (
        ("plain", False, (128, 256, 512, 1024)),
        ("gradient checkpointing per step", True, (512, 1024, 2048, 4096)),
    ):
        print(f"\n  {label}")
        print("  " + f"{'envs':>6s} " + " ".join(f"{'T=' + str(w):>16s}" for w in windows))
        for envs in env_counts:
            print("  " + f"{envs:6d} " + " ".join(one(envs, w, checkpointing) for w in windows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", help="time and size a TBPTT step on the GPU")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--extero", type=int, default=32)
    parser.add_argument("--proprio-ch", type=int, default=8)
    parser.add_argument("--proprio-dim", type=int, default=45)
    parser.add_argument("--envs", type=int, default=512, help="for the memory figure")
    parser.add_argument("--batch", type=int, default=8, help="for the forward check")
    args = parser.parse_args()

    device = torch.device("cpu")
    keep, num_x, num_y = keep_index(device)
    total = num_x * num_y

    print("=== grid ===")
    print(f"  {num_x} x {num_y} = {total} cells, {RESOLUTION * 100:.0f} cm")
    print(f"  observation keeps {keep.numel()}, body footprint drops {total - keep.numel()}")

    model = TerrainEncoder(
        grid_shape=(num_x, num_y),
        proprio_dim=args.proprio_dim,
        proprio_channels=args.proprio_ch,
        extero_channels=args.extero,
        hidden_channels=args.hidden,
        keep_index=keep,
    )

    print("\n=== equivalence ===")
    check_conv_gru_matches_nn_gru()

    print("\n=== round trip ===")
    flat = torch.randn(args.batch, keep.numel())
    grid = model.scatter_observation(flat)
    back = model.gather_prediction(grid)
    print(f"  flat -> grid -> flat    max abs diff {(back - flat).abs().max():.2e}")
    body = model.body_mask(device)
    print(f"  body_mask cells {int(body.sum())}, grid[body] all zero: "
          f"{bool((grid[:, body] == 0).all())}")

    print("\n=== forward ===")
    hidden = model.init_hidden(args.batch, device)
    height = model.scatter_observation(flat)
    mask = model.scatter_observation(torch.ones_like(flat))
    mean, log_std, hidden = model(height, mask, torch.randn(args.batch, args.proprio_dim), hidden)
    print(f"  mean {tuple(mean.shape)}  log_std {tuple(log_std.shape)}  hidden {tuple(hidden.shape)}")
    print(f"  log_std range [{log_std.min():.2f}, {log_std.max():.2f}] "
          f"(bounds {model.log_std_bounds})")

    target = torch.randn(args.batch, num_x, num_y)
    weight = torch.ones_like(target)
    weight[:, body] = 0.5
    loss = gaussian_nll_loss(mean, log_std, target, weight)
    loss.backward()
    grads = [p for p in model.parameters() if p.grad is None]
    print(f"  loss {loss.item():.4f}, parameters without grad: {len(grads)}")

    print("\n=== memory carries terrain ===")
    # A step is fed a measurement, then the same encoder is run with the
    # measurement blanked. If the second output still resembles the first, the
    # information survived in the state rather than in the input.
    model.zero_grad()
    with torch.no_grad():
        h = model.init_hidden(1, device)
        obs = torch.zeros(1, num_x, num_y)
        obs[:, 20:24, 8:13] = 0.2  # a patch of raised ground ahead
        m = torch.ones_like(obs)
        m[:, body] = 0.0
        seen, _, h = model(obs, m, torch.zeros(1, args.proprio_dim), h)
        blind, _, h = model(
            torch.zeros_like(obs), torch.zeros_like(m), torch.zeros(1, args.proprio_dim), h
        )
        cold, _, _ = model(
            torch.zeros_like(obs),
            torch.zeros_like(m),
            torch.zeros(1, args.proprio_dim),
            model.init_hidden(1, device),
        )
    carried = (blind - cold).abs().mean().item()
    print(f"  blank step with state vs with zero state: mean abs diff {carried:.2e}")
    print("  (untrained, so only the path is checked, not the accuracy)")

    print("\n=== cost ===")
    n_param = sum(p.numel() for p in model.parameters())
    per_module = {
        "extero_conv": model.extero_conv,
        "proprio_fc": model.proprio_fc,
        "rnn": model.rnn,
        "gate_conv": model.gate_conv,
        "belief_conv": model.belief_conv,
        "head_conv": model.head_conv,
        "head_out": model.head_out,
    }
    for name, mod in per_module.items():
        print(f"  {name:14s} {sum(p.numel() for p in mod.parameters()):>8d}")
    print(f"  {'total':14s} {n_param:>8d}")

    state_floats = args.hidden * num_x * num_y
    mib = args.envs * state_floats * 4 / 2**20
    print(f"\n  hidden state {state_floats} floats/env, {args.envs} envs -> {mib:.1f} MiB")

    prop = 1.0  # cells per step, from one 3x3 recurrent kernel
    robot = MAX_SPEED * CONTROL_DT / RESOLUTION
    print(f"  propagation {prop:.1f} cells/step vs robot {robot:.1f} cells/step at "
          f"{MAX_SPEED} m/s -> {'OK' if prop > robot else 'TOO SLOW, add a level'}")

    if args.bench:
        print("\n=== one TBPTT training step ===")
        bench((num_x, num_y), args.hidden, args.extero, args.proprio_dim)


if __name__ == "__main__":
    main()
