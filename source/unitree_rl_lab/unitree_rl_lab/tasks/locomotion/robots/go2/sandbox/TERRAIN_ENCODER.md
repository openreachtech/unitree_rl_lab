# Terrain encoder: results and the architecture comparison

Running record for the terrain encoder that reconstructs the true terrain from the
noisy LiDAR grid behind a frozen blind walking policy. Kept here so the numbers
survive the next context switch.

Code: `assets/models/terrain_encoder.py`, `assets/models/modules/conv_gru.py`,
`scripts/rsl_rl/train_terrain_encoder.py`, `scripts/rsl_rl/play_terrain_encoder.py`,
`tasks/locomotion/robots/go2/velocity_env_cfg_terrain_encoder.py`.

## Setup

The walking policy is frozen and blind. It drives the robot; the encoder watches the
grid the fan produces and learns to reconstruct the terrain, supervised by a clean
top-down scan. No gradient reaches the policy and the policy never reads the encoder,
so the state distribution is independent of the encoder -- which is why no DAgger is
needed and why the same seed gives two encoders byte-identical data.

One encoder phase per walking phase, each driven by the policy trained on that
phase's own terrain, chained by `--resume`. Phase 1 is skipped: flat ground has
nothing to reconstruct.

| Encoder phase | Terrain | Frozen policy |
|---|---|---|
| Phase 2 | rough / boxes | `go2_blind_gru_phase2/2026-08-23_23-35-29/model_6497.pt` |
| Phase 3 | stairs | `go2_blind_gru_phase3/2026-08-25_20-42-02/model_9995.pt` |
| Phase 4 | thin walls | `go2_blind_gru_phase4/2026-08-26_01-06-04/model_7300.pt` |

512 environments, TBPTT window 24 (= `num_steps_per_env`), 2 epochs per rollout,
Adam 5e-4, LiDAR noise at full strength from the first iteration. 1000 iterations per
phase, roughly 24 minutes each.

## Metrics

Three regions, reported as RMSE in cm against the clean 609-cell scan. The 5 cm cell
size is the unit to read them against.

| Region | Cells | What it measures |
|---|---|---|
| measured | of 388 | cells a beam reached this step |
| occluded | of 388 | in the observation band but unmeasured this step -- shadow and sparsity |
| body | 221 | under the robot: the fan can *never* reach these, so the error there is the error of a value produced from memory alone |

The body region is the one to watch. It is a permanent, always-on test of whether the
recurrence is doing anything, which is why the blind-sensor test Miki et al. run on
hardware was considered and dropped -- it would measure the same thing intermittently
that this measures continuously.

## ConvGRU results (the architecture built first)

`(16, 29, 21)` recurrent state, 22,802 parameters. Mean +- SD over each phase's last
200 iterations.

| Phase | measured | occluded | body |
|---|---|---|---|
| Phase 2 (rough / boxes) | **2.56 ± 0.46** | **3.84 ± 0.31** | **3.53 ± 0.36** |
| Phase 3 (stairs) | **3.76 ± 0.27** | **6.45 ± 0.35** | **4.59 ± 0.31** |
| Phase 4 (thin walls) | **3.43 ± 0.05** | **4.46 ± 0.07** | **4.41 ± 0.08** |

Environment conditions were held constant across all three: terrain level ~4.4 (spread
uniformly over all 10 rows and frozen there), commanded speed ~0.72 m/s (full range
from iteration 0), episode length ~470, unobserved rate 42-45%.

**These are training-time numbers, not a held-out evaluation.** They are recorded to
show the encoder works and roughly where it lands; the architecture comparison uses
the paired protocol below instead. The three rows are also not comparable to each
other -- different terrain, not different encoders.

### Reading them

* Every region lands under one cell (5 cm) on every terrain.
* The body region at 3.5-4.6 cm comes from memory alone: no beam has ever reached
  those cells, so the ConvGRU is carrying past observations forward and shifting them
  by however far the robot moved, without being given odometry.
* Occluded is consistently the worst region, which is expected -- 42-45% of the band
  has no return at any instant.
* Stairs are the hard case: occluded 6.45 cm against 3.84 on rough ground. A riser
  cuts the sightline, so the shadow is deep, and what is inside it is a continuing
  staircase rather than the flat continuation that works on rough ground.
* Walls came out easier than stairs (occluded 4.46). A 5 cm wall is thin, so its
  shadow is narrow and the far side is close.

### Convergence

Slope of the last 400 iterations, with the standard error of the slope:

| Phase | measured | occluded | body | verdict |
|---|---|---|---|---|
| Phase 2 | +0.024 (t=+1.39) | +0.010 (t=+0.82) | -0.004 (t=-0.30) | flat |
| Phase 3 | +0.026 (t=+2.65) | -0.004 (t=-0.35) | -0.029 (t=-2.55) | ~flat |
| Phase 4 | -0.020 (t=-7.66) | -0.029 (t=-9.56) | -0.017 (t=-4.32) | still falling |

Units are cm per 100 iterations. Phase 4 is the only one still improving, but at
0.02-0.03 cm/100it another 1000 iterations buys 0.2-0.3 cm, so 1000 was called enough
on magnitude rather than on significance.

Two traps in reading these curves, both hit once:

1. **Single-point comparisons are meaningless.** The per-iteration value swings
   2.10-4.96 cm at constant performance, because one iteration is a snapshot of 512
   robots x 24 steps and the terrain draw varies.
2. **The NLL is not a usable convergence signal.** Its SD over a window is 400-770
   against a window-mean difference of ~30, and structurally it falls when sigma
   shrinks whether or not the mean improves. Use the RMSEs, which are in cm.

Consecutive iterations are autocorrelated (lag-1 ~ +0.24), so the effective sample
size is about 60% of the nominal one. Correcting for it moves the Phase 2
600-800 vs 800-1000 difference from t=+2.45 to t=+1.91 -- which is the point at which
the honest answer stops being "significant or not" and becomes "0.10 cm is 2% of a
cell, so it does not matter either way."

## The architecture comparison (in progress)

Question: does the spatial recurrence earn its complexity against the paper's flat
one?

### The two arms

Only the network differs. Input (388-cell noisy grid + 45-dim proprioception), target
(609-cell clean scan), loss (Gaussian NLL), training schedule and terrain chain are
all held identical, so the comparison isolates the recurrence and not the input
representation.

| | ConvGRU | Belief encoder/decoder |
|---|---|---|
| entry | `Conv3x3 1 -> 16` | `Linear 388 -> 80 -> 60 -> 24` |
| recurrence | `ConvGRU 3x3, 16ch` | `GRU 2 x 50` |
| state | 9,744 floats/env | 100 floats/env |
| exit | `Conv1x1 16 -> 2` | `Linear 50 -> 64 -> 64 -> 1218` |
| parameters | 22,802 | ~208,100 |

The baseline is the project's own implementation of Miki et al. 2022 --
`assets/models/student_actor.py`'s `BeliefEncoder` / `BeliefDecoder` with
`config.yaml`'s dimensions -- rather than a fresh transcription, so "close to the
paper" means something checkable. Only its output width changes, to 609 mean + 609
log sigma, so the loss can be shared.

Note it is nine times the parameters and one four-hundredth of the state. The
comparison is not big-versus-small; it is where the capacity sits.

Two adaptations of the paper's own numbers are already baked into that
implementation and worth knowing about:

* `l_e` is 24, not the paper's 96. The paper encodes height samples *per foot* --
  24 x 4 feet -- and the Go2 map is one body-centred grid, so one encoder's 24 is the
  faithful translation. It does mean the exteroceptive compression is 388 -> 24
  (16:1) where the paper's was 208 -> 96 (2.2:1).
* The belief state is 24 and the addition needs no zero-padding, where the paper has
  120 = 96 + 24 with padding.

The decoder there also settles a reading the paper's text leaves open. "The same gate
is used in the decoder" turns out to mean a *separate* gate of the same shape, and its
skip carries the **raw noisy scan** at full 388 width rather than the encoded latent
-- so where the sensor is trustworthy the decoder can copy it and only learn the
corrections. Figure 7D, which would settle this directly, is an image the parsed
markdown in `doc/papers/` does not include.

### Evaluation protocol

Region-wise RMSE, as above. No blind-sensor test: the body region already is one.

The frozen policy ignores the encoder, so a fixed seed produces an identical rollout
for both arms -- and both can be run in the *same* process on the *same* rollout,
which makes the comparison paired. That matters: the per-iteration swing of 2.10-4.96
cm is common to both arms and cancels in the difference.

* evaluation seed different from training, so terrain tiles and noise draws are unseen
  (in-distribution generalisation, not out-of-distribution -- same generators)
* all three terrains, Phase 2 / 3 / 4, scored separately
* report the paired difference with a confidence interval, corrected for
  autocorrelation, and in cm against the 5 cm cell rather than as a p-value
* report accuracy per parameter alongside raw RMSE, given the 9x gap

Held in reserve: if the two arms come out close on the body region, a blind-sensor
test would still separate memory horizons longer than the ~42 steps a cell spends
under the body. Not needed if the difference is clear.
