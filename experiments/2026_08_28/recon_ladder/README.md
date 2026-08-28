# `recon_ladder/` — the reconstruction ladder (Exp 1)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_recon3.py` — the three-layer stack, train + reconstruction ladder. Also the shared
  module (`import run_recon3 as R`) that almost every other experiment of the day imports
  for geometry, encoders, galleries and data loading.
- `results/` — metrics.json, report.md, galleries, weights.npz, run log.

    .venv/bin/python experiments/2026_08_28/recon_ladder/run_recon3.py

## Exp 1 — run_recon3.py: the reconstruction ladder

The classic stack, no labels, no top memory, whole training set presented
once, unsupervised:

    L1   8x8 px windows, stride 2 -> 11x11, K=64    learns DENSE
    L2   3x3 over L1,    stride 2 ->  5x5,  K=64    learns SKELETON
    L3   3x3 over L2,    stride 1 ->  3x3,  K=100   learns SKELETON

Top-1 learning only; output = dense positive correlations (relu). The
decode path is rebuilt fresh and simple — no squelch, no feathering, no
confidence weighting (the old renderer was tuned to make label-drawings
pretty, not to minimize error): expand codes down level by level, plain
average where footprints overlap, each L1 window rebuilt as
mean + norm x (unit direction voted by its code).

Instrument: reconstruct from the L1 code, the L2 code, the L3 code (each
rung prices that layer's bottleneck), in two read modes (graded / top-1),
scored by pixel MSE and shape correlation on 5000 held-out images.
Controls: predict-the-mean-image, and side-channels-only (every window flat
at its stored mean — how much of MSE the bookkeeping alone explains).
Also per-window quantization residual per layer: for unit vectors
||window - c*w||^2 = 1 - c^2, so the mean top-1 correlation per window IS
the local reconstruction quality that top-1 learning implicitly optimizes.

Predictions (recorded before the run):
1. The ladder falls with depth — each stride-2 bottleneck costs; from-L1
   should be very good (corr > 0.9), from-L3 above the old 0.651 because
   the graded read and plain decode remove self-inflicted damage.
2. Graded read beats top-1 at every rung (read-mode law: reconstruction of
   a specific input is evidence-carrying).
3. Side channels alone already explain a large share of MSE (MNIST is
   mostly flat background).

Smoke-run flag (300 train images): at L1 the top-1 read BEAT graded
(0.933 vs 0.907 corr) — the dense vote over 64 correlated templates blurs
the direction; opposite at L2/L3 where graded won. Watching whether this
survives full training; prediction 2 may be wrong at L1.

## Results (55k train x 1 epoch, 5k test, ~14 min CPU)

All units used (64/64/100), no monopoly. `results/`.

The ladder (pixel MSE in [0,1] images / shape corr vs the input):

| rung | graded read | top-1 read |
|---|---|---|
| from L1 code | 0.0157 / 0.909 | **0.0113 / 0.934** |
| from L2 code | 0.0184 / 0.892 | 0.0178 / 0.895 |
| from L3 code | **0.0247 / 0.855** | 0.0289 / 0.827 |

Controls: mean-image predictor 0.0671 / 0.545; side-channels-only (flat
windows at their means) 0.0633 / 0.593. Per-window top-1 corr (the
quantity top-1 learning implicitly optimizes): L1 0.807, L2 0.443,
L3 0.448.

Prediction scorecard:
1. CORRECT — the ladder falls with depth, and from-L3 graded (0.855)
   beats the old static3 roundtrip (0.651, same depth) by +0.20 corr:
   the old squelch/feather/harden renderer plus missing side channels
   were self-inflicted damage worth a fifth of the correlation.
2. WRONG at L1/L2, right only at L3 — READ MODE FLIPS WITH DEPTH.
   Reconstructing a window from its OWN code, top-1 wins (0.934 vs
   0.909): the dense vote over 64 mutually-correlated templates blurs
   the direction, and the winner alone is the quantizer's honest
   output. Two expansion steps later the preference reverses: hardened
   codes compound errors on the way down, graded overlaps correct each
   other. L2 (one expansion) sits at the crossover, a near-tie.
3. WRONG in spirit — side channels alone explain almost nothing beyond
   the mean image (0.0633 vs 0.0671): per-window means at 8x8 are just
   a blur. The template content does the real work (MSE 0.063 -> 0.016,
   a 4x drop). The side channels matter as the SCALE the templates get
   re-applied at, not as content.

By eye (roundtrip_graded.png): from-L1 and from-L2 are faithful; from-L3
still clearly the same digit but softer, and the 8 degrades toward an
a-like blur (2/4/8 the historic weak trio again). templates_L2.png =
clean motifs (bars, corners, hooks). templates_L3.png: with a 3x3 window
over the 5x5 L2 grid the footprint is nearly the whole image — L3 units
render as NEAR-WHOLE DIGIT PROTOTYPES, not mid-scale parts. So this
stack has no true mid-scale compositional level; the from-L3 rung is
"reconstruct from ~9 overlapping near-global prototype votes."
samples_random_code.png confirms it: independent random parts per
position = spatially incoherent stroke soup (generation needs one
coherent source — the known law, now visible without any labels).

Open levers this immediately suggests: (a) make L1's own rung better
with more than one template per window (winner explains, subtract,
let a second winner cover the remainder) — the residual idea reopened
under the new objective; (b) fix L3's footprint so the stack has a real
mid-scale level; (c) the L2/L3 expansion step is the biggest single
loss (0.934 -> 0.895 -> 0.855) — a better downward mapping is worth as
much as a better dictionary.
