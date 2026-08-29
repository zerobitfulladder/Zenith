# Blurry numbers: does one hypercolumn become a regressor?

2026-08-29. Testing the idea from the second-layer drone session — that
**inputs overlapping semantically is what lets this thing generalize**
— on the simplest task where the claim can be checked with a ruler:
plain (x, y) pairs from a known function.

## The setup

`f(x) = 1.5·sin(1.3x) + 0.35x + 0.6·cos(0.7x)` on x ∈ [-6, 6], 800
noisy pairs (σ = 0.08). **A slice x ∈ [1.2, 2.8] is cut out of
training entirely** — no pair from that band is ever shown. That slice
is the interpolation test.

Everything else is the single-layer drone's machinery with two channels
instead of ten:

* A number becomes a **bump**: its own cell is brightest and neighbours
  fade with distance (raised cosine, half-width 10 of 96 cells ≈ ±1.25
  in x). So 2.0 and 2.6 share most of their cells and -3.0 shares none.
* x and y each own a random scatter of 96 positions in one 8192-cell
  array, and the two bumps are **OR'ed** into it — 40 cells on, 0.5%
  density. One vector says *"at this x, the answer was this y."*
* One hypercolumn, 256 minicolumns, competing on each vector; only the
  winner learns (geodesic rotation, centred unit templates). Identical
  to `sl_drone.Hypercolumn`.
* **Testing:** write only the x cells, leave the y cells empty, let the
  minicolumns compete, take the winner's own y cells, emit the
  brightest one. Same as reading the drone's motors back out.

Training takes 3.4 s. Whole experiment, all sweeps: 98 s.

## It works

| | unseen x inside the range | inside the held-out slice |
|---|---|---|
| **hypercolumn, brightest cell** | **0.107** | **0.729** |
| hypercolumn, sub-cell read | 0.102 | 0.740 |
| k-nearest neighbours (k=5) | 0.037 | 0.739 |
| random forest (300 trees) | 0.056 | 0.765 |
| MLP (64-64, tanh) | 0.138 | 0.122 |
| *(just hold the value at the nearer rim)* | — | *0.696* |

RMSE. The layer beats the MLP on x values inside the training range and
loses to k-NN and the forest by ~3x — the gap is the staircase: 256
minicolumns means 256 steps, and the y axis is only 96 cells wide.

`readout_field.png` is the picture worth looking at: the answer comes
out **blurry**, a band of asserted y values, and that band tracks f(x)
across the whole range.

## What the sweeps say

**1. Overlap is doing the work — but only where the data is thin.**
With 800 pairs the training set is dense enough (spacing 0.013 in x)
that even a one-cell code just memorises: RMSE 0.097 at half-width 1.
Rerun with **60 pairs**, where an unseen x genuinely has no example next
to it, and the claim shows up cleanly:

| half-width (cells) | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| RMSE, 60 pairs | 0.576 | 0.347 | **0.163** | 0.168 | 0.217 | 0.447 | 0.824 |

3.5x better purely from making the code overlap. Same U-shape inside
the held-out slice with the full dataset (1.91 → 0.56). Too wide and it
breaks the other way: past ~16 cells different x values stop being
distinguishable and the layer collapses onto a handful of minicolumns.

**2. It does not interpolate — it holds the nearest rim.**
`gap_sweep.png` cuts holes of increasing width and measures inside
them. The hypercolumn lands **on top of the "hold whatever f was at the
nearer rim of the hole" line** at every width. It answers with its
nearest stored piece of curve; it does not continue the curve through
the hole. k-NN and the forest do the same. The MLP bridges holes up to
~2.2 wide because it fitted a function — and then fails worse than
everything at width 3.0, inventing a curve that is not there.

So the honest statement of what the blurry code buys: **an input it has
never seen still means something** (it lands on top of inputs it has
seen, and retrieves their answer). That is generalization. It is not
interpolation, and top-1 retrieval structurally cannot be —
one winner can only ever emit one stored answer.

**3. The graded read is worth 2x, and it says "I don't know" properly.**
Reading every positively-matching minicolumn's y cells, magnitudes
intact (the project's settled graded speech, applied to the read
instead of the message) instead of only the winner's:

| read | unseen x in range | held-out slice |
|---|---|---|
| top-1 cell (as specified) | 0.094 | 0.748 |
| graded, top-4 | 0.065 | 0.748 |
| **graded, top-16** | **0.046** | 0.768 |
| graded, top-64 | 0.103 | 0.889 |
| graded, every positive match | 0.488 | 0.946 |

Top-16 halves the in-support error and beats the random forest. It does
not help in the hole — nothing does — but look at the bottom-right
panel of `readout_field_best.png`: inside the hole the graded answer is
visibly **bimodal**, one peak at the left rim's y and one at the right
rim's, with the truth sitting between them. The layer is saying "it is
either this or that, I have never been here." Top-1 throws that away
and emits one of the two peaks with full confidence.

## Two mechanism findings that apply to the drone rig

**A. Collisions in the shared array are not harmless.** Each channel
drawing its own random positions means x cells and y cells sometimes
land on the same array position. One shared position puts a phantom
peak on the **y row of every template whose x bump covers it** — and
the top-1 read then emits that phantom as the answer, confidently, for
a whole band of x. Those were the only catastrophic errors this rig
ever made. Carving all channels out of one permutation instead:

| array size | shared positions | RMSE with | RMSE disjoint |
|---|---|---|---|
| 4096 (first version) | 3 | 0.675 | 0.160 |
| 8192 (current) | 2 | 0.157 | 0.107 |

Worth checking on the drone, which reads its motor commands out exactly
this way: **31 of its 256 motor cells share an array position with a
sensory cell** (83 shared positions overall in its 8192 array).

**B. A partial query needs the template's norm over the queried cells.**
The drone scores a partial query — sensory cells written, motor cells
blank — with a bare dot product against unit-norm templates. That
silently assumes every template carries the same norm inside the
queried cells, and it does not: a minicolumn that has become **vague
about the missing half** keeps its whole norm in the queried half and
outbids the minicolumns that actually match. It is self-reinforcing —
winning more queries makes it vaguer still.

The damage scales with capacity, because finer tiling means smaller
honest score differences for the bias to overturn:

| minicolumns | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| plain dot product | 0.102 | 0.107 | 0.396 | 0.546 |
| ÷ template norm over queried cells | 0.092 | 0.105 | 0.098 | **0.094** |
| *minicolumns actually used (dot)* | 109 | 166 | 164 | 130 |
| *minicolumns actually used (÷norm)* | 123 | 236 | 377 | 471 |

**Adding minicolumns makes the plain rule worse.** With the correction
— which is just the same cosine rule applied to the cells that are
actually present — capacity helps again and the layer uses 46% of its
minicolumns instead of 13%. This is worth checking on the drone, whose
inference is exactly this partial-query read.

## Files

- `pop_regress.py` — data, encoder, hypercolumn (self-contained)
- `regress_viz.py` — every picture
- `run_regress.py` — the run; writes `results/`
- `results/run.log`, `results/metrics.json` — every number above

```
.venv/bin/python experiments/2026_08_29/regress/run_regress.py
```

`results/`: `dataset.png` · `encoding.png` (the fall-off, sample pairs
as codes, the raw OR'ed array) · `templates.png` (16 busiest
minicolumns) · `template_map.png` (all 256 decoded onto the curve) ·
`predictions.png` (model vs the three regressors) · `readout_field.png`
(the blurry answer, as is) · `width_sweep.png` · `selection_rule.png` ·
`gap_sweep.png` · `*_best.png` (1024 minicolumns, norm-corrected
selection, graded top-16 read)

The unit this experiment uses is written up in full, with the formulas,
in [`HypercolumnReference.md`](HypercolumnReference.md) (kept beside this
README; it also covers what a layer may say to the next layer, from
[`../two_layer/`](../two_layer/README.md)).
