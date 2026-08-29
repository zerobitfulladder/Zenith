# Part three: a convolutional layer one, and a stack that is content-based throughout

*Two writeups of this experiment were kept, one after the other; both are
reproduced below as they were. The scripts import `pop_regress` from
[`../regress/`](../regress/README.md) and `two_layer` from
[`../two_layer/`](../two_layer/README.md).*

Part two's layer two worked, but it cheated in a way worth naming: it
heard layer one's *decoded answer*, re-encoded as a fresh blurry strip.
The invariance came from a hand-written reference subtraction, at one
level only. The user's objection was that a real hierarchy should have
layer one emit a code whose *overlap structure* already carries the
data's structure, so layer two abstracts patterns over patterns.

## What was wrong with the old layer one, measured

Every old layer-one template was tied to one place on the x axis. There
was no template meaning "a descent", only "the descent at x = 1.4" and
"the descent at x = -7.6", and those two are strangers.

| window pair | old L1's graded code | relative-value code |
|---|---|---|
| x=2.0 vs x=2.4 — nearly the same place | 0.427 | 0.235 |
| x=2.0 vs x=-7.6 — **same shape**, 2 periods apart | **-0.005** | 0.802 |
| x=2.0 vs x=-3.0 — different shape and place | -0.005 | 0.341 |

105 and 188 templates active in the two windows; **0 shared**. Its
margin between "same shape" and "unrelated" is exactly **0.000**. A code
built from place-bound identities can only answer *where am I*.

## The fix: weight sharing

`conv_stack.py`. ONE hypercolumn whose input is a **5-sample patch, 0.8
wide** — far too small to span the 1.6-wide hole, so it cannot cheat —
**applied at every x position**, each patch written relative to **its
own mean**. Its 64 templates become a vocabulary of local shapes with no
location attached (`l1_vocabulary.png`: steep descents through flats to
steep rises, every one of them winning hundreds of times).

Vocabulary size controls how much sharing is forced. Measured with an
order-preserving window code:

| K1 | positions per template | reuse span in x | same shape | diff shape | margin |
|---|---|---|---|---|---|
| 8 | 21 | 19.8 | 0.548 | 0.596 | -0.048 |
| 16 | 12 | 19.4 | 0.518 | 0.422 | +0.096 |
| 32 | 6 | 19.2 | 0.537 | 0.353 | +0.184 |
| **64** | 3 | 14.3 | 0.414 | 0.193 | **+0.221** |
| 128 | 1 | 0.0 | 0.355 | 0.145 | +0.209 |
| *old place-bound (1024)* | 1 | 0.0 | *-0.004* | *-0.004* | ***0.000*** |

K1 = 8 goes negative — too few shapes and everything looks alike. Note
K1 = 128 keeps a good margin despite each template *winning* at one
place: the sharing lives in the graded code, not in who wins.

`l1_code_field.png` is the picture of it — the code field is periodic,
the same rows lighting up every wave-period, and blank across the hole.

## Layer one alone can carry the curve about one receptive field

Walking into the hole one step at a time, the new point as the patch's
last sample, four already-known samples as the cue:

| | RMSE across the hole |
|---|---|
| old place-bound layer 1 | 0.817 |
| hold the value at the nearer rim | 0.743 |
| **conv layer 1 alone, walking** | **0.376 - 0.43** |

Real transfer — the shared vocabulary is worth 0.82 → 0.38. But it
drifts: errors of +0.06, -0.03, +0.06, +0.02, +0.12, +0.04 through
x = 1.8, then +0.33 at 2.2 and +1.20 at 2.6. Each prediction becomes the
next step's cue and 0.8 of context cannot tell it where in the larger
wave it sits. Which is exactly the job left for the layer above.

**One reference bug found here, the third of its kind.** Training used
the mean of all 5 samples as a patch's reference; the walk had only 4
and used the mean of those. For a falling patch the mean of the first
four sits above the mean of all five, so every query rode high and
matched templates that turn *upward* — the layer predicted an upturn
from four clearly descending samples. Fixing the reference subset:
**1.022 → 0.428**. Whatever you subtract or divide by must come from the
same subset of the input at training and at read time.

## The full stack

`run_conv_stack.py`, 16 s.

    level 0   raw pairs read through a small window
    layer 1   shared hypercolumn over a 0.8-wide patch, at every x
              -> graded code over 64 shared shapes
    layer 2   window of 51 of those CODES (5.0 wide), 96 templates
              -> completes the codes layer one could not emit
    decode    each completed patch says how its samples differ from
              one another; ONE least-squares solve turns every such
              difference into absolute values at once, pinned by the
              known data. No marching, so nothing accumulates drift.

Layer one has no code across x in [0.9, 3.1] — 23 positions, wider than
the 1.5-wide data hole because the patch footprint eats 0.4 either side.
Layer two completed all 23 from a single template (#65).

| in the hole | RMSE |
|---|---|
| old place-bound layer 1 | 0.817 |
| hold the value at the nearer rim | 0.743 |
| conv layer 1 alone, walking | 0.428 |
| MLP (64-64, tanh) | 0.144 |
| **conv layer 1 + layer 2 over its codes** | **0.086** |
| *part two's layer 2 on decoded values, hand-written reference* | *0.033* |

The stack that is content-based end to end beats the fitted network and
is 5x better than layer one alone. It does not quite reach part two's
0.033, and the reason is honest: everything upward passes through a
64-word discrete vocabulary, so the reconstruction inherits that
quantisation, whereas part two's layer two worked on continuous values
with the invariance supplied by hand. Trading a little accuracy for an
invariance that is structural rather than hand-written is the whole
point of the exercise.

`l2_templates.png` decodes layer two's templates back into curve: they
are recognisable full wave-cycles built out of layer one's local shapes.
(The grid-scale sawtooth on them is a decode artifact — stitching
per-tap *argmax* shapes rather than the graded code — not something in
the templates. It averages out in the least-squares fill, which is why
`regeneration.png` is smooth.)

## Files

- `conv_stack.py` — level 0 sampling, `PatchLayer` (shared conv L1),
  `CodeWindowLayer` (L2 over codes), `stitch` (least-squares decode)
- `conv_viz.py`, `run_conv_stack.py` — the run; writes
  `results/`

```
.venv/bin/python experiments/2026_08_29/conv_stack/run_conv_stack.py
```

`results/`: `l1_vocabulary.png` (all 64 shared shapes) ·
`l1_code_field.png` (what layer one says everywhere — periodic) ·
`l2_templates.png` (layer two's templates as curve) ·
`completed_codes.png` (given vs invented codes) ·
`regeneration.png` (the filled hole) · `metrics.json`

---

# Part three: the convolutional stack — shared shapes below, structure above

Part two's second layer worked (0.033) but it was not the architecture
the user was after. Its two layers talked in **values**: layer two heard
layer one's decoded answer re-encoded as a fresh bump. The intended
design was for layer one to emit a *sparse code whose overlap mirrors
the data's own structure*, and for layer two to learn patterns over
those patterns — hierarchy all the way up, in one idiom.

That design was implemented literally in part two (the "graded"
interface) and it scored 2.783. Part three finds out why, and fixes it.

## The diagnosis: the old layer one overlaps by PLACE, not by content

Every template in the old layer one was tied to one spot on the x axis.
There was no template meaning "a descent", only "the descent at x=1.4"
and "the descent at x=-7.6", and those two are strangers.

Measured, with order preserved, on 21-tap windows:

| | same shape, 2 periods apart | different shape | margin |
|---|---|---|---|
| old place-bound layer 1 | −0.004 | −0.004 | **0.000** |
| conv layer 1 (64 shapes) | 0.414 | 0.193 | **+0.221** |

**The old layer one's margin is exactly zero** — it cannot distinguish
"same shape" from "unrelated", so nothing above it can transfer. The
two windows over the same shape shared **0 of their 105 and 188 active
templates**.

## The fix: weight sharing

One hypercolumn over a **5-sample patch, 0.8 wide** — far too small to
span the 1.6-wide hole — applied at *every* x position, each patch
written relative to **its own mean**. Templates become a vocabulary of
local shapes with no location attached (`l1_vocabulary.png`: 64 shapes
sorted by slope, from steep descents through crests and troughs to
steep rises).

`l1_code_field.png` is the picture worth keeping. Because the templates
are sorted by slope, layer one's graded code field draws the derivative
of the curve — and **the same rows light up at every crest and every
trough across the whole axis**, though the curve trends upward
throughout. That is content-overlap, visible. The old layer one would
have drawn a diagonal that never repeats.

## The stack

    level 0   raw pairs read off the axis by local averaging. Not
              learned. Undefined inside the deleted slice.
    layer 1   one shared hypercolumn, 5-sample patch, every position.
              Emits a graded code over 64 shared shapes.
    layer 2   a 51-tap window (5.0 wide) of those codes, one grid step
              apart. Blank the positions layer one cannot speak for,
              complete them, and decode the completed shape-words back
              to values.

The decode solves all overlapping patches **at once** by least squares —
each patch contributes its adjacent-sample differences as equations,
pinned by the known values — rather than marching left to right. No
marching, so nothing accumulates drift.

**The only subtraction anywhere is each patch's own mean.** No
privileged reference tap, no knowledge of periods or trends. The same
local rule at every position and every layer.

## It regenerates the missing stretch

| in the hole | RMSE |
|---|---|
| old place-bound layer 1 alone | 0.817 |
| hold the value at the nearer rim | 0.743 |
| conv layer 1 alone, walking step by step | 0.428 |
| MLP (64-64, tanh) | 0.144 |
| **conv layer 1 + layer 2 over its codes** | **0.086** |
| *(part two's hand-relative window layer 2)* | *0.033* |

Better than the fitted network, 9.5× better than the old layer one, and
5× better than the same shared vocabulary run on its own. The
hand-relative rig still wins on the raw number — but it was told to
measure each window from a chosen tap, whereas this one was told
nothing but "each patch is written relative to its own mean".

Layer two did not invent one shape; it wrote the descent out as a
**sentence of shape-words**, one per position:

| x | word | the shape it means | slope |
|---|---|---|---|
| 0.9 | #15 | `[-0.43 -0.06 0.12 0.17 0.21]` | +0.64 |
| 1.1 | #51 | `[-0.26 0.03 0.13 0.13 -0.03]` | +0.23 |
| 1.2 | #24 | `[-0.07 0.05 0.10 0.03 -0.10]` | −0.03 |
| 1.4 | #14 | `[0.10 0.14 0.08 -0.07 -0.24]` | −0.34 |
| 1.7 | #55 | `[0.42 0.31 0.13 -0.24 -0.62]` | −1.04 |
| 2.1–2.6 | #53 | `[0.76 0.39 0.03 -0.43 -0.74]` | −1.51 |
| 2.9–3.1 | #21 | `[0.57 0.26 -0.05 -0.31 -0.47]` | −1.04 |

Still rising into the crest, flat at x≈1.2 (the true crest is 1.25),
steepest at 2.1–2.6, easing toward the trough. 13 distinct words over
23 positions, repeating exactly where the slope is genuinely constant.

## The vocabulary must be smaller than the number of places

Sweeping the shared vocabulary size, with the same order-preserving
measurement:

| K1 | positions per template | reuse span in x | same shape | diff shape | margin |
|---|---|---|---|---|---|
| 8 | 21 | 19.8 | 0.548 | 0.596 | −0.048 |
| 16 | 12 | 19.4 | 0.518 | 0.422 | +0.096 |
| 32 | 6 | 19.2 | 0.537 | 0.353 | +0.184 |
| 64 | 3 | 14.3 | 0.414 | 0.193 | **+0.221** |
| 128 | 1 | 0.0 | 0.355 | 0.145 | +0.209 |

Too few shapes (8) and everything looks alike — the margin goes
negative. Note that K1=128 still has a good margin even though each
template *wins* at only one place: the sharing that matters lives in the
**graded** code, not in who wins.

## A third instance of the same bug

Walking the conv layer one across the hole gave 1.022 — worse than
holding the rim — until the cause showed up: training computed each
patch's reference as the mean of all 5 samples, while the walk had only
4 and used the mean of those. For a falling patch the mean of the first
four sits *above* the mean of all five, so every query rode high and
matched templates that turn upward. The layer predicted an upturn from
four clearly descending samples. Fixing the reference subset: **1.022 →
0.428**.

That is the third time this class of bug has cost real accuracy here
(position collisions, the partial-cue norm, now this). The rule:
**whenever part of the input is missing, whatever you subtract or divide
by must be computed from the same subset in training and at read time.**

## Files

- `conv_stack.py` — level-0 sampling, the shared `PatchLayer`, the
  `CodeWindowLayer` over its codes, and the least-squares `stitch`
- `conv_viz.py`, `run_conv_stack.py` — the run; writes
  `results/`, 16 s

```
.venv/bin/python experiments/2026_08_29/conv_stack/run_conv_stack.py
```

`results/`: `l1_vocabulary.png` (the 64 shared shapes) ·
`l1_code_field.png` (what layer one says everywhere — the repeat is
visible) · `l2_templates.png` · `completed_codes.png` (the codes layer
two invented) · `regeneration.png` · `metrics.json`

The full from-scratch account of this stack is kept beside this README:
[`ConvStack.md`](ConvStack.md).
