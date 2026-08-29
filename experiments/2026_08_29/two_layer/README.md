# Part two: a second layer that carries the wave

The one-layer finding was that top-1 retrieval holds the nearest rim
rather than continuing the curve. That is structural: a single (x, y)
pair contains nothing about which way the curve is going — "x = 2.0,
y = 1.6" does not know it is on a descent. The wave is a property
*across* pairs. So layer two is handed several pairs at once.

## The rig

Extended to x ∈ [-12, 12], 1600 pairs, same hole [1.2, 2.8]. Layer one
is unchanged (1024 minicolumns, norm-corrected selection, graded top-16
read; in-support RMSE 0.067).

**Layer two's input is a WINDOW**: 21 taps spaced 0.2 apart along x,
each tap one channel holding that tap's y as a bump, all OR'ed into one
array with disjoint scatters. So one vector is a whole *piece of
curve*, and a template is a stored shape. Same hypercolumn, same
competition, same geodesic learning.

A tap is **known** if there is training data within half a tap of it —
the honest test-time signal, not the experimenter pointing at the hole.
Bridging is one window centred on the hole: rims written, middle blank,
partial-cue read (norm-corrected, contrast floor 0.25), and the winner's
cells at the blank taps are the answer. 7 taps have to be invented.

`trend_shapes.png` shows what layer two became: crests, troughs, rising
and falling ramps — the vocabulary of shapes the wave makes.

## It bridges, and it beats the fitted network

| in the hole | RMSE |
|---|---|
| layer 1 alone | 0.817 |
| hold the value at the nearer rim | 0.819 |
| MLP (64-64, tanh) | 0.149 |
| **layer 1 + layer 2** | **0.033** |

25x better than one layer, and 4.5x better than the MLP, out of the same
machinery — competition, one-winner rotation, partial-cue read.

## What a tap may SAY is the whole question

Four interfaces, everything else identical. Two measurements per
interface: bridging the real hole, and a **blank control** — blank the
same 7 taps in windows sitting *inside* the training data, where layer
two has definitely seen the shape. The control separates "cannot
complete" from "never saw this shape".

| what a tap says | cells/tap | blank control | the real hole (trended) | the real hole (periodic) |
|---|---|---|---|---|
| value, relative to the first known tap | 96 | **0.059** | **0.033** | 0.102 |
| value, absolute | 96 | 0.055 | 3.221 | **0.063** |
| the winning minicolumn (one-hot) | 1024 | 0.638 | 4.317 | 0.427 |
| graded top-16, magnitudes intact | 1024 | 0.143 | 2.783 | 1.594 |

Two separate effects, and they are worth keeping apart.

**1. Overlap in the message — the blank-control column.** One-hot is
4.5x worse than graded speech at ordinary completion (0.638 vs 0.143;
0.763 vs 0.141 on the periodic task). This is the failure at its
purest: layer one's own input code overlaps beautifully, but if layer
one reports only its winner, the code arriving at layer two is one lit
cell out of 1024, and minicolumn #237 shares nothing with #238 even
when they stand for almost the same thing. The overlap that made layer
one work is thrown away at its own output. Graded, sparse speech
restores it. `trend_window_codes.png` shows this directly — in the
one-hot and graded panels the vertical axis is *minicolumn index*,
which is arbitrary, so the wave is scrambled into unreadable dots.

**2. Expressible invariance — the two hole columns.** Overlap is not
sufficient. Both value interfaces overlap identically, and on the
trended wave one gets 0.033 and the other 3.221. The difference is that
the relative code can say *"this shape, at whatever height"* and the
absolute one cannot, so the absolute one needs to have seen this exact
descent at this exact height — and on a wave with a trend, it never
has. The periodic control proves the mechanism is fine: make the wave
strictly repeat, so the absolute pattern really does occur elsewhere,
and absolute becomes the best interface (0.063).

Note that neither activation interface can be made relative *even in
principle*: a minicolumn's identity binds an x to a y absolutely, so
no message assembled from minicolumn identities has a word for "the
same shape, higher up". The invariance is not lost in transmission,
it is unrepresentable.

## Files

- `two_layer.py` — window layer and the four tap interfaces
- `two_layer_viz.py`, `run_two_layer.py` — the run; writes
  `results/`, 137 s

```
.venv/bin/python experiments/2026_08_29/two_layer/run_two_layer.py
```

`results/`: `trend_*` and `periodic_*` × `bridge` (the hole,
four interfaces side by side) · `window_codes` (one window said four
ways) · `shapes` (layer two's templates as curve pieces) ·
`interfaces` (all methods ranked) · `metrics.json`

## One correction to Part one's machinery

(Part one is [`../regress/`](../regress/README.md); `Hypercolumn` lives in
`../regress/pop_regress.py`.)

`Hypercolumn.scores(mode="masked")` now takes a **contrast floor**
(default 0.25). Dividing by a template's norm over the queried cells is
right when every template has real mass there, but a template holding
almost nothing at those cells gets a tiny overlap divided by a tiny
norm and bids absurdly high. floor=0 is the bare masked rule, floor=1
is exactly the plain dot product. It changes none of Part one's numbers
(in-support RMSE 0.0941 at every floor from 0 to 0.5, and 0.5463 at
floor=1, reproducing the dot product exactly).

---

The run's console output is `results/run_two_layer.log`.

A plain-language walkthrough of this experiment (and of Part one, which it
builds on) is kept beside this README:
[`TwoLayerWalkthrough.md`](TwoLayerWalkthrough.md).
