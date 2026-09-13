# 2026-09-13

## `search_mlp/` — search in place of backprop, same MLP shape

Same 784-256-128-64 ReLU net, same init, same data. One arm learns by
backprop, the others by the search: pick the unit with the largest projection
onto what is left, keep the projection as its activation, subtract, pick
again, with a beam of 4 over the order; only chosen units learn, toward the
residual they were shown. The label never touches a weight in the search
arms; it is read by a tally over which units fired, or, in one arm, fed to
the top layer as a second stream.

```
                     units on    selectivity    tally on firing    probe    head
backprop   L1        205/256         0.15            0.53           0.93    0.912
search-b4  L1          5/256         0.41            0.83           0.87       -
search-b4  L2/L3                                0.78 / 0.70
label-b4   label stream read                    0.69
```

**Identity falls out of the search with no rule asking for it**, and the
firing pattern alone reads the label at 0.84 against 0.53. Under search the
probe adds nothing over the tally (the meaning is in who fired); under
backprop it adds forty points (the meaning is in the blend).

**The middle layers are the first layer again.** Expanded down to pixels,
layer-2 and layer-3 templates are whole digits, and 93 to 100% of them put
over 90% of their weight on one lower unit: wrappers, yesterday's `encoder`
result in a new coat. Depth loses seven points a layer because the vocabulary
shrinks instead of composing. A beam of 4 explains more energy and hires
fewer units (half of layer 1 dead); the label as a second stream at the top
attaches to wrappers and reads at 0.69. Backprop's 0.912 against the
search's 0.84 is what top-down credit is worth.

Next: give the upper layer *what goes with each other* (carve from a pair
tally over lower units, refuse single-member hires).

Full writeup: [`search_mlp/README.md`](search_mlp/README.md).

---

## `ownership/` — one layer, a search over configurations, ownership by strength

The search is the forward pass and the search is over configurations: a set
of templates, each pixel owned by the sharpest one on it, cost = unexplained
energy + price per template, beam of 4 over add / drop / swap, stop when
nothing shortens the description. Nothing fixes how many are on. Three
learning rules from the winning configuration, same search, price 0.02:

```
                    pixels/template   on/image   unexplained   cost    tally
order (residual)          77             2.2        0.273     0.317   0.845
hard  (own pixels)        14             8.9        0.254     0.432   0.867
soft  (split by weight)   41             4.8        0.268     0.364   0.856
```

**Parts fall out of ownership with no locality imposed.** But the hard rule
carves to the price's resolution (7 / 14 / 27 px at price 0.01 / 0.02 / 0.04),
not to the data's, and its vocabulary costs MORE under the objective than
whole digits: the "toward zero on lost pixels" term over-carves, and the
objective's own gradient only touches owned pixels. The order vocabulary is
57% wholes and 43% corrections and is the cheapest description; soft gives
strokes, the vocabulary a second layer could compose from. Backprop's layer
under this search leaves 73% unexplained even at zero price: a dense code
cannot exist under ownership. Beam 1 / 4 / 16 cost 0.4251 / 0.4198 / 0.4189;
best-first never beat beam 4 and never proved it. Next: the owned-only rule,
corrections per image, own-fit gating.


**Second round in `ownership/`:** the EM reference (search alternated with
the exact best templates, six passes) says the objective at price 0.02 wants
wholes plus strokes with three on per image; the online order rule is within
0.027 of its own optimum (0.317 vs 0.290), the hard rule's dots are a basin
EM cannot leave (0.343). Names priced at −log2 usage spread reuse across many
parts (usage entropy 6.9 / 7.8 bits), but taking the noise level from the
residual is a runaway: the price fell to 0.003 and templates on rose to 12-18
per image over the pass, the opposite of the practice effect. Owned-only
online rules lost to order (0.365, 0.405); ownership by fit inside the search
is disqualified (synthesis peeks at the input). Running: usage-priced names
with the mean price held at 0.02, and owned with per-pixel counts.
Follow-ups: usage-priced names at a fixed mean price are benign (0.310 vs
0.317, same vocabulary, no runaway, no practice effect); per-pixel counts take
the owned rule from 0.365 to 0.342, still behind order. Every pixel-sharing
rule drifts online toward a parts basin; the objective at 0.02 prefers the
wholes basin; the residual rule is the one that stays there.

Full writeup: [`ownership/README.md`](ownership/README.md).

---

## `tower/` — two layers, one search over the tower, the label at the top

Layer 1 = this morning's stroke vocabulary, layer 2 = groups over layer-1
identities plus the label, both read by the configuration search, the top's
groups expanded down to make expected units cheap, names priced by usage.
Held out: label read at the top 0.611, tally on layer-1 identities 0.843;
feedback on and off identical to three decimals. **Layer 2 is 82% wrappers**
(members median 1, label share 0.08): the top renames the bottom, so the
label cannot climb and the feedback has nothing to say. Third appearance of
the same wall (09-12 encoder, search_mlp L2/L3, here): a template learning
the average of what accompanies it erodes to its most reliable member, and
selection from above cannot cure it. Next, and only this: carve layer-2
groups from the pair counts over layer-1 identities, no single-member hires.

Full writeup: [`tower/README.md`](tower/README.md).

---

## `settle/` — two layers settling together, no beam

The user's picture: every unit starts a little on, units explaining the same
thing inhibit each other, the threshold is the price, the top's expectation
is extra drive, both layers relax together, the units on at the end learn.
Layer 1 = the frozen stroke vocabulary; layer 2 groups by three rules. Four
seconds of training per arm.

```
                     label at top   tally L1   wrappers   members   label share
avg (no 1-member hires)   0.749       0.834       34%        1.6        0.66
gated by own fit          0.749       0.835       17%        1.7        0.68
carve from pair counts    0.620       0.847        0%        2.4        0.37
beam tower, earlier       0.611       0.843       82%        1.5        0.08
```

Wrappers fell from 82% to 17% and to 0%. The label climbed to 0.75 and still
stops below the tally on the strokes (0.84): a group is a hard vote from two
coarse units, the tally a soft vote from four. The carve's groups are the
first honest groups in three days and the worst label readers. Top-down drive
lifts the label at the top by a point and turns strokes off below. Next: groups
chosen with the label in the counts from the start.

Full writeup: [`settle/README.md`](settle/README.md).

---

## `neurons/` — whole-image cells, settling, learn what you saw

The user's proposal: no residual; after settling, each active cell learns the
input it fired on, thresholds held by homeostasis. Result: clean whole-digit
prototypes (249/256 wholes), overlapping 0.79 on average, and the settling
breaks on them (unexplained 3.3, activities thrash, some images end with no
cell on). The residual rule beside it: noisy wholes, unexplained 0.29-0.41.
With whole-image cells neither rule produces parts. Next: limited receptive
fields, the anatomical answer.

Pearson + geodesic rotation on the same cells: still wholes (242/256),
overlap 0.62. A shared weight budget per input line (share and divide, never
assign): no dither (smoothness 0.86), overlap 0.38, blow-ups gone, cells
thinner with a negative surround, still wholes (203/220), 15% dead; the
budget's scale cancels against row normalisation, only its profile acts.

**Share the input, then scale (user's design, `neurons/share.py`):** raw
pixels, nonnegative weights, a total-weight budget per template and a
shared budget per pixel, after settling each input pixel divided among the
cells on by their claim, each cell moves toward its share, then both budgets
enforced by scaling. 144 cells: **all strokes** (0 wholes / 136 strokes / 6
dots, 30 px, smoothness 0.80), unexplained 0.24 (best of the day), five on
per image, 1% dead, every held-out rebuild legible. No windows, no residual,
no rotation, no Pearson, no winner picked.
**Two layers of share-then-scale with the label at the top** (`neurons/tower2.py`, now `share_tower/tower.py`):
layer 1 strokes again; layer 2 94% wrappers, label share 0, label read at
the top 0.38, generations one or two strokes. Same mechanism as the strokes:
the smallest piece of a discrete code is one line, and one layer-1 line
(3.5% rate) already meets a layer-2 cell's 3% rate target. A cell is a
conjunction only when it is rarer than its inputs; the layer-2 rate must sit
well below the layer-1 rate. Not run.

Full writeup: [`neurons/README.md`](neurons/README.md).

---

## `windows/` — layer 1 as a grid of windows

Cells that each see only one window of the image (12x12 windows at stride 4,
12 cells per window, no weight sharing), Pearson matching, settling, every cell
on after settling rotates toward the patch it saw, thresholds hold each cell's
firing rate. Built as the "limited receptive fields" step named at the end of
the first `neurons/` round; the line then went on without windows. No writeup
was kept. Held out, at target rates 0.04 / 0.08 / 0.15: 10.3 / 16.8 / 21.1
cells on per image, 61% / 48% / 45% of inked windows left silent, patch
variance unexplained 0.78 / 0.73 / 0.72 (`results/layer1_r*.json`).

Full writeup: [`windows/README.md`](windows/README.md).

---

## `share_tower/` — two layers of share-then-scale, settled drawings

Moved out of `neurons/`. Top-cell pictures and generations are now settled
(layer 1 relaxed under the expectation, then painted), not weighted sums.
A top cell puts 79% of its weight on one stroke and 11% on a second; settled
under it, only the main stroke comes on. Label read 0.38, generations one to
two strokes. Same mechanism as the strokes: the smallest piece of a discrete
code is one line, and one line already meets the top's rate target. Lever:
layer-2 rate far below layer-1 rate (not run).
Two phases (competition without the label, then the top learns free with
the label, layer 1 still sharing): label read at the top 0.38 -> 0.70, 100%
covered; top cells = one stroke plus a halo plus a label mix (1.5 bits);
generations = one typical stroke per class. The vote works; no cell is a
conjunction; more voters is the read lever.
Label as a random sparse pattern: OR-ed onto layer 1's lines, the top cells
become whole-digit groups and the generations from a class pattern are real
digits (first of the day), but the read drops to 0.47-0.62 with 18-32% of
the top dead, since the pattern is absent at read time. On 100 lines of its
own (concat), the read is 0.71 but the top cells are single labelled strokes
again. OR composes, concat reads.
Top sharing in phase two: each label bit went to a cell of its own (OR: 37-48%
of the top dead, read 0.24 / 0.08; concat: label-bit cells never fire without
the bits, stroke cells carry no label, read undefined). Under sharing a cell
holds one line unless one line cannot clear its threshold.

Full writeup: [`share_tower/README.md`](share_tower/README.md).
