# Depth, feedback, readouts and controls on the counted stack

2026-09-01 / 09-02. Parts 5-9 of the 09-01 day writeup and two of the 09-02
addenda, moved here verbatim. Some of the scripts for Parts 5-8 live in
[`../experts/`](../experts/README.md) (see its script table); the GPU versions
and everything from Part 9 on are in this folder.

## Part 5 — depth. Six layer-2 designs, six losses

Every configuration we could think of, against layer 1 alone:

```
design                                templates  positions  rf     L2     L1 alone
whole image                              200         1      28px  0.9540  0.9650
13x13 pooled, sliding                    200        36      13px  0.9593  0.9650
8x8 pooled, 2x2 sub-regions             1000       121       8px  0.9490  0.9697
sparse-sampled 4 points, S=2            1000       484       7px  0.9547  0.9693
sparse-sampled 4 points, S=4            1000       400       9px  0.9533  0.9693
merged window, w=4                      1000       441       8px  0.9623  0.9697
merged window, w=6                      1000       361      10px  0.9620  0.9697
```

Receptive field 7 to 28 pixels, positions 1 to 484, templates 200 to 1000,
pooled and unpooled, sub-regions and merged. **All lose.**

Two structural observations came out of it:

**Merging beats sub-regions.** Superimposing a whole window into one 400-vector
(the L1 code is one-hot in depth, so 16 positions collide only ~27% of the time
and max-pooling keeps the stronger) gave +1.3 points at **a quarter of the
template size**. I had been paying for fine position inside a window that the
window's own location already supplied.

**The table is already a second layer.** It holds 144,000 learned parameters and
maps a 230,400-slot sparse code to a class. So "one layer" was already two
stages, and adding L2 inserts a **1-of-1000 bottleneck between two things that
were already communicating well**. Cortex passes population codes between
layers; we pass a single index.

### Does L2 add anything at all?

The right question, asked late: sum both tables rather than replacing one.

```
MNIST     L1 alone 0.9690;  + L2(w=6) at validated weight 0.2 -> 0.9707
          sweep 0.05..1.5 all >= baseline, peak 0.9723
Fashion   L1 alone 0.8353;  every combination <= baseline
```

+0.17 on MNIST (five images), nothing on Fashion. And a **scale sweet spot**
only on MNIST: 10 pixels adds, 8 and 12 add nothing. Digits have mid-scale
structure (junctions, crossbars); garments have information at the very local
(texture) and very global (silhouette) scale with little in between — Fashion's
L2 gets monotonically worse as the window grows.

---

## Part 6 — feedback. Five designs, none helped

```
whole-image L2 -> L1, projected        read-time +2.1 on the code, training -0.9
sliding L2 -> L1, projected            read-time +0.3, training -0.5
top-k mixture (5, 20) softening        no better; read-time benefit shrank
counted exact table T_down             single run +0.50 -> THREE SEEDS: -0.0002
```

`T_down[L1 template, relative offset, L2 template]`, counted, base-rate
normalised, **exact** ("you, at offset 2, should be template 87") rather than
blurred — the only version that didn't cost accuracy. Its log-ratios span
[-9.55, +7.14], so the connection carries strong constraints. Three seeds:
**-0.0002 +- 0.0013**. Nothing.

### Why: measured, not guessed

At 25% of positions the winner changes under feedback, and:

```
|cosine| between swapped-out and swapped-in template   0.754   (random pair 0.300)
class evidence change, relative to its own size        0.745
predicted class UNCHANGED at                           43.8% of swaps
```

The swaps are **not** informationally empty — they change the class vote at a
quarter of all positions, in both directions, and cancel exactly. The
feedback's information is *appearance* co-occurrence; the table's is *class*.
Those turn out to be uncorrelated, and **appearance-similar templates have very
different class evidence** — the map from template to class statistics is jagged,
so every appearance-plausible nudge is a lottery ticket.

Also: movement saturates instantly (32.1% at beta 0.1, 35.9% at beta 2.0). About
a third of layer 1's winners are near-ties that flip on any nudge; the other two
thirds have a decisive ink winner that no bias overcomes.

---

## Part 7 — readout extensions

```
                             MNIST              Fashion
pairs at weight 0.05      0.9650 -> 0.9663   0.8293 -> 0.8357
  (at weight 1.0)                  0.9333            0.8060
magnitude, weighted       0.9690 -> 0.9707   0.8363 -> 0.8103
magnitude, 4 bins         0.9690 -> 0.9653   0.8363 -> 0.8447
discriminability weights  nearest-mean 0.9100 -> 0.9283 (subsumed by pressure)
hierarchical class code   no gain at any ancestor weight, either dataset
```

**The Fashion class tree is the nicest artifact of the day.** Clustering the ten
classes purely by the similarity of their feature distributions, with no
semantics supplied:

```
{Coat, Shirt}                                0.994
{Pullover, Coat, Shirt}                      0.987
{Dress, Pullover, Coat, Shirt}               0.960
{T-shirt, Dress, Pullover, Coat, Shirt}      0.939   <- all clothing
{Sandal, Ankle boot}                         0.925
{Sneaker, Sandal, Ankle boot}                0.877   <- all footwear
```

A perfect clothing/footwear split, assembled in exactly the order those classes
confuse each other. It bought **no accuracy**, because the ancestor evidence is
computed from the same features as the leaf evidence and adds nothing when the
leaves aren't data-starved. Hierarchical codes should pay with many classes and
few examples each; not here.

---

## Part 8 — controls

**Counting raw pixels, no layer at all**, with the identical scoring rule:

```
bins        2       4       8      16
MNIST    0.8343  0.8340  0.8327  0.8277   <- FEWER is better
Fashion  0.6377  0.6953  0.7253  0.7330   <- MORE is better
```

The template layer is worth **+13.6 points on MNIST and +9.6 on Fashion** over
the same machinery on pixels. And the two datasets want opposite things, which
is the fourth independent measurement of the same split:

```
                                        MNIST        Fashion
position (bag -> cell-conditioned)      +39 pts       +19 pts
intensity/magnitude resolution          hurts         +0.8
L2 window growing                       peaks at 10px  monotonically worse
patch contrast spread (median/90th)     1.87/2.19      1.01/1.86
```

**Digits are arrangement; garments are material.**

**Scaling the data**, fixed test set:

```
train    K1    counts/bin   accuracy
12,000   400      27.4       0.9673
12,000   800      13.7       0.9700
40,000   400      91.5       0.9687
40,000   800      45.7       0.9730
```

800 templates beats 400 at both sizes; more data adds +0.14 to +0.30. Neither
axis has flattened.

---

## Part 9 — how much is the readout leaving on the table? (`readouts_ladder.py`)

argmax over the table's ten class scores is the cheapest possible readout. What
does a trained classifier get on top of the same evidence?

```
L1 + table, argmax                        0.9700
L1 + table + linear on 10 summed          0.9487    <- my bug, see below
L1 + table + linear on 360 PER-CELL       0.9770    +0.70

L2 + table, argmax                        0.9617
L1+L2 tables + linear on 20 summed        0.9593
L1+L2 tables + linear on 720 per-cell     0.9753    WORSE than L1's 360

probe on the raw pooled L1 code           0.9840    the ceiling
```

**The 10-number row is an optimisation failure, not a result.** A linear map
*contains* argmax (set W to the identity), so it cannot genuinely be 2 points
worse. Raw evidence sums have magnitudes in the tens to hundreds — totals of
~331 log-ratios — and feeding them unstandardised into a softmax trained with
Adam at lr 1e-3 is badly conditioned. Standardise before believing that row.

**Summing costs 0.70 points.** The table produces evidence at each of 36 regions
and we then add it all up, throwing away *which region voted for what*. A
classifier given the 360 per-cell numbers recovers most of a point — it can learn
that the top-left is more reliable for 7s than the bottom-right is. Cheap to
exploit and not yet exploited.

**And this is the fairest test layer 2 ever got.** A weight sweep can only scale
its evidence uniformly; this classifier can weight it per class *and* per region,
360 free parameters' worth. Given all that, adding layer 2 makes things **worse**
(0.9753 vs 0.9770) — more parameters, no more information. That is the eighth
independent way of asking, and the cleanest negative of them.

**Where the ceiling is:**

```
probe on the raw pooled code    0.9840   everything L1's output contains
table + per-cell linear         0.9770   -0.70   the summing
table + argmax                  0.9700   -1.40   + coarse cells, + independence
```

The counted table sits 1.4 points below what is extractable from layer 1's code,
and half of that is just the summing.

---

## Split-MNIST on the champion (`forget.py`) — templates never forget

Train the winner rig on digits 0-4 (3 epochs), continue the same weights and
the same accumulating table on 5-9, three seeds:

```
                                   old (0-4)      new (5-9)   all
after phase A                       0.9843
final, online table                 0.9767         0.0574     0.514
final, per-phase rebuilt counts     0.7152+-0.056  0.9793     0.848
final, full rebuild (= replay)      0.9736         0.9645     0.969
frozen after A, counts stream       0.9725         0.9634     0.9679
joint training (ceiling)            0.9734         0.9634     0.9683
```

**All of the forgetting is in the table; none of it is in the weights.** With a
full recount, sequential equals joint to the third digit — even though 54% of
the old data's winners changed under the final templates. The vocabulary
drifted and stayed exactly as good; what broke is that phase A's counts
describe indices that no longer mean the same thing (class 3 falls to 0.584,
absorbed by the new classes). The online table fails the other way round: phase
A's mass already owns the shared (template, cell) bins, so the NEW classes are
the ones that cannot be expressed (0.0574).

The reuse has structure: zero templates were dead after phase A, and 69% of
phase B's wins landed on A's *heavy* templates. 5x5 features are class-agnostic,
so new classes rent the old vocabulary instead of colonising free space —
the opposite of the dense-experts split, where new classes went to dead experts.

> The continual recipe this licenses: train the vocabulary on the first task,
> **freeze it, and let only the counts stream.** A frozen code keeps every old
> count valid forever — counting never forgets — and costs 0.04 points against
> joint training. A moving vocabulary costs 26.

## Per-cell readout + width x data on Fashion (`percell.py`) — the sum was the bottleneck

The two moves that paid on MNIST and were never run on Fashion, as one grid.
Champion 4-bin table, both readouts, 3 seeds, the usual fixed test set (extra
training images drawn from beyond the test block):

```
                       argmax (counted)     per-cell linear
fashion   400 x 12k    0.8434 +- 0.0014     0.8820 +- 0.0012
          800 x 12k    0.8476 +- 0.0023     0.8823 +- 0.0019
          400 x 40k    0.8461 +- 0.0016     0.8989 +- 0.0017   <- new best
          800 x 40k    0.8537 +- 0.0010     0.8939 +- 0.0004
mnist     800 x 40k    0.9743 +- 0.0007     0.9869 +- 0.0009   <- new best
```

**Summing costs 3.9 points on Fashion at 12k and 5.3 at 40k — five to seven
times its MNIST price (+0.70).** Fashion's classes disagree about where their
information lives (silhouette at the outline, texture in the interior), and the
flat sum averaged those witnesses together. One standardised linear map on the
360 per-cell numbers, nothing else changed, and Fashion goes 0.8447 -> 0.8989 —
**4.8 points above logistic-on-pixels (0.8512)**. The "loses to a linear
baseline" diagnosis is closed.

The two readouts want different resources. Argmax follows width (800 templates
is its best row, 0.8537); the per-cell probe follows DATA and is actively hurt
by width at 40k (0.8989 at 400 vs 0.8939 at 800) — doubling the templates
halves every (template, cell, bin) count, and the probe feeds on the quality of
those log-ratios. The statistics budget again: the readout can only be as good
as the counts beneath it.

And the MNIST bonus row: 0.9869 at 800 x 40k with the per-cell readout — above
the 12k "ceiling" probe on the raw pooled code (0.9840). The honest row
(counted features, counted evidence, one fitted linear map) keeps rising with
data; neither axis has flattened.

## Scripts

| script | part |
|---|---|
| `gpu_stack.py`, `gpu_sparse.py`, `gpu_merge.py` | Part 5, the layer-2 designs |
| `gpu_combine.py` | Part 5, does L2 add anything (both tables summed) |
| `gpu_bidir.py`, `swaps.py` | Part 6, the counted feedback table and why its swaps cancel |
| `magnitude.py` | Part 7, magnitude weighted and binned |
| `scale.py` | Part 8, scaling the data |
| `readouts_ladder.py` | Part 9 |
| `forget.py` | 09-02, split-MNIST on the champion |
| `percell.py` | 09-02, per-cell readout + width x data |
| `run.py` | the whole stack end to end on any dataset (`results/full_stack_fashion_mnist.json`) |
| `templates.py` | layer-1 templates, MNIST beside Fashion (`results/l1_templates.png`) |

All write to `results/`.
