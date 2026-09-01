# 2026-09-01 / 09-02 — attention, depth, and what counting can do

A long day. It began by trying to build the attention loop that yesterday's
walls seemed to demand, and ended having established something narrower and
sturdier: **one wide layer of competing templates plus counted tables is the
best thing this project has, and nothing built on top of it has beaten it.**

Twenty-odd experiments, in four folders: `scenes/`, `experts/`, `fashion/`,
`stack/`. Every number below is on this repo's MNIST split (12,000 train /
3,000 test carved from the 60k training file) unless stated.

---

## The headline numbers

```
MNIST     0.9730    fully counted, no gradients anywhere
                    L1, 800 templates, counted table, 40,000 training images
          0.9770    counted table + ONE trained linear layer on per-cell evidence
          0.9840    ceiling: trained linear layer on the raw pooled code, no table
Fashion   0.8447    L1, 400 templates, counted table, magnitude in 4 bins

controls
  count raw pixels, no layer          MNIST 0.8343   Fashion 0.7330
  logistic regression on pixels       MNIST 0.9074   Fashion 0.8512
```

The middle MNIST row is probably the most honest "what this system can do": the
features and the evidence are counted in one pass, and only the final combination
is fitted.

The architecture, in full:

```
patches     every 5x5 window, stride 1        -> 24 x 24 = 576 positions
            centred, L2-normalised; flat ones dropped (~331 survive)
templates   400-800, shared across positions, competing
            winner-take-all; ONLY the winner learns
            winner moves toward the patch, step = max(1/n, floor)
message     one winner index + that patch's contrast, per position
            = 576 x 400 slots with ~331 active (a 294x EXPANSION of 784 pixels)
table       T[template, coarse cell (6x6), class]
            = log P(template t in cell c | class) / P(t in c)
            counted in one pass; no gradients anywhere
read        sum the ~331 lookups, take the largest of the ten
pressure    the table's class belief biases which patches each template wins
            during training (beta = 1.5)
```

Total: ~10,000-20,000 template numbers plus a 144,000-360,000 entry table.

---

## `scenes/` — can one pass answer a question the image aims?

Three-digit scenes, split by arrangement, to test whether the attention loop
that yesterday seemed to demand is needed at all. Position does the binding for
"which is leftmost": an MLP scored 0.8797 on trained arrangements and 0.8793 on
never-seen ones. The anchored question ("what is left of the 7?") did not
collapse (0.6799 seen, 0.6311 held out) but sat on its recognition ceiling, so
it says nothing. The oracle control showed one identity out of 300 is worth 55%
of a ten-way digit call, thirty of them 74%, a graded code 94%. The attention
loop was never built.

Full writeup: [`scenes/README.md`](scenes/README.md).

---

## `experts/` — aggregation, basis-pinning, and pressure

30 experts of 8 templates on 5x5 patches. Putting the label into the patch
competition was catastrophic (0.4143 -> 0.1543). Then, without touching the
experts, six ways of combining ~576 per-patch opinions went from 0.41 to 0.97;
position alone was worth 39 points. Training twice from different starts showed
only non-negative templates and winner-take-all competition pin a basis
(0.7813 and 0.7140 against a 0.4756 chance line); sparsity and labels do not.
Letting the counted table bias which patches each template wins gave +3.6
points on match-against-average, saturating at beta=1.5 — the one intervention
that reliably helped. Table and templates learning together in one pass is five
times faster, but a table accumulated while templates move must be rebuilt
before judging anything.

Full writeup: [`experts/README.md`](experts/README.md).

---

## `stack/` — depth, feedback, readouts, controls

Six layer-2 designs all lost to layer 1 alone; summing both tables adds +0.17
on MNIST and nothing on Fashion. Five feedback designs, none helped (the exact
counted table: -0.0002 over three seeds), because appearance-similar templates
have very different class evidence. Counting raw pixels with no layer scores
0.8343 / 0.7330, so the template layer is worth +13.6 / +9.6 points. The readout
ladder puts the ceiling at 0.9840 and shows summing the per-cell evidence costs
0.70 points. On 09-02: split-MNIST shows all the forgetting is in the table and
none in the templates, and a per-cell linear readout lifts Fashion from 0.8447
to **0.8989** (above logistic's 0.8512) and MNIST to **0.9869**.

Full writeup: [`stack/README.md`](stack/README.md).

---

## `fashion/` — the pipeline on Fashion, and brightness channels

The MNIST pipeline moved to Fashion unchanged, then (09-02) four counted
brightness channels laid beside the champion. Nothing, at any weight, on either
dataset: 0.8440 anchor against 0.8437 with the channels combined. The
brightness the pixel control priced is already carried by which positions are
live, in which cell, at which contrast bin.

Full writeup: [`fashion/README.md`](fashion/README.md).

---

## `pooling_lab/` — an interactive kernel and pooling playground

Not an experiment: a small window for looking. It shows a picture
(`cat.jpg`, greyscale), the same picture after one kernel is swiped over it,
and that result after pooling; everything recomputes as you move a control.
No results.

Full writeup: [`pooling_lab/README.md`](pooling_lab/README.md).

---

## The laws this day produced

**A counted table is exquisitely sensitive to what gets to shout.** Four
separate times, extra information *lost* points at full weight and *gained* them
at a small weight or as a separate index:

```
pairs, free vote               -3.2      at weight 0.05   +0.1 / +0.6
top-3 reporting                -0.5      (never recovered)
magnitude by multiplication    -2.6      by BINNING       +0.8
multi-scale by averaging       -1.4      per-scale weights untested
```

Conditioning is safe; scaling is dangerous. Anything whose range is wide will
dominate a sum of log-ratios.

**Identity coding is vector quantisation, so dimensions are not information.**
A one-hot over 400 carries 8.6 bits whether written as one integer or 400 slots.
L1's 294x expansion of the input adds no information — it makes the existing
information linearly accessible, which is why counting works on it (0.97) and
not on pixels (0.83). The expansion is worth doing **exactly once**; after it,
every further quantisation is pure loss.

**The label belongs above, not inside.** Three measurements: it must not enter
the patch competition (-26 pts); the expert's own label block loses to its bare
identity (-5.7 pts); and at K=1 it makes no difference at all to the templates
(0.9627 vs 0.9613), because a single unit vector has no subspace for the label
to bend.

**Competition creates identity — now with a number.** 0.7140 cross-seed
reproducibility against a 0.4756 chance line.

---

## Methodological notes

**GPU non-determinism is ~0.3-0.4 points.** `cupyx.scatter_add` uses float
atomics, so identical seed and code give different results run to run (0.9667 vs
0.9703 for one configuration). Two confident conclusions today were noise:
"feedback helps, +0.50" and "expansion peaks at 400 templates". Both reversed
under a fixed split and three seeds.

> **Any difference under about half a point needs a fixed test split and three
> seeds, or it is not a finding.**

**Class purity is inflated by near-dead units.** The 180-template run reported
purity 0.350; the busiest 60 templates sat at 0.158 (chance 0.10) while the
rarest 60 won **two patches each** and scored 0.539 by arithmetic. Always report
purity alongside win counts.

**Batch size, not seeding, drove the starvation.** 40% of templates were
starved with 2,048-patch batches (11 per template, against a MIN_S gate of 4);
the co-adaptive loop's 64-image batches (~21,000 patches) left 16 of 180 dead on
MNIST and **0 of 180** on Fashion.

---

## What is open

- **the attention loop is still unbuilt.** Part 1 never cleared its recognition
  floor, so the premise that motivated the whole day is neither supported nor
  refuted
- **configuration C** — the pure chain `L1 <-> L2 <-> class` with no L1-to-class
  shortcut. In every test so far layer 1 could bypass layer 2, so layer 2 never
  had to carry anything
- **width and data have not flattened.** 1,600 templates and the full 57,000
  images are the cheapest untried gains
- **a fast/short-term component on the table**, so it is modulated by context
  rather than merely indexed by it. Blocked by data: `T_down` already has 6.4M
  bins and under one count each
- **class-clustered features.** Every code we built is *discriminable* but not
  *clustered* — nearest-mean trails a linear probe by 4.5 to 6.8 points
  throughout, because classes are stretched cigars rather than balls, and
  reconstruction has no reason to shrink the directions that don't matter
