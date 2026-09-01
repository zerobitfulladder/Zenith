# Competing experts on 5x5 patches: aggregation, basis-pinning, pressure

2026-09-01. Parts 2-4 of the 09-01 day writeup, moved here verbatim. Many of
the later experiments (depth, feedback, readout extensions, controls) also have
their scripts in this folder; their writeup is in
[`../stack/README.md`](../stack/README.md) — see the script table at the end.

## Part 2 — the aggregation discovery (`experts/`)

30 experts of 8 templates on 5x5 patches, trained jointly with the label.

**Putting the label into the competition was catastrophic**: image accuracy
0.4143 at lambda=0 against **0.1543** at lambda=1. The mechanism is exact —
training chose winners using ink *and* label, reading has no label, so the
partition could not be reproduced. Refined into a law:

> the label may enter the competition only as far as the sensory block can
> reproduce that same choice alone

(Yesterday's whole-image joint k-means worked *because* 784 pixels are
class-informative; a 5x5 patch is not — per-patch class accuracy 0.1162 against
0.10 chance.)

### Then the aggregation, which was the real finding

Same experts, same weights, six ways of combining ~576 per-patch opinions:

```
1  sum of the experts' own label rebuilds            0.4143
2  log-ratio bag, position-free                      0.5460
3  log-ratio conditioned on a 4x4 cell               0.9423
4  same, only the 40 most diagnostic patches         0.8207
5  pooled expert map + trained linear probe          0.9717
6  pooled label-opinions + probe                     0.9150
   reference: the same probe on 64 k-means templates 0.9817
```

**0.41 to 0.97 without touching the experts.** Three things fell out:

- **position is worth 39 points** (0.5460 -> 0.9423)
- **max is the wrong instinct**: restricting to the 40 most diagnostic patches
  *costs* 12 points. The evidence is spread thin, and the log-ratio already
  silences uninformative patches automatically (they score log 1 = 0)
- rule 1's confusion matrix shows the failure mode: class 2 recall **0.010**,
  everything collapsing onto a couple of attractors, because 500 near-flat
  opinions drown the few informative ones

**Experts are good reporters and bad interpreters.** Reporting what they *saw*
(their index) beats reporting what they *meant* (their label block) by 8 points.

---

## Part 3 — what actually pins a basis (`rotation*.py`)

The morning's claim was that dense codes have no identity because the objective
is rotation-invariant. Tested directly: train twice from different random
starts, match the two template sets one-to-one by cosine, against the
similarity of two *random* sets.

```
chance line (two random sets of 100 directions in 25 dims)   0.4756

non-negative TEMPLATES, raw patches                0.7813   +0.306   PINS
winner-take-all competition                        0.7140   +0.238   PINS
non-negative coefficients, raw patches             0.4895   +0.014
sparse coding, best sparsity (4.7 of 100 active)   0.4882   +0.013
non-negative coefficients, centred patches         0.4819   +0.006
unconstrained (control)                            0.4651   -0.011
LABELS in the input vector                         0.4130   +0.002
```

The unconstrained control rebuilds almost perfectly (error **0.0027**) with
templates that are **at chance** — the sharpest statement of *crisp rebuilds are
not evidence* this project has.

Three of my hypotheses were refuted by this table: sparsity does not pin a
basis (flat across the whole range, including 4.7% active), non-negative
*coefficients* do not either, and the opposite-polarity escape I proposed does
not exist (zero anti-pairs, minimum cosine -0.63).

**Two things pin a basis, and both give each template its own territory in the
data**: non-negativity (a template can only ever *add*, so it must correspond to
something present) and competition (a template becomes the average of a cluster
it owns). Everything that fails constrains the *collective* — reconstruction
cares only about the span; the label readout is linear so rotations pass
straight through; sparsity constrains the shape of the code, not which template
owns which data.

Labels in the input steer *which subspace* you land in and say nothing about the
axes drawn on it. This was already proved in this repo's own comment:
`(QW)_lab^T (QW)_img = W_lab^T Q^T Q W_img`, and the Q's cancel.

---

## Part 4 — pressure, shaping, co-adaptation

### The one intervention that reliably helped

Let the counted table bias **which patches each template wins** during training,
while leaving the learning rule untouched:

```
winner = argmin( ink error  -  beta * sum_c q_c T[t, cell, c] )
```

```
beta      0     0.75    1.5     2.5     4.0     6.0
match-avg 0.9100 0.9380 0.9460 0.9450 0.9453 0.9477
linear    0.9767 0.9780 0.9810 0.9787 0.9783 0.9800
```

**+3.6 points on match-against-average**, saturating at beta=1.5, never
destabilising even at four times the useful pressure. And the *trained probe*
rose too (0.9767 -> 0.9810), which is the check against circularity: if the
templates were merely absorbing the classifier's opinion, a probe couldn't
improve, since it already had that opinion.

Why this worked when four other attempts to bend the fitting failed: **the
learning rule is untouched**. A winner still moves toward the patch it won,
undistorted. Only the assignment changed — and the same procedure runs at
training and at reading, so no partition exists that can't be reproduced.

### Read-time shaping, and a warning about the metric

Feeding the class belief back to reshape the map at READ time (never letting the
map re-decide) gives, for free:

```
             accuracy   code separation
feedforward   0.9423        +0.119
beta 1.0      0.9423        +0.297
```

But the readout ladder showed the code separation flatters interventions that
inject the answer:

```
                 nearest class mean   1-NN     linear    mlp
feedforward           0.8957         0.9717   0.9717   0.9740
shaped, beta 1.0      0.9237         0.9677   0.9687   0.9697
```

Shaping moved the separation 2.5x and nearest-mean only 2.8 points, while
*costing* a little at the higher rungs. And on images where the injected belief
was wrong, matching against class averages recovered the true class **17.9% of
the time either way** — shaping neither helps nor hurts there, it inherits the
decision's errors.

**Nearest-mean accuracy is the honest measure of a "stable class code"; the
similarity gap is not.**

### Co-adaptation, and the table-staleness trap

Templates and table learning together, from random templates and an empty table,
one pass:

```
co-adaptive, 1 pass, from scratch   table 0.9580  match-avg 0.9487   151 s
alternating, 6 rounds, pre-trained  table 0.9540  match-avg 0.9460   757 s
```

Five times faster, no pre-training, same or better. But the **online**
accumulated table read only 0.66 while the same weights with a **rebuilt** table
read 0.9587. Three separate diagnoses of mine were wrong before the fourth
stuck; the practical rule is simply:

> a table accumulated while its templates are still moving is a mixture of
> statistics about templates that no longer exist. Rebuild it before judging
> anything.

Count decay did **not** help (no decay 0.9487, decay 0.999 0.9463, decay 0.995
0.9430) — it traded staleness for noise. Annealing the template step
(`eta = max(1/n, 0.02)`) lifted the online table from 0.6467 to **0.8947** after
one epoch, so "the templates never settle" was a real factor.

## Scripts

Which part of the day writeup each script belongs to, going by its docstring.
Everything writes to `results/`.

| script | what it asks | written up in |
|---|---|---|
| `experts.py` | 30 experts of 8 templates on 5x5 patches, trained jointly with the label | Part 2 (here) |
| `vote.py`, `probe.py`, `nolabel.py`, `patterns.py`, `misread.py` | how to gather ~576 per-patch opinions into one class call; the readout ladder; did the label in the templates matter | Part 2 (here) |
| `rotation.py`, `rotation2.py`, `sparsity.py`, `compete_rot.py` | what pins a basis | Part 3 (here) |
| `pressure.py`, `pressure2.py`, `weighting.py`, `shaped.py`, `coadapt.py` | pressure from the table, read-time shaping, table and templates learning together | Part 4 (here) |
| `layer2.py`, `layer2_save.py`, `conv2layer.py` | layer 2 over the pooled layer-1 code | Part 5, [`../stack/README.md`](../stack/README.md) |
| `settle.py`, `stack.py`, `stack2.py` | feedback from layer 2 / the class down to layer 1 | Part 6, [`../stack/README.md`](../stack/README.md) |
| `pairs.py`, `blank.py` | readout extensions: pairs, class codes, absence | Part 7, [`../stack/README.md`](../stack/README.md) |
| `pixels.py` | the control: count raw pixels, no layer | Part 8, [`../stack/README.md`](../stack/README.md) |
| `single.py`, `capacity.py`, `expand.py`, `posexp.py` | one hypercolumn per patch; budget split; widening layer 1; experts over [ink ; position ; label] | the laws and methodological notes in the day README |
