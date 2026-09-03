# Two rungs on Fashion-MNIST: a stroke layer read by an object layer

2026-09-03. Single arm. The question is stacking: does an object layer over
the pooled stroke code beat reading the stroke layer directly?

```
L1     9x9 patches, 400 templates, top-1 per position, cnt/n step, no class
       pressure; trained without labels on all 12k images, 3 epochs, frozen
pool   winners into 4x4 cells, one value per (template, cell) = max match
       score on collision; 6400-long code, ~172 entries on per image
L2     400 object templates over the code, cosine, no centring; label 0.25 +
       belief 0.5 in the competition, purity step, hire on error; 20 epochs
read   L2 winner's table row plus the belief
```

Fashion 12k/3k, seeds 7 and 8.

```
                                      seed 7   seed 8    mean
L1 alone, 4x4 per-cell table read     0.8027   0.8027   0.8027
L2 recount                            0.8163   0.8103   0.8133
L2 online table                       0.8167   0.8120   0.8143
dead L2 templates                          0        0
hires during training                   4525     4542
downward residual, read right / wrong  0.18/0.26  0.18/0.26
```

- **The object layer beats reading the strokes directly, by a point.** The
  August two-rung stack on MNIST (k-means in the upper layer) fell *below* the
  L1 table; with today's object-layer machinery the upper layer is above it on
  Fashion, and its online table equals its recount.
- **It is 7 points below the per-cell linear probe** on the same kind of L1
  (0.882 at 12k, `2026_09_01/stack/percell.py`). The object layer
  is a nonlinear per-cell readout, and 400 prototypes over a 6400-long code do
  not yet match one fitted linear map. Template count at L2 is the first knob
  the dimensionality law points at.
- **Hiring churns.** ~4500 hires over 400 templates in 480 batches: Fashion is
  read wrong one time in five, every misread class hires every batch, and the
  cheapest template is re-hired about eleven times over. It still ends with
  no dead templates and a table that matches the recount, but on a stationary
  distribution most misreads are ambiguity, and the rule as written treats
  every one as novelty.
- **The downward residual separates.** Cells where the L2 winner's most
  expected L1 identity is absent: 18% of cells on correctly read images, 26%
  on misread ones, both seeds. That is a usable message from L2 to L1, logged
  here and not yet used.

## L2 template count: flat

`--k2 400 / 800 / 1600`, same everything else, 2 seeds.

```
L2 templates        400      800     1600
L2 recount       0.8203   0.8200   0.8173
seed 7 / 8    .818/.822  .815/.825  .813/.822
dead                  0        0        0
hires              4552     4462     4150
```

Template count is not the lever. And note the run-to-run noise: the same
seed-8 run read 0.8103 in the first run and 0.8223 here (GPU scatter order
changes tie-breaks in the code), so differences under a point between these
columns mean nothing. The stack sits at 0.82 against the L1 table's 0.80 and
the per-cell linear probe's 0.88, at any width.

What is left between 0.82 and 0.88 is therefore the read, not the capacity:
top-1 identity at L2 throws away the graded information that a fitted linear
map keeps, and the hire rule churns on ambiguity. Untested: top-k read at L2;
hiring only when no template of the true class was close; L2 over the L1
table's per-cell evidence map (360 numbers, what the probe reads) instead of
the raw 6400-long code.

## Read, hire, input: none of the three

`--topk`, `--strict`, `--evidence`; 400 L2 templates, 2 seeds. Baseline is
top-1 read, plain hire, raw code: **0.8203**.

```
                                     L2 recount   hires   note
top-3 read                             0.7910     4514
top-5 read                             0.7692     4542    reading more object templates hurts
strict hire, top-1 read                0.8120     4419    same churn, same accuracy
strict hire, top-3 read                0.7638     1970
evidence-map input                     0.8178     4164    = raw code, within noise
evidence map + top-3                   0.8118     4145
evidence map + top-3 + strict          0.7853     1702
```

- **Top-k hurts here**, 3 points at k=3 and 5 at k=5, where at whole-digit
  scale on pixels it helped. The second-nearest object template over the
  pooled code is a different object, not more evidence for the same one.
- **Strict hiring changes nothing.** "True class below chance in the read"
  filters almost no hires with a top-1 read, because a single row's belief is
  peaked; with a top-3 read it halves the hires and the accuracy falls with
  the read, not the hiring. Plain hire at ~4500 per run is not what limits
  this layer.
- **The evidence map is the raw code**, 0.818 vs 0.820. Twenty times fewer
  inputs, same answer: the object layer reads the same information either way.

So the stack sits at 0.82 whichever way it is read, fed, or hired, against
0.80 for the L1 table and 0.88 for the fitted linear map on the same evidence.
What the linear map has that no arm here has is a weight per (cell, class)
learned from errors, not a prototype. That is the gap.

## The message down: L2's residual guiding L1

`joint_rung.py`. Both layers train together. For every image L2 misreads with
a committed winner, the residual cells are found (the L2 winner's most-expected
L1 identity is absent from the code) and the patches at those positions are
handed to L1 in one of two ways: `residual` -> the nearest L1 template that is
uncommitted at that cell (allocate a new stroke identity there); `expected` ->
the identity L2 expected learns the patch it should have won (attraction only,
one template, one patch). 2 seeds, 20 epochs.

```
arm          L2 recount   L1 table   patches reassigned per run
frozen         0.8160      0.8027           0            L1 frozen (= two_rung.py)
learns         0.8177      0.8040           0            L1 keeps learning, no message
residual       0.8167      0.8030     ~5,000,000
expected       0.8185      0.8032     ~4,700,000
```

**Nothing.** Four arms inside the one-point noise band, the L1 table unmoved
at 0.80, and five million reassigned patches per run (about 5% of all patch
events) with no effect on either layer. The residual separates (0.20 vs 0.28
in every arm) but acting on it at L1 changes neither what L1 becomes nor what
L2 reads. On 9x9 Fashion strokes, as on MNIST, the stroke vocabulary is
already complete for what the object layer can use; the object layer's limit
is its own read, not its input.

## The message that works: SoftHebb's contrast code upward

`--contrast`, `--contrast-std`. After reading Journé et al. 2023: the stroke
layer is untouched, but what L2 receives is every template's match score
minus the mean over all templates at that position, rectified (mean-centre
THEN ReLU), max-pooled into the 4x4 cells. Dense: ~4600 of 6400 entries
nonzero. `std` standardises each feature on the training set and lets L2
match with a signed cosine. 400 L2 templates, top-1 read, plain hire, 2 seeds.

```
message to L2                              L2 recount   seeds          online   dead
identity of the winner, max score (base)     0.8203    .818 / .822     0.8170     0
contrast, raw, cos^2 matching                0.8390    .839 / .839     0.8395     0
contrast, standardised, signed matching      0.8462    .849 / .844     0.8445     0
```

**+2.6 points, the first stacking gain from anything today**, and both seeds
sit clear of the one-point noise band. Width, read, hiring and input form all
left the identity message at 0.82; changing the message moved it. What the
object layer needed was not the winner's name or its match strength but "which
templates matched *more than the rest*" at each position — the shared "how
well does this patch match anything" component removed. That is the August
note ("identity messages cannot express what the next layer needs") answered:
the next layer needs contrast.

The gap to the fitted per-cell linear probe (0.882) is now 3.6 points instead
of 6, with a counted read and no gradient anywhere. Untested on top of this
message: L2 width (the message is dense now, so the width law may finally
bite), top-k read (L2 neighbours over a contrast code may be the same object),
and the split.

## Width and top-k, retried under the contrast message

`--contrast-std` with `--k2` and `--topk`, 2 seeds.

```
L2 templates      400      800     1600     3200     6400
L2 recount     0.8462   0.8458   0.8597   0.8590   0.8610
seeds        .849/.844 .846/.845 .856/.864 .857/.861 .861/.861
dead              0        0        0      731     3877

read at 400    top-1    top-3    top-5
               0.8462   0.8382   0.8280
read at 800             0.8397
```

- **Width pays under the dense message and saturates at ~0.86.** The width
  law was dormant under the identity message (§ above: 400/800/1600 flat at
  0.82) and bites here: +1.4 from 400 to 1600, both seeds clear of noise. Past
  1600 nothing: at 6400 more than half the templates never win. With 12k
  training images and ~2500 live object templates each template stands for
  about five images, which is nearest-neighbour territory; the next lever on
  this axis is data, not width.
- **Top-k still hurts** (-0.8 at three, -1.8 at five), under the contrast
  message as under identity. The second-nearest object template is a
  different object, whichever message it was matched on.

Standing: L1 table 0.803, object layer 0.861, fitted linear probe 0.882 (12k)
and 0.899 (40k). The stack has gone from 6 points under the probe to 2, with a
counted read and no gradient anywhere. Untested: 40k data; the split under the
contrast message; contrast against the nearest few templates instead of the
global mean.

## A linear probe at the end, like the paper's head

`probe.py`. Softmax regression trained by gradient on frozen features
(standardised, SGD + momentum, weight decay, 100 epochs, final-epoch test
accuracy), at three depths, next to the counted reads. L2 at 1600, 2 seeds.

```
                          counted    probe
L1 per-cell evidence (160)   0.8027   0.8747      the 09-02 probe (0.882 there, 3 seeds, 12k)
L1 contrast message (6400)      --    0.8937      "one Hebbian layer + head", the paper's protocol
L2 object layer (1600)       0.8572   0.8603      "two layers + head"
```

- **The paper's protocol reproduces on our stroke layer: 0.894**, the best
  Fashion number in the project at 12k, two points above the per-cell probe
  and within half a point of the 40k record (0.899). A fitted head over the
  dense contrast message is strong, exactly as they report, and the message
  costs nothing to make: same templates, centre then rectify.
- **Our object layer is a good classifier and a poor feature layer.** Probed,
  its output reads 0.860, no better than counting it (0.857) and 3.3 points
  *below* probing its own input. A layer of prototypes, however well it
  counts, hands the head less than it received. That is the paper's central
  measurement, hard-winner layers read worse than what feeds them, now on our
  own rig. It also says where the stack's 2-point gap to the probe lives: in
  the quantisation at L2, not in L1.

So two readouts, two verdicts, side by side: for a counted read the object
layer is worth +5.5 over the stroke table; for a fitted head it is worth -3.3
under the stroke message. Which one the project wants is a choice about
gradients and streams, not about accuracy.

## Files

| | |
|---|---|
| `two_rung.py` | the rig. `--smoke` |
| `joint_rung.py` | both layers training, with L2's residual reassigning L1 patches (`residual`, `expected`) |
| `probe.py` | linear probe at three depths next to the counted reads. `--k2` |
| `results/two_rung_*.json`, `k2_sweep.log`, `read_sweep.log` | per seed, per L2 width; `two_rung.log` is the first run, which included a split that was dropped |

    uv run python two_rung.py [--k2 800] [--topk 3] [--strict] [--evidence] [--contrast | --contrast-std]    # ~30 s
    uv run python joint_rung.py    # ~2 min
    uv run python probe.py --k2 1600    # ~45 s
