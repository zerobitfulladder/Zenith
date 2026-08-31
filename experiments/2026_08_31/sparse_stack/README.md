# Layer one on a place code, against the 0.57 / 0.93 board

2026-08-31. The sparse column from `../sparse_column` put in as layers one
and two of `2026_08_30/stack3/stack3.py`, and asked the same two
questions: write the image and read the label, write the label and read the
image. Geometry, layer three, both reads and the judge are **imported** from
`stack3.py`, not reimplemented, so the endpoints are produced by the code
that produced them. rho fixed at 0.6.

**The baselines reproduce exactly** — dense 0.9232 and wta 0.5708, matching
08-30's rho=0.6 row to four decimals. The harness is faithful.

## The board

20k train, 5k held out, judge (logistic on unit-norm pixels) 0.9074.

| | accuracy | fraction of the L2 code lit | image→code→image | gen top-1 | gen graded |
|---|---|---|---|---|---|
| dense | **0.9232** | 0.4917 | 0.733 | 0.8 | 1.0 |
| wta (k=1) | 0.5708 | 0.0075 | 0.755 | 0.5 | 0.9 |
| sparse k=2 | 0.7132 | 0.0153 | 0.661 | 0.8 | 0.9 |
| sparse k=4 | 0.7494 | 0.0301 | 0.567 | 0.7 | **1.0** |
| sparse k=8 | 0.7646 | 0.0595 | **0.475** | **1.0** | **1.0** |
| **sparse k=4, settled read** | **0.7946** | 0.0289 | 0.637 | 0.8 | **1.0** |
| sparse k=4, no L3 centring | 0.7494 | 0.0301 | 0.567 | **1.0** | **1.0** |

## Identity is not free

The middle beats the codebook decisively — 0.571 to 0.795, most of the way
across — and **does not reach the dense arm.** The best sparse number is
0.795 against 0.923. That is a real 13-point price, and the earlier hope
that identity might come for free is answered: it does not.

The reason is visible in the second column. **Accuracy tracks how much of
the code is lit, not k:**

    lit 0.0075 -> 0.571     lit 0.0301 -> 0.749
    lit 0.0153 -> 0.713     lit 0.0595 -> 0.765     lit 0.4917 -> 0.923

Monotone, with sharply diminishing returns — doubling the density from k=2
to k=4 buys 3.6 points, k=4 to k=8 buys 1.5. Extrapolating, matching the
dense arm needs roughly its density, which is to say it needs to be dense.

That is 08-30's diagnosis, now with numbers on it: **layer three is a
correlation matcher over whole vectors and it wants numbers, not names.** It
never asks which template. So this run prices what identity *costs* against
a readout with no use for it, and none of what identity buys — indexing,
counting, one-shot writes, a gallery you can read — is on the table.

## But drawing and rebuilding go the other way

The sparse stack is a **much better autoencoder** than either endpoint —
0.475 at k=8 against dense's 0.733 and wta's 0.755. 08-30's own complaint
was that "the stack is not a good autoencoder in either mode"; this fixes
that, and the place code is most of the reason, since the down path now
recovers a pixel value by centroid instead of inverting two normalisations.

And **generation matches or beats dense**: graded 1.0 for every sparse arm
at k>=4, and top-1 1.0 at k=8 where dense manages 0.8. Layer three's
templates are class-pure at 0.999-1.000 throughout.

So the sparse stack draws better and rebuilds better. It only classifies
worse, and only on a readout that consumes coordinates.

## The settled read is worth 4.5 points

Same learning rule, same dictionary, same k, **slightly sparser code**
(0.0289 vs 0.0301 lit) — and 0.7494 -> **0.7946**.

That is the `../sparse_column` finding converting into a downstream number.
The settled read's support survives a 5% nudge 89% of the time against the
pursuit's 79%, and here that stability is worth more than doubling k is
(k=4 -> k=8 bought 1.5 points; settling bought 4.5). Worth its 3.4x on CPU,
and it is the arm to carry forward.

## Two things that did not go as expected

**Removing layer three's mean-centring changed classification by nothing** —
0.749400 both, the same 3747 of 5000. The two layers demonstrably differ
(their joins differ by up to 0.18, and top-1 generation moves 0.7 -> 1.0),
so the identical count is most likely a coincidence rather than identical
predictions, but it was not checked at the prediction level and should not
be leaned on. Either way the claim that centring meaningfully damages a
sparse code at layer three is **not supported** by this run.

**k barely matters for classification.** 0.713 to 0.765 across a 4x change
in density. Since the small end is where the support is countable — 612
distinct supports at k=2 against 3440 at k=8, on 4000 patches — the flatness
argues for k=2, not k=8. The one combination not run is the one the table
now points at: **k=2 with the settled read.**

## Files

| | |
|---|---|
| `sparse_stack.py` | `SparseStack` (place code + sparse columns), `NoCenterBound` |
| `run_sparse_stack.py` | all seven arms, the figures, `results/metrics.json` |
| `results/01_board.png` | the table above |
| `results/02_generated.png` | the label alone, back down to pixels |
| `results/l1_templates_*.png` | what layer one learned, per arm |
| `results/confusion_sparse_k4.png`, `classification_report.txt` | per class |

    python run_sparse_stack.py     # ~13 min, 7 arms

## What this settles, and what it does not

Settled: a sparse identity-carrying layer one carries a working stack, draws
better than dense, rebuilds better than dense, and classifies 13 points
worse. The cost of identity is real and now priced.

Not settled, and this is the whole question: **nothing above has yet tried
to use identity.** The next thing worth building is a layer three that reads
names rather than coordinates — counting co-occurrences over template pairs,
or a partial-cue read addressed by index — because that is where the 13
points are supposed to be bought back, and no run in this repo has ever
asked for it.

---

# Part two: scoring in the dictionary's geometry — it does not work

The hypothesis was that the 13 points were **geometry, not information**.
Argument: in the dense arm the code *is* `Wx`, so `c1 . c2 = x1' W'W x2` is
the true patch similarity (08-30 part two). In the sparse arm the code is a
selection, so `c1 . c2` only asks "did we pick the same templates" — two
near-identical templates are different coordinates and contribute nothing.
Fix: score as `(c1 W2) . (c2 W2) = c1' (W2 W2') c2`, one matmul, keeping the
code sparse and nameable and moving only the comparison.

| | accuracy |
|---|---|
| dense | 0.9232 |
| sparse k=4, cell-wise score | 0.7494 |
| **sparse k=4, gram-space score** | **0.7164** |
| wta | 0.5708 |

**It lost 3.3 points.** Layer three's input went from 1152 cells at 3% lit
to 5184 at 63% lit — denser than the dense arm — and got *worse*, which by
itself kills the "accuracy tracks how much is lit" reading from part one as
a causal story. Purity stayed 1.000 and generation stayed 0.8 / 1.0, so
nothing is broken; it is simply a worse classifier.

The reason the fix cannot work, in hindsight: `c @ W2` **adds no
information**. A code with 4 nonzeros of 128 expands to a sum of 4 template
rows — 576 numbers carrying the same 4 magnitudes and 4 indices. Re-spreading
information over more cells cannot recreate what selection discarded. And if
layer two's templates were near-orthogonal the transform would be close to a
rotation, changing nothing except how `BoundLayer.join`'s centring and
normalisation land on 5194 cells instead of 1162 — which is a confound this
arm does not separate, and the reason the number moved at all may be that
rather than geometry.

**So the honest revision: the gap is real information loss, not a change of
coordinates.** The dense code keeps all 128 projections of each window; the
sparse code keeps 4. Layer three is a correlation matcher and uses all of
them, so it is simply better served by 128 numbers than by 4. There is no
free lunch to be had by re-expanding.

What still stands from part one is the **settled read**, +4.5 points at
identical sparsity, which is the only thing that has bought accuracy back so
far — and it does so by choosing *which* templates fire, not by adding cells.

The way to buy back 13 points is therefore not to make the sparse code look
dense to a matcher that wants coordinates. It is to put something above that
wants names: counting co-occurrences over template pairs, or a partial-cue
read addressed by index. Until that exists, the sparse arm is being scored
entirely on the axis it is worst at.

`run_gram.py`, `results/gram.json`, `results/03_gram_generated.png`. ~2.5 min.

---

# Part three: classify by counting, and the ordering inverts

Layer three is a nearest-neighbour matcher over whole vectors. It consumes
coordinates and never asks which template fired, which is why parts one and
two could only ever measure what identity *costs*. `probe_tally.py` asks the
opposite question — what if the only thing read is **which** templates fired?

    tally[position, template, label] = how often that template fired
                                       there on a digit with that label

Every active cell in a test code votes for the labels it has historically
co-occurred with; sum, argmax. **No layer three at all** — one counting pass
and one matmul, no prototypes, no winner-take-all, nothing learned.

## The inversion

| | layer three (matching) | tally (counting) | change |
|---|---|---|---|
| dense | **0.9232** | 0.7060 | **−0.2172** |
| wta (k=1) | 0.5708 | 0.7852 | **+0.2144** |
| **sparse k=4** | 0.7494 | **0.8672** | **+0.1178** |
| sparse k=4, settled read | 0.7946 | 0.8670 | +0.0724 |

Under matching, dense wins by a mile and the codebook is worst. Under
counting, **the order reverses exactly**: sparse wins, dense is worst, and
the arm that was bottom of the board gains 21 points.

So the readout a code wants depends entirely on whether it carries identity.
A dense code is coordinates and needs a matcher. A sparse code is names and
wants a counter. Scoring either one on the other's readout measures nothing
about the code.

## Identity alone carries it — the magnitudes add nothing

The variant grid was build-the-tally by {presence, magnitude} x score by
{presence, magnitude} x {log, linear}. For **every** sparse arm the winner is
`count/log/count` — pure binary presence, no magnitudes anywhere — and it is
also the best overall number in the table.

Dense's best (0.7060) needs magnitudes (`count/linear/mass`); its pure-count
score collapses to 0.6252, because with 49% of cells lit every cell votes on
every image and the tally decays toward the class prior.

**Which templates fired is the whole signal.** How loudly they fired is
worth nothing to a counter. That is the sharpest statement of what a sparse
code is *for* that this repo has produced.

## Three things that follow

**A table beat the learned layer.** 0.8672 against 0.7494, on the same
codes, replacing 1024 geodesically-learned prototypes with counting. Layer
three's winner-take-all prototype learning was actively losing points on a
sparse code.

**The settled read stops paying.** It was worth +4.5 under matching
(0.7494 -> 0.7946) and is worth **nothing** under counting (0.8672 vs
0.8670). Its advantage was specifically about making the argmax of a
*matcher* stable. A counter does not take an argmax over templates, so it
does not care. If the tally is the readout, the pursuit's 3.4x speed
advantage can be taken for free.

**The gap is now 5.6 points, not 13.4.** Best sparse (0.8672, a lookup
table) against best dense (0.9232, a thousand learned prototypes). Nobody has
yet combined the two readouts.

## Caveats

Naive Bayes over singles — co-occurrence is ignored entirely, which is half
of what identity is supposed to buy, and counting *pairs* is the obvious next
step. The tally is also position-specific, so it has no spatial tolerance
where the dense code interpolates. And MNIST may not discriminate the
question well: a bag of local strokes at known positions is already
discriminative for digits, so this result may understate how much
co-occurrence matters for anything where arrangement is the content.

The dense arm's magnitude-built variants produce negative counts (dense codes
are signed) and their logs are undefined; those combinations are invalid and
were not selected as any arm's best. The reported numbers are all from valid
variants.

`probe_tally.py`, `results/tally.json`. 943 s, most of it the settled arm.
