# Part two — Fashion-MNIST, and what correction costs

The prediction above was that Fashion-MNIST would give composition a real gap
to close. It gave something better: the architecture failed, and the failure
was legible.

## The wall (`fashion/`)

The competitive40 architecture, unchanged, moved from digits to clothes:

| | MNIST | Fashion |
|---|---|---|
| competitive + fit gate | 0.9390 | 0.5952 |
| conscience + fit gate | 0.9392 | 0.6090 |
| best off-the-shelf | — | 0.8926 (RBF SVM) |
| nearest centroid | — | 0.6950 |

Below every baseline including the dumbest one, and competition *reversed sign*
— the control arm, where everyone learns from everything, beat both competitive
arms (0.6604 vs 0.6090).

The cause is not in the training. Every live hypercolumn stayed class-pure and
all ten classes had specialists. It is in the **read**: the right-class expert
ranked 1st for only 59.5% of test images but was in the **top 3 for 81.6%**.
The fit gate asks "who rebuilds this image best", and on MNIST that question
happened to also answer "which digit is this". On Fashion it does not.

Measured geometry: Fashion has *lower* within-class spread than MNIST (0.769 vs
0.662 similarity to the class mean) and *larger* average class separation. The
difficulty is one clump — Coat/Shirt 0.966, Pullover/Coat 0.957, Sneaker/Sandal
0.879, all tighter than MNIST's worst pair (4/9 at 0.898). Coat scored 0.014.
Coats are called shirts.

## Two competitions, and we trained the wrong one

The sandal expert wins the fit gate on **73.2%** of sneaker images while the
sneaker expert wins 12.8% — and during training no expert ever won both
(sneaker experts `[7,18,29,30]`, sandal experts `[2,4,6,9,10,15,39]`, disjoint).

It never had to see a sneaker. Its 36 templates span a shoe-shaped subspace, and
a sneaker is a shoe-shaped thing. Fitting is about what your subspace covers,
not what you trained on, and nothing ever charged it for covering a neighbour.

    training   winner = most confident in the TRUE LABEL   (answer key in hand)
    reading    winner = best reconstruction fit            (no label)

Two different orderings. We optimised one and read the other.

## Correction: what each kind buys, and what it costs (`dopamine.py`, `reject.py`)

| | what it edits | Fashion |
|---|---|---|
| nothing | | 0.5952 |
| **A — reluctance**: a learned offset per hypercolumn at the gate | nothing in the content | 0.7086 |
| **B — repulsion**: the wrong gate winner rotated away from that image | the templates | 0.7874 |
| A + B | | **0.8186** |

Repulsion cost the model something real: own-class rebuild error 0.6734 ->
0.7708. But best-other-class went 0.6964 -> 0.8294, so the gap widened 2.6x and
the right expert became the best fit for 78.7% of images instead of 59.5%. It
narrows every subspace: worse at fitting anything, better at being one thing.

The generations got *crisper* while reconstruction got worse — **rebuild error
measures flexibility, not fidelity to a concept.** Stop using it as a quality
proxy.

## Grading on both, which is where the best number came from (`both.py`)

    score = image error + λ · (1 − confidence in the true label)

λ=0 is "who draws it best", λ→∞ is "who names it best". The sweep says there is
no tradeoff between the two goals — imagination is *worst* at λ=0 (0.319) and
peaks in the middle (0.452), while accuracy plateaus from λ=0.25 upward.

| λ | 0 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| accuracy + reluctance | 0.772 | 0.819 | 0.820 | 0.821 | 0.825 | **0.828** | 0.826 |
| imagination | 0.319 | 0.416 | **0.452** | 0.393 | 0.422 | 0.398 | 0.400 |

**0.8278 at λ=4 is the best result of the day.** But an honest correction: with
repulsion switched off, λ=4 scores 0.6010/0.7066 against the old rule's
0.5952/0.7086 — essentially identical. The grading only pays *given* repulsion.

## Picking the winner by drawing instead of by label (`draw.py`)

Selecting the learner by reconstruction alone, label not consulted:

    draw 0.3544   +conscience 0.4940   +repulsion 0.6440   +both 0.7726

Fit-selected clusters divide by *silhouette*, not class — one hypercolumn took
40206 of 120000 training steps, purity 0.591, and the own-class rebuild gap went
*negative*. Adding fairness and a rejection signal repairs it to within 1.5
points of the label-picked rule. The answer key in the selection was standing in
for two separate things: nobody hogging, and someone saying no.

## The generative half (`coef_stats.py`, `gen_ui.py`)

Training learns *which* directions a hypercolumn spans, never *how they are
combined*. Measuring each one's coefficient mean and covariance and sampling
from it (realism = cosine to the nearest real image of the class):

| sampler | realism | self-similarity |
|---|---|---|
| mean (prototype) | 0.8743 | 1.0000 (identical) |
| diagonal | 0.5589 | 0.3499 |
| full covariance | 0.7082 | 0.3525 |
| real images | 0.8961 | 0.6194 |

Covariance buys 15 points of plausibility at the same spread. Pure random
coefficients give noise (−0.024) — the subspace alone is not a generator.

`gen_ui.py` is a DearPyGui explorer for this: pick a class and a hypercolumn,
draw from the mean / diagonal / covariance, drag the twelve highest-variance
coefficients, and round-trip an image through the unit. Not a fork of the drone
`viewer.py` — no simulation in it.

## Capacity (`capacity.py`)

1,143,360 parameters against logistic's 7,850, and 2.3 points worse. But only
**14 of 40 hypercolumns ever win the gate**; keeping the 15 the gate can reach
gives 0.8278 to four decimals at 428,760 parameters — two-layer-MLP scale.
Minicolumns are the opposite of idle: accuracy climbs all the way to K=36 with
no plateau.

The 22 trained-but-unreachable hypercolumns are not waste, though — see below.

The stream version of this rule (split-MNIST with and without punishment) is
written up in [`../continual_newrule/README.md`](../continual_newrule/README.md).

## The teacher-free stack

The user's constraint, which settles what any rule may contain: **the label is a
second sensory stream, not a verdict.** Grading is legitimate (it is
reconstruction error on the other modality) and so is sleep (it uses each unit's
own learned association). Repulsion and reluctance are not.

| | Fashion |
|---|---|
| graded, no punishment | 0.6010 |
| + habitual calibration (no teacher) | 0.6414 |
| + sleep + habitual calibration | **0.6868** |
| + reluctance (teacher) | 0.7066 |
| graded + repulsion + reluctance (teacher) | 0.8278 |

Habitual calibration — subtract each hypercolumn's mean error over all inputs —
recovers 38-63% of the reluctance's gain with no labels. Calibrating on "the
inputs I already win" collapses to 0.1886, and equalising airtime costs 15
points: loudness is not evidence of over-claiming when one class has seven
experts and another has one.

**And dropping punishment takes the forgetting problem with it.** 0.9620
retained. Forgetting was an artefact of the teacher signals, not a property of
the architecture.

## Input geometry, checked rather than assumed (`norm_ablation.py`)

| | raw | + reluctance | coherence | DC alignment |
|---|---|---|---|---|
| centre + L2 (current) | 0.8170 | 0.8278 | 0.0287 | 0.022 |
| centre only | 0.7928 | 0.8098 | 0.0289 | 0.023 |
| L2 only | 0.7316 | 0.8082 | 0.0288 | 0.051 |
| raw | 0.7052 | 0.8062 | 0.0288 | 0.051 |

Both operations earn their place. The predicted mechanism was wrong, though:
removing the centring does *not* collapse templates into the positive orthant —
coherence is identical to four decimals in all four arms, because the geodesic
rule learns from residuals, which are signed even when the data is not.

Note purity *rises* while accuracy falls without centring: brightness splits the
training set cleanly without carrying the real distinction, which is what the
centring was there to remove.

## Confidence is input-blind (`confidences.py`)

The 14 working hypercolumns say their own class at ~0.99 on their own images —
and at 0.79-0.98 on *other classes'* images. The coat expert says "coat" at
0.971 for a boot. Confidence measures commitment, not evidence, which is why the
confidence gate scores 0.31 where the fit gate scores 0.82. The cause is our own
code: the label completion is L2-normalised before reading, and the magnitude
thrown away by that normalisation *is* the evidence.

## Files

Every script writes its numbers to a json in `results/`; `board.py` redraws the
single board `results/20_board.png` from all of them. `common.py` is the shared
loader (20000 train / 5000 test). `baselines.py` gives the off-the-shelf numbers
(`results/baselines_fashion_mnist.json`).
