# 2026-09-01 — can one pass answer a question the image aims?

## Why this day exists

Yesterday ended with two walls that turned out to be the same wall. An index
cannot compose (the woman with no mustache), and a single global code cannot
carry a property of a small part (CelebA's local attributes, below their own
base rates). The conversation that followed proposed a way out that is not a
better code but a different *act*:

> the top of the hierarchy holds the thing you are attending to, not the scene.
> Parts are reached by querying — the query produces a spatial mask, the mask
> re-runs the input with that region amplified, and the top now holds the part.
> "Nose is part of face" is then not written in any vector. It is written in the
> fact that from the face state, a nose-query lands on the nose.

If that is right, relations move out of the representation and into the
transitions, and the binding problem dissolves rather than being solved: you
never represent "nose-of-face", you *visit* it. It also explains, with no extra
rule, why you can attend the nose of a face but not the nose of a nose — the
legality of a query is a property of a learned mapping, and that mapping only
ever saw transitions that occurred.

It has a cost. Recognition becomes serial, the output is a trajectory rather
than a vector, and something has to decide what to attend next.

**Before building any of that, this folder asks whether the gap it is meant to
close actually exists.**

## The correction that shaped the task

The first design was two digits and "what is to the left of the 7". Working
through it, that is too easy for the baseline, for a reason that is the whole
point: our pooled map is **retinotopic**. "Which digit is on the left" is
answered by a readout wired to the left columns of the grid. Position does the
binding, for free, exactly as claimed. And with two digits, "left of the 7" is
just "the other one".

The distinction that matters:

    position-addressed   "what is in the left region?"        fixed cells
                                                              a flat code is fine
    content-addressed    "what is to the left OF THE 7?"      the cells holding
                                                              the answer move with
                                                              the image

Only the second needs the anchor found before you know where to look. So:
**three digits**, and the query names one of them.

## The setup (`scenes/`)

    canvas       48 x 100, three MNIST digits of distinct classes
    placement    x-centres >= 20px apart so left/right is never close; y jitters
    layer 1      the 64 5x5 templates already trained on MNIST patches
                 (2026_08_31/kmeans), winner-take-all per position,
                 carrying patch contrast
    pooling      max per channel onto a 4 x 8 grid of 11 x 12 blocks -> 2048 dims
                 the grid is KEPT, not collapsed: the baseline gets full
                 positional information, so the comparison is honest

Three questions on the same map:

    Q1  presence           is there a 3?                    a set property
    Q2  global order       which is leftmost / rightmost?   a fixed region
    Q3  anchored relation  what is left of the 7?           a moving region

Three readouts, because the baseline should be the strongest fair opponent:
the **quantiser** (yesterday's second rung — nearest template over
`[map ; query ; answer]`, read the answer half), **logistic**, and a one-hidden-
layer **mlp**.

## The splits are the measurement

720 ordered triples of distinct classes. 20 unordered triples are held out
whole. Of the remaining 100, four of the six orderings train and two are
reserved. So:

    seen     400 orderings, trained on, tested with digit images never seen
    order    200 orderings never trained, whose classes were trained together
             repeatedly — same three digits, same ink, new arrangement
    triple   120 orderings of three classes that never co-occurred at all

The headline is not any accuracy. It is the **gap between `seen` and `order`**.
Anything that answered by memorising configurations is fine on the first and
lost on the second.

## Predictions, stated before running

| | Q1 presence | Q2 order | Q3 anchored |
|---|---|---|---|
| quantiser, seen | good | good | moderate |
| quantiser, order | good | **falls** | **falls** |
| logistic, seen | good | good | poor |
| logistic, order | good | good | poor |
| mlp, seen | good | good | good |
| mlp, order | good | good | **degrades** |

Chance: Q1 top-3 exact set 1/120; Q2 1/10; Q3 **1/3** — knowing only which
three digits are present and which is the query, the answer is one of the other
two or "none". 1/3 is the number a set-only reader gets, so it is the real floor.

**Honest limit, recorded in advance.** There is no task a large enough network
provably fails. An mlp can approximate "find the anchor, then look left". What
these splits measure is not impossibility but how much the answer *depends on
having seen the combination*. And the comparison this project actually needs is
narrower: our own stack, one pass, versus our own stack with a query-and-look
loop. The mlp is here to say what conventional machinery does on the same task.

**If Q3 shows no gap, the premise is wrong and the attention loop is unnecessary.**

## Results

_(This section read "pending — `baseline.py` running" when the page was
written. The results were written up afterwards in the day README, as its
Part 1, and are moved here verbatim.)_

## Part 1 — the attention premise, tested and left unproven (`scenes/`)

Yesterday's argument was that the top of the hierarchy holds the *attended*
thing, and parts are reached by querying — a query makes a mask, the mask
re-runs the input, the top now holds the part. Relations then live in the
transitions rather than in any vector, which dissolves the binding problem
instead of solving it.

To test whether that gap exists, we built three-digit scenes on a 48x100 canvas
and asked three questions of increasing difficulty:

```
Q1  presence         "is there a 3?"                     a set property
Q2  global order     "which is leftmost/rightmost?"      a FIXED region
Q3  anchored         "what is left of the 7?"            a region the IMAGE picks
```

Split by *arrangement*: 400 ordered triples trained, 200 orderings held out,
120 triples never co-occurring. The measurement is the gap between seen and
held-out arrangements, not the accuracy.

**Q2 confirmed retinotopy outright.** An MLP scored 0.8797 on trained
arrangements and **0.8793** on arrangements never seen. Zero gap. Position does
the binding for free, exactly as claimed.

**Q3 did not collapse** — 0.6799 seen, 0.6311 held-out — but the run was
recognition-limited and therefore uninterpretable: per-digit presence accuracy
was 0.6970, so Q3 was sitting on its recognition ceiling.

### Chasing the recognition limit (three refuted hypotheses)

```
finer pooling grid       REFUTED   4x8: 0.8465, 11x12: 0.8065, 11x16: 0.7715
                                   monotonically WORSE
position is the cost     CONFIRMED one digit fixed 0.9790, free 0.8205
sliding layer 2 fixes it REFUTED   0.6145, worse than what it replaced
```

The oracle control then separated the causes:

```
A  oracle window, graded code         0.9410   the FRAME is sound
B  oracle window, one identity K=300  0.5505   one symbol is a weak description
C  all windows, one identity          0.7425   the POPULATION carries it
D  all windows, top-3                 0.8370
```

**A single identity out of 300 is worth 55% of a ten-way digit call; thirty of
them are worth 74%; graded is worth 94%.** The information lives in the
population across the map, not in any one index — direct support for "N indices
per thing, and which fires gives the style."

The attention loop was never built. The premise remains untested, because the
task never got clear of its recognition floor.
