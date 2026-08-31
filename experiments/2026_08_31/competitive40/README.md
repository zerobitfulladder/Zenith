# Only the winner learns, and identity appears

2026-08-31. 40 hypercolumns x 36 templates on whole MNIST, `[image ; label]`
at rho=1.0, the 08-30 dense `weighted` rule unchanged. One line differs from
`../dense_ensemble`:

    per sample, write the image and blank the label, ask every hypercolumn
    to complete it, and let ONLY the hypercolumn whose completion best
    supports the TRUE label take a learning step.

**40 x 36 = 1440 templates — the same budget as `../competitive`'s 10 x 144.**
Only the partition changed.

## The board

| | fit gate | confidence gate | oracle | dead | subspace overlap |
|---|---|---|---|---|---|
| **competitive 40x36** | **0.9390** | 0.2944 | **1.0000** | 25/40 | 0.0682 |
| **+ conscience** | **0.9392** | 0.3106 | **1.0000** | 17/40 | 0.0929 |
| control 40x36 (all learn) | 0.7944 | 0.8162 | 0.9304 | — | — |
| competitive 10x144 | 0.8698 | 0.1658 | 0.9932 | 2/10 | 0.1912 |
| dense + layer three (08-30 harness) | 0.9232 | | | | |
| logistic regression on pixels | 0.9074 | | | | |
| sparse + tally | 0.8672 | | | | |

**0.9392 is the best number this project has produced**, past dense +
layer three and past the logistic judge — with **no learning at the
selector at all**, just "which hypercolumn explains this image best".

## Three findings

**Partition beats capacity.** Identical 1440 templates: 10x144 gives 0.8698,
40x36 gives 0.9390. Seven points from redistributing the same parameters into
more, narrower experts. A 36-dimensional subspace is far more selective about
what it explains well than a 144-dimensional one, and selectivity is exactly
what the gate needs.

**The gate must match what the experts are, and getting it backwards is
catastrophic.** Specialists want a *fit* gate (0.939) and collapse under a
*confidence* gate (0.294). Generalists are the reverse — the control scores
0.816 on confidence and 0.794 on fit. A class specialist says its digit
confidently for every input, including inputs it has never seen, because
nothing in training ever penalised it for being sure about those. Confidence
is only meaningful among experts that are all competent.

**The 08-30 speckle problem dissolves.** `02_templates_competitive.png`:
every live hypercolumn's templates are variations of the digit it claimed,
at purity 1.00. The rotation invariance 08-30 diagnosed is *still present* --
any rotation within a hypercolumn would do just as well -- but the subspace
being rotated is now one class's, so every basis of it looks like that class.
**Meaning came from restricting the data, not from fixing the rotation.**

## What the conscience buys

A win-frequency penalty (`gamma=1.0`) revives starving hypercolumns: dead
falls 25 -> 17, live specialists rise 15 -> 23, and every digit gets 1-3
experts instead of 1-2. Accuracy is **unchanged** (0.9390 -> 0.9392), so the
revived columns are redundant on MNIST -- about 15 experts saturate this
task. Worth keeping anyway: on a harder problem the spare capacity is the
point, and 25 dead columns out of 40 is capacity paid for and unused.

## The gate is now the only thing left

The oracle is **1.0000** — on every one of the 5000 held-out digits, at
least one hypercolumn gives the right answer. The fit gate recovers 0.939 of
it. The remaining **6 points are entirely a selection problem**, and a
learned gate is the obvious attack.

## Files

| | |
|---|---|
| `compete40.py` | all three arms, both gates, the figures |
| `figs.py` | regenerates the template sheets from saved weights |
| `results/weights.npz` | all three arms plus the win matrices |
| `results/01_results.png` | win matrices and the board |
| `results/02_templates_*.png` | the live specialists, labelled by what they claimed |

    python compete40.py     # ~4 min, three arms

## Next

A learned gate (6 points on the table). And the honest caveat: MNIST is
saturated at 15 experts, so the interesting version of this needs a task
where the specialists have to carve something less obvious than ten digits.
