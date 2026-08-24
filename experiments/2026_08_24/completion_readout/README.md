# 2026-08-24 — `completion_readout/`: Completion readout: classify by voting memories

Part of the day log [`../README.md`](../README.md).

## Completion readout — predictions (before running, 2026-08-24)

User proposal: classify by GENERATING the label, not by copying the
single nearest memory's label — present [code ; empty label], every
memory votes its label weighted by its match, strongest class wins.
Lineage: the batch-1 readout ladder on the old rig (hard 0.621 -> soft
0.665 -> top-10 voting 0.712). Never run on today's rigs. Mechanism
bet: pooling over ~20 same-class memories averages out the clone
coin-flip noise that single-nearest matching eats at full strength.

`run_completion_readout.py`: retrain the three 8x8 K1=512 arms
(dense / top3 / top1 L1 output — weights were not saved), score each
with three readouts on identical codes: top-1 (baseline), top-10
vote, all-200 vote.

1. Voting beats top-1 in every arm (top-10 by >= 2pp, the precedent).
2. THE POINT: sparse arms gain MORE than dense — the dense-sparse gap
   narrows from ~5pp to <= 2.5pp under top-10 voting.
3. Dense arm gains too; plausible new record above .9094.
4. All-200 vote underperforms top-10 (old lesson: class-allocation
   bias — popular classes shout).

### Outcome (same day) — ALL FOUR PREDICTIONS LOST; the winner's max
### is a sufficient statistic on a sharp matcher

`results/` (two passes; second adds sharpened
weightings and saves corrs_{mode}.npz so future readout variants are
numpy one-liners, no retrain):

| L1 out | top-1 | top-10 raw vote | top-10 margin vote | softmax vote t=.02 | all-200 vote |
|---|---|---|---|---|---|
| dense | .9080 | .8468 | .8984 | .9080 | .6894 |
| top3 | .8560 | .8094 | .8598 | .8584 | .7294 |
| top1 | .8594 | .7610 | .8220 | .8594 | .7682 |

(retrain jitter between passes ~0.5-1pp; top3 ~ top1 again)

- Raw-correlation voting costs 5-10pp EVERYWHERE — correlations to the
  top-10 memories are too flat (e.g. 0.45 vs 0.40), so ten mediocre
  wrong-class matches outshout two excellent right-class ones.
- Sharpened weightings (margin-relative, sharp softmax) recover to
  within ~1pp of top-1 / exactly top-1 — but NEVER beat it. The
  evidence beyond the winner adds nothing on this rig.
- The batch-1 precedent (+9pp for voting) was context-bound: pooling
  rescues a WEAK matcher (hard 0.62 there); today's matcher is sharp
  (0.86-0.91), and the max IS the signal. Completion/voting readout:
  closed for this rig class.
- The sparse arms' ~5pp deficit is untouched by any readout variant —
  it is not recoverable at the readout; it is in the code/matching.
  Back to the two live levers: keep dense speech, or residual
  competition. (Learned readout organ remains the long-standing
  ceiling-closer for classification per the working theory.)
