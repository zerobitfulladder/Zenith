# 2026-08-24 — `residual/`: Residual competition in speech

Part of the day log [`../README.md`](../README.md).

## Residual competition — predictions (before running, 2026-08-24)

The inhibition the user proposed, made concrete: at a position, the
argmax winner speaks with its correlation, SUBTRACTS what it explained
from the (centered, normalized) window, and the remaining templates
compete over the residue; repeat until the residue drops below the
contrast floor (reused as the stop rule) or 3 speakers. Learning
untouched everywhere. Sorted-gallery ground truth: the 512 palette is
a few dozen parts x fine-variant families (median adjacent cosine
0.816, 13% of adjacent pairs > 0.9) — top-k picks k tiles from one
family; residual competition cannot.

`run_residual.py`, 8x8 K1=512: arm A = residual speech at L1 only
(references: dense .9094 / top3 .8642 / top1 .8590); arm B = residual
speech at all three levels (reference: all-top3 .7596). Measures the
usual + mean speakers per active position.

1. L1-resid beats top3 clearly: hard .87-.89 (complementary speakers
   recover most of the coverage the clones wasted).
2. NOT full rescue: the FIRST winner is still an argmax among clones —
   the coin flip survives, so a gap to dense (.9094) remains. If
   resid ~= dense, flips barely mattered and redundancy was almost
   the whole disease (would be a surprise worth having).
3. All-layer resid clearly above all-top3 (.7596), still below dense —
   complementarity helps coverage, cannot stabilize naming.
4. Probes >= the top3 arms' (unhurt); generation stays clean; speakers
   per position adaptive, mean ~1.5-2.5 at L1.

### Outcome

No outcome section was written for this experiment. The run's own report,
[`results/report.md`](results/report.md), has the numbers:

| arm | probe L2 | probe L3 | hard | consistent | speakers L1/L2/L3 |
|---|---|---|---|---|---|
| L1resid | 0.9658 | 0.9682 | 0.8502 | 10/10 | 2.97/0.00/0.00 |
| allresid | 0.9624 | 0.9618 | 0.7898 | 10/10 | 2.97/3.00/3.00 |

The later sections of the day quote the L1 result as "resid speech .8502"
(see `../resid_learning/README.md`).
