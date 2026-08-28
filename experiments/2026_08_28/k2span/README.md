# `k2span/` — is L2's ceiling its vocabulary span? (Exp 7)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_k2span.py`
- `results/` — metrics.json, run log.

## Exp 7 — run_k2span.py: is L2's ceiling its vocabulary span?

K2 in {64,256,512} (L3 input dim follows), GPU minibatch, L1-residual
learning, L2/L3 plain. Prediction: bigger K2 raises round-1 match and
the L2 residual read climbs past 2 voices. 48s for all three arms.
CONFIRMED — and it cuts BOTH ways:

- L2 residual read climbs with K2: R6 0.846 -> 0.870 -> 0.876 (still
  rising at 6 voices at K2=512); matchq round 1 0.507 -> 0.528, round 2
  0.363 -> 0.405. All units used at every K2 (no monopoly).
- But the DENSE reads worsen with K2 (L2_parity 0.877 -> 0.859 ->
  0.829 — a 512-template dense sum is blurrier: the rich-palette
  lesson, expansion edition), and L3 COLLAPSES (parity 0.814 -> 0.742,
  res 0.786 -> 0.690): growing K2 made L3's blocks 4608-dim under the
  same 100 templates — 9x undercomplete became 46x.

SPAN LAW (both experiments agree): a level's residual-read ceiling is
its dictionary's span relative to its input dimension; widening one
level's vocabulary deepens the starvation of the level above unless
its vocabulary grows too. Balanced ratios across the stack, or
footprint reduction, are the actual lever — not one big K2.
