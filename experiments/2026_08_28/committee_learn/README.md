# `committee_learn/` — committee-LOO learning, CPU-online (Exp 12)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_committee_learn.py`
- `results/` — metrics.json, ladder and template galleries, weights.npz, run log.

## Exp 12 — run_committee_learn.py: committee-LOO learning (CPU-online)

Sequential SELECTION kept (soloist order); ATTRIBUTION exact: committee
coefficients via Gram inverse, each member's geodesic target = window
minus what the OTHER members explain (the user's learning3 LOO formula
restricted to the chosen 4; selection breaks the symmetry that
dissolved units in the full-bank version). User's prediction: could
lift round-2+ match; risk = ownership re-blur.

RESULT: THE STRONGEST LEARNING RULE OF THE DAY. vs fullseq_cpu:

| read | fullseq | committee |
|---|---|---|
| L1 res R6 | 0.969 | **0.974** |
| L1 graded | 0.938 | **0.945** |
| L2 parity | 0.892 | **0.921** |
| L2 res R4 | 0.909 | **0.916** |
| L3 parity | 0.866 | **0.899** |
| L3 res | 0.867 | 0.866 |

matchq_L1 rounds 2-3 up exactly as the user predicted (0.588 -> 0.635,
0.473 -> 0.516; round 1 pays a bigger duty split 0.773 -> 0.723). The
ownership risk RESOLVED by the gallery: templates became compact,
LOCALIZED, complementary stroke pieces — a tiling of window content,
sharper ownership, not blurrier (exact blame -> disjoint
responsibilities). All units used. NOTE: at L3 the plain graded
expansion (0.899) now BEATS the residual read (0.866) — read
preference flipped on tiled vocabularies; re-tune reads per bank.
FINAL LADDER: **0.974 / 0.921 / 0.899** (best-read per rung, committee
weights). The standing recipe amends: committee-LOO targets at all
levels. Untested: committee learning under GPU B=2 (should port —
solve is batchable); resolve-read (Exp 11) on committee weights.
