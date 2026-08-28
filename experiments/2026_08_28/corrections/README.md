# `corrections/` — across-level corrections, pure read (Exp 6)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_corrections.py` — on `../residual_learning/results/weights_seq.npz`.
- `results/` — metrics.json, corrections_ladder.png, run log.

## Exp 6 — run_corrections.py: across-level corrections (pure read)

The residual principle rotated sideways: each level residual-codes the
GAP between its actual blocks and what the level above's expansion
already explains, in its own vocabulary (Exp 3 = the degenerate case
where the level above says nothing). Chain on frozen Exp 4 weights:
L3 voices (R3=4) -> predicted L1-code map -> L2 gap-voices (R2c) ->
corrected map -> L1 gap-voices (R1c) -> decode. Prediction (stated in
discussion): L2 corrections recover a chunk of 0.857->0.9+; (4,4)
approaches the from-L1 ceiling. Result (5000 imgs, corr / stored
structure numbers):

| arm | corr | numbers |
|---|---|---|
| coarse only (0,0) | 0.856 | 72 |
| +L2 corr (2,0)/(4,0) | 0.884 / 0.887 | 172 / 272 |
| +L1 corr (0,4) | 0.955 | 1040 |
| (2,2) | 0.941 | 656 |
| (4,4) | 0.957 | 1240 |
| [flat from-L1 R4 / R6] | 0.964 / 0.969 | 968 / 1452 |

Verdicts: (0,0)==L3_res baseline (plumbing faithful). L2 gap-voices
worth +3 points for ~100-200 numbers but SATURATE AT 2 VOICES (span
ceiling again). RATE-DISTORTION LAW: the hierarchy wins at low rates
(72 numbers -> 0.856 vs flat R1's 242 -> 0.930... i.e. coarse story =
by far the cheapest description), flat local coding wins at high rates
(flat R4 968/0.964 beats (0,4) 1040/0.955 — at L1 grain, a top-down
warm start is worth less than one fresh voice). Prediction ~held;
high-rate crossover was not predicted.
