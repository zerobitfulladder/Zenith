# `fullseq_cpu/` — CPU-online full-stack residual learning (Exp 8)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_fullseq_cpu.py`
- `results/` — metrics.json, ladder_fullseq_cpu.png, weights.npz (loaded by resolve_read), run log.

## Exp 8 — run_fullseq_cpu.py: CPU-online full-stack (in flight)

Same full-stack rule as Exp 5's refuted arm, but in the CPU online
regime that produced the good Exp 4 weights (L2/L3 seq-residual on
skeleton views, R=4). Prediction: if minibatch dulling was the
confound, fullseq >= exp4seq on deep residual reads; if it still
degrades, transfer-failure is regime-independent (off-manifold
leftovers / skeleton-vs-dense view mismatch).

RESULT: EXP 5'S REFUTATION IS OVERTURNED — the mini-batch regime was
the whole story. CPU-online full-stack residual learning helps at
every deep rung and costs nothing (vs exp4seq): L2_parity 0.892 vs
0.884, L2_res 0.909 vs 0.904, L3_parity 0.866 vs 0.847 (+1.9),
L3_res 0.867 vs 0.857; L1 rungs identical (same pathway); all units
used. Subtlety mirroring L1's round-1 dip: matchq_L2 round 1 slightly
LOWER (0.451 vs 0.482) yet every read better — the vocabulary
reshapes toward usefulness, not per-round match. The off-manifold-
leftovers worry did not materialize as harm. FINAL LADDER OF THE DAY:
**L1 0.969 / L2 0.909 / L3 0.867** (fullseq_cpu weights + residual
reads). Standing recipe amended: sequential residual learning at ALL
levels (online regime), R_TRAIN=4; the GPU trainer needs a
committed-structure fix before it can train this recipe (eval on GPU
remains verified and free).
