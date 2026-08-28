# `resolve_read/` — greedy selection + closed-form coefficients (Exp 11)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_resolve_read.py` — on `../fullseq_cpu/results/weights.npz`.
- `results/` — metrics.json, run log.

## Exp 11 — run_resolve_read.py: greedy selection + closed-form coefficients

Pure read on the fullseq_cpu weights: selection unchanged (sequential
rounds pick the committee), then the committee's strengths re-solved
JOINTLY via the Gram inverse (orthogonal projection onto the
committee's span; the partial-correlation operation from the user's
CSHL node). Same ids, same storage. Prediction: small consistent free
win, biggest at L2/L3 (near-tie committees).

RESULT: win CONFIRMED at L1 — 0.969 -> **0.972** at R=6 (MSE 0.00539
-> 0.00503), consistent at every R>=2, free. But the "biggest at
depth" half was WRONG, reversed: L2 0.909 -> 0.908, L3 0.867 flat.
The diagnostics say why: mean coefficient shift at L1 is 2x the deep
levels' (0.053 vs 0.027/0.020) and negative resolved coefficients
exist only at L1 (0.6%). The overlap worth disentangling lives at L1
(64-in-64, crowded, near-ties); the deep banks are so undercomplete
that greedy voices grab near-orthogonal pieces of a huge leftover —
nothing to re-attribute. Adopted for the from-L1 read; a no-op
elsewhere. From-L1 record: 0.972.
