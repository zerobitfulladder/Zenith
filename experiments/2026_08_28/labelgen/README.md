# `labelgen/` — label-conditioned generation (Exp 9)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_labelgen.py` — on `../residual_learning/results/weights_seq.npz`.
- `results/` — metrics.json, generation galleries, weights_top.npz, run log.

## Exp 9 — run_labelgen.py: label-conditioned generation (user design)

L1/L2 frozen from Exp 4 seq weights; a label-bearing top trained on the
L2 code (dense joint learning, per-pathway centering, lam=0.5, K=200,
online, one pass). Two arms since the design admits both; predictions
stated in advance: global = known-good pattern, clean digits; conv
(label at every L3 position) = position degeneracy risk (the
generation query [empty;label] is identical at all 9 positions over a
shared bank).

GLOBAL: hard readout **0.9276 = NEW PROJECT RECORD** (prev 0.9182,
which needed supervised LVQ; this is one unsupervised pass + label
concat on the residual-trained banks), 10/10 label->own-memory,
balanced ownership (15-24 memories/label), all 200 units used.
Generation: all ten digits clean in both descent modes;
generation_global_variants.png (6 memories/label, hardened) is the
best generation gallery of the project — real style variety (crossed
and plain 7s, open and closed 4s, slanted 0s and 1s) and legible 8s.
Reconstruction-first training transfers to generation+recognition for
free.

CONV: degeneracy confirmed exactly as predicted — every read mode
(top1/top5/graded) yields grid-textured mush, no legible digit;
per-position vote readout only 0.7546. The identical label query
cannot place content; a shared bank + label-only cue has no WHERE.
(Matches the temporal composition law: composition needs locality of
DECISION — here every local decision is the same decision.) Fix
directions if pursued: position channel beside the label, or global
gate -> local refine (the two-movies gate-then-match pattern).
