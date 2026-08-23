# `celeba_big/` — bigger dictionaries, 1000-memory top layer, zero-count attribute pairs

(2026-08-23, from the day log.) Scripts: `run_celeba_big.py`, `show_anatomy.py` (per-level galleries from the saved weights).
Results: `results/`.

## Big faces run — predictions (before running, 2026-08-23)

`run_celeba_big.py`: dictionaries 48/96/128, archive KTOP=1000 (~51M
params, capacity where knowledge lives), 6 epochs, clean births,
rehearsal-gated contrast retrieval. Generation program: (a) existing-
label portraits, (b) zero-count attribute pairs (census-verified)
composed via PURIFIED deltas (attribute delta computed within its
natural gender, applied across — removes the Male confound seen in the
raw mustache delta).

1. Portraits: >= 9/10 distinct real faces; attribute signals more
   legible than v3 (finer clusters from 5x archive).
2. Zero-pair compositions: >= 2 of 4 pairs show the added attribute's
   localized structure on the base identity at moderate beta; purified
   deltas visibly reduce gender-morphing vs the raw delta.
3. Training <= ~25 min on GPU.
4. Risk to watch: 6 epochs of batch-mean updates may over-consolidate
   memories toward cluster means (blurrier archive).

### Outcome (same night) — attributes legible, two clean impossible faces

822s train (14 min). Census: 2 target pairs occur ZERO times, 2 occur
2-3 times in 39,910. Portraits 10/10 distinct AND attribute-legible:
m943 wears a visible dark mustache, m623 the glasses band, m114 a bald
dome, m907 a dark goatee patch, m610 a huge wavy-hair frame. P1 ✓.

Zero-pair compositions (purified deltas): **lipstick+smiling+mustache
works cleanly** (dark upper-lip band on an intact feminine face,
+0.5-0.8, less gender-morphing than the raw delta — purification did
its job) and **lipstick+wavy-hair+goatee works** (dark chin patch under
the mouth, wavy frame intact). Makeup-on-mustached-man partial (eye
darkening). **Bald-on-woman fails informatively**: baldness is a
SUBTRACTIVE attribute — success requires deleting the base's hair
strokes, and although deltas carry negative components, they are too
weak at these betas to cancel strong base structure before the relu.
New concept on record: additive vs subtractive composition; subtractive
edits need stronger negative-delta weighting or explicit
suppress-then-add. P2 ✓ (2 clean of 4), P3 ✓, P4 risk did not
materialize (no over-consolidation blur).
