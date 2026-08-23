# `celeba_faces/` — the flagship on CelebA faces + composition by code arithmetic

(2026-08-23, from the day log.) Scripts: `prepare_celeba.py` (downloads CelebA and writes the 48x48 arrays into `<repo>/data/`),
`run_celeba_faces.py`. Results: `results/` (portraits v1-v3, delta heatmaps, composition,
`weights.npz`).

## CelebA faces + composition — predictions (before running, 2026-08-23)

`run_celeba_faces.py`: the flagship (consolidated unit, stride-1, GPU
mini-batch) on 39,910 48x48 faces, top layer over [code ; 0.5 *
40-attr multi-hot]. Deliverables: attribute-conditioned portraits,
attribute-delta heatmaps, and the composition test. Note:
woman_with_mustache = 0 in the data — retrieval cannot fake success.

1. The L1 dictionary learns face-edge primitives; attribute-conditioned
   queries retrieve and render recognizable face-like structure
   (edge-drawing style — hollow interiors expected, contrast coding).
2. THE falsifiable one: mustache-delta = mean(code | mustache) -
   mean(code | no mustache), rendered, localizes on the upper-lip
   region. Eyeglasses-delta localizes around the eyes.
3. Composition: female-base code + mustache-delta, hardened and
   rendered, shows added dark upper-lip structure on an otherwise
   female face. Success = visible LOCALIZED addition; honest failure =
   the delta smears globally or overwrites the base.
4. Training <= ~4 min on GPU despite 2025 L1 windows per face.

### Outcome (same night) — composition demonstrated

178s training on 39,910 faces (grids 45/22/20, code3 40,000).

- P1 corrected by the user's eye: v1 portraits collapsed to TWO
  prototypes (one male, one female) — attribute conditioning failed
  beyond the Male axis. Mechanism: stored label halves are dominated by
  shared base rates (Young/No_Beard/etc.), so rare attributes carry no
  discriminative mass. v2 fix (portraits_v2.png, retrieval-time only):
  contrast retrieval (subtract the across-memory mean label half) +
  IDF-weighted queries -> 6 distinct winners incl. genuinely different
  female faces. New failure mode exposed: contrast scoring over-favors
  under-rehearsed outlier memories, three of which render as noise (the
  style-grid consolidation phenomenon). Full fix, queued: gate
  retrieval on rehearsal count (save win_counts in weights.npz) or
  blend top-k consolidated matches.
- v3 (portraits_v3.png) — RESOLVED, three stacked fixes: (1) the user's
  "stripes" observation exposed a dimension-scaling BUG — bootstrap
  noise at 0.01/component has norm ~2.0 at dim 40,040, so every memory
  was born two-thirds noise; fixed to fixed-total-length ~0.05
  (../rig/run_gpu_minibatch.py), retrained in 173s; (2) contrast retrieval +
  IDF query weighting; (3) rehearsal gate on saved win_counts (median
  15, max 603). Result: **10/10 distinct, all real faces, zero
  stripes**, with visibly attribute-appropriate retrievals (glasses
  band, bald dome, wavy hair, gendered faces). Note: the same noise bug
  mildly affected the MNIST GPU runs (norm ~0.4 at dim 1,610) — free
  small improvement whenever those rerun.
- P2: eyeglasses delta localizes CLEANLY (dark frame-band across the
  eyes). Mustache delta's strongest element is the dark upper-lip band,
  embedded in male-face correlates — partial; the raw
  difference-of-means confounds Mustache with Male. Obvious refinement:
  purified delta (mustached men MINUS clean-shaven men).
- P3 ✓ **THE composition works in the moderate regime**: at +0.3 (and
  0.6) the female-memory base keeps its feminine hair and structure and
  acquires a distinct dark upper-lip band — a woman with a mustache,
  from a dataset containing zero. At +1.0 the Male confound in the
  delta takes over and coherence degrades — the failure mode is
  understood and matches the confound.
- P4 ✓ under 3 minutes.

Composition-by-code-arithmetic is real in this architecture. Refinements
queued: purified deltas, per-attribute LAM scaling for finer retrieval,
more top memories for 40k-face granularity.
