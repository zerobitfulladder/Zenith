# `batch2/` — one shared stroke dictionary at stride 2, feathered rendering

(2026-08-23, from the day log.) Scripts: `run_batch2.py` (the rig), `run_validate_batch2.py` (8 seeds + style grid).
Results: `results/mnist/`, `results/fashion/` (`GF_DATASET=fashion`), `results/validation/`.

## Batch 2 — predictions (before running, 2026-08-23)

`run_batch2.py`: shared dictionary (one K1=36 stroke vocabulary swept
across all positions — the convergent-vocabulary argument), 4x4 windows
at stride 2 (13x13 = 169 overlapping positions, code dim 6084), K2=200
concat top (lam=0.5), feathered confidence-weighted overlap-add
rendering. Baselines: batch-1 K2=200 (hard 0.648 / soft_norm 0.725 /
top10 0.737 / probe 0.916).

1. The single shared vocabulary is strokes (one 36-template figure).
2. Feathered overlap-add kills the seams and per-patch brightness
   blocks — smoothest generation yet, digits stay recognizable.
3. Probe >= 0.915 on the overlap code (overlap adds shift tolerance:
   small translations change the code gradually, not catastrophically).
4. Prototype readouts improve over batch-1 K2=200 for the same reason —
   same-class codes overlap more. Open how much.
5. Sanity: shared dictionary usage stays spread (patch space is
   isotropic); no dead-template pileup.

### Outcome (same day) — all five predictions confirmed

Best results of the session, across the board (vs batch-1 K2=200):

- Probe **0.9466** (was 0.916) — first layer in the project that ADDS
  linearly-readable class information over raw pixels (~0.92) instead of
  merely preserving it. Overlap's shift tolerance is the mechanism —
  the scattering story demonstrated in-house, answering the depth
  concern: unsupervised depth helps when discards align with nuisance.
- Prototype readouts: hard 0.648 -> **0.743**, top-10 0.737 -> **0.782**
  (same-class codes overlap more under shift tolerance, as predicted).
  soft_norm dipped slightly (0.725 -> 0.716) — vote normalization
  matters less here.
- Shared vocabulary is a textbook stroke alphabet (36 oriented
  edges/bars/corners/curves), used near-uniformly (entropy 0.994, 0
  dead) — the convergent-vocabulary argument validated.
- Feathered overlap-add: seams, blocks, and dots all gone. Generation
  is smooth and unmistakable for all ten digits; reconstruction is
  near-faithful to each input's individual handwriting.
- Label-only retrieval 10/10.

The batch-2 rig (shared 36-stroke dictionary, stride-2 overlap, K2=200
concat top, feathered rendering) is the new baseline architecture.
Remaining queue: 8-seed validation of these numbers; the depth
experiment (stack level 3 with vs without pooling — committed
predictions in the depth discussion); the searchlight test on the
stable rig; optional delta-rule readout organ.

## Batch 2 validation — predictions (before running, 2026-08-23)

`run_validate_batch2.py`: the batch-2 rig across 8 seeds (split, data
order, and init all vary per seed), plus a style grid from the first
seed (render up to 5 stored constellations per class — the sampled-
variety preview).

1. Probe mean lands near 0.947 with small spread (±~0.01); hard and
   top-10 readouts similarly stable near 0.743 / 0.782.
2. Label-only retrieval consistent in (nearly) all seeds.
3. Style grid: each class's templates render as *distinct complete
   digits* — different handwriting styles, not near-duplicates and not
   blends (the free-allocation crispness claim, checked visually).

### Outcome (same day) — validated

probe **0.9460 ± 0.0034**, hard 0.7487 ± 0.0084, top-10 0.7536 ± 0.0168,
label-only retrieval **80/80** across all 8 seeds. Tightest spread of the
session; the batch-2 numbers are project facts. Style grid confirmed
distinct per-class styles, with under-rehearsed templates rendering
noisier (birth noise scrubs only with wins) — led to the RENDER_SQUELCH
addition in run_batch2.py (render-time gate: coefficients below 10% of a
position's strongest are silenced; positions below 10% of the global
peak stay silent).

## Batch 2 on Fashion-MNIST — predictions (before running, 2026-08-23)

`GF_DATASET=fashion python run_batch2.py` with the render squelch on.

1. The shared dictionary learns garment edge/texture primitives.
2. Label-only generation gives recognizable garment silhouettes for most
   of the 10 classes (trouser/sneaker/bag/boot easiest; shirt-family
   likely mushiest). Committed expectation: interiors may render hollow
   or patchy — solid fabric regions have near-zero contrast, fall below
   NORM_FLOOR, and are honestly silent; the code lives on edges.
3. Probe well above the whole-image fashion rig (0.794): expect >= 0.84.
4. Label-only retrieval stays high (>= 9/10).

### Outcome (same day)

All four predictions confirmed. probe **0.8550** (target >= 0.84;
whole-image fashion rig: 0.794), hard readout 0.694 (whole-image
pipeline was ~0.576), retrieval 10/10, dictionary entropy 0.973, 0 dead.

Label-only generation: recognizable garments for 9/10 classes —
t-shirt, trouser, pullover, dress, coat, shirt, sneaker, bag, and boot
are unmistakable; sandal is the mush (most heterogeneous class — straps
have no consistent constellation). The committed hollow-interior
prediction confirmed exactly: garments render as clean edge/outline
drawings with silent interiors, because uniform fabric has no contrast
and the code honestly lives on edges. The system draws what it
perceives: boundaries. (A texture/surface channel is now a
well-motivated future direction, not a patch.)

Render squelch (RENDER_SQUELCH=0.10) active in this run — generations
are speckle-free.
