# `rebuild/` — 7x7 grid of patch Hypercolumns + concat top layer

(2026-08-23, from the day log.) Script: `run_rebuild.py`. Results: `results/first/` (the first rebuild run, written by an earlier
version of the script), `results/batch1/` (batch-1 fixes, K2=100), `results/batch1_k200|k400|k800/`
(the K2 capacity sweep, `GF_K2=...`).

## Rebuild — predictions (before running, 2026-08-23)

`run_rebuild.py`: the architecture the whole discussion pointed at.
L1 = 7x7 grid of independent Zenith nodes, each seeing one non-overlapping
4x4 patch (25 templates each, top-1, bootstrap init). Code = concatenated
per-position top-1 activations (49 nonzero of 1225). L2 = free-allocation
Zenith (K2=100) over [code ; lam * onehot], lam in {0.5, 1.0}. MNIST,
no feedback anywhere (that chapter is closed).

1. L1 templates look like strokes/local features, not digits (the
   small-window claim, checked visually).
2. L2 allocates multiple templates per class at BOTH lams — the sparse
   code carries within-class variation, so label dominance can't collapse
   allocation ("sparsity does what labels couldn't", March, re-tested).
3. Label-only compositional generation renders recognizable digits from
   stroke constellations — parts + arrangement, generatively.
4. Classification without labels is at least comparable to the
   whole-image rig (selector 0.699 / concat 0.727); the code probe vs the
   whole-image probe (0.871) is left open — sparsity may cost pixel
   detail or may pay in class structure.
5. Sanity: per-position usage stays spread (no within-node monopolies;
   patch space is far more isotropic than digit space).

### Outcome (same day)

Four of five predictions confirmed; the fifth split in an instructive way.

- P1 ✓ (visual, unambiguous): the 4x4 templates are oriented edges,
  bars, corners — strokes, zero mini-digits (`l1_templates_center.png`).
- P2 ✓: 4-15 templates per class at both lams; label dominance never
  collapses allocation over the sparse code.
- P3 ✓ — **first compositional generation in the project**: label-only
  queries render all ten digits recognizably from 49 independent stroke
  dictionaries (`generation_labelonly_lam0p5.png`). Grid seams and
  blank-corner artifacts present but identity is unmistakable.
- P4 split: the sparse local code is the **best representation measured
  all day** — probe 0.9170 vs 0.871 whole-image. But the end-to-end
  readout (nearest L2 template's stored label) lags: 0.655 at lam=0.5,
  0.489 at lam=1.0, vs 0.727 whole-image concat. The code got better;
  the single-winner readout wastes it (sparse constellations overlap
  partially, so the nearest stored sub-variant is often wrong-class;
  lam=1.0 templates also spend norm on label dims that are zeroed at
  test). Cheap in-architecture fix to try: soft label readout — sum all
  templates' label halves weighted by relu(correlation) instead of
  taking the single winner.
- P5 ✓: mean per-position usage entropy 0.944, no within-node
  monopolies — patch space is isotropic enough for plain WTA.

## Rebuild batch 1 — predictions (before running, 2026-08-23)

Three fixes to the rebuild rig, same seed/data, results in
`results/batch1/`: contrast floor (NORM_FLOOR=0.15 — blank
patches neither activate nor become templates), render-active-only (clip
the stored code's negative floor when drawing), soft label readout (all
L2 templates vote their label halves weighted by relu correlation).

1. Dotted-paper artifact and corner squares gone from generation figures.
2. Soft readout closes most of the probe-readout gap: well above the hard
   readout's 0.655, target >= the whole-image concat's 0.727.
3. Code probe stays >= 0.917 (floor removes noise dimensions).
4. Hard readout improves somewhat on cleaner codes.

### Outcome (same day)

- P1 ✓✓: dotted paper and corner squares gone; the cleanest, most
  legible generation grid the project has produced. Only per-patch
  brightness offsets remain (batch 2 overlap/feathering territory).
- P3 ✓: probe held (0.9146 vs 0.9170) — the floor removed noise, not
  information.
- P4 ✗: the contrast floor COST the hard readout (0.655 -> 0.621). The
  silenced faint patches carried weak-but-real signal that
  nearest-constellation matching depended on; the probe, which weighs
  evidence properly, barely noticed their loss. Cleanliness for
  generation, brittleness for matching — an unpredicted trade-off.
- P2 partial: the readout ladder at lam=0.5 — hard 0.621, soft 0.665
  (allocation-biased), class-normalized 0.672, top-10 voting 0.712.
  Each fix helped; the best is +5.7pp over the original readout but
  misses the 0.727 target and leaves a ~20pp gap to the probe.
  (lam=1.0 stays worse everywhere: templates spend norm on label dims
  that are zeroed at query time.)

Reading: the concat top layer is excellent as associative memory and
generator (100% label retrieval, best-yet generation) and mediocre as a
classifier — prototype matching over sparse codes saturates around
~0.71 while the code itself carries 0.91. Candidate explanations for a
later test: K2=100 cells is under-provisioned for a 1225-dim sparse
code (capacity sweep: does prototype readout approach the probe as K2
grows?); or classification simply wants a discriminatively learned
readout as its own organ — consistent with the day's theory (dictionary
by observation, task skill in the readout).

## K2 capacity sweep — predictions (before running, 2026-08-23)

Same batch-1 rig, lam=0.5 only, K2 in {100 (done), 200, 400, 800} via
`GF_K2=... GF_LAM=0.5 python run_rebuild.py`. The question: is the
readout ceiling (top-10 voting 0.712 vs probe 0.915) a capacity problem
or a wrong-tool problem?

1. Capacity story: top-10 readout climbs steadily with K2 toward the
   probe; allocation per class grows; generation stays good or sharpens
   (finer sub-variants).
2. Wrong-tool story: readout plateaus around 0.71-0.75 despite 8x the
   cells — prototype lookup is simply not a classifier, and a learned
   readout organ becomes the standing conclusion.
3. Watch: at K2=800 each template averages ~50 training hits (2 epochs,
   20k samples) — enough to refine but bootstrap eats the first 800
   samples; probe should stay ~0.915 throughout (K2 does not affect L1).

### Outcome (same day)

**Wrong-tool story wins.** Best readout (class-normalized voting) by K2:
100 -> 0.672, 200 -> 0.725, 400 -> 0.763, 800 -> 0.764. The first
doubling was real capacity relief; the curve then flattens hard — 8x the
cells recovered 5 of the 20 points to the probe, and the last doubling
bought +0.001. Probe stable at 0.915-0.916 throughout (sanity ✓);
label-only retrieval 100% at every K2.

Standing conclusion, now formalized: the concat top layer is a memory +
generation organ (perfect retrieval, best-yet generation, cheap at
K2~200); classification saturates in the mid-0.7s for the entire
prototype-lookup family regardless of capacity. Task skill belongs to a
separate learned readout — which can stay local and backprop-free (a
single delta-rule layer on the codes is exactly what the probe measures,
made into a living component). Batch 2 sizing: K2=200 suffices for the
top layer's actual jobs.
