# `wide3/` — wide-shallow 3-level faces rig (64/256/256) + showcase and attribute probes

(2026-08-23/24, from the day log.) The rig is the pyramid code (`../pyramid/run_pyramid.py`,
`../pyramid/run_pyramid_eval.py`) with three levels; `results/report.md` is that eval's output (it keeps the
pyramid's header). `run_wide3_showcase.py` sets the same level settings
(`GF_LEVELS=[[4,1,64],[5,2,256],[5,1,256]]`, `GF_KTOP2=1000`, `GF_OUT=wide3/results`) and makes the
showcase figures from the saved weights. Results: `results/` (weights, roundtrip, unit galleries,
showcase figures, `classification_report.md`).

## Wide-3 rig — first attempt status (2026-08-23, late)

Config B (64/256/256, archive 1000 x 74k = 76M params, grids 45/21/17)
launched with an 18-min time budget. Estimate was 4x optimistic: epochs
cost ~790s, budget bought only 2 — L2/L3 drift still 0.38/0.89 (wet).
Eval OOM'd on L3 window tensors. Weights saved (609MB).

ROOT CAUSE, bigger than this run: the whole GPU trainer runs float64,
and GeForce executes f64 at 1/32 of f32 throughput.

FIX DONE + VERIFIED: float32 trainer reproduces the f64 MNIST flagship
within ±0.001 on probes (hard +0.01) and delivers **6x** on the wide
config (790s -> ~130s/epoch). Permanent speedup for all future runs.

RETRAIN VERDICT (8 epochs, 17.5 min, chunked eval clean):
- Male-probes L2 0.886 / L3 0.876 — L2 BEATS the deep pyramid's best
  level (0.878): wide-shallow wins the representation contest at
  matched data.
- Roundtrip: every reconstruction is a FACE with real SHADING — cheeks,
  eye sockets, tonal hair mass — the luminance channel working end to
  end for the first time. Massive improvement over the pyramid's
  smudges; instance identity is softer than the sketch-style 4-layer
  recon (surface fidelity gained, structure-tracing fidelity partly
  traded — different axes).
- Archive sample: shaded prototype faces, but the top-rehearsed 12 are
  notably HOMOGENEOUS (near-identical frontal faces). Open question:
  modal-cluster sampling artifact vs archive-level content collapse
  (the archive learns from dense graded codes — the collapse mechanism
  could apply one level up) vs still-wet L3 (drift 0.30 at cutoff)
  destabilizing the archive's input. Diagnosis path: render LOW-win
  memories, an archive peakiness analog, L3-only continuation.

## Wide-3 showcase + classification report (2026-08-24, small hours)

`run_wide3_showcase.py` -> results/: portraits_gated,
archive_lowwin, sampled_variety, morphs + classification_report.md.

- CLASSIFICATION (40 probes on L3 code, 4000/2000): every near-balanced
  attribute learned strongly — Smiling +0.36 over baseline (F1 .874),
  Lipstick +0.34, Male +0.33 (.910), Mouth_Open +0.29, Cheekbones +0.26,
  Makeup +0.22, and (prediction miss) Attractive +0.21. Rare attributes:
  ceiling-capped gains, respectable recall (Eyeglasses F1 .655 at 7%
  base). Mean gain +0.05 misleads: it averages in the capped rare tail.
- ARCHIVE MYSTERY RESOLVED (archive_lowwin): quality/diversity is a
  GRADIENT BY REHEARSAL — top-win memories are homogeneous because they
  are the modal cluster; median-win (~50) memories show real variety;
  low-win (4-7) are blotchy proto-faces. With clean births, the low-win
  degradation is a NONSTATIONARITY artifact: they were adopted in epoch
  1 against still-wet dictionary codes and the code distribution moved
  out from under them. Fix ideas queued: delayed archive bootstrap, or
  re-adoption of starved memories. Rehearsal-gated retrieval (already
  standard) remains the right practice.
- MORPHS: continuous code-space interpolation between memories renders
  smooth face-to-face transitions (hair recedes, features shift) with
  minor artifacts at extremes — a new capability: continuous paths
  through face space, no training.
