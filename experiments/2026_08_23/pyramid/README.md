# `pyramid/` — 8-level pyramid with a luminance channel

(2026-08-23, from the day log.) Scripts: `run_pyramid.py` (training + eval), `run_pyramid_eval.py` (chunked eval from saved
weights). Both write to `results/` by default; `GF_OUT` (relative to the day folder) redirects
them, which is how `../wide3/` was trained.

## 8-level pyramid + luminance — predictions (before running, 2026-08-23)

`run_pyramid.py`: 7 dictionaries (4x4s1/48, 3x3s2/64, 3x3s1/96,
3x3s2/128, 3x3s1/128, 3x3s2/160, 2x2s1/192 -> grids
45/22/20/9/7/3/2) + archive 1000 over [2x2x192 ; 40] — ~2M params total
(22x fewer than the 4-layer big run: deep downsampling shrinks the
archive's code). LUMINANCE CHANNEL: each L1 window appends its raw mean
brightness (LUM_W-scaled) as channel K1+1 — outside the Pearson metric
(triangle-safe), inside the message, stored by memories, re-lit at
render. Skeleton learning excludes the lum channel (learn structure,
speak shaded). EARLY STOP: per-epoch mean template drift angle; stop
when < 0.003 rad (max 12 epochs).

1. Early stopping triggers before the cap (drift decays monotonically).
2. The deep ladder continues: Male-probe at L6 >= L4.
3. NEW SCALES: L5 (20px RF) units show eye/mouth-scale structure; L6
   (34px) face regions — the first face-semantic parts.
4. THE realism check: luminance makes renders SHADED — light/dark
   regions on faces, visibly less pure-edge than the big run.
5. Risks on record: 7 hardened expansions may compound loss (roundtrip
   is the stress test); the 4-slot archive attachment may be brittle
   (MNIST's 16-slot lesson) — attach-point sweep is the fallback.

### Outcome (same night) — a discovery, a beauty, and the first hard wall

1,457,536 params total ✓ (35x leaner than the 4-layer big run). 12
epochs, 21 min; **early stop never fired — settling is a CASCADE**: L1
frozen by epoch 3 (drift 0.006), L2-L3 calm, L5-L7 still swinging
0.2-0.4 rad/epoch at cutoff. Convergence propagates upward; deep levels
need freeze-and-continue scheduling or more epochs.

- Male-probes L4-L7: 0.878/0.866/0.821/0.759 — the sag returns
  precisely in the UNSETTLED levels (P2 failed, mechanism attached:
  under-training, not depth per se).
- **units_L5.png is the night's beauty**: ~20px-RF units are large,
  smooth, SHADED curve-parts — brows, jaw-curves, shoulders, junctions
  — the luminance channel visibly working at mid levels. Diverse, zero
  collapse at level 5 of a 7-dictionary stack.
- **Roundtrip through all 7 levels FAILS** (near-empty smudges with a
  faint eye-line): the stress test lost — compounding hardened
  expansions + the 2x2x192 top + unsettled upper templates. First hard
  generative wall of the project.
- Follow-ups implied: (a) freeze L1-L3, continue training the top
  (tests the cascade hypothesis directly); (b) reconstruction depth
  profile — render roundtrips from EACH level's code to localize the
  break; (c) archive attachment at L5/L6 instead of L7.
