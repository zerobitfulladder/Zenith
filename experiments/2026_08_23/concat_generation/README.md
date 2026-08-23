# `concat_generation/` — label as evidence: a free-competition top layer over [code ; label]

(2026-08-23, from the day log.) Script: `run_concat_generation.py`. Results: `results/` (generation grids per lam, `report.md`).

## Concat generation test — predictions (before running, 2026-08-23)

`run_concat_generation.py`: L2 becomes a free-competition Zenith (K2=50)
over [L1 code ; lam * onehot] — label as evidence, association learned,
allocation free. Generation queried with the label alone (code half
zeroed), lam in {0.25, 0.5, 1.0}. MNIST.

1. Label-only generation produces recognizable digits at some lam — the
   March single-layer result survives one level up (query completes from
   the label half, code half read out through W1).
2. lam large -> the one-template-per-digit geometry returns (label
   dominates, zero within-class variance): ~1 template per class,
   generation ~= the selector baseline's class blends. lam small -> more
   templates per class, but label-only retrieval risks failing (label too
   quiet after normalization). Sweet spot in between.
3. Classification without labels (winner's stored label half) lands near
   the selector pipeline's ~0.70 — ranking preserved without the label,
   as the March notes claimed.

### Outcome (same day)

- Prediction 1 confirmed: label-only queries retrieve the right class
  100% of the time at every lam, and the read-out code half renders
  recognizable digits at lam=0.5 and 1.0. Generation survives
  concatenation one level up from pixels.
- Prediction 2 wrong, instructively: one-template-per-digit did NOT
  return even at lam=1.0 — allocation stayed at 4-10 templates per class.
  Likely cause: bootstrap init adopts 50 real (code+label) samples, so
  allocation *starts* multi-template-per-class and top-1 competition
  maintains it; the March collapse arose when allocation had to emerge by
  drift. Bootstrap init changes allocation dynamics, not just stability.
- Prediction 3 exceeded: label-free classification via the winner's
  stored label half reaches 0.727 at lam=1.0 — *above* the wired
  selector pipeline (0.699) — while also providing generation. Free
  allocation + learned association beats one-blended-prototype-per-class
  on both jobs.
- Caveat: the gen-corr metric (vs class means) favors blends by
  construction — the selector's 0.698 vs concat's 0.532 reflects that
  bias; judge generation by the grids.
