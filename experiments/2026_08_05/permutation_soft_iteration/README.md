# `permutation_soft_iteration/` — refining a population without a verifier

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

A scheme proposed by Lavender: read a Y-score distribution from [X, 0], sample
100 candidates, feed each [X, candidate] back through the ensemble, average the
100 regenerated Y-score profiles into the next distribution, repeat for 6
rounds. No energy ranking anywhere. Near-negative model; T=0.5, T=1.0 and an
annealed schedule.

## Result

From [`results/report.md`](results/report.md): decode Jaccard falls every
round (T=0.5: 0.2725 → 0.1840; T=1.0: 0.2830 → 0.2144; annealed: 0.2830 →
0.1836), the best sample in the pool falls as well (T=0.5: 0.4999 → 0.2975), and the pool grows unanimous
(annealed: 0.8194 → 0.9553). Anchors: one-shot 0.3995, hard iteration 0.196.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_soft_iteration/permutation_soft_iteration.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md), [`permutation_hard_negatives.py`](../permutation_hard_negatives/README.md) and [`permutation_propose_verify.py`](../permutation_propose_verify/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
