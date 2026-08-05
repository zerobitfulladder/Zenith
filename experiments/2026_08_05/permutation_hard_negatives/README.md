# `permutation_hard_negatives/` — near-miss negatives

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

The verifier fails in the near field: lures a few bits away from the truth
outscore it. So train with negatives at exactly that distance: the item's own
correct pair with 1-3 random Y bits swapped, reversed rotation. Modes: `near`
(all near-miss) and `mixed` (half near-miss, half mismatched).

## Result

From [`results/report.md`](results/report.md): `near` gives ranking 98.60% and
one-shot **0.3995** (the best global model of the arc), `mixed` 99.30% and
0.3780. In both, the search still beats the truth on 100% of items and the true
answer never wins the propose-and-verify ranking (0%).

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_hard_negatives/permutation_hard_negatives.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md), [`permutation_contrastive.py`](../permutation_contrastive/README.md), [`permutation_iterative_completion.py`](../permutation_iterative_completion/README.md) and [`permutation_propose_verify.py`](../permutation_propose_verify/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
