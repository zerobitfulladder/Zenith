# `permutation_dreams/` — the model's own fake peaks as negatives

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

Mismatched pairs never visit the states the search exploits. So take the
negatives from the model itself: 6 rounds of (wake) train on correct pairs plus
the current negative pool, then (dream) run the energy search with the current
model and collect the fake peaks it finds as the next pool. Same reversed
rotation for unlearning.

## Result

From [`results/report.md`](results/report.md): the search beat the truth on
100% of items in every one of the 6 rounds. Final: ranking 98.70%, one-shot
0.3639, iterative 0.1963, energy 0.1809, gap +0.9345. The arc calls it
whack-a-mole: point negatives cannot cover the space of wrong answers.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_dreams/permutation_dreams.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md), [`permutation_contrastive.py`](../permutation_contrastive/README.md) and [`permutation_iterative_completion.py`](../permutation_iterative_completion/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
