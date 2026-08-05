# `permutation_propose_verify/` — sample candidates, let the energy pick

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

Never optimise the energy; only use it to rank. Per test item: read the Y
scores out of [X, 0], sample N sparse candidates from them at temperature T,
add the one-shot guess, and pick the candidate with the highest energy.
Diagnostics: the best candidate in the pool (what the proposer covers), how
often the pick is that best one, and whether the true answer, secretly added,
wins. Models: plain and contrastive (negative scale 0.5); T in 0.5 / 1 / 2,
N in 100 / 1000.

## Result

From [`results/report.md`](results/report.md): the pool contains good answers
(best 0.5757 for the contrastive model at T=0.5, N=1000), but the pick is never
better than one-shot (worse in 7 of the 12 cells, equal in the rest), the true
answer wins the ranking in 0% of items in all 12 cells, and N=1000 picks worse
than N=100 at T=0.5 and T=1.0 (equal at T=2.0).

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_propose_verify/permutation_propose_verify.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md), [`permutation_contrastive.py`](../permutation_contrastive/README.md) and [`permutation_iterative_completion.py`](../permutation_iterative_completion/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
