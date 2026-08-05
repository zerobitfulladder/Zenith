# `permutation_iterative_completion/` — feeding the answer back in

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

The ensemble verifies well but produces poorly (about 50% precision reading Y
out of the blended templates). Does iteration close the gap? One ensemble
(March's direct model: 100 hypercolumns, k=10, 300 pairs, 600 epochs), four
readouts: nearest training source; one-shot readout of [X, 0]; iterative
(feed [X, Y] back in, re-read Y, repeat until it stops changing); and a greedy
bit-swap search on Y that climbs the energy.

## Result

From [`results/report.md`](results/report.md) (Jaccard): nearest source
0.2261, one-shot **0.3362**, iterative 0.1957 (4.31 rounds to a fixed point on
average), energy search 0.1817. The search ends above the truth's energy on
100% of items (mean gap +0.9805): the verifier can be fooled.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_iterative_completion/permutation_iterative_completion.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md) and [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
