# `permutation_consensus/` — voting over the top candidates

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

If the verifier is only noisy, not wrong, a per-bit majority vote over the
top-M candidates should keep the shared (true) bits and cancel private errors.
Near-negative model, pools of 1000 at T=0.5. Decodes: argmax, vote over the
top 10/25/50/100, energy-weighted average, and the plain pool average as a
control.

## Result

From [`results/report.md`](results/report.md): the truth's energy sits at the
0.7th percentile (median; mean 3.1%) of its pool, so the near field is inverted,
not noisy. Jaccard: one-shot 0.3995, pool average 0.3972, best vote (top 100)
0.3518, energy-weighted 0.3088, argmax 0.2810.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_consensus/permutation_consensus.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md), [`permutation_hard_negatives.py`](../permutation_hard_negatives/README.md), [`permutation_iterative_completion.py`](../permutation_iterative_completion/README.md) and [`permutation_propose_verify.py`](../permutation_propose_verify/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
