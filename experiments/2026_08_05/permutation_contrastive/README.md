# `permutation_contrastive/` — mismatched pairs as negatives

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

Training only ever raised the energy of correct pairs. Add a negative phase:
for each [X, P(X)] also show [X, P(X')] (someone else's answer) and apply the
same rule with the rotation reversed, so the winning template turns away.
Measured at negative scale 0.0 (control) and 0.5.

## Result

From [`results/report.md`](results/report.md):

| neg scale | ranking | one-shot J | iterative J | energy J | search beats truth |
|---|---|---|---|---|---|
| 0.0 | 97.70% | 0.3362 | 0.1957 | 0.1817 | 100% |
| 0.5 | 99.40% | 0.3880 | 0.2241 | 0.2090 | 100% |

Negatives help recognition and one-shot readout, but the search still finds
fake peaks everywhere.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_contrastive/permutation_contrastive.py 0.0 0.5
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md) and [`permutation_iterative_completion.py`](../permutation_iterative_completion/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
