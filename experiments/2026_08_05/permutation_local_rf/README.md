# `permutation_local_rf/` — each hypercolumn sees only a window

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

Global templates blend about 30 whole pairs each. The rule is 128 separate
wire facts (x_i goes to y_pi(i)), which whole-pattern recall cannot mix. So
each hypercolumn sees only F bits (F/2 random X bits + F/2 random Y bits), with
the number of hypercolumns M set so every wire is covered about 5 times:
F=16 / M=1280, F=32 / M=320, F=64 / M=80. Plain training, no negatives. The
answer is assembled from many local readouts.

## Result

From [`results/report.md`](results/report.md):

| F | M | true wire templates (null) | one-shot J | far-field ranking | truth percentile (median) |
|---|---|---|---|---|---|
| 16 | 1280 | 4.4% (0.8%) | 0.6611 | 29.0% | 29.1% |
| 32 | 320 | 17.0% (0.8%) | 0.5879 | 62.0% | 7.1% |
| 64 | 80 | 60.8% (0.4%) | 0.6411 | 100.0% | 21.6% |

Global anchors: one-shot 0.336, far-field 97.7%, truth percentile 0.7%. Small
windows generate, large windows verify; the truth still never wins the
near-field ranking (0%).

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_local_rf/permutation_local_rf.py
```

Pure numpy on synthetic data; writes `results/report.md`. It imports March's
[`permutation_projection.py`](../../2026_03_29/permutation_projection/README.md), [`permutation_relation_scoring.py`](../permutation_relation_scoring/README.md) and [`permutation_propose_verify.py`](../permutation_propose_verify/README.md) as
modules (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, `permutation_relation_scoring.py` stubs them out (see the day README).
