# `permutation_relation_scoring/` — the code's size is the signal

Part of the hidden-permutation arc; the writeup is the day README
[`../README.md`](../README.md). This README is rebuilt from the script's
docstring and its generated report.

## What it tests

The March test [`permutation_relation`](../../2026_03_29/permutation_relation/README.md)
found that nearest-neighbour lookup on the ensemble's normalised code ranks the
true P(X) at chance. Hypothesis: the size (norm) of the code carries the "my
templates explain this pair" signal, and normalising deletes it. Same data,
seeds and trained ensemble; only the scoring changes:

- `raw_nn`: best cosine of the candidate pair to a training pair (control)
- `act_cos`: best cosine of the normalised code to stored codes (the March method)
- `act_dot`: the same without normalising the candidate's code
- `act_norm`: the size of the code alone, no lookup
- `template`: mean top-1 correlation across hypercolumns (control)

## Result

From [`results/scoring_variants.md`](results/scoring_variants.md), ranking the
truth against 15 distractors (chance 6.25%): `raw_nn` 84.54%, `act_cos` 4.17%,
`act_dot` 55.96%, `act_norm` **96.21%**, `template` 92.54%. The size of the
code alone is the best verifier. This is experiment 1 of the arc and the
"energy" used by every later one.

## Running it

```
.venv/bin/python experiments/2026_08_05/permutation_relation_scoring/permutation_relation_scoring.py
```

Pure numpy on synthetic data; writes `results/scoring_variants.md`. It imports March's
[`permutation_relation.py`](../../2026_03_29/permutation_relation/README.md) as a
module (the `sys.path` lines at the top find them). If tqdm or matplotlib are
missing, this script stubs them out; the other August scripts import it for
that reason (see the day README).
