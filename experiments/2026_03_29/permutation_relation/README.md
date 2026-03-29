# `permutation_relation/` — can the ensemble recognise a hidden permutation?

No writeup was kept for this experiment beyond the script's report. This
README is rebuilt from the docstring and [`results/report.md`](results/report.md).

## What it tests

One fixed hidden permutation P of 128 bits. 30 hypercolumns of 10 templates
train only on 400 correct pairs [X, P(X)] (sparse X, 12 active bits). At test
time X is new, and the true P(X) must be ranked above 1, 3, 7 or 15 wrong
candidates. Three scores are compared: raw nearest neighbour on the pair,
nearest neighbour on the (normalised) joint code, and the direct template
score (mean top-1 correlation across hypercolumns, no lookup).

## Result (15 distractors, chance 6.25%)

| score | accuracy |
|---|---|
| raw pair nearest neighbour | 86.38% |
| code nearest neighbour | 4.17% |
| direct template score | 92.54% |

The direct template score beats raw similarity at every distractor count,
which the report takes as evidence that the templates hold something of the
permutation itself. The code nearest neighbour sits at chance (below it with 15 distractors); the
reason (the normalisation throws away the code's size, which carries the
signal) was found in August, in
[`../../2026_08_05/permutation_relation_scoring/`](../../2026_08_05/permutation_relation_scoring/README.md).
Figure: [`results/permutation_relation.png`](results/permutation_relation.png).

## Running it

```
.venv/bin/python experiments/2026_03_29/permutation_relation/permutation_relation.py
```

Pure numpy on synthetic data; writes `results/report.md` and
`results/permutation_relation.png`. The August script
`permutation_relation_scoring.py` imports this file as a module.
