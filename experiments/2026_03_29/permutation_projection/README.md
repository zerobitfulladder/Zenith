# `permutation_projection/` — can the ensemble produce P(X) for a new X?

No writeup was kept for this experiment beyond the script's report. This
README is rebuilt from the docstring and [`results/report.md`](results/report.md).

## What it tests

Same hidden 128-bit permutation as [`../permutation_relation/`](../permutation_relation/README.md),
but now the answer must be generated, not just recognised. 100 hypercolumns of
10 templates, 300 training pairs, 600 epochs. Three methods on 100 unseen X:

- **source nearest neighbour**: return the stored P(X) of the closest training X.
- **direct**: train on [X, P(X)], show [Y, 0], and read the second half of the
  reconstruction (projection) as the answer.
- **two-stage**: first learn a code S(X) from [X, 0], then train a second
  ensemble on [S(X), P(X)] and read out from [S(Y), 0].

## Result

| method | Jaccard | precision |
|---|---|---|
| source nearest neighbour | 0.2261 | 0.3669 |
| direct | 0.3438 | 0.5054 |
| two-stage | 0.0764 | 0.1385 |

No method gets any target exactly right. Direct readout beats memory lookup by
0.12 Jaccard; putting a learned code in between destroys the information.
Figure: [`results/permutation_projection.png`](results/permutation_projection.png).

This direct model (M=100, k=10, 300 pairs, 600 epochs) is the starting point
of the August hidden-permutation arc in [`../../2026_08_05/`](../../2026_08_05/README.md).

## Running it

```
.venv/bin/python experiments/2026_03_29/permutation_projection/permutation_projection.py
```

Pure numpy on synthetic data; writes `results/report.md` and
`results/permutation_projection.png`. Eight of the nine August scripts import
this file as a module (`import permutation_projection as pp`) and reuse its
functions and settings.
