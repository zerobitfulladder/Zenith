# `generalization_test/` — does the ensemble code see through nuisance bits?

No writeup was kept for this experiment beyond the script's reports. This
README is rebuilt from the docstring and [`results/report.md`](results/report.md).

## What it tests

Each pattern is 100 "signal" bits shared by every member of one of 20
categories, plus 200 "nuisance" bits drawn fresh for every exemplar. 30
hypercolumns of 10 templates train on 15 exemplars per category; testing uses
10 new exemplars per category with brand-new nuisance bits. Category is read
by nearest neighbour, once on the hypercolumns' joint code and once on the raw
input. If the code beats the raw input, it has picked up the category
structure rather than mirroring raw similarity. The same comparison is repeated
with part of the input masked.

## Results

Two runs were kept: [`results/report1.md`](results/report1.md) (nuisance
sparsity 0.05) and [`results/report.md`](results/report.md) (nuisance
sparsity 0.2, the later one; its figure is
[`results/generalization.png`](results/generalization.png)).

Unmasked, both methods get 100% in both runs. Under masking the code pulls
ahead, and more so with denser nuisance:

| masked | nuisance 0.05: code / raw / gap | nuisance 0.2: code / raw / gap |
|---|---|---|
| 40% | 99.88% / 99.80% / +0.07 | 98.83% / 89.78% / +9.05 |
| 60% | 97.40% / 93.62% / +3.77 | 92.83% / 77.65% / +15.17 |
| 80% | 77.05% / 61.82% / +15.22 | 69.92% / 48.05% / +21.88 |

## Running it

```
.venv/bin/python experiments/2026_03_29/generalization_test/generalization_test.py
```

Pure numpy on synthetic data; writes `results/report.md` and
`results/generalization.png`.
