# `pattern_completion/` — one hypercolumn as a memory for sparse patterns

No writeup was kept in this folder; the findings are in
[`../ensemble_completion/EnsemblePatternCompletion.md`](../ensemble_completion/EnsemblePatternCompletion.md)
("Single-Node Pattern Completion"). This README points there.

## What it tests

One hypercolumn of k templates (a "node" in the script; top-1: only the
winning template learns, rotated on the unit sphere) is trained on N random
sparse binary patterns of 200 bits. Then a fraction of each pattern's active
bits is zeroed and the question is whether the right template still wins.
The script sweeps k (10, 25, 50), N (5, 10, 20, 40), sparsity (5%, 10%, 20%)
and the masked fraction (0-80%), and draws everything into two figures:
[`results/pattern_completion.png`](results/pattern_completion.png) and
[`results/capacity_heatmaps.png`](results/capacity_heatmaps.png).

## What came out (from the note)

- With few patterns per template (load N/k about 0.2) retrieval stays perfect
  even with 80% of the bits masked.
- As the load approaches 1 and beyond, accuracy drops; at load 2.0 and 80%
  masking it is about 61%.
- Sparser patterns (5%) hold up better than denser ones (20%).
- Retention (masked similarity / full similarity) is the fair measure: a drop
  from 0.8 to 0.6 is not a drop from 1.0 to 0.6.

## Running it

```
.venv/bin/python experiments/2026_03_29/pattern_completion/pattern_completion.py
```

Pure numpy on synthetic data; writes the two figures into `results/`.
