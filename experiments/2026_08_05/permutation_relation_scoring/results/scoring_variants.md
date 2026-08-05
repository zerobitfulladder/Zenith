# Zenith — Permutation Relation, Scoring Variants

Same data, seeds, and trained ensemble as `permutation_relation.py`.
Only the scoring of the activation code differs between variants.

| Distractors | Chance | raw_nn | act_cos | act_dot | act_norm | template |
|---|---|---|---|---|---|---|
| 1 | 50.00% | 98.58% | 53.75% | 92.71% | 99.79% | 99.58% |
| 3 | 25.00% | 96.04% | 25.17% | 82.04% | 99.33% | 98.67% |
| 7 | 12.50% | 90.88% | 12.33% | 69.50% | 98.46% | 96.50% |
| 15 | 6.25% | 84.54% | 4.17% | 55.96% | 96.21% | 92.54% |

`act_cos` is the original activation-space NN (normalized cosine).
`act_dot` keeps the activation's magnitude in the NN lookup.
`act_norm` is the magnitude alone — no lookup.
