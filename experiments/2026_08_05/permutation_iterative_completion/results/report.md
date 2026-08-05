# Zenith — Iterative Completion of the Hidden Permutation

Config identical to permutation_projection.py direct model: M=100, k=10, N_train=300, epochs=600.

| Method | Jaccard | Precision | Recall | Exact |
|---|---|---|---|---|
| source_nn | 0.2261 | 0.3669 | 0.3669 | 0.00% |
| one_shot | 0.3362 | 0.4977 | 0.4977 | 0.00% |
| iterative | 0.1957 | 0.3231 | 0.3231 | 0.00% |
| energy | 0.1817 | 0.3023 | 0.3023 | 0.00% |

Iterative loop: mean iterations to fixed point = 4.31 (max allowed 20).
Energy search final energy minus true-target energy: mean +0.9805 (>0 means search found a state the verifier likes MORE than the truth: 100% of items).
