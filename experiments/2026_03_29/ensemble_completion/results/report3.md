# Zenith Ensemble — Pattern Completion Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality | 200 |
| Sparsity | 0.05 |
| Templates per node (k) | 10 |
| Stored patterns (N) | 200 |
| Load (N/k) | 20.0 |
| Ensemble nodes (M) | 50 |
| Learning rate | 0.03 |
| Training epochs | 1000 |
| Mask trials per pattern | 50 |

## Training Summary

- Final avg top-1 cosine similarity: **-0.0132**
- Final code uniqueness: **100.00%** (200/200 unique codes)

## Retrieval Results

| Mask % | Retrieval Acc | Cos Sim (masked vs full) | Avg Rank | Retention |
|--------|--------------|--------------------------|----------|-----------|
| 0% | 100.00% | 1.0000 | 0.0 | 100.00% |
| 10% | 100.00% | 0.9752 | 0.0 | 97.52% |
| 20% | 99.96% | 0.9462 | 0.0 | 94.62% |
| 40% | 99.61% | 0.8751 | 0.0 | 87.51% |
| 60% | 90.91% | 0.7708 | 0.2 | 77.08% |
| 80% | 42.22% | 0.5996 | 2.7 | 59.96% |

## Interpretation

- **Retrieval Acc**: fraction of masked trials where the correct pattern's full-input activation was the nearest neighbor (by cosine similarity).
- **Cos Sim**: average cosine similarity between masked-input activation and the same pattern's full-input activation. Higher = more stable representation.
- **Avg Rank**: where the correct pattern falls in the similarity ranking (0 = best). Lower is better.
- **Retention**: ratio of masked cos sim to full cos sim (mask=0%). Shows how much of the activation structure is preserved under masking.
