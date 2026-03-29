# Zenith Ensemble — Pattern Completion Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality | 200 |
| Sparsity | 0.05 |
| Templates per node (k) | 10 |
| Stored patterns (N) | 40 |
| Load (N/k) | 4.0 |
| Ensemble nodes (M) | 50 |
| Learning rate | 0.03 |
| Training epochs | 1000 |
| Mask trials per pattern | 50 |

## Training Summary

- Final avg top-1 cosine similarity: **-0.0107**
- Final code uniqueness: **100.00%** (40/40 unique codes)

## Retrieval Results

| Mask % | Retrieval Acc | Cos Sim (masked vs full) | Avg Rank | Retention |
|--------|--------------|--------------------------|----------|-----------|
| 0% | 100.00% | 1.0000 | 0.0 | 100.00% |
| 10% | 100.00% | 0.9902 | 0.0 | 99.02% |
| 20% | 100.00% | 0.9781 | 0.0 | 97.81% |
| 40% | 100.00% | 0.9458 | 0.0 | 94.58% |
| 60% | 99.90% | 0.8897 | 0.0 | 88.97% |
| 80% | 95.25% | 0.7759 | 0.1 | 77.59% |

## Interpretation

- **Retrieval Acc**: fraction of masked trials where the correct pattern's full-input activation was the nearest neighbor (by cosine similarity).
- **Cos Sim**: average cosine similarity between masked-input activation and the same pattern's full-input activation. Higher = more stable representation.
- **Avg Rank**: where the correct pattern falls in the similarity ranking (0 = best). Lower is better.
- **Retention**: ratio of masked cos sim to full cos sim (mask=0%). Shows how much of the activation structure is preserved under masking.
