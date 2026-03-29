# Zenith Ensemble — Pattern Completion Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality | 200 |
| Sparsity | 0.1 |
| Templates per node (k) | 10 |
| Stored patterns (N) | 40 |
| Load (N/k) | 4.0 |
| Ensemble nodes (M) | 6 |
| Learning rate | 0.03 |
| Training epochs | 1000 |
| Mask trials per pattern | 50 |

## Training Summary

- Final avg top-1 cosine similarity: **-0.0373**
- Final code uniqueness: **97.50%** (39/40 unique codes)

## Retrieval Results

| Mask % | Retrieval Acc | Cos Sim (masked vs full) | Avg Rank | Retention |
|--------|--------------|--------------------------|----------|-----------|
| 0% | 100.00% | 1.0000 | 0.0 | 100.00% |
| 10% | 100.00% | 0.9814 | 0.0 | 98.14% |
| 20% | 100.00% | 0.9609 | 0.0 | 96.09% |
| 40% | 99.75% | 0.9093 | 0.0 | 90.93% |
| 60% | 95.65% | 0.8375 | 0.1 | 83.75% |
| 80% | 83.70% | 0.7084 | 0.6 | 70.84% |

## Interpretation

- **Retrieval Acc**: fraction of masked trials where the correct pattern's full-input activation was the nearest neighbor (by cosine similarity).
- **Cos Sim**: average cosine similarity between masked-input activation and the same pattern's full-input activation. Higher = more stable representation.
- **Avg Rank**: where the correct pattern falls in the similarity ranking (0 = best). Lower is better.
- **Retention**: ratio of masked cos sim to full cos sim (mask=0%). Shows how much of the activation structure is preserved under masking.
