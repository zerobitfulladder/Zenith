# Zenith Ensemble — Generalization Test Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Total dimensionality | 300 |
| Signal dimensions | 100 |
| Nuisance dimensions | 200 |
| Signal sparsity | 0.1 |
| Nuisance sparsity | 0.2 |
| Categories | 20 |
| Train per category | 15 |
| Test per category | 10 |
| Templates per node (k) | 10 |
| Ensemble nodes (M) | 30 |
| Load (N_train/k) | 30.0 |
| Learning rate | 0.03 |
| Training epochs | 1000 |

## Data Design

Each pattern consists of two parts:
- **Signal** (first 100 dims): sparse binary pattern shared by all exemplars in the same category. Defines category identity.
- **Nuisance** (last 200 dims): sparse binary pattern randomized independently per exemplar. Creates misleading raw similarity.

Training uses one set of exemplars. Testing uses held-out exemplars with the same signal bits but completely new nuisance bits.

## Generalization Test (no masking)

| Method | Category Accuracy |
|--------|-------------------|
| Raw input NN | 100.00% |
| Ensemble activation NN | 100.00% |
| **Gap (ensemble - raw)** | **+0.00%** |

Ensemble and raw similarity perform similarly — the representation neither helps nor hurts category retrieval.

## Generalization Under Masking

| Mask % | Ensemble Acc | Raw NN Acc | Gap |
|--------|-------------|------------|-----|
| 0% | 100.00% | 100.00% | +0.00% |
| 20% | 99.95% | 97.78% | +2.17% |
| 40% | 98.83% | 89.78% | +9.05% |
| 60% | 92.83% | 77.65% | +15.17% |
| 80% | 69.92% | 48.05% | +21.88% |

## Interpretation

- **Positive gap** = generalization. The ensemble learned structure that raw similarity misses.
- **Zero gap** = the ensemble merely reflects input similarity (no generalization).
- **Negative gap** = the ensemble representation is worse than raw input for this task (possibly overtrained or too few nodes).

The key question: does the gap grow as nuisance increases relative to signal? If so, the ensemble is genuinely filtering noise.
