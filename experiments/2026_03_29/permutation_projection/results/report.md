# Zenith — Fixed Permutation Completion via Projection Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Source dimensionality | 128 |
| Target dimensionality | 128 |
| Source/target sparsity | 0.1 |
| Active bits per vector | 12 |
| Training examples | 300 |
| Test examples | 100 |
| Stage-1 nodes | 100 |
| Stage-1 templates per node | 10 |
| Stage-1 code dimensionality | 1000 |
| Stage-2 nodes | 100 |
| Stage-2 templates per node | 10 |
| Direct-model nodes | 100 |
| Direct-model templates per node | 10 |
| Learning rate | 0.03 |
| Training epochs | 600 |

## Task

One fixed hidden permutation P is used for all examples.
Training sources X are random sparse vectors. Targets are P(X).
The test sources Y are unseen sparse vectors; the goal is to reconstruct P(Y).

## Methods

- **Source NN**: nearest neighbor in source space, return that training example's P(X).
- **Direct**: train Zenith on [X, P(X)], then test with [Y, 0] and reconstruct by projection.
- **Two-stage**: train Stage 1 on [X, 0] to get S(X); train Stage 2 on [S(X), P(X)]; test with [S(Y), 0] and reconstruct by projection.

## Results

| Method | Jaccard | Precision | Recall | Exact Match | Avg Response |
|--------|---------|-----------|--------|-------------|--------------|
| Source NN | 0.2261 | 0.3669 | 0.3669 | 0.00% | 0.2954 |
| Direct | 0.3438 | 0.5054 | 0.5054 | 0.00% | 0.1159 |
| Two-stage | 0.0764 | 0.1385 | 0.1385 | 0.00% | 0.2421 |

## Interpretation

- Two-stage minus direct (Jaccard): **-0.2674**
- Two-stage minus source NN (Jaccard): **-0.1498**
- Direct minus source NN (Jaccard): **+0.1176**

- **Jaccard** measures overlap between predicted and true active target bits.
- **Precision** asks: of the predicted active target bits, how many are correct?
- **Recall** asks: of the true target bits, how many were recovered?
- **Exact match** is the fraction of targets reconstructed perfectly.

The stage-1 code hurts or discards useful information. Direct completion from [Y, 0] works better than reconstructing through S(Y).

### Caveat

This is still prototype completion, not symbolic rule execution.
A good result means the learned code and templates support completion of a fixed transformation on unseen sparse vectors.
