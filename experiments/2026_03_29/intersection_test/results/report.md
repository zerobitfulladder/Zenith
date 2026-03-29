# Zenith Ensemble — Intersection (AND) Learning Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Part dimensionality | 100 |
| Total dimensionality | 300 |
| Sparsity | 0.2 |
| Active bits per part | 20 |
| Expected \|A∩B\| | ~4.0 |
| Actual mean \|A∩B\| (train) | 4.0 |
| Training triplets | 500 |
| Test triplets | 200 |
| Templates per node (k) | 10 |
| Ensemble nodes (M) | 30 |
| Load (N_train/k) | 50 |
| Learning rate | 0.03 |
| Training epochs | 1000 |

## Task

Train on [A, B, A∩B] triplets. Test: present [X, Y, 0] with never-seen X, Y.
Retrieve nearest stored pattern, take its C part. Compare to true X∩Y.

## Results

| Method | Jaccard | Precision | Recall | Exact Match |
|--------|---------|-----------|--------|-------------|
| Raw NN | 0.0718 | 0.1471 | 0.1171 | 0.00% |
| Ensemble NN | 0.0333 | 0.0777 | 0.0457 | 0.00% |
| **Gap** | **-0.0386** | **-0.0694** | **-0.0714** | **+0.00%** |

## Interpretation

- **Jaccard**: |retrieved ∩ true| / |retrieved ∪ true|. 1.0 = perfect reconstruction.
- **Precision**: of the bits the retrieved C has active, how many are correct?
- **Recall**: of the true intersection bits, how many were retrieved?
- **Exact match**: fraction where retrieved C is bit-for-bit identical to true X∩Y.

Raw NN outperforms the ensemble. The learned representation does not help with AND reconstruction — the ensemble's partition structure doesn't align with intersection computation.

### Important caveat

Even if the ensemble outperforms raw NN, this doesn't necessarily mean it 
"learned AND". Both methods retrieve the C part of the nearest stored 
training pattern. The question is whether the ensemble's activation space 
groups training patterns in a way that makes the nearest neighbor's C part 
a better approximation of the true X∩Y than raw similarity provides.

True AND computation would require outputting X∩Y for inputs far from any 
training pair — which nearest-neighbor retrieval fundamentally cannot do.
